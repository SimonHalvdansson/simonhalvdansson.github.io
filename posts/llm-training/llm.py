import os, math, time
from contextlib import nullcontext
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# caching, dataset
HF_HOME = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "hub"))

HF_DS_CACHE = os.environ["HF_DATASETS_CACHE"]
HF_HUB_CACHE = os.environ["TRANSFORMERS_CACHE"]
print(f"HF datasets cache: {HF_DS_CACHE}\nHF hub cache:      {HF_HUB_CACHE}")

# device stuff
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
is_cuda = (device == "cuda")
use_amp = is_cuda
amp_dtype = torch.bfloat16 if use_amp else None

# slices
train_split = "train[:5%]"
val_split   = "validation[:2%]"

PACKED_TRAIN_PATH = "tinystories_train_packed.pt"
PACKED_VAL_PATH   = "tinystories_val_packed.pt"

# global hyperparams
context_len   = 256
batch_size   = 16

# --------------------------
# Model
# --------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_drop = nn.Dropout(dropout)
        self.attn_p = dropout

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, context_len, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.context_len = context_len
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(context_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.context_len
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# --------------------------
# Data helpers
# --------------------------
def to_packed_ids(tok_ds, context_len):
    ids_list = tok_ds["input_ids"]
    total_len = sum(len(a) for a in ids_list)
    ids = torch.empty(total_len, dtype=torch.long)
    pos = 0
    for arr in ids_list:
        n = len(arr)
        ids[pos:pos+n] = torch.as_tensor(arr, dtype=torch.long)
        pos += n
    n = (len(ids) - 1) // context_len
    if n <= 0:
        raise RuntimeError("Not enough tokens; enlarge dataset or reduce context_len.")
    x = ids[: n * context_len].view(n, context_len)
    y = ids[1 : n * context_len + 1].view(n, context_len)
    return x, y

class PackedDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return self.x.size(0)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

def build_loaders(x_train, y_train, x_val, y_val):
    train_ds = PackedDataset(x_train, y_train)
    val_ds   = PackedDataset(x_val,   y_val)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=is_cuda, persistent_workers=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=is_cuda, persistent_workers=False, drop_last=False,
    )
    return train_loader, val_loader

def load_or_build_packed(context_len):
    print("Loading TinyStories… (will reuse local cache if present)")
    ds_train = load_dataset(
        "roneneldan/TinyStories",
        split=train_split, cache_dir=HF_DS_CACHE,
        download_mode="reuse_cache_if_exists",
    )
    ds_val = load_dataset(
        "roneneldan/TinyStories",
        split=val_split, cache_dir=HF_DS_CACHE,
        download_mode="reuse_cache_if_exists",
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=HF_HUB_CACHE)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e9)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], add_special_tokens=True, truncation=False)

    print("Tokenizing (single-process; safer on Windows/Spyder)...")
    tok_train = ds_train.map(
        tokenize_batch, batched=True,
        remove_columns=ds_train.column_names
    )
    tok_val = ds_val.map(
        tokenize_batch, batched=True,
        remove_columns=ds_val.column_names
    )

    if os.path.exists(PACKED_TRAIN_PATH) and os.path.exists(PACKED_VAL_PATH):
        pk_train = torch.load(PACKED_TRAIN_PATH, map_location="cpu")
        pk_val   = torch.load(PACKED_VAL_PATH,   map_location="cpu")
        x_train, y_train = pk_train["x"], pk_train["y"]
        x_val,   y_val   = pk_val["x"],   pk_val["y"]
    else:
        x_train, y_train = to_packed_ids(tok_train, context_len)
        x_val,   y_val   = to_packed_ids(tok_val,   context_len)
        torch.save({"x": x_train, "y": y_train}, PACKED_TRAIN_PATH)
        torch.save({"x": x_val,   "y": y_val},   PACKED_VAL_PATH)

    if is_cuda:
        x_train = x_train.pin_memory(); y_train = y_train.pin_memory()
        x_val   = x_val.pin_memory();   y_val   = y_val.pin_memory()

    vocab_size = tokenizer.vocab_size
    return x_train, y_train, x_val, y_val, vocab_size

# --------------------------
# Train/Eval
# --------------------------
def make_autocast_and_scaler():
    autocast_ctx = (lambda: torch.amp.autocast(device_type=device, dtype=amp_dtype)) if use_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    return autocast_ctx, scaler

def run_epoch(model, loader, args, optimizer, train=True,
              autocast_ctx=nullcontext, scaler=None,
              deadline=None, allow_early_stop=False):
    """
    Returns: (avg_loss, total_tokens, stopped_early)
    - For TRAIN: if allow_early_stop and deadline is reached mid-epoch, stop early.
    - For EVAL: deadline is ignored; we always finish the loader.
    """
    model.train(train)
    total_loss_sum, total_tokens = 0.0, 0
    stopped_early = False

    # small rolling window tok/s meter
    window_s = 2.0
    t_stamps = deque()
    tok_hist = deque()
    toks_in_window = 0

    pbar = tqdm(
        loader, leave=False, desc="train" if train else "eval",
        dynamic_ncols=True, mininterval=0.1, maxinterval=1.0, miniters=1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    first_draw = True

    for x, y in pbar:
        x = x.to(device, non_blocking=is_cuda)
        y = y.to(device, non_blocking=is_cuda)

        with (autocast_ctx() if use_amp else nullcontext()):
            logits = model(x)
            vocab_size = model.head.out_features
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if args["grad_clip"] is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args["grad_clip"] is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                optimizer.step()

        if is_cuda:
            torch.cuda.synchronize()

        toks = x.numel()
        t_now = time.perf_counter()

        # throughput
        t_stamps.append(t_now); tok_hist.append(toks); toks_in_window += toks
        while t_stamps and (t_now - t_stamps[0]) > window_s:
            t_stamps.popleft(); toks_in_window -= tok_hist.popleft()
        window_dt = max(t_now - (t_stamps[0] if t_stamps else t_now), 1e-9)
        tok_s = toks_in_window / window_dt

        total_loss_sum += loss.item() * toks
        total_tokens += toks

        pbar.set_postfix(loss=f"{loss.item():.3f}", tok_s=f"{tok_s:,.0f}")
        if first_draw:
            pbar.refresh()
            first_draw = False

        if train and allow_early_stop and (deadline is not None) and (t_now >= deadline):
            stopped_early = True
            break

    avg_loss = total_loss_sum / max(total_tokens, 1)
    return avg_loss, total_tokens, stopped_early

def train_limited_time(model, train_loader, val_loader, lr, time_limit_s, args):
    """
    Train for up to `time_limit_s` seconds (early-stopping inside epoch if needed),
    then ALWAYS do a full validation pass.
    Returns: (val_loss, tokens_this_run)
    """
    autocast_ctx, scaler = make_autocast_and_scaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args["weight_decay"])

    start = time.perf_counter()
    deadline = start + time_limit_s
    total_train_tokens = 0

    # Loop across as many epochs as fit into the time budget
    stopped = False
    while True:
        # If we’re at/over time before starting a new epoch, stop training
        if time.perf_counter() >= deadline:
            break
        train_loss, train_toks, stopped_early = run_epoch(
            model, train_loader, args=args, optimizer=optimizer, train=True,
            autocast_ctx=autocast_ctx, scaler=scaler,
            deadline=deadline, allow_early_stop=True
        )
        total_train_tokens += train_toks
        if stopped_early:
            stopped = True
            break
        # otherwise continue to next epoch loop

    # Validation: always full pass, regardless of time
    val_loss, val_toks, _ = run_epoch(
        model, val_loader, optimizer=None, train=False,
        autocast_ctx=autocast_ctx, scaler=None,
        deadline=None, allow_early_stop=False
    )

    print(f"Train_tokens={total_train_tokens:,} | "
          f"Val_bpc={val_loss*math.log2(math.e):.3f} | "
          f"Val_ppl={math.exp(val_loss):.2f} | "
          f"{'stopped early' if stopped else 'clean epoch end'}")

    tokens_this_run = total_train_tokens + val_toks
    return val_loss, tokens_this_run

def test_setup(args, train_loader, val_loader, n_runs=15, per_run_seconds=300):
    """
    Runs `n_runs` independent trainings. Each run:
      - builds a FRESH model via `make_model_fn()`
      - trains for up to `per_run_seconds`
      - then runs full validation
    Returns:
      - val_losses: np.ndarray shape (n_runs,)
      - tokens_per_run: np.ndarray shape (n_runs,)
    """
    def make_model():
        return TinyGPT(vocab_size, context_len, args["n_layer"], args["n_head"], args["n_embd"], args["dropout"])
    
    vals, toks = [], []
    for i in range(n_runs):
        print(f"\n=== Run {i+1}/{n_runs} (time budget: {per_run_seconds}s) ===")
        model = make_model().to(device)
        val_loss, tokens_this_run = train_limited_time(
            model, train_loader, val_loader, lr=lr, time_limit_s=per_run_seconds
        )
        vals.append(val_loss)
        toks.append(tokens_this_run)
        # (model will be GC'd; a new one is created next loop)
    return np.array(vals, dtype=np.float64), np.array(toks, dtype=np.int64)

# --------------------------
# Main
# --------------------------
if __name__ == '__main__':
    x_train, y_train, x_val, y_val, vocab_size = load_or_build_packed(context_len)
    train_loader, val_loader = build_loaders(x_train, y_train, x_val, y_val)


    args = {"lr":           3e-4,
            "n_layer":      2,
            "n_head":       2,
            "n_embd":       128,
            "dropout":      0.0,
            "weight_decay": 0.1,
            "grad_clip":    None,
    }

    val_losses, tokens_per_run = test_setup(
        args, train_loader, val_loader,
        n_runs=5, per_run_seconds=300
    )
    
    #14 -> std=0.0058

    # Summary
    print("\n=== Summary over runs ===")
    print(f"Total tokens processed: {int(tokens_per_run.sum()):,}")
    print(f"Val losses (mean ± std): {val_losses.mean():.4f} ± {val_losses.std():.4f}")
