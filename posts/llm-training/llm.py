# tiny_llm_tinystories_sdpa.py
# Minimal deps: torch>=2.1, datasets, transformers, tqdm

import os, math, random, time
from contextlib import nullcontext
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# --------------------------
# Explicit Hugging Face cache (helps on some Windows/IDE setups)
# --------------------------
HF_HOME = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "hub"))
HF_DS_CACHE = os.environ["HF_DATASETS_CACHE"]
HF_HUB_CACHE = os.environ["TRANSFORMERS_CACHE"]
print(f"HF datasets cache: {HF_DS_CACHE}\nHF hub cache:      {HF_HUB_CACHE}")

# --------------------------
# Config (small & fast)
# --------------------------
#seed = 1337
#random.seed(seed); torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 256
batch_size = 16  # per request
lr = 3e-4
weight_decay = 0.1
epochs = 1           # bump to 2–5 for better results
n_layer = 4
n_head  = 4
n_embd  = 256
dropout = 0.0

train_split = "train[:5%]"       # keep tiny for demo; use "train" for full
val_split   = "validation[:2%]"

# --------------------------
# Data: download + tokenize + pack
# --------------------------
print("Loading TinyStories… (will reuse local cache if present)")
ds_train = load_dataset(
    "roneneldan/TinyStories",
    split=train_split,
    cache_dir=HF_DS_CACHE,
    download_mode="reuse_cache_if_exists",
)
ds_val = load_dataset(
    "roneneldan/TinyStories",
    split=val_split,
    cache_dir=HF_DS_CACHE,
    download_mode="reuse_cache_if_exists",
)

# Pretrained tokenizer (no training)
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=HF_HUB_CACHE)
tokenizer.pad_token = tokenizer.eos_token  # for batching/generation convenience
# Silence the "sequence length > 1024" warning for GPT-2 tokenizer during preprocessing
tokenizer.model_max_length = int(1e9)  # effectively "no cap" for offline tokenization

def tokenize_batch(batch):
    # Keep full text; we pack into block_size later anyway
    return tokenizer(batch["text"], add_special_tokens=True, truncation=False)

tok_train = ds_train.map(tokenize_batch, batched=True, remove_columns=ds_train.column_names)
tok_val   = ds_val.map(tokenize_batch,   batched=True, remove_columns=ds_val.column_names)

def to_packed_ids(tok_ds):
    # concat all token ids then cut into contiguous blocks
    ids = []
    for arr in tok_ds["input_ids"]:
        ids.extend(arr)
    ids = torch.tensor(ids, dtype=torch.long)
    # create (inputs, targets) pairs via one-token shift
    n = (len(ids) - 1) // block_size
    if n <= 0:
        raise RuntimeError("Not enough tokens; enlarge dataset or reduce block_size.")
    x = ids[: n * block_size].view(n, block_size)
    y = ids[1 : n * block_size + 1].view(n, block_size)
    return x, y

x_train, y_train = to_packed_ids(tok_train)
x_val,   y_val   = to_packed_ids(tok_val)

class PackedDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return self.x.size(0)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

pin_mem = (device == "cuda")
train_loader = DataLoader(PackedDataset(x_train, y_train),
                          batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=pin_mem)
val_loader   = DataLoader(PackedDataset(x_val, y_val),
                          batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=pin_mem)

vocab_size = tokenizer.vocab_size

# --------------------------
# Tiny GPT model (PyTorch, SDPA)
# --------------------------
class CausalSelfAttention(nn.Module):
    """Uses torch.nn.functional.scaled_dot_product_attention with is_causal=True."""
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_drop = nn.Dropout(dropout)
        self.attn_p = dropout  # used by SDPA when training

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)
        # reshape to (B, nH, T, Hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # fused attention; uses Flash/Math kernels when available
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_p if self.training else 0.0,
            is_causal=True
        )  # -> (B, nH, T, Hd)
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
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
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
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

model = TinyGPT(vocab_size, block_size, n_layer, n_head, n_embd, dropout).to(device)
print(f"Device: {device}; params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# --------------------------
# Optimizer
# --------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# --------------------------
# Modern AMP: torch.amp.autocast(...) + new GradScaler API
# --------------------------
use_amp = device == "cuda"
if device == "cuda":
    amp_dtype = torch.bfloat16
else:
    amp_dtype = None

autocast_ctx = (lambda: torch.amp.autocast(device_type=device, dtype=amp_dtype)) if use_amp else nullcontext
# New API (fixes deprecation warning); only needed for CUDA fp16
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device == "cuda" and amp_dtype == torch.float16)
)

# --------------------------
# Training loop (tokens/sec with short rolling window, Spyder-friendly refresh)
# --------------------------
bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"

def run_epoch(loader, train=True):
    model.train(train)
    total_loss_sum, total_tokens = 0.0, 0

    # short rolling window for tok/s
    window_s = 2.0
    t_stamps = deque()
    tok_hist = deque()
    toks_in_window = 0

    pbar = tqdm(
        loader,
        leave=False,
        desc="train" if train else "eval",
        dynamic_ncols=True,
        mininterval=0.1,
        maxinterval=1.0,
        miniters=1,
        bar_format=bar_format,
    )

    first_draw = True
    for x, y in pbar:
        step_t0 = time.perf_counter()

        x = x.to(device, non_blocking=(device == "cuda"))
        y = y.to(device, non_blocking=(device == "cuda"))

        with (autocast_ctx() if use_amp else nullcontext()):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # timing & throughput (synchronize on CUDA for accurate timings)
        if device == "cuda":
            torch.cuda.synchronize()
        step_t1 = time.perf_counter()
        toks = x.numel()  # B*T tokens this step

        # maintain ~2 s rolling window
        t_stamps.append(step_t1); tok_hist.append(toks); toks_in_window += toks
        while t_stamps and (step_t1 - t_stamps[0]) > window_s:
            t_stamps.popleft(); toks_in_window -= tok_hist.popleft()
        window_dt = max(step_t1 - (t_stamps[0] if t_stamps else step_t1), 1e-9)
        tok_s = toks_in_window / window_dt

        total_loss_sum += loss.item() * toks
        total_tokens += toks

        pbar.set_postfix(loss=f"{loss.item():.3f}", tok_s=f"{tok_s:,.0f}")
        if first_draw:  # Spyder sometimes delays the first render
            pbar.refresh()
            first_draw = False

    return total_loss_sum / total_tokens

if __name__ == '__main__':
    for ep in range(1, epochs+1):
        train_loss = run_epoch(train_loader, train=True)
        val_loss   = run_epoch(val_loader,   train=False)
        print(f"Epoch {ep}: train_bpc={train_loss*math.log2(math.e):.3f}, "
              f"val_bpc={val_loss*math.log2(math.e):.3f}, "
              f"val_ppl={math.exp(val_loss):.2f}")
    
    # --------------------------
    # Quick sample
    # --------------------------
    prompt = "Once upon a time"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=120, temperature=0.9, top_k=50)
    print("\n=== SAMPLE ===\n", tokenizer.decode(out[0].tolist(), skip_special_tokens=True))
