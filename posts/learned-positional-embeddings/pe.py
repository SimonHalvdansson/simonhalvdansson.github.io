import math, time, os
from contextlib import nullcontext
from collections import deque
import schedulefree

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from data_helpers import load_or_build_packed, build_loaders

from model import LLM

# device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
is_cuda = (device == "cuda")
use_amp = is_cuda
amp_dtype = torch.bfloat16

# global hyperparams
context_len = 256
batch_size = 16

# --------------------------
# Train/Eval
# --------------------------
def make_autocast_and_scaler():
    autocast_ctx = (lambda: torch.amp.autocast(device_type=device, dtype=amp_dtype)) if use_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    return autocast_ctx, scaler

def run_epoch(model, loader, args, optimizer, train=True,
              autocast_ctx=nullcontext, scaler=None,
              deadline=None,
              selfsim=None):
    """
    Returns: (avg_loss, total_tokens, stopped_early)
    - For TRAIN: if deadline is not None and deadline is reached mid-epoch, stop early.
    - For EVAL: deadline is ignored; we always finish the loader.
    """
    model.train(train)
    if args["optimizer"] == "schedulefree" and optimizer is not None:
        if train:
            optimizer.train()
        else:
            optimizer.eval()
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

        with autocast_ctx():
            logits = model(x)
            vocab_size = model.head.out_features
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if args["grad_clip"] is not None:
                    scaler.unscale_(optimizer)
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

        total_loss_sum += loss.item() * toks
        total_tokens += toks

        pbar.set_postfix(loss=f"{loss.item():.3f}")
        if first_draw:
            pbar.refresh()
            first_draw = False

        # Periodic self-similarity snapshots during training (step-based)
        if train and selfsim is not None and "interval_steps" in selfsim:
            try:
                # count completed optimizer steps
                selfsim["step"] = selfsim.get("step", 0) + 1
                while (
                    selfsim["step"] >= selfsim.get("next_step", selfsim["interval_steps"]) and
                    selfsim.get("next_step", selfsim["interval_steps"]) <= selfsim.get("max_steps", float("inf"))
                ):
                    step_for_name = selfsim["next_step"]
                    fname = f"{selfsim['prefix']}_step{step_for_name:05d}.png"
                    fpath = os.path.join(selfsim["save_dir"], fname)
                    os.makedirs(selfsim["save_dir"], exist_ok=True)
                    plot_positional_self_similarity(
                        model,
                        save_path=fpath,
                        dpi=300,
                        figsize=(5, 4.5),
                        show=False,
                        title=f"Self-similarity, step: {step_for_name}"
                    )
                    selfsim["next_step"] = step_for_name + selfsim["interval_steps"]
            except Exception:
                # Don't break training if plotting fails; ignore.
                pass

        if train and deadline is not None and t_now >= deadline:
            stopped_early = True
            break

    avg_loss = total_loss_sum / max(total_tokens, 1)
    return avg_loss, total_tokens, stopped_early

def train_one_epoch(model, train_loader, val_loader, args, selfsim=None):
    """
    Train for exactly one epoch over `train_loader`, then run a full validation pass.
    Also emits step-based self-similarity plots as configured by `selfsim`.
    Returns: val_loss
    """
    autocast_ctx, scaler = make_autocast_and_scaler()
    if args["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])
    else:
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args["lr"])

    train_loss, train_tokens, _ = run_epoch(
        model, train_loader, args=args, optimizer=optimizer, train=True,
        autocast_ctx=autocast_ctx, scaler=scaler, deadline=None, selfsim=selfsim
    )

    # Final self-similarity snapshot at the last step before training finishes
    if selfsim is not None and "interval_steps" in selfsim:
        try:
            last_step = selfsim.get("step", 0)
            fname = f"{selfsim['prefix']}_step{last_step:05d}_final.png"
            fpath = os.path.join(selfsim["save_dir"], fname)
            os.makedirs(selfsim["save_dir"], exist_ok=True)
            plot_positional_self_similarity(
                model, save_path=fpath, dpi=300, figsize=(5, 4.5), show=False,
                title=f"Self-similarity, step: {last_step}"
            )
        except Exception:
            pass

    # Full validation pass
    val_loss, _, _ = run_epoch(
        model, val_loader, args, optimizer=None, train=False, autocast_ctx=autocast_ctx
    )

    print(
        f"Train_tokens={train_tokens:,} | Train_ce={train_loss:.3f} | "
        f"Val_ce={val_loss:.3f} | Val_ppl={math.exp(val_loss):.2f}"
    )

    return val_loss

# Backward-compatible shim for previous time-limited API
def train_limited_time(model, train_loader, val_loader, time_limit_s, args, selfsim=None):
    return train_one_epoch(model, train_loader, val_loader, args, selfsim)

def test_setup(args, train_loader, val_loader, n_runs, per_run_seconds):
    """
    Runs `n_runs` independent trainings. Each run:
      - builds a FRESH model via `make_model()`
      - trains for up to `per_run_seconds`
      - then runs full validation
    Returns:
      - val_losses: np.ndarray shape (n_runs,)
      - tokens_per_run: np.ndarray shape (n_runs,)
    """
    def make_model():
        return LLM(vocab_size=vocab_size,
                       context_len=context_len, 
                       n_layer=args["n_layer"],
                       n_head=args["n_head"],
                       d_model=args["d_model"],
                       dropout=args["dropout"],
                       norm=args["norm"],
                       ffn=args["ffn"],
                       prepost=args["prepost"],
                       pos_emb=args["pos_emb"])    
    
    vals = []
    for i in range(n_runs):
        print(f"\n=== Run {i+1}/{n_runs} (time budget: {per_run_seconds}s) ===")
        model = make_model().to(device)
        
        val_loss = train_limited_time(
            model, train_loader, val_loader, per_run_seconds, args
        )
        vals.append(val_loss)
        # (model will be GC'd; a new one is created next loop)
        
    return np.array(vals, dtype=np.float64)


@torch.no_grad()
def _sample_ids(model, start_ids, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    x = torch.as_tensor(start_ids, dtype=torch.long, device=device).view(1, -1)
    for _ in range(max_new_tokens):
        logits = model(x[:, -context_len:])[:, -1, :]  # (1, vocab)
        logits = logits / max(temperature, 1e-8)
        if top_k is not None and top_k > 0 and top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        x = torch.cat([x, next_id], dim=1)
    return x.squeeze(0).tolist()

def train_and_sample_once(
    train_loader,
    val_loader,
    args,
    per_run_seconds,
    start_ids,
    max_new_tokens,
    decode_fn,
    temperature=1.0,
    top_k=None,
):
    """
    Trains a fresh model for up to `per_run_seconds`, then samples `max_new_tokens`.
    start_ids: list[int] seed tokens (e.g., BOS or tokenized prompt).
    decode_fn: callable(list[int]) -> str
    Returns dict with val_loss, out_ids, text.
    """
    # build fresh model
    model = LLM(
        vocab_size=vocab_size,
        context_len=context_len,
        n_layer=args["n_layer"],
        n_head=args["n_head"],
        d_model=args["d_model"],
        dropout=args["dropout"],
        norm=args["norm"],
        ffn=args["ffn"],
        prepost=args["prepost"],
        pos_emb=args["pos_emb"],
    ).to(device)

    # time-limited train + full validation
    val_loss = train_limited_time(
        model, train_loader, val_loader, per_run_seconds, args
    )

    # sampling
    out_ids = _sample_ids(
        model,
        start_ids=start_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    text = decode_fn(out_ids)
    return {"val_loss": val_loss, "out_ids": out_ids, "text": text}


# --------------------------
# Visualization helpers
# --------------------------
def _get_positional_weights(model: nn.Module) -> np.ndarray:
    """
    Returns positional embedding weights as a numpy array of shape (T, D).
    Supports both learned (nn.Embedding) and sinusoidal buffer variants.
    """
    # Try learned embedding first
    if isinstance(getattr(model, "pos_emb", None), nn.Embedding):
        return model.pos_emb.weight.detach().cpu().numpy()

    # Fallback: sinusoidal buffer stored under model.pos_emb.pos_embedding with shape (1, T, D)
    pe = getattr(getattr(model, "pos_emb", object()), "pos_embedding", None)
    if pe is not None:
        arr = pe.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        return arr

    raise ValueError("Model does not expose recognizable positional embeddings under model.pos_emb.")


def plot_positional_embeddings_and_fft(model: nn.Module, dims=(0, 1, 2), save_path: str = None, *, figsize=(10, 8), dpi=240, show=True):
    """
    Produces a 3x2 grid: each row is one embedding dim in `dims`.
    Left column: positional embedding values over positions.
    Right column: magnitude of the 1D FFT of that curve.
    """
    import matplotlib.pyplot as plt

    pe = _get_positional_weights(model)  # (T, D)
    T = pe.shape[0]

    fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)

    for r, d in enumerate(dims):
        curve = pe[:, d]
        # Left: time/position domain
        ax_time = axes[r, 0]
        ax_time.plot(curve, lw=1.2)
        ax_time.set_title(f"Pos Emb dim {d}")
        ax_time.set_xlabel("Position")
        ax_time.set_ylabel("Value")

        # Right: frequency domain (magnitude spectrum)
        ax_freq = axes[r, 1]
        spec = np.fft.rfft(curve)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(T, d=1.0)
        ax_freq.plot(freqs, mag, lw=1.0)
        ax_freq.set_title(f"FFT magnitude dim {d}")
        ax_freq.set_xlabel("Frequency (cycles/pos)")
        ax_freq.set_ylabel("|FFT|")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


def plot_positional_self_similarity(model: nn.Module, save_path: str = None, *, figsize=(5, 4.5), dpi=300, show=True, title=None):
    """
    Plots cosine self-similarity across positions using L2-normalized position vectors.
    """
    import matplotlib.pyplot as plt

    pe = _get_positional_weights(model)  # (T, D)
    # L2-normalize each position vector
    norms = np.linalg.norm(pe, axis=1, keepdims=True) + 1e-12
    pe_norm = pe / norms
    # Cosine similarity matrix
    sim = pe_norm @ pe_norm.T

    plt.figure(figsize=figsize)
    im = plt.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1, origin="upper", interpolation="nearest")
    plt.title(title if title is not None else "Positional Self-Similarity (cosine)")
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Default hyperparameters for constructing a model if none is provided.
    args = {"lr":           1.1e-3,
            "optimizer":    "adamw",
            "n_layer":      2,
            "n_head":       4,
            "d_model":      256,
            "dropout":      0.0,
            "grad_clip":    None,
            "norm":         "layer",
            "ffn":          "mlp",
            "prepost":      "pre",
            "pos_emb":      "learned"}

    # Where to save figures: under this post's media folder
    base_dir = os.path.dirname(__file__)
    media_dir = os.path.join(base_dir, "media")
    grid_path_learned = os.path.join(media_dir, "positional_embeddings_fft_learned_d0_127_255.png")
    sim_path_learned = os.path.join(media_dir, "positional_self_similarity_learned.png")  # kept for backward compat (unused below)
    grid_path_sinus = os.path.join(media_dir, "positional_embeddings_fft_sinusoidal_d0_127_255.png")
    sim_path_sinus = os.path.join(media_dir, "positional_self_similarity_sinusoidal.png")
    sim_during_dir = os.path.join(media_dir, "self_similarity_during_training")


    placeholder_vocab = 50257
    
    # Sinusoidal baseline model (no training) for comparison plots
    sinus_args = dict(args)
    sinus_args["pos_emb"] = "sinusoidal"
    sinus_model = LLM(
        vocab_size=placeholder_vocab,
        context_len=context_len,
        n_layer=sinus_args["n_layer"],
        n_head=sinus_args["n_head"],
        d_model=sinus_args["d_model"],
        dropout=sinus_args["dropout"],
        norm=sinus_args["norm"],
        ffn=sinus_args["ffn"],
        prepost=sinus_args["prepost"],
        pos_emb=sinus_args["pos_emb"],
    ).to(device)

    # Plots for sinusoidal: self-similarity and 3x2 grid for dims 0,127,255
    plot_positional_self_similarity(sinus_model, save_path=sim_path_sinus, dpi=300, figsize=(5, 4.5))
    plot_positional_embeddings_and_fft(sinus_model, dims=(0, 127, 255), save_path=grid_path_sinus, dpi=300, figsize=(10, 8))

    # Learned positional embeddings model (to train)
    model = LLM(
        vocab_size=placeholder_vocab,
        context_len=context_len,
        n_layer=args["n_layer"],
        n_head=args["n_head"],
        d_model=args["d_model"],
        dropout=args["dropout"],
        norm=args["norm"],
        ffn=args["ffn"],
        prepost=args["prepost"],
        pos_emb=args["pos_emb"],
    ).to(device)
    
    x_train, y_train, x_val, y_val, vocab_size, tokenizer = load_or_build_packed(context_len, is_cuda)
    train_loader, val_loader = build_loaders(x_train, y_train, x_val, y_val, context_len, batch_size, is_cuda)
    
    # Configure periodic self-similarity snapshots: every 50 steps, stop after 1000 steps
    selfsim_cfg = {
        "interval_steps": 50,
        "next_step": 50,
        "max_steps": 5000,
        "step": 0,
        "save_dir": sim_during_dir,
        "prefix": "positional_self_similarity_learned",
    }

    # Train for 100 epochs, with step-based snapshots every 50 steps (max 1000),
    # and save an additional self-sim figure at the end of each epoch.
    for epoch in range(1, 2):
        val_loss = train_one_epoch(model, train_loader, val_loader, args=args, selfsim=selfsim_cfg)
        print(f"Finished epoch {epoch}. Val loss: {val_loss:.3f}")
        # Save a per-epoch self-sim figure in main media dir
        epoch_sim_path = os.path.join(media_dir, f"positional_self_similarity_learned_epoch{epoch}.png")
        plot_positional_self_similarity(
            model,
            save_path=epoch_sim_path,
            dpi=300,
            figsize=(5, 4.5),
            title=f"Self-similarity, step: {selfsim_cfg['step']}"
        )

    # After all epochs, save the FFT grid at dims 0,127,255 for the learned model
    plot_positional_embeddings_and_fft(model, dims=(0, 127, 255), save_path=grid_path_learned, dpi=300, figsize=(10, 8))
