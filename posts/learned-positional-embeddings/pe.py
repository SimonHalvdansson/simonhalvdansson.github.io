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
batch_size = 8
max_epochs = 3

def print_model_parameter_count(model: nn.Module, label: str = "LLM") -> int:
    """Prints and returns the total number of parameters in `model`."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{label} parameters: {total_params:,}")
    return total_params

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
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
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
                while selfsim["step"] >= selfsim.get("next_step", selfsim["interval_steps"]):
                    step_for_name = selfsim["next_step"]
                    fname = f"{selfsim['prefix']}_step{step_for_name:05d}.png"
                    fpath = os.path.join(selfsim["save_dir"], fname)
                    os.makedirs(selfsim["save_dir"], exist_ok=True)
                    plot_positional_self_similarity(
                        model,
                        save_path=fpath,
                        dpi=250,
                        figsize=(5, 4.5),
                        show=False,
                        title=f"Self-similarity ({step_for_name})",
                        raw_title=f"Raw positional embeddings (step {step_for_name})",
                    )
                    base, ext = os.path.splitext(fpath)
                    ext = ext if ext else ".png"
                    density_path = f"{base}_density{ext}"
                    plot_embedding_density_map(
                        model,
                        save_path=density_path,
                        dpi=250,
                        figsize=(8, 4.5),
                        show=False,
                        title=f"Learned positional embedding density (step {step_for_name})",
                        value_range=(-0.1, 0.1),
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
    def make_model(label=None):
        model = LLM(vocab_size=vocab_size,
                       context_len=context_len, 
                       n_layer=args["n_layer"],
                       n_head=args["n_head"],
                       d_model=args["d_model"],
                       dropout=args["dropout"],
                       norm=args["norm"],
                       ffn=args["ffn"],
                       prepost=args["prepost"],
                       pos_emb=args["pos_emb"]).to(device)
        print_model_parameter_count(model, label or "LLM")
        return model
    
    vals = []
    for i in range(n_runs):
        print(f"\n=== Run {i+1}/{n_runs} (time budget: {per_run_seconds}s) ===")
        model = make_model(label=f"Run {i+1} model")
        
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
    print_model_parameter_count(model, "train_and_sample_once LLM")

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


def _plot_embedding_value_heatmap(
    pe: np.ndarray,
    *,
    save_path: str = None,
    figsize=(5, 4.5),
    dpi=250,
    show=True,
    title=None,
    value_range=None,
):
    """Plots raw positional embedding values as a heatmap."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    if value_range is None:
        value_max = float(np.max(pe))
        value_min = float(np.min(pe))
        display = pe
    else:
        value_min, value_max = value_range
        value_min = float(value_min)
        value_max = float(value_max)
        display = np.clip(pe, value_min, value_max)
    data = display.T  # rows: embedding dims, columns: positions
    im = plt.imshow(data, aspect="auto", cmap="coolwarm", origin="lower", vmin=value_min, vmax=value_max)
    plt.title(title or "Raw positional embeddings")
    plt.xlabel("Position")
    plt.ylabel("Embedding dimension")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Value")
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


def plot_positional_embeddings_and_autocorr(
    model: nn.Module, dims=(0, 1, 2), save_path: str = None, *, figsize=(10, 8), dpi=250, show=True
):
    """
    Produces a 3x2 grid: each row is one embedding dim in `dims`.
    Left column: positional embedding values over positions.
    Right column: autocorrelation of that curve (all lags).
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

        # Right: autocorrelation across lags
        ax_auto = axes[r, 1]
        centered = curve - np.mean(curve)
        autocorr = np.correlate(centered, centered, mode="full")
        lags = np.arange(-T + 1, T)
        ax_auto.plot(lags, autocorr, lw=1.0)
        ax_auto.set_title(f"Autocorrelation dim {d}")
        ax_auto.set_xlabel("Lag")
        ax_auto.set_ylabel("Autocorr")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


def plot_mean_abs_positional_autocorr(
    model: nn.Module,
    save_path: str = None,
    *,
    max_lag: int = 20,
    figsize=(6, 4),
    dpi=250,
    show=True,
    title=None,
):
    """
    Plots the average absolute autocorrelation across all embedding dimensions for lags 0..`max_lag`.
    """
    import matplotlib.pyplot as plt

    pe = _get_positional_weights(model)  # (T, D)
    centered = pe - np.mean(pe, axis=0, keepdims=True)
    T = centered.shape[0]
    if max_lag > T - 1:
        max_lag = T - 1

    # Collect absolute autocorrelations per dimension for positive lags
    pos_corrs = []
    for d in range(centered.shape[1]):
        corr = np.correlate(centered[:, d], centered[:, d], mode="full")
        pos_corr = np.abs(corr[T - 1 : T + max_lag])
        pos_corrs.append(pos_corr)
    mean_abs_corr = np.mean(pos_corrs, axis=0)

    lags = np.arange(0, max_lag + 1)
    plt.figure(figsize=figsize)
    plt.plot(lags, mean_abs_corr, lw=1.5)
    plt.title(title if title is not None else f"Mean |autocorr| up to lag {max_lag}")
    plt.xlabel("Lag")
    plt.ylabel("Mean |autocorr|")
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


def plot_positional_self_similarity(
    model: nn.Module,
    save_path: str = None,
    *,
    figsize=(5, 4.5),
    dpi=250,
    show=True,
    title=None,
    raw_save_path: str = None,
    raw_title: str = None,
    raw_value_range=(-0.1, 0.1),
):
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

    if raw_save_path is None and save_path is not None:
        base, ext = os.path.splitext(save_path)
        ext = ext if ext else ".png"
        raw_save_path = f"{base}_values{ext}"

    heatmap_title = raw_title or "Raw positional embeddings"
    _plot_embedding_value_heatmap(
        pe,
        save_path=raw_save_path,
        figsize=figsize,
        dpi=dpi,
        show=show,
        title=heatmap_title,
        value_range=raw_value_range,
    )


def plot_embedding_density_map(
    model: nn.Module,
    *,
    num_bins: int = 128,
    save_path: str = None,
    figsize=(8, 4.5),
    dpi=250,
    show=True,
    title: str = None,
    value_range=None,
):
    """
    Plots a per-token normalized histogram (density) of embedding values.
    Each column corresponds to a token position; each column is normalized
    independently so the maximum bin density is 1 (relative density).
    If `value_range` is provided, the y-axis and bins are fixed to that range.
    """
    import matplotlib.pyplot as plt

    pe = _get_positional_weights(model)
    seq_len, _ = pe.shape
    if value_range is None:
        flat = pe.reshape(-1)
        value_min = float(np.min(flat))
        value_max = float(np.max(flat))
        if value_min == value_max:
            value_min -= 1e-1
            value_max += 1e-1
    else:
        value_min, value_max = value_range
    bins = np.linspace(value_min, value_max, num_bins + 1)

    density = np.zeros((num_bins, seq_len), dtype=np.float32)
    for idx in range(seq_len):
        row = pe[idx]
        if value_range is not None:
            row = np.clip(row, value_min, value_max)
        hist, _ = np.histogram(row, bins=bins)
        hist = hist.astype(np.float32)
        max_hist = np.max(hist)
        if max_hist > 0:
            hist /= max_hist
        density[:, idx] = hist

    plt.figure(figsize=figsize)
    im = plt.imshow(
        density,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=(0, seq_len - 1, value_min, value_max),
        vmin=0.0,
        vmax=1.0,
    )
    plt.xlabel("Position")
    plt.ylabel("Embedding value")
    plt.title(title or "Normalized embedding value density")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Relative density")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Quick toggles for individual experiments.
    RUN_SINUSOIDAL_BASELINE = 1
    RUN_LEARNED_TRAINING = 1
    RUN_LEARNED_FFT_GRID = 1
    RUN_DENSITY_PLOTS = 1

    # Default hyperparameters for constructing a model if none is provided.
    args = {"lr":           2e-4,
            "optimizer":    "adamw",
            "n_layer":      12,
            "n_head":       4,
            "d_model":      512,
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
    grid_path_sinus = os.path.join(media_dir, "positional_embeddings_fft_sinusoidal_d0_127_255.png")
    mean_abs_autocorr_path_learned = os.path.join(media_dir, "positional_autocorr_mean_abs_learned.png")
    mean_abs_autocorr_path_sinus = os.path.join(media_dir, "positional_autocorr_mean_abs_sinusoidal.png")
    sim_path_sinus = os.path.join(media_dir, "positional_self_similarity_sinusoidal.png")
    raw_values_path_sinus = os.path.join(media_dir, "positional_embedding_values_sinusoidal.png")
    density_path_sinus = os.path.join(media_dir, "positional_embedding_density_sinusoidal.png")
    density_path_learned = os.path.join(media_dir, "positional_embedding_density_learned.png")
    training_snapshots_dir = os.path.join(media_dir, "training_snapshots")

    placeholder_vocab = 50257
    learned_model = None

    if RUN_SINUSOIDAL_BASELINE:
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
        print_model_parameter_count(sinus_model, "Sinusoidal positional LLM")

        # Plots for sinusoidal: self-similarity, raw values, FFT grid, and density
        plot_positional_self_similarity(
            sinus_model,
            save_path=sim_path_sinus,
            raw_save_path=raw_values_path_sinus,
            dpi=250,
            figsize=(5, 4.5),
            title="Sinusoidal positional embeddings self-similarity",
            raw_title="Raw sinusoidal positional embeddings",
            raw_value_range = (-1, 1),
        )
        plot_positional_embeddings_and_autocorr(
            sinus_model, dims=(0, 127, 255), save_path=grid_path_sinus, dpi=250, figsize=(10, 8)
        )
        plot_mean_abs_positional_autocorr(
            sinus_model,
            save_path=mean_abs_autocorr_path_sinus,
            dpi=250,
            figsize=(6, 4),
            max_lag=20,
            title="Mean |autocorr| (sinusoidal)",
        )
        if RUN_DENSITY_PLOTS:
            plot_embedding_density_map(
                sinus_model,
                save_path=density_path_sinus,
                dpi=250,
                figsize=(8, 4.5),
                value_range=(-1,1, 1.1),
                title="Sinusoidal positional embedding density",
            )

    if RUN_LEARNED_TRAINING:
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
        print_model_parameter_count(model, "Learned positional LLM")

        x_train, y_train, x_val, y_val, vocab_size, tokenizer = load_or_build_packed(context_len, is_cuda)
        train_loader, val_loader = build_loaders(x_train, y_train, x_val, y_val, context_len, batch_size, is_cuda)

        # Configure periodic self-similarity snapshots: every 300 steps
        selfsim_cfg = {
            "interval_steps": 300,
            "next_step": 300,
            "step": 0,
            "save_dir": training_snapshots_dir,
            "prefix": "learned",
        }

        # and save an additional self-sim figure at the end of each epoch.
        for epoch in range(1, max_epochs + 1):
            val_loss = train_one_epoch(model, train_loader, val_loader, args=args, selfsim=selfsim_cfg)
            print(f"Finished epoch {epoch}. Val loss: {val_loss:.3f}")
            # Save a per-epoch self-sim figure in main media dir
            epoch_sim_path = os.path.join(media_dir, f"positional_self_similarity_learned_epoch{epoch}.png")
            plot_positional_self_similarity(
                model,
                save_path=epoch_sim_path,
                dpi=250,
                figsize=(5, 4.5),
                title=f"Self-similarity, step: {selfsim_cfg['step']}",
                raw_title=f"Raw learned positional embeddings (step {selfsim_cfg['step']})",
            )

        learned_model = model

    if RUN_LEARNED_FFT_GRID:
        if learned_model is None:
            learned_model = LLM(
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
            print_model_parameter_count(learned_model, "Learned positional LLM (fresh)")

        plot_positional_embeddings_and_autocorr(
            learned_model,
            dims=(0, 127, 255),
            save_path=grid_path_learned,
            dpi=250,
            figsize=(10, 8),
        )
        plot_mean_abs_positional_autocorr(
            learned_model,
            save_path=mean_abs_autocorr_path_learned,
            dpi=250,
            figsize=(6, 4),
            max_lag=20,
            title="Mean |autocorr| (learned)",
        )
        if RUN_DENSITY_PLOTS:
            plot_embedding_density_map(
                learned_model,
                save_path=density_path_learned,
                dpi=250,
                figsize=(8, 4.5),
                title="Learned positional embedding density",
                value_range=(-0.1, 0.1),
            )
