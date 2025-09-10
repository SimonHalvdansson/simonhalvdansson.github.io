import math, time
from contextlib import nullcontext
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import LLM
from data_helpers import load_or_build_packed, build_loaders

# device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
is_cuda = (device == "cuda")
use_amp = is_cuda
amp_dtype = torch.bfloat16 if use_amp else None


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
              deadline=None):
    """
    Returns: (avg_loss, total_tokens, stopped_early)
    - For TRAIN: if deadline is not None and deadline is reached mid-epoch, stop early.
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

        pbar.set_postfix(loss=f"{loss.item():.3f}")
        if first_draw:
            pbar.refresh()
            first_draw = False

        if train and deadline is not None and t_now >= deadline:
            stopped_early = True
            break

    avg_loss = total_loss_sum / max(total_tokens, 1)
    return avg_loss, total_tokens, stopped_early

def train_limited_time(model, train_loader, val_loader, time_limit_s, args):
    """
    Train for up to `time_limit_s` seconds (early-stopping inside epoch if needed),
    then ALWAYS do a full validation pass.
    Returns: (val_loss, tokens_this_run)
    """
    autocast_ctx, scaler = make_autocast_and_scaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    start = time.perf_counter()
    deadline = start + time_limit_s
    total_train_tokens = 0

    # Loop until run_epoch returns stopped_early=True
    stopped = False
    while True:
        train_loss, train_toks, stopped_early = run_epoch(
            model, train_loader, args=args, optimizer=optimizer, train=True,
            autocast_ctx=autocast_ctx, scaler=scaler,
            deadline=deadline
        )
        total_train_tokens += train_toks
        if stopped_early:
            stopped = True
            break
        # otherwise continue to next epoch loop

    # Validation: always full pass, regardless of time
    val_loss, val_toks, _ = run_epoch(
        model, val_loader, args, optimizer=None, train=False,
        autocast_ctx=autocast_ctx, scaler=None,
        deadline=None
    )    

    print(f"Train_tokens={total_train_tokens:,} | "
          f"Val_bpc={val_loss*math.log2(math.e):.3f} | "
          f"Val_ppl={math.exp(val_loss):.2f} | "
          f"{'stopped early' if stopped else 'clean epoch end'}")

    return val_loss

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
                       n_embd=args["n_embd"],
                       dropout=args["dropout"],
                       norm=args["norm"],
                       ffn=args["ffn"],
                       prepost=args["prepost"])
    
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

def plot_mean_with_ci(xs, ys, alpha=0.05, ax=None, label="mean ± CI", dpi=180):
    """
    xs: list/1D array of x-values, length N
    ys: list of length N where ys[i] is a 1D array-like of samples at x=xs[i]
    alpha: 1 - confidence level (0.05 -> 95% CI)
    ax: optional matplotlib Axes
    label: legend label
    dpi: figure DPI

    Plots the point estimate (sample mean) with two-sided (1-alpha) CI on the mean
    using Student's t with unknown variance.
    """
    from scipy.stats import t as student_t
    
    xs = np.asarray(xs)
    assert len(xs) == len(ys), "len(xs) must equal len(ys)"

    means = np.empty(len(xs), dtype=float)
    se = np.empty(len(xs), dtype=float)
    df = np.empty(len(xs), dtype=int)

    for i, samp in enumerate(ys):
        y = np.asarray(samp, dtype=float)
        y = y[np.isfinite(y)]
        n = y.size
        if n < 2:
            raise ValueError(f"Need at least 2 samples at xs[{i}]={xs[i]} to form a CI, got {n}.")
        m = y.mean()
        s = y.std(ddof=1)
        means[i] = m
        se[i] = s / np.sqrt(n)
        df[i] = n - 1

    
    tcrit = np.array([student_t.ppf(1 - alpha/2, d) for d in df])

    err = tcrit * se
    yerr = np.vstack([err, err])  # symmetric

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    else:
        fig = ax.figure

    ax.errorbar(xs, means, yerr=yerr, fmt="o-", capsize=3, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.set_title(f"Mean with {(1-alpha)*100:.0f}% CI on the mean")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.show()
    return fig, ax, means, err
    

if __name__ == '__main__':
    x_train, y_train, x_val, y_val, vocab_size = load_or_build_packed(context_len, is_cuda)
    train_loader, val_loader = build_loaders(x_train, y_train, x_val, y_val, context_len, batch_size, is_cuda)

    args = {"lr":           3e-4,
            "n_layer":      2,
            "n_head":       2,
            "n_embd":       128,
            "dropout":      0.0,
            "weight_decay": 0.1,
            "grad_clip":    None,
            "norm":         "layer",
            "ffn":          "mlp",
            "prepost":      "pre"
    }

    def objective(lr):
        local_args = args.copy()
        local_args["lr"] = lr
        
        val_losses = test_setup(
            args, train_loader, val_loader,
            n_runs=10, per_run_seconds=30
        )
        
        return val_losses
    
    xs = np.linspace(1e-3, 1e-4, 5)
    ys = []
    for x in xs:
        ys.append(objective(x))
        
    plot_mean_with_ci(xs, ys)
    

    #val_losses = test_setup(
    #    args, train_loader, val_loader,
    #    n_runs=5, per_run_seconds=300
    #)
    
    # Summary
    #print("\n=== Summary over runs ===")
    #print(f"Val losses (mean ± std): {val_losses.mean():.4f} ± {val_losses.std():.4f}")
