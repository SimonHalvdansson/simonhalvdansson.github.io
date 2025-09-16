import math, time
from contextlib import nullcontext
from collections import deque
import schedulefree

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

from model import LLM
from data_helpers import load_or_build_packed, build_loaders

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
              deadline=None):
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
    if args["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])
    else:
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args["lr"])

    start = time.perf_counter()
    deadline = start + time_limit_s
    total_train_tokens = 0

    # Loop until run_epoch returns stopped_early=True
    stopped = False
    while True:
        _, train_toks, stopped_early = run_epoch(
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
    val_loss, _, _ = run_epoch(model, val_loader, args, optimizer=None, train=False, autocast_ctx=autocast_ctx)    

    print(f"Train_tokens={total_train_tokens:,} | "
          f"Val_ce={val_loss:.3f} | "
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


def plot_lr_sweep_both(train_loader, val_loader,
                       min_lr, max_lr, n_points,
                       n_runs=3, per_run_seconds=3,
                       base_args=None, alpha=0.05, dpi=180,
                       title="LR sweep: AdamW vs ScheduleFree"):
    """
    Runs LR sweeps for AdamW and ScheduleFree on the SAME model/config.
    Learning rates are sampled logarithmically from min_lr to max_lr.

    Arguments:
      min_lr, max_lr: bounds for learning rate sweep (both > 0)
      n_points: number of learning rate points between min_lr and max_lr
      n_runs: independent runs per LR
      per_run_seconds: training time budget per run
      base_args: dict with base model args
      alpha: significance level for CI
      dpi: plot resolution

    Returns: (fig, ax, results)
      results = {
        "adamw": {"vals": list[np.ndarray], "means": np.ndarray, "err": np.ndarray, "lrs": np.ndarray},
        "schedulefree": {...}
      }
    """
    from scipy.stats import t as student_t

    assert base_args is not None
    if min_lr <= 0 or max_lr <= 0:
        raise ValueError("min_lr and max_lr must be > 0 for log spacing")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    xs = np.logspace(np.log10(min_lr), np.log10(max_lr), n_points)

    def run_sweep(optimizer_name):
        all_vals = []
        for lr in xs:
            cfg = base_args.copy()
            cfg["optimizer"] = optimizer_name
            cfg["lr"] = float(lr)
            vals = test_setup(cfg, train_loader, val_loader,
                              n_runs=n_runs, per_run_seconds=per_run_seconds)
            all_vals.append(vals)
        # stats
        means = np.array([v.mean() for v in all_vals], dtype=float)
        ns = np.array([len(v) for v in all_vals], dtype=int)
        s = np.array([v.std(ddof=1) for v in all_vals], dtype=float)
        se = s / np.sqrt(np.maximum(ns, 1))
        df = np.maximum(ns - 1, 1)
        tcrit = np.array([student_t.ppf(1 - alpha/2, d) for d in df])
        err = tcrit * se
        return {"vals": all_vals, "means": means, "err": err, "lrs": xs}

    res_adamw = run_sweep("adamw")
    res_sf    = run_sweep("schedulefree")

    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    ax.set_xscale("log", base=10)

    # AdamW
    ax.errorbar(xs, res_adamw["means"], yerr=res_adamw["err"], fmt="o-",
                capsize=3, label=f"AdamW (n={n_runs} per LR)")
    # ScheduleFree
    ax.errorbar(xs, res_sf["means"], yerr=res_sf["err"], fmt="s--",
                capsize=3, label=f"ScheduleFree (n={n_runs} per LR)")

    ax.set_xlabel("learning rate")
    ax.set_ylabel("validation loss (cross-entropy)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

    return fig, ax, {"adamw": res_adamw, "schedulefree": res_sf}


def plot_val_loss_hist(samples, *, bins="auto", dpi=180, density=False, title=None):
    """
    samples: 1D array-like of validation losses (cross-entropy) for one config
    Returns: (fig, ax)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        raise ValueError("No samples.")

    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    ax.hist(x, bins=bins, density=density)
    ax.set_xlabel("validation loss (cross-entropy)")
    ax.set_ylabel("density" if density else "count")
    ax.set_title(title or f"Validation loss histogram (n={x.size})")
    ax.grid(True, alpha=0.3)
    plt.show()
    return fig, ax

def plot_binary_option_bars(train_loader, val_loader, *,
                            option_key, option_values,       # e.g. ("layer","rmsnorm")
                            n_runs=5, per_run_seconds=3,
                            base_args=None, alpha=0.05, dpi=180,
                            title=None):
    """
    For a single optimizer (from base_args), compare two settings of a binary option.
    Each bar = mean ± (1-alpha) CI of validation loss over n_runs.
    """
    from scipy.stats import t as student_t
    assert base_args is not None
    assert len(option_values) == 2, "option_values must be a pair"
    assert n_runs >= 2, "n_runs >= 2 needed for CI"

    results = {}

    for opt_val in option_values:
        cfg = base_args.copy()
        cfg[option_key] = opt_val
        vals = test_setup(cfg, train_loader, val_loader,
                          n_runs=n_runs, per_run_seconds=per_run_seconds)
        n = len(vals)
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
        se = s / np.sqrt(n)
        df = n - 1
        tcrit = float(student_t.ppf(1 - alpha/2, df))
        ci = tcrit * se
        results[opt_val] = {"vals": vals, "mean": m, "ci": ci, "n": n}

    # plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    x = np.arange(len(option_values))
    means = [results[v]["mean"] for v in option_values]
    errs  = [results[v]["ci"]   for v in option_values]

    ax.bar(x, means, yerr=errs, capsize=3, width=0.5, align="center")
    ax.set_xticks(x, option_values)
    ax.set_ylabel("validation loss (cross-entropy)")
    ax.set_title(title or f"{option_key}")
    ax.grid(True, axis="y", alpha=0.3)

    plt.show()
    return fig, ax, results

def plot_option_bars(train_loader, val_loader, *,
                     option_key, option_values,                 # list/tuple of values
                     n_runs=5, per_run_seconds=3,
                     base_args=None, alpha=0.05, dpi=180,
                     title=None, label_fmt=str):
    """Single-optimizer N-category sweep. Bars = mean ± t-CI of val CE."""
    from scipy.stats import t as student_t
    assert base_args is not None
    assert len(option_values) >= 2
    assert n_runs >= 2

    stats = []
    for v in option_values:
        cfg = base_args.copy()
        cfg[option_key] = v
        vals = test_setup(cfg, train_loader, val_loader,
                          n_runs=n_runs, per_run_seconds=per_run_seconds)
        n = len(vals); m = float(np.mean(vals)); s = float(np.std(vals, ddof=1))
        se = s / np.sqrt(n); df = n - 1
        tcrit = float(student_t.ppf(1 - alpha/2, df))
        stats.append((v, m, tcrit * se))

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=dpi)
    x = np.arange(len(stats))
    means = [m for _, m, _ in stats]
    errs  = [e for _, _, e in stats]
    labels = [label_fmt(v) for v, _, _ in stats]

    ax.bar(x, means, yerr=errs, capsize=3, width=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("validation loss (cross-entropy)")
    ax.set_title(title or f"{option_key} sweep")
    ax.grid(True, axis="y", alpha=0.3)
    plt.show()
    return fig, ax, stats

def _default_title_for_binary(option_key, option_values):
    key = option_key.lower()
    vals = tuple(str(v).lower() for v in option_values)
    if key == "norm" and set(vals) == {"layer", "rmsnorm"}:
        return "LayerNorm vs RMSNorm"
    if key == "ffn" and set(vals) == {"mlp", "swiglu"}:
        return "MLP vs SwiGLU"
    if key == "prepost" and set(vals) == {"pre", "post"}:
        return "Pre-norm vs Post-norm"
    # fallback
    return f"{option_key}: {option_values[0]} vs {option_values[1]}"

def plot_binary_option_bars(train_loader, val_loader, *,
                            option_key, option_values,       # e.g. ("layer","rmsnorm")
                            n_runs=5, per_run_seconds=3,
                            base_args=None, alpha=0.05, dpi=180,
                            title=None):
    """Single-optimizer comparison of two settings. Bars = mean ± t-CI of val CE."""
    from scipy.stats import t as student_t
    assert base_args is not None
    assert len(option_values) == 2
    assert n_runs >= 2

    results = {}
    for opt_val in option_values:
        cfg = base_args.copy()
        cfg[option_key] = opt_val
        vals = test_setup(cfg, train_loader, val_loader,
                          n_runs=n_runs, per_run_seconds=per_run_seconds)
        n = len(vals); m = float(np.mean(vals)); s = float(np.std(vals, ddof=1))
        se = s / np.sqrt(n); df = n - 1
        tcrit = float(student_t.ppf(1 - alpha/2, df))
        results[opt_val] = {"vals": vals, "mean": m, "ci": tcrit * se, "n": n}

    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    x = np.arange(2)
    means = [results[v]["mean"] for v in option_values]
    errs  = [results[v]["ci"]   for v in option_values]
    ax.bar(x, means, yerr=errs, capsize=3, width=0.5)
    ax.set_xticks(x, [str(v) for v in option_values])
    ax.set_ylabel("validation loss (cross-entropy)")
    ax.set_title(title or _default_title_for_binary(option_key, option_values))
    ax.grid(True, axis="y", alpha=0.3)
    plt.show()
    return fig, ax, results



def plot_gradclip_bars(train_loader, val_loader, *,
                       clips=(None, 0.5, 1.0, 2.0),
                       n_runs=5, per_run_seconds=3,
                       base_args=None, alpha=0.05, dpi=180):
    return plot_option_bars(
        train_loader, val_loader,
        option_key="grad_clip", option_values=clips,
        n_runs=n_runs, per_run_seconds=per_run_seconds,
        base_args=base_args, alpha=alpha, dpi=dpi,
        title="Gradient clipping sweep",
        label_fmt=lambda v: "None" if v is None else f"{v:g}"
    )

def plot_dropout_bars(train_loader, val_loader, *,
                      drops=(0.0, 0.05, 0.10, 0.15, 0.20),
                      n_runs=5, per_run_seconds=3,
                      base_args=None, alpha=0.05, dpi=180):
    return plot_option_bars(
        train_loader, val_loader,
        option_key="dropout", option_values=drops,
        n_runs=n_runs, per_run_seconds=per_run_seconds,
        base_args=base_args, alpha=alpha, dpi=dpi,
        title="Dropout sweep",
        label_fmt=lambda v: f"{v:.2f}"
    )




if __name__ == '__main__':
    x_train, y_train, x_val, y_val, vocab_size = load_or_build_packed(context_len, is_cuda)
    train_loader, val_loader = build_loaders(x_train, y_train, x_val, y_val, context_len, batch_size, is_cuda)

    args = {"lr":           1e-3,
            "optimizer":    "adamw",
            "n_layer":      2,
            "n_head":       2,
            "n_embd":       128,
            "dropout":      0.0,
            "grad_clip":    None,
            "norm":         "layer",
            "ffn":          "mlp",
            "prepost":      "pre"
    }

    per_run_seconds = 30
    n_runs = 10

    def objective(key, value, n_runs):
        local_args = args.copy()
        if key is not None:
            local_args[key] = value
            local_args["optimizer"] = "schedulefree"
        
        val_losses = test_setup(
            local_args, train_loader, val_loader,
            n_runs=n_runs, per_run_seconds=per_run_seconds
        )
        
        return val_losses
    
    
    
    #do histogram for base case
    if False:
        vals = objective(key=None, value=None, n_runs=5)
        plot_val_loss_hist(vals)

    #sweep lr for adamw + schedulefree, same figure
    if False:
        plot_lr_sweep_both(train_loader, val_loader,
                        min_lr=1e-5, max_lr=1e-2, n_points=10,
                        n_runs=n_runs, per_run_seconds=per_run_seconds,
                        base_args=args)

    if True:
        # norm
        plot_binary_option_bars(train_loader, val_loader,
            option_key="norm", option_values=("layer","rmsnorm"),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args)

        # ffn
        plot_binary_option_bars(train_loader, val_loader,
            option_key="ffn", option_values=("mlp","swiglu"),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args)

        # prepost
        plot_binary_option_bars(train_loader, val_loader,
            option_key="prepost", option_values=("pre","post"),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args)
        
        # Dropout sweep
        plot_dropout_bars(train_loader, val_loader,
            drops=(0.0, 0.05, 0.10, 0.15, 0.20),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args)
        
        # Gradient clipping sweep
        plot_gradclip_bars(train_loader, val_loader,
            clips=(None, 0.5, 1.0, 2.0),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args)

    