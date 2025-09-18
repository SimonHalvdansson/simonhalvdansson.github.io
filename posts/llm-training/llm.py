import math, time
from contextlib import nullcontext
from collections import deque
import schedulefree

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from plotting import (
    plot_layers_heads_dims_heatmaps,
    plot_layers_heads_dims_bars,
    plot_val_loss_hist,
    plot_lr_sweep_both,
    plot_binary_option_bars,
    plot_dropout_bars,
    plot_gradclip_bars,
)

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

    #LEARNINGS:
    #5 runs is too few for 20 s at least for 
    
    per_run_seconds = 60
    n_runs = 10
    lr_points = 5
    histogram_runs = 10

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
    
    if True:
        layers = [2, 4, 6, 8]
        heads  = [2, 4, 8, 16]
        d_models   = [128, 256, 384, 512]  # many cells will be masked when d_model % heads != 0

        _, _, sweep_results = plot_layers_heads_dims_heatmaps(
            train_loader, val_loader,
            layers=layers, heads=heads, d_models=d_models,
            base_args=args, n_runs=n_runs, per_run_seconds=per_run_seconds,
            annotate=False, test_setup_fn=test_setup
        )

        plot_layers_heads_dims_bars(
            sweep_results["stats"], alpha=0.05, dpi=150,
        )

    
    #do histogram for base case
    if True:
        vals = objective(key=None, value=None, n_runs=histogram_runs)
        plot_val_loss_hist(vals)

    #sweep lr for adamw + schedulefree, same figure
    if True:
        plot_lr_sweep_both(train_loader, val_loader,
                        min_lr=1e-4, max_lr=1e-2, n_points=lr_points,
                        n_runs=n_runs, per_run_seconds=per_run_seconds,
                        base_args=args, test_setup_fn=test_setup)

    if True:
        # norm
        plot_binary_option_bars(train_loader, val_loader,
            option_key="norm", option_values=("layer","rmsnorm"),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args, test_setup_fn=test_setup)

        # ffn
        plot_binary_option_bars(train_loader, val_loader,
            option_key="ffn", option_values=("mlp","swiglu"),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args, test_setup_fn=test_setup)

        # prepost
        plot_binary_option_bars(train_loader, val_loader,
            option_key="prepost", option_values=("pre","post"),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args, test_setup_fn=test_setup)
        
        # Dropout sweep
        plot_dropout_bars(train_loader, val_loader,
            drops=(0.0, 0.05, 0.10, 0.15, 0.20),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args, test_setup_fn=test_setup)
        
        # Gradient clipping sweep
        plot_gradclip_bars(train_loader, val_loader,
            clips=(None, 0.5, 1.0, 2.0),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args, test_setup_fn=test_setup)

    