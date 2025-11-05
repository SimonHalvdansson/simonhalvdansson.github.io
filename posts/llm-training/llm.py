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
    text_to_png
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
            
            print(vocab_size)
            
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




if __name__ == '__main__':
    args = {"lr":           1.1e-3,
            "optimizer":    "schedulefree",
            "n_layer":      2,
            "n_head":       4,
            "d_model":      256,
            "dropout":      0.0,
            "grad_clip":    None,
            "norm":         "layer",
            "ffn":          "mlp",
            "prepost":      "pre",
            "pos_emb":      "learned"
    }
    
    per_run_seconds = 300
    n_runs = 20
    lr_points = 8
    histogram_runs = 100
    
    do_hist     = 0
    do_lhd      = 0
    do_lr       = 0
    do_norm     = 0
    do_ffn      = 0
    do_prepost  = 0
    do_pos_emb  = 0
    do_dropout  = 0
    do_gradclip = 0
        
    min_time = 0
    if do_hist:
        min_time += histogram_runs*per_run_seconds
        
    if do_lhd:
        min_time += 4*4*4*per_run_seconds*n_runs
        
    if do_lr:
        min_time += lr_points*per_run_seconds*n_runs*2
        
    if do_norm:
        min_time += 2*per_run_seconds*n_runs
        
    if do_ffn:
        min_time += 2*per_run_seconds*n_runs
        
    if do_prepost:
        min_time += 2*per_run_seconds*n_runs
        
    if do_pos_emb:
        min_time += 2*per_run_seconds*n_runs

    if do_dropout:
        min_time += 5*per_run_seconds*n_runs

    if do_gradclip:
        min_time += 5*per_run_seconds*n_runs
        
    print(f"Minimum time: {min_time/60/60} hours")
    
    x_train, y_train, x_val, y_val, vocab_size, tokenizer = load_or_build_packed(context_len, is_cuda)
    train_loader, val_loader = build_loaders(x_train, y_train, x_val, y_val, context_len, batch_size, is_cuda)


    if do_lhd:
        layers = [1, 2, 4, 6]
        heads  = [1, 2, 4, 8]
        d_models   = [64, 128, 256, 384]

        _, _, sweep_results = plot_layers_heads_dims_heatmaps(
            train_loader, val_loader,
            layers=layers, heads=heads, d_models=d_models,
            base_args=args, n_runs=n_runs, per_run_seconds=per_run_seconds,
            annotate=False, test_setup_fn=test_setup
        )

        plot_layers_heads_dims_bars(
            sweep_results["stats"], alpha=0.05, dpi=150,
        )

    
    if do_hist:
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
        
        vals = objective(key=None, value=None, n_runs=histogram_runs)
        plot_val_loss_hist(vals)

    if do_lr:
        plot_lr_sweep_both(train_loader, val_loader,
                        min_lr=4e-4, max_lr=3e-3, n_points=lr_points,
                        n_runs=n_runs, per_run_seconds=per_run_seconds,
                        base_args=args, test_setup_fn=test_setup)

    if do_norm:
        plot_binary_option_bars(train_loader, val_loader,
            option_key="norm", option_values=("layer","rmsnorm"),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args, test_setup_fn=test_setup)
        
    if do_ffn:
        plot_binary_option_bars(train_loader, val_loader,
            option_key="ffn", option_values=("mlp","swiglu"),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args, test_setup_fn=test_setup)
    
    if do_prepost:
        plot_binary_option_bars(train_loader, val_loader,
            option_key="prepost", option_values=("pre","post"),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args, test_setup_fn=test_setup)
    
    if do_pos_emb:
        plot_binary_option_bars(train_loader, val_loader,
            option_key="pos_emb", option_values=("sinusoidal","learned"),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args, test_setup_fn=test_setup)
    
    if do_dropout:
        plot_dropout_bars(train_loader, val_loader,
            drops=(0.0, 0.05, 0.10, 0.15, 0.20),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args, test_setup_fn=test_setup)
    
    if do_gradclip:
        plot_gradclip_bars(train_loader, val_loader,
            clips=(0.5, 1.0, 1.5, 2.0, None),
            n_runs=n_runs, per_run_seconds=per_run_seconds, base_args=args, test_setup_fn=test_setup)

    
    
    
    # Build model
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

    
    # Train for a time budget
    val_loss = train_limited_time(model, train_loader, val_loader, time_limit_s=per_run_seconds, args=args)
    print("Finished training. Val loss:", val_loss)
    
    prompt = "Once upon a time"
    idx = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)],
                       dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=120, temperature=0.9, top_k=50, context_len=context_len)
    
    sample = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    print("\n=== SAMPLE ===\n", sample)
    
    text_to_png(str(sample))

    