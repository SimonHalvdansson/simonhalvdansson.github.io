import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import t as student_t

import numpy as np

def plot_lr_sweep_both(train_loader, val_loader,
                       min_lr, max_lr, n_points,
                       n_runs=3, per_run_seconds=3,
                       base_args=None, alpha=0.05, dpi=180,
                       title="LR sweep: AdamW vs ScheduleFree",
                       test_setup_fn=None):
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
            vals = test_setup_fn(cfg, train_loader, val_loader,
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

def _tight_ylim(ax, means, errs, pad_frac=0.08):
    m = np.asarray(means, dtype=float)
    e = np.asarray(errs, dtype=float)
    y_lo = float(np.min(m - e))
    y_hi = float(np.max(m + e))
    rng = max(1e-12, y_hi - y_lo)
    ax.set_ylim(y_lo - pad_frac*rng, y_hi + pad_frac*rng)

def plot_option_bars(train_loader, val_loader, *,
                     option_key, option_values,
                     n_runs=5, per_run_seconds=3,
                     base_args=None, alpha=0.05, dpi=180,
                     title=None, label_fmt=str,
                     test_setup_fn=None):
    """Single-optimizer N-category sweep. Bars = mean ± t-CI of val CE.
       Tight y-limits, adjacent bars, distinct colors, lowest highlighted."""
    assert base_args is not None
    assert len(option_values) >= 2
    assert n_runs >= 2

    stats = []
    for v in option_values:
        cfg = base_args.copy()
        cfg[option_key] = v
        vals = test_setup_fn(cfg, train_loader, val_loader,
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

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(x))]

    bars = ax.bar(x, means, yerr=errs, capsize=3, width=0.9, color=colors, edgecolor="none")

    # pack bars tightly and remove side padding
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.margins(x=0.0)

    # tight y-limits around data + CI
    _tight_ylim(ax, means, errs)

    # highlight lowest bar
    best = int(np.argmin(means))
    bars[best].set_edgecolor("black")
    bars[best].set_linewidth(2.5)
    ax.scatter([x[best]], [means[best]], s=150, marker="*", color="black", zorder=4)
    ax.annotate("lowest", xy=(x[best], means[best]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x, labels)
    ax.set_ylabel("validation loss (cross-entropy)")
    ax.set_title(title or f"{option_key} sweep")
    ax.grid(True, axis="y", alpha=0.3)
    plt.show()
    return fig, ax, stats

def plot_binary_option_bars(train_loader, val_loader, *,
                            option_key, option_values,       # e.g. ("layer","rmsnorm")
                            n_runs=5, per_run_seconds=3,
                            base_args=None, alpha=0.05, dpi=180,
                            title=None, test_setup_fn=None):
    """Single-optimizer comparison of two settings. Bars = mean ± t-CI of val CE.
       Tight y-limits, adjacent bars, distinct colors, lowest highlighted."""
    from scipy.stats import t as student_t
    assert base_args is not None
    assert len(option_values) == 2
    assert n_runs >= 2

    results = {}
    for opt_val in option_values:
        cfg = base_args.copy()
        cfg[option_key] = opt_val
        vals = test_setup_fn(cfg, train_loader, val_loader,
                          n_runs=n_runs, per_run_seconds=per_run_seconds)
        n = len(vals); m = float(np.mean(vals)); s = float(np.std(vals, ddof=1))
        se = s / np.sqrt(n); df = n - 1
        tcrit = float(student_t.ppf(1 - alpha/2, df))
        results[opt_val] = {"vals": vals, "mean": m, "ci": tcrit * se, "n": n}

    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    x = np.arange(2)
    means = [results[v]["mean"] for v in option_values]
    errs  = [results[v]["ci"]   for v in option_values]

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(2)]

    bars = ax.bar(x, means, yerr=errs, capsize=3, width=0.9, color=colors, edgecolor="none")

    # pack bars tightly and remove side padding
    ax.set_xlim(-0.5, 1.5)
    ax.margins(x=0.0)

    # tight y-limits around data + CI
    _tight_ylim(ax, means, errs)

    # highlight lowest bar
    best = int(np.argmin(means))
    bars[best].set_edgecolor("black")
    bars[best].set_linewidth(2.5)
    ax.scatter([x[best]], [means[best]], s=150, marker="*", color="black", zorder=4)
    ax.annotate("lowest", xy=(x[best], means[best]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x, [str(v) for v in option_values])
    ax.set_ylabel("validation loss (cross-entropy)")
    ax.set_title(title or _default_title_for_binary(option_key, option_values))
    ax.grid(True, axis="y", alpha=0.3)
    plt.show()
    return fig, ax, results

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


def plot_layers_heads_dims_heatmaps(train_loader, val_loader, *,
                                    layers, heads, dims,
                                    base_args, n_runs=3, per_run_seconds=3,
                                    dpi=180, annotate=False, cmap_name="viridis",
                                    test_setup_fn=None):
    """
    Four small heatmaps (one per layer count) in a single figure.
    x-axis: n_head (len=4), y-axis: n_embd 'dims' (len=4).
    Invalid combos where dims % heads != 0 are masked.
    Cell value = mean validation cross-entropy over n_runs.
    """
    assert len(layers) == 4 and len(heads) == 4 and len(dims) == 4, "Use 4 options per dimension."

    # Collect results per layer into 4 matrices of shape (len(dims), len(heads))
    matrices = []
    for L in layers:
        M = np.full((len(dims), len(heads)), np.nan, dtype=float)
        for i, d_model in enumerate(dims):
            for j, h in enumerate(heads):
                if d_model % h != 0:
                    continue  # invalid: head_dim not integer
                cfg = base_args.copy()
                cfg["n_layer"] = int(L)
                cfg["n_head"]  = int(h)
                cfg["n_embd"]  = int(d_model)
                vals = test_setup_fn(cfg, train_loader, val_loader,
                                  n_runs=n_runs, per_run_seconds=per_run_seconds)
                M[i, j] = float(np.mean(vals))
        matrices.append(M)

    # Global color limits across panels, ignore NaNs
    vmin = np.nanmin([np.nanmin(M) for M in matrices])
    vmax = np.nanmax([np.nanmax(M) for M in matrices])

    # Colormap with masked color for invalid cells
    cmap = mpl.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#d9d9d9")  # light gray for invalid

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=dpi, constrained_layout=True)
    axes = axes.ravel()

    last_im = None
    for idx, (L, M) in enumerate(zip(layers, matrices)):
        ax = axes[idx]
        data = np.ma.array(M, mask=np.isnan(M))
        im = ax.imshow(data, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax, cmap=cmap)
        last_im = im

        # Ticks and labels
        ax.set_xticks(np.arange(len(heads)))
        ax.set_yticks(np.arange(len(dims)))
        ax.set_xticklabels([str(h) for h in heads])
        ax.set_yticklabels([str(d) for d in dims])
        ax.set_xlabel("n_heads")
        ax.set_ylabel("dims (n_embd)")
        ax.set_title(f"Layers = {L}")

        # Mark per-panel minimum
        if np.any(~np.isnan(M)):
            ii, jj = np.unravel_index(np.nanargmin(M), M.shape)
            ax.plot(jj, ii, marker="*", markersize=10, color="black", zorder=3)
            ax.annotate("min", (jj, ii), textcoords="offset points", xytext=(6, 6),
                        ha="left", va="bottom", fontsize=9, weight="bold", color="black")

        # Optional numeric annotations
        if annotate:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                                fontsize=8, color="white",
                                path_effects=[mpl.patheffects.withStroke(linewidth=1.5, foreground="black")])

    # One shared colorbar
    cbar = fig.colorbar(last_im, ax=axes.tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Validation loss (cross-entropy)")

    plt.show()
    return fig, axes, {"layers": layers, "heads": heads, "dims": dims, "matrices": matrices}