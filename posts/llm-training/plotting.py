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
    from scipy.stats import norm

    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        raise ValueError("No samples.")

    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    counts, bin_edges, _ = ax.hist(x, bins=bins, density=density,
                                   color="#4c72b0", edgecolor="white", alpha=0.85)

    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    xs = np.linspace(bin_edges[0], bin_edges[-1], 256)
    if sigma > 0:
        pdf = norm.pdf(xs, loc=mu, scale=sigma)
        if density:
            curve = pdf
        else:
            # Scale to histogram counts by expected bin width
            bin_widths = np.diff(bin_edges)
            avg_width = float(np.mean(bin_widths)) if bin_widths.size > 0 else 1.0
            curve = pdf * x.size * avg_width
        ax.plot(xs, curve, color="#dd8452", linewidth=2.0)

    ax.set_xlabel("validation loss (cross-entropy)")
    if density:
        ax.set_ylabel("density")
    else:
        ax.set_ylabel("")
    ax.set_title(title or f"Validation loss histogram (n={x.size})")
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
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
       Tight y-limits, adjacent bars, distinct colors."""
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
       Tight y-limits, adjacent bars, distinct colors."""
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

    labels = _labels_for_binary_option(option_key, option_values)
    ax.set_xticks(x, labels)
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


def _labels_for_binary_option(option_key, option_values):
    key = option_key.lower()
    pretty = {
        "norm": {
            "layer": "LayerNorm",
            "rmsnorm": "RMSNorm",
        },
        "ffn": {
            "mlp": "MLP",
            "swiglu": "SwiGLU",
        },
        "prepost": {
            "pre": "Pre-norm",
            "post": "Post-norm",
        },
    }
    if key in pretty:
        mapping = pretty[key]
        return [mapping.get(str(v).lower(), str(v)) for v in option_values]
    return [str(v) for v in option_values]

def plot_gradclip_bars(train_loader, val_loader, *,
                       clips=(None, 0.5, 1.0, 2.0),
                       n_runs=5, per_run_seconds=3,
                       base_args=None, alpha=0.05, dpi=180,
                       test_setup_fn=None):
    return plot_option_bars(
        train_loader, val_loader,
        option_key="grad_clip", option_values=clips,
        n_runs=n_runs, per_run_seconds=per_run_seconds,
        base_args=base_args, alpha=alpha, dpi=dpi,
        title="Gradient clipping sweep",
        label_fmt=lambda v: "None" if v is None else f"{v:g}",
        test_setup_fn=test_setup_fn
    )

def plot_dropout_bars(train_loader, val_loader, *,
                      drops=(0.0, 0.05, 0.10, 0.15, 0.20),
                      n_runs=5, per_run_seconds=3,
                      base_args=None, alpha=0.05, dpi=180,
                      test_setup_fn=None):
    return plot_option_bars(
        train_loader, val_loader,
        option_key="dropout", option_values=drops,
        n_runs=n_runs, per_run_seconds=per_run_seconds,
        base_args=base_args, alpha=alpha, dpi=dpi,
        title="Dropout sweep",
        label_fmt=lambda v: f"{v:.2f}",
        test_setup_fn=test_setup_fn
    )


def plot_layers_heads_dims_heatmaps(train_loader, val_loader, *,
                                    layers, heads, d_models,
                                    base_args, n_runs=3, per_run_seconds=3,
                                    dpi=150, annotate=False, cmap_name="viridis",
                                    test_setup_fn=None):
    """
    Four stacked heatmaps (one per layer count) in a single figure.
    x-axis: n_head (len=4), y-axis: d_model (n_embd, len=4).
    Invalid combos where d_model % n_head != 0 are masked.
    Cell value = mean validation cross-entropy over n_runs.
    """
    assert len(layers) == 4 and len(heads) == 4 and len(d_models) == 4, "Use 4 options per dimension."

    # Collect results per layer into 4 matrices of shape (len(d_models), len(heads))
    matrices = []
    stats = {}
    for L in layers:
        M = np.full((len(d_models), len(heads)), np.nan, dtype=float)
        for i, d_model in enumerate(d_models):
            for j, h in enumerate(heads):
                if d_model % h != 0:
                    continue  # invalid: head_dim not integer
                cfg = base_args.copy()
                cfg["n_layer"] = int(L)
                cfg["n_head"]  = int(h)
                cfg["n_embd"]  = int(d_model)
                vals = test_setup_fn(cfg, train_loader, val_loader,
                                  n_runs=n_runs, per_run_seconds=per_run_seconds)
                vals = np.asarray(vals, dtype=float)
                if vals.size == 0:
                    continue
                mean_val = float(np.mean(vals))
                M[i, j] = mean_val
                stats[(int(L), int(h), int(d_model))] = {
                    "values": vals,
                    "mean": mean_val,
                }
        matrices.append(M)

    # Global color limits across panels, ignore NaNs
    vmin = np.nanmin([np.nanmin(M) for M in matrices])
    vmax = np.nanmax([np.nanmax(M) for M in matrices])

    # Colormap with masked color for invalid cells
    cmap = mpl.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#d9d9d9")  # light gray for invalid

    fig, axes = plt.subplots(len(layers), 1, figsize=(9, 12), dpi=dpi, constrained_layout=True)
    axes = np.atleast_1d(axes)

    # Determine global best configuration across all combinations
    global_key = None
    if stats:
        global_key = min(stats.keys(), key=lambda k: stats[k]["mean"])

    last_im = None
    for idx, (L, M) in enumerate(zip(layers, matrices)):
        ax = axes[idx]
        data = np.ma.array(M, mask=np.isnan(M))
        im = ax.imshow(data, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax, cmap=cmap)
        last_im = im

        # Ticks and labels
        ax.set_xticks(np.arange(len(heads)))
        ax.set_yticks(np.arange(len(d_models)))
        ax.set_xticklabels([str(h) for h in heads])
        ax.set_yticklabels([str(d) for d in d_models])
        ax.set_xlabel("n_heads")
        ax.set_ylabel("d_model")
        ax.set_title(f"Layers = {L}")

        # Mark per-panel minimum
        if np.any(~np.isnan(M)):
            ii, jj = np.unravel_index(np.nanargmin(M), M.shape)
            ax.scatter([jj], [ii], marker="o", s=110,
                       facecolors="none", edgecolors="white", linewidths=2.2, zorder=3)
            ax.scatter([jj], [ii], marker="o", s=40, color="black", zorder=4)

            combo = (int(L), int(heads[jj]), int(d_models[ii]))
            if global_key is not None and combo == global_key:
                ax.scatter([jj], [ii], marker="X", s=140, color="#ff6f59",
                           edgecolors="white", linewidths=1.2, zorder=5)

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
    return fig, axes, {
        "layers": layers,
        "heads": heads,
        "d_models": d_models,
        "matrices": matrices,
        "stats": stats,
        "global_best": global_key,
    }


def plot_layers_heads_dims_bars(results, *, alpha=0.05, dpi=150,
                                title="Validation loss by (layers, heads, d_model)"):
    if not results:
        raise ValueError("No results provided.")

    entries = []
    for (layers, heads, d_model), info in results.items():
        vals = np.asarray(info.get("values", []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean = float(np.mean(vals))
        if vals.size > 1:
            s = float(np.std(vals, ddof=1))
            se = s / np.sqrt(vals.size)
            tcrit = float(student_t.ppf(1 - alpha / 2, vals.size - 1))
            ci = tcrit * se
        else:
            ci = 0.0
        entries.append(((layers, heads, d_model), mean, ci, vals.size))

    if not entries:
        raise ValueError("Results did not contain any finite values.")

    entries.sort(key=lambda item: item[1])
    cfg_labels = [f"L={cfg[0]}\nH={cfg[1]}\nd_model={cfg[2]}" for cfg, *_ in entries]
    means = [mean for _, mean, _, _ in entries]
    cis = [ci for _, _, ci, _ in entries]

    width = max(9, 0.6 * len(entries))
    fig, ax = plt.subplots(figsize=(width, 6), dpi=dpi)

    x = np.arange(len(entries))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(entries) - 1)) for i in range(len(entries))]
    ax.bar(x, means, yerr=cis, capsize=3, color=colors, edgecolor="none")

    ax.set_xticks(x, cfg_labels)
    ax.set_ylabel("validation loss (cross-entropy)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    plt.show()

    return fig, ax, entries
