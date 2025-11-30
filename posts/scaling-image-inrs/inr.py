import math
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import colors, ticker
from pathlib import Path
from models import GeneralModel
from utils import (
    load_target_image,
    save_loss_curve,
    evaluate_and_save_output,
    plot_moe_activations,
    print_sweep_summary,
)

DEFAULT_BATCH_SIZE = 4096
IMAGE_SIZE = 512
DEFAULT_LR = 1e-4
PLOT_INTERVAL = 5

DEFAULT_MAX_EPOCHS = 160
MAX_SECONDS = 60

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(
    image_name,
    model_args,
    lr,
    batch_size,
    max_epochs,
    report_progress=False,
):

    base_path = Path(__file__).resolve().parent
    image_path = base_path / "images" / image_name
    media_path = base_path / "media"
    media_path.mkdir(exist_ok=True)

    target = load_target_image(image_path, IMAGE_SIZE).to(device) 
    height, width, _ = target.shape
    target_cpu = target.detach().cpu().numpy()

    y_coords = torch.linspace(0.0, 1.0, steps=height, device=device)
    x_coords = torch.linspace(0.0, 1.0, steps=width, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    coords = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)  # (N, 2)

    pixels = target.reshape(-1, 3)  # (N, 3),

    num_samples = coords.shape[0]
    indices = torch.arange(num_samples, device=device)

    model = GeneralModel(**model_args).to(device)
    if report_progress:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_batch_size = batch_size
    eval_batch_size = batch_size

    train_losses = []
    best_val_loss = float("inf")
    start_time = time.time()

    epoch_iterable = (
        tqdm(range(1, max_epochs + 1), desc="Epochs", leave=False)
        if report_progress
        else range(1, max_epochs + 1)
    )

    for epoch in epoch_iterable:
        model.train()
        running_loss = 0.0
        seen_samples = 0

        perm = indices[torch.randperm(num_samples, device=device)]

        for start_idx in range(0, num_samples, train_batch_size):
            end = min(start_idx + train_batch_size, num_samples)
            batch_idx = perm[start_idx:end]

            batch_coords = coords[batch_idx]
            batch_pixels = pixels[batch_idx]

            optimizer.zero_grad()
            preds = model(batch_coords)
            loss = criterion(preds, batch_pixels)
            loss.backward()
            optimizer.step()

            current_batch = batch_coords.size(0)
            running_loss += loss.detach() * current_batch
            seen_samples += current_batch

        train_losses.append(running_loss.item() / max(seen_samples, 1))

        # Compute full-image loss for progress reporting
        model.eval()
        full_loss = 0.0
        full_samples = 0
        with torch.no_grad():
            for start in range(0, num_samples, eval_batch_size):
                end = min(start + eval_batch_size, num_samples)
                batch_coords = coords[start:end]
                batch_pixels = pixels[start:end]
                preds = model(batch_coords)
                current_batch = batch_coords.size(0)
                full_loss += criterion(preds, batch_pixels).item() * current_batch
                full_samples += current_batch

        avg_full_loss = full_loss / max(full_samples, 1)
        best_val_loss = min(best_val_loss, avg_full_loss)
        if report_progress:
            epoch_iterable.set_postfix(loss=f"{avg_full_loss:.2e}")

        if report_progress and epoch % PLOT_INTERVAL == 0:
            save_loss_curve(train_losses, media_path)
            evaluate_and_save_output(
                model,
                criterion,
                eval_batch_size,
                epoch,
                num_samples,
                indices,
                coords,
                pixels,
                media_path,
                target_cpu,
                report_progress,
            )
            if getattr(model, "n_experts", 1) > 1:
                plot_moe_activations(
                    model,
                    coords,
                    indices,
                    height,
                    width,
                    eval_batch_size,
                    media_path,
                )

        # Stop early if we exceed the wall-clock budget
        if MAX_SECONDS is not None:
            elapsed = time.time() - start_time
            if elapsed >= MAX_SECONDS:
                break

    final_val_loss = evaluate_and_save_output(
        model,
        criterion,
        eval_batch_size,
        max_epochs,
        num_samples,
        indices,
        coords,
        pixels,
        media_path,
        target_cpu,
        report_progress,
    )
    best_val_loss = min(best_val_loss, final_val_loss)
    
    if report_progress and getattr(model, "n_experts", 1) > 1:
        plot_moe_activations(
            model,
            coords,
            indices,
            height,
            width,
            eval_batch_size,
            media_path,
        )
    
    if report_progress:
        save_loss_curve(train_losses, media_path)

    return {"best_val_loss": best_val_loss, "train_losses": train_losses}

def sweep_model(
    image_name,
    base_model_args,
    sweep_param,
    sweep_values,
    n_train=5,
    lr=DEFAULT_LR,
    batch_size=DEFAULT_BATCH_SIZE,
    max_epochs=DEFAULT_MAX_EPOCHS,
):
    base_path = Path(__file__).resolve().parent
    media_path = base_path / "media"
    media_path.mkdir(exist_ok=True)

    def apply_param(
        model_args,
        current_lr,
        current_batch_size,
        current_max_epochs,
        param_name,
        param_value,
    ):
        lower_name = str(param_name).lower()
        if lower_name in {"lr", "learning_rate"}:
            current_lr = float(param_value)
        elif lower_name in {"batch_size", "bs"}:
            current_batch_size = int(param_value)
        elif lower_name in {"max_epochs", "epochs"}:
            current_max_epochs = int(param_value)
        else:
            model_args[param_name] = param_value
        return model_args, current_lr, current_batch_size, current_max_epochs

    def format_tick(val):
        try:
            num = float(val)
            if abs(num) < 1:
                return f"{num:.1e}"
            return f"{num:g}"
        except (TypeError, ValueError):
            return str(val)

    is_grid_sweep = isinstance(sweep_param, (list, tuple))

    if is_grid_sweep:
        if not isinstance(sweep_values, (list, tuple)):
            raise ValueError("Grid sweep expects sweep_values to be a sequence.")
        if len(sweep_param) not in {2, 3}:
            raise ValueError("Grid sweep expects two or three parameters.")
        if len(sweep_values) != len(sweep_param):
            raise ValueError("Grid sweep expects sweep_values to match sweep_param length.")

        if len(sweep_param) == 2:
            param_x, param_y = sweep_param
            values_x, values_y = sweep_values

            mean_grid = []
            result_grid = []

            total_runs = len(values_x) * len(values_y) * n_train
            desc = f"Sweeping {param_x} vs {param_y}"
            with tqdm(total=total_runs, desc=desc) as progress:
                for val_x in values_x:
                    row_means = []
                    row_results = []
                    for val_y in values_y:
                        losses = []
                        for run_idx in range(n_train):
                            torch.manual_seed(run_idx)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(run_idx)

                            run_model_args = dict(base_model_args)
                            run_lr, run_bs, run_epochs = lr, batch_size, max_epochs

                            run_model_args, run_lr, run_bs, run_epochs = apply_param(
                                run_model_args, run_lr, run_bs, run_epochs, param_x, val_x
                            )
                            run_model_args, run_lr, run_bs, run_epochs = apply_param(
                                run_model_args, run_lr, run_bs, run_epochs, param_y, val_y
                            )

                            run_result = train_model(
                                image_name,
                                run_model_args,
                                lr=run_lr,
                                batch_size=run_bs,
                                max_epochs=run_epochs,
                                report_progress=False,
                            )
                            losses.append(run_result["best_val_loss"])
                            progress.set_postfix(
                                {
                                    str(param_x): format_tick(val_x),
                                    str(param_y): format_tick(val_y),
                                    "run": f"{run_idx + 1}/{n_train}",
                                }
                            )
                            progress.update(1)

                        mean_loss = sum(losses) / len(losses)
                        if len(losses) > 1:
                            variance = sum((loss - mean_loss) ** 2 for loss in losses) / (len(losses) - 1)
                            std_error = math.sqrt(variance) / math.sqrt(len(losses))
                        else:
                            std_error = 0.0

                        row_means.append(mean_loss)
                        row_results.append(
                            {
                                "values": (val_x, val_y),
                                "losses": losses,
                                "mean_loss": mean_loss,
                                "std_error": std_error,
                            }
                        )

                    mean_grid.append(row_means)
                    result_grid.append(row_results)

            # Let the layout and size adapt to the grid so labels/colorbar fit
            width = max(6.0, 0.9 * len(values_y) + 3.0)
            height = max(4.5, 0.9 * len(values_x) + 2.5)
            fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
            im = ax.imshow(mean_grid, origin="lower", cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(values_y)))
            ax.set_xticklabels([format_tick(v) for v in values_y], rotation=45, ha="right")
            ax.set_yticks(range(len(values_x)))
            ax.set_yticklabels([format_tick(v) for v in values_x])
            ax.set_xlabel(param_y)
            ax.set_ylabel(param_x)
            ax.set_title(f"Grid sweep: {param_x} vs {param_y} (n={n_train})")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mean best validation loss (MSE)")
            cbar.formatter = ticker.FormatStrFormatter("%.1e")
            cbar.update_ticks()
            plot_path = media_path / f"sweep_{param_x}_vs_{param_y}.png"
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)

            return {
                "param": sweep_param,
                "params": [param_x, param_y],
                "values": [list(values_x), list(values_y)],
                "results": result_grid,
                "plot_path": plot_path,
                "bar_plot_path": plot_path,
                "loss_curve_plot_path": None,
            }

        param_a, param_b, param_c = sweep_param
        values_a, values_b, values_c = sweep_values

        panel_results = []
        all_mean_losses = []

        total_runs = len(values_a) * len(values_b) * len(values_c) * n_train
        progress = tqdm(total=total_runs, desc=f"Sweeping {param_a} vs {param_b} vs {param_c}")

        for val_a in values_a:
            mean_grid = []
            result_grid = []
            for val_b in values_b:
                row_means = []
                row_results = []
                for val_c in values_c:
                    losses = []
                    for run_idx in range(n_train):
                        torch.manual_seed(run_idx)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(run_idx)

                        run_model_args = dict(base_model_args)
                        run_lr, run_bs, run_epochs = lr, batch_size, max_epochs

                        run_model_args, run_lr, run_bs, run_epochs = apply_param(
                            run_model_args, run_lr, run_bs, run_epochs, param_a, val_a
                        )
                        run_model_args, run_lr, run_bs, run_epochs = apply_param(
                            run_model_args, run_lr, run_bs, run_epochs, param_b, val_b
                        )
                        run_model_args, run_lr, run_bs, run_epochs = apply_param(
                            run_model_args, run_lr, run_bs, run_epochs, param_c, val_c
                        )

                        run_result = train_model(
                            image_name,
                            run_model_args,
                            lr=run_lr,
                            batch_size=run_bs,
                            max_epochs=run_epochs,
                            report_progress=False,
                        )
                        losses.append(run_result["best_val_loss"])
                        progress.update(1)

                    mean_loss = sum(losses) / len(losses)
                    if len(losses) > 1:
                        variance = sum((loss - mean_loss) ** 2 for loss in losses) / (len(losses) - 1)
                        std_error = math.sqrt(variance) / math.sqrt(len(losses))
                    else:
                        std_error = 0.0

                    row_means.append(mean_loss)
                    all_mean_losses.append(mean_loss)
                    row_results.append(
                        {
                            "values": (val_a, val_b, val_c),
                            "losses": losses,
                            "mean_loss": mean_loss,
                            "std_error": std_error,
                        }
                    )
                mean_grid.append(row_means)
                result_grid.append(row_results)

            panel_results.append(
                {
                    "param_value": val_a,
                    "mean_grid": mean_grid,
                    "results": result_grid,
                }
            )

        progress.close()

        vmin = min(all_mean_losses) if all_mean_losses else None
        vmax = max(all_mean_losses) if all_mean_losses else None
        cmap = "viridis"

        n_panels = len(values_a)
        n_cols = max(1, math.ceil(math.sqrt(n_panels)))
        n_rows = math.ceil(n_panels / n_cols)
        width = max(6.0, 3.4 * n_cols)
        height = max(5.0, 3.2 * n_rows + 0.6)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(width, height),
            squeeze=False,
            layout="constrained",
        )
        axes_list = [ax for ax in axes.flatten()]

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for idx, (val_a, panel) in enumerate(zip(values_a, panel_results)):
            ax = axes_list[idx]
            im = ax.imshow(panel["mean_grid"], origin="lower", cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(range(len(values_c)))
            ax.set_xticklabels([format_tick(v) for v in values_c], rotation=45, ha="right")
            ax.set_yticks(range(len(values_b)))
            ax.set_yticklabels([format_tick(v) for v in values_b])
            ax.set_xlabel(param_c)
            ax.set_ylabel(param_b)
            ax.set_title(f"{param_a} = {format_tick(val_a)}")

        for ax in axes_list[len(values_a) :]:
            ax.axis("off")

        suptitle = f"Grid sweep: {param_a} vs {param_b} vs {param_c} (n={n_train})"
        fig.suptitle(suptitle)
        used_axes = axes_list[: len(values_a)]
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=used_axes,
            fraction=0.035,
            pad=0.02,
        )
        cbar.set_label("Mean best validation loss (MSE)")
        cbar.formatter = ticker.FormatStrFormatter("%.1e")
        cbar.update_ticks()
        plot_path = media_path / f"sweep_{param_a}_vs_{param_b}_vs_{param_c}.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        return {
            "param": sweep_param,
            "params": [param_a, param_b, param_c],
            "values": [list(values_a), list(values_b), list(values_c)],
            "results": panel_results,
            "plot_path": plot_path,
            "bar_plot_path": plot_path,
            "loss_curve_plot_path": None,
        }

    results = []
    mean_losses = []
    std_errors = []
    loss_curves_by_value = {}

    total_runs = len(sweep_values) * n_train
    with tqdm(total=total_runs, desc=f"Sweeping {sweep_param}") as progress:
        for val in sweep_values:
            losses = []
            loss_curves = []
            for run_idx in range(n_train):
                torch.manual_seed(run_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(run_idx)

                model_args = dict(base_model_args)
                run_lr, run_bs, run_epochs = lr, batch_size, max_epochs
                model_args, run_lr, run_bs, run_epochs = apply_param(
                    model_args, run_lr, run_bs, run_epochs, sweep_param, val
                )

                run_result = train_model(
                    image_name,
                    model_args,
                    lr=run_lr,
                    batch_size=run_bs,
                    max_epochs=run_epochs,
                    report_progress=False,
                )
                losses.append(run_result["best_val_loss"])
                loss_curves.append(run_result["train_losses"])
                progress.set_postfix(
                    {
                        str(sweep_param): format_tick(val),
                        "run": f"{run_idx + 1}/{n_train}",
                    }
                )
                progress.update(1)

            mean_loss = sum(losses) / len(losses)
            if len(losses) > 1:
                variance = sum((loss - mean_loss) ** 2 for loss in losses) / (len(losses) - 1)
                std_error = math.sqrt(variance) / math.sqrt(len(losses))
            else:
                std_error = 0.0

            results.append(
                {
                    "value": val,
                    "losses": losses,
                    "mean_loss": mean_loss,
                    "std_error": std_error,
                }
            )
            mean_losses.append(mean_loss)
            std_errors.append(std_error)
            loss_curves_by_value[val] = loss_curves

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_pos = list(range(len(sweep_values)))
    ax.bar(x_pos, mean_losses, yerr=std_errors, capsize=5, color="#4C72B0")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in sweep_values])
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("Best validation loss (MSE)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title(f"Sweep over {sweep_param} (n={n_train})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    bar_plot_path = media_path / f"sweep_{sweep_param}.png"
    fig.savefig(bar_plot_path, dpi=200)
    plt.close(fig)

    # Plot aggregated loss curves with continuous color mapping
    fig, ax = plt.subplots(figsize=(8, 5))
    # Use a blue-to-green palette with good contrast across values
    sweep_blue_green = colors.LinearSegmentedColormap.from_list(
        "sweep_blue_green", ["#6e1eb1", "#1311b1", "#128d2d", "#831212"]
    )
    try:
        numeric_values = [float(v) for v in sweep_values]
        norm = colors.Normalize(vmin=min(numeric_values), vmax=max(numeric_values))
        color_map = {val: sweep_blue_green(norm(float(val))) for val in sweep_values}
    except Exception:
        # Fall back to discrete blue-to-green tones for non-numeric sweeps
        palette = ["#1f77b4", "#3a89b8", "#57a0bc", "#62b2a3", "#56b28f", "#2ca25f"]
        color_map = {
            val: palette[i % len(palette)]
            for i, val in enumerate(sweep_values)
        }

    for val in sweep_values:
        curves = loss_curves_by_value[val]
        if not curves:
            continue
        # Align curves to the shortest run length to handle early stopping
        min_len = min(len(c) for c in curves)
        trimmed = [c[:min_len] for c in curves]

        # Compute mean and standard error per epoch across runs
        mean_curve = []
        std_err_curve = []
        for epoch_losses in zip(*trimmed):
            n = len(epoch_losses)
            mean_epoch = sum(epoch_losses) / n
            if n > 1:
                variance = sum((loss - mean_epoch) ** 2 for loss in epoch_losses) / (n - 1)
                std_err_epoch = math.sqrt(variance) / math.sqrt(n)
            else:
                std_err_epoch = 0.0
            mean_curve.append(mean_epoch)
            std_err_curve.append(std_err_epoch)

        epochs = list(range(1, min_len + 1))
        ax.plot(
            epochs,
            mean_curve,
            label=str(val),
            color=color_map[val],
            linewidth=2,
        )
        upper = [m + s for m, s in zip(mean_curve, std_err_curve)]
        lower = [m - s for m, s in zip(mean_curve, std_err_curve)]
        ax.fill_between(
            epochs,
            lower,
            upper,
            color=color_map[val],
            alpha=0.2,
            linewidth=0,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (MSE)")
    ax.set_title(f"Loss for different {sweep_param}")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title=str(sweep_param))
    fig.tight_layout()
    loss_curve_plot_path = media_path / f"sweep_{sweep_param}_loss_curves.png"
    fig.savefig(loss_curve_plot_path, dpi=200)
    plt.close(fig)

    return {
        "param": sweep_param,
        "values": list(sweep_values),
        "results": results,
        "bar_plot_path": bar_plot_path,
        "loss_curve_plot_path": loss_curve_plot_path,
    }

if __name__ == "__main__":

    """
    CONCEPTS:
    - dont use layernorm and siren together
    """
    
    base_model_args = dict(
        position_encoding=None,
        trainable_embeddings=True,
        freq_scale=50,
        encoding_dim=256,
        n_layers=6,
        n_experts=2,
        hidden_dim=512,
        layernorm="layernorm_post",
        layer_type="siren",
    )

    #sweep_param = ("hidden_dim", "lr")
    #sweep_values = ([256, 384, 512, 768, 1024], [2e-4, 3e-4, 4e-4, 5e-4])

    #sweep_param="batch_size"
    #sweep_values=[512, 1024, 2048, 4096, 8192]

    #sweep_param="lr"
    #sweep_values=[1e-4, 2e-4, 3e-4, 4e-4, 5e-4]

    sweep_param="layernorm"
    sweep_values=["layernorm_post", "layernorm_pre", None]

    #sweep_param="layer_type"
    #sweep_values=["relu", "swish", "gelu"]

    #sweep_param="n_experts"
    #sweep_values=[1, 2, 4, 8]

    #sweep_param="position_encoding"
    #sweep_values=["fourier", "gabor"]

    #sweep_param=("position_encoding", "hidden_dim", "n_layers")
    #sweep_values=(["fourier", "gabor"], [768, 1024, 1280], [3, 4])

    #sweep_param="freq_scale"
    #sweep_values=[50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    #train_model(
    #    image_name="image1.jpg",
    #    model_args=base_model_args,
    #    lr=DEFAULT_LR,
    #    batch_size=DEFAULT_BATCH_SIZE,
    #    max_epochs=DEFAULT_MAX_EPOCHS,
    #    report_progress=True,
    #)

    
    sweep_result = sweep_model(
        image_name="image4.jpg",
        base_model_args=base_model_args,
        sweep_param=sweep_param,
        sweep_values=sweep_values,
        n_train=2,
        lr=DEFAULT_LR,
        batch_size=DEFAULT_BATCH_SIZE,
        max_epochs=DEFAULT_MAX_EPOCHS,
    )

    print_sweep_summary(sweep_result)

# gabor 7e-5
# pre norm layernorm is better for long training sessions
