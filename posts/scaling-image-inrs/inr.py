import math
import time
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from pathlib import Path
from models import GeneralModel
from utils import load_target_image, save_loss_curve, evaluate_and_save_output, plot_moe_activations

DEFAULT_BATCH_SIZE = 4096
IMAGE_SIZE = 256
DEFAULT_LR = 2e-4
PLOT_INTERVAL = 5

DEFAULT_MAX_EPOCHS = 60
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
        if len(sweep_param) != 2:
            raise ValueError("Grid sweep expects exactly two parameters.")
        if not isinstance(sweep_values, (list, tuple)) or len(sweep_values) != 2:
            raise ValueError("Grid sweep expects sweep_values to be a pair of value lists.")

        param_x, param_y = sweep_param
        values_x, values_y = sweep_values

        mean_grid = []
        result_grid = []

        for val_x in tqdm(values_x, desc=f"Sweeping {param_x}"):
            row_means = []
            row_results = []
            for val_y in tqdm(values_y, desc=f"{param_x}={val_x}", leave=False):
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

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(mean_grid, origin="lower", cmap="viridis")
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
        fig.tight_layout()
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

    results = []
    mean_losses = []
    std_errors = []
    loss_curves_by_value = {}

    outer_progress = tqdm(sweep_values, desc=f"Sweeping {sweep_param}")

    for val in outer_progress:
        losses = []
        loss_curves = []
        inner_progress = tqdm(range(n_train), desc=f"{sweep_param}={val}", leave=False)
        for run_idx in inner_progress:
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
        "sweep_blue_green", ["#1f77b4", "#2ca25f"]
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
    base_model_args = dict(
        position_encoding="gabor",
        trainable_embeddings=True,
        freq_scale=70,
        encoding_dim=256,
        n_layers=6,
        n_experts=2,
        hidden_dim=512,
        layernorm=True,
        layer_type="relu",
    )

    sweep_param = ("hidden_dim", "lr")
    sweep_values = ([256, 384, 512, 768, 1024], [2e-4, 3e-4, 4e-4, 5e-4])

    #sweep_param="batch_size"
    #sweep_values=[512, 1024, 2048, 4096, 8192]

    #sweep_param="layernorm"
    #sweep_values=[True, False]

    #sweep_param="layer_type"
    #sweep_values=["relu", "swish", "gelu"]

    #sweep_param="n_experts"
    #sweep_values=[1, 2, 4, 8]

    #sweep_param="position_encoding"
    #sweep_values=["fourier", "gabor"]


    sweep_result = sweep_model(
        image_name="image4.jpg",
        base_model_args=base_model_args,
        sweep_param=sweep_param,
        sweep_values=sweep_values,
        n_train=4,
        lr=DEFAULT_LR,
        batch_size=DEFAULT_BATCH_SIZE,
        max_epochs=DEFAULT_MAX_EPOCHS,
    )






    print(f"Sweep finished. Bar plot: {sweep_result['bar_plot_path']}")
    if sweep_result.get("loss_curve_plot_path"):
        print(f"Loss curves plot: {sweep_result['loss_curve_plot_path']}")
    if sweep_result.get("params"):
        p_x, p_y = sweep_result["params"]
        flat_results = [item for row in sweep_result["results"] for item in row]
        for res in flat_results:
            vx, vy = res["values"]
            print(
                f"{p_x}={vx}, {p_y}={vy} "
                f"mean={res['mean_loss']:.3e} stderr={res['std_error']:.3e}"
            )
    else:
        for res in sweep_result["results"]:
            print(
                f"{sweep_result['param']}={res['value']} "
                f"mean={res['mean_loss']:.3e} stderr={res['std_error']:.3e}"
            )

# gabor 7e-5
