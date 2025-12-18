import math
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import colors, ticker
from pathlib import Path
import hashlib
import schedulefree
from models import GeneralModel
from utils import (
    load_target_image,
    save_rgb_image,
    save_target_image,
    save_loss_curve,
    save_loss_curves_combined,
    evaluate_model,
    save_prediction_with_overlay,
    plot_moe_activations,
    create_mp4_from_frames,
    print_sweep_summary,
)

DEFAULT_BATCH_SIZE = 4096
IMAGE_SIZE = 768
DEFAULT_LR = 4e-4
PLOT_INTERVAL = 5

DEFAULT_MAX_EPOCHS = 200
MAX_SECONDS = 180

device = "cuda" if torch.cuda.is_available() else "cpu"

def _safe_image_tag(name):
    stem = Path(str(name)).stem.strip().lower().replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)
    return safe or "image"

def _combined_tag(image_names, max_len=80):
    tags = [_safe_image_tag(name) for name in image_names]
    combined = "__".join(tags)
    if len(combined) <= max_len:
        return combined
    digest = hashlib.sha1(combined.encode("utf-8")).hexdigest()[:8]
    return combined[: max_len - 11] + "__" + digest

def _save_combined_video_frame(target_paths, predicted_paths, titles, output_path):
    n_images = len(target_paths)
    fig, axes = plt.subplots(2, n_images, figsize=(4 * n_images, 8), squeeze=False)

    for col in range(n_images):
        target_img = plt.imread(str(target_paths[col]))
        pred_img = plt.imread(str(predicted_paths[col]))

        ax_top = axes[0][col]
        ax_bottom = axes[1][col]

        ax_top.imshow(target_img)
        ax_top.axis("off")
        if titles is not None:
            ax_top.set_title(str(titles[col]))

        ax_bottom.imshow(pred_img)
        ax_bottom.axis("off")

    fig.tight_layout(pad=0.2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def train_model(
    image_name,
    model_args,
    lr,
    batch_size,
    max_epochs,
    report_progress=False,
    save_video=False,
    video_fps=10,
):
    if isinstance(image_name, (list, tuple)):
        base_path = Path(__file__).resolve().parent
        media_path = base_path / "media"
        media_path.mkdir(exist_ok=True)

        per_image_results = []
        for name in image_name:
            result = train_model(
                image_name=name,
                model_args=model_args,
                lr=lr,
                batch_size=batch_size,
                max_epochs=max_epochs,
                report_progress=report_progress,
                save_video=save_video,
                video_fps=video_fps,
            )
            result["image_name"] = name
            per_image_results.append(result)

        combined_loss_curve_path = None
        if len(per_image_results) > 1:
            losses_by_label = {}
            for res in per_image_results:
                label = _safe_image_tag(res.get("image_name", "image"))
                losses_by_label[label] = res.get("train_losses", [])
            combined_loss_curve_path = media_path / "loss_curve_combined.png"
            save_loss_curves_combined(
                losses_by_label,
                combined_loss_curve_path,
                title="Training Loss Curves (Combined)",
            )

        combined_frames_dir = None
        combined_video_path = None
        if save_video:
            combined_frames_dir = media_path / f"snapshots_combined_{_combined_tag(image_name)}"
            combined_frames_dir.mkdir(parents=True, exist_ok=True)
            for existing in combined_frames_dir.glob("frame_*.png"):
                existing.unlink(missing_ok=True)

            targets = []
            per_image_frames = []
            titles = []
            for res in per_image_results:
                frames_dir = res.get("frames_dir")
                if not frames_dir:
                    continue
                frames_dir = Path(frames_dir)
                target_path = frames_dir / "target.png"
                frame_paths = sorted(frames_dir.glob("frame_*.png"))
                if not target_path.exists() or not frame_paths:
                    continue
                targets.append(target_path)
                per_image_frames.append(frame_paths)
                titles.append(_safe_image_tag(res.get("image_name", "image")))

            if targets and per_image_frames and len(targets) == len(per_image_frames):
                max_len = max(len(paths) for paths in per_image_frames)
                for idx in range(max_len):
                    predicted = []
                    for paths in per_image_frames:
                        if idx < len(paths):
                            predicted.append(paths[idx])
                        else:
                            predicted.append(paths[-1])
                    _save_combined_video_frame(
                        targets,
                        predicted,
                        titles,
                        combined_frames_dir / f"frame_{idx:06d}.png",
                    )

                combined_video_path = media_path / f"training_combined_{_combined_tag(image_name)}.mp4"
                try:
                    create_mp4_from_frames(combined_frames_dir, combined_video_path, fps=video_fps)
                except Exception as exc:
                    print(f"[save_video] Failed to create combined mp4: {exc}")
                    combined_video_path = None

        return {
            "per_image_results": per_image_results,
            "combined_loss_curve_path": combined_loss_curve_path,
            "combined_frames_dir": combined_frames_dir,
            "combined_video_path": combined_video_path,
        }

    base_path = Path(__file__).resolve().parent
    image_path = base_path / "images" / image_name
    media_path = base_path / "media"
    media_path.mkdir(exist_ok=True)

    image_tag = _safe_image_tag(image_name)
    final_output_path = media_path / f"final_output_{image_tag}.png"
    loss_curve_path = media_path / f"loss_curve_{image_tag}.png"
    moe_activations_path = media_path / f"moe_activations_{image_tag}.png"

    frames_dir = None
    video_path = None
    if save_video:
        frames_dir = media_path / f"snapshots_{image_tag}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for existing in frames_dir.glob("frame_*.png"):
            existing.unlink(missing_ok=True)
        video_path = media_path / f"training_{image_tag}.mp4"

    target = load_target_image(image_path, IMAGE_SIZE).to(device) 
    height, width, _ = target.shape
    target_cpu = target.detach().cpu().numpy()

    if frames_dir is not None:
        save_target_image(target_cpu, frames_dir / "target.png")

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

    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_batch_size = batch_size
    eval_batch_size = batch_size

    train_losses = []
    val_psnrs = []
    best_val_loss = float("inf")
    start_time = time.time()
    last_epoch = 0
    frame_idx = 0

    epoch_iterable = (
        tqdm(range(1, max_epochs + 1), desc="Epochs", leave=False)
        if report_progress
        else range(1, max_epochs + 1)
    )

    for epoch in epoch_iterable:
        model.train()
        optimizer.train()
        running_loss = 0.0
        seen_samples = 0
        last_epoch = epoch

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
        optimizer.eval()
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
        # Compute PSNR on the full image (pixels are in [0,1])
        psnr_full = 10.0 * math.log10(1.0 / max(avg_full_loss, 1e-12))
        val_psnrs.append(psnr_full)
        best_val_loss = min(best_val_loss, avg_full_loss)
        if report_progress:
            epoch_iterable.set_postfix(loss=f"{avg_full_loss:.2e}")

        if epoch % PLOT_INTERVAL == 0:
            if report_progress:
                save_loss_curve(
                    train_losses,
                    media_path,
                    filename=loss_curve_path.name,
                    title=f"Training Loss Curve ({image_tag})",
                )

            snapshot_path = None
            if frames_dir is not None:
                snapshot_path = frames_dir / f"frame_{frame_idx:06d}.png"
                frame_idx += 1
            elif report_progress:
                snapshot_path = media_path / f"last_output_{image_tag}.png"

            if snapshot_path is not None:
                predicted_image, mse, psnr = evaluate_model(
                    model,
                    criterion,
                    eval_batch_size,
                    num_samples,
                    indices,
                    coords,
                    pixels,
                    target_cpu.shape,
                )
                save_prediction_with_overlay(predicted_image, epoch, mse, psnr, snapshot_path)
                if report_progress and getattr(model, "n_experts", 1) > 1:
                    plot_moe_activations(
                        model,
                        coords,
                        indices,
                        height,
                        width,
                        eval_batch_size,
                        media_path,
                        filename=moe_activations_path.name,
                    )

        # Stop early if we exceed the wall-clock budget
        if MAX_SECONDS is not None:
            elapsed = time.time() - start_time
            if elapsed >= MAX_SECONDS:
                break

    predicted_final, final_val_loss, final_psnr = evaluate_model(
        model,
        criterion,
        eval_batch_size,
        num_samples,
        indices,
        coords,
        pixels,
        target_cpu.shape,
    )
    save_rgb_image(predicted_final, final_output_path)
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
            filename=moe_activations_path.name,
        )
    
    if report_progress:
        save_loss_curve(
            train_losses,
            media_path,
            filename=loss_curve_path.name,
            title=f"Training Loss Curve ({image_tag})",
        )

    if frames_dir is not None:
        final_frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
        save_prediction_with_overlay(
            predicted_final,
            max(last_epoch, 1),
            final_val_loss,
            final_psnr,
            final_frame_path,
        )
        try:
            create_mp4_from_frames(frames_dir, video_path, fps=video_fps)
        except Exception as exc:
            print(f"[save_video] Failed to create mp4 for {image_name}: {exc}")
            video_path = None

    return {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_psnrs": val_psnrs,
        "final_output_path": final_output_path,
        "video_path": video_path,
        "frames_dir": frames_dir,
    }

def sweep_model(
    image_name,
    base_model_args,
    sweep_param,
    sweep_values,
    n_train=5,
    default_lr=DEFAULT_LR,
    default_batch_size=DEFAULT_BATCH_SIZE,
    default_max_epochs=DEFAULT_MAX_EPOCHS,
    print_estimate=True,
):
    base_path = Path(__file__).resolve().parent
    media_path = base_path / "media"
    media_path.mkdir(exist_ok=True)

    def _normalize_component(value):
        text = str(value).strip().lower().replace(" ", "_")
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
        return safe or "sweep"

    def _sweep_folder_name(param):
        if isinstance(param, (list, tuple)):
            return "_vs_".join(_normalize_component(p) for p in param)
        return _normalize_component(param)

    sweep_media_path = media_path / _sweep_folder_name(sweep_param)
    sweep_media_path.mkdir(parents=True, exist_ok=True)

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

    def safe_image_tag(name):
        return Path(name).stem.replace(" ", "_")

    def mean_and_stderr(values):
        if not values:
            return float("nan"), float("nan")
        mean_val = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((val - mean_val) ** 2 for val in values) / (len(values) - 1)
            stderr = math.sqrt(variance) / math.sqrt(len(values))
        else:
            stderr = 0.0
        return mean_val, stderr

    def finite_mean_and_stderr(values):
        finite = [val for val in values if math.isfinite(val)]
        return mean_and_stderr(finite)

    def run_means(per_image_series, n_runs):
        means = []
        for run_idx in range(n_runs):
            run_vals = []
            for series in per_image_series:
                if series is None or run_idx >= len(series):
                    continue
                val = series[run_idx]
                if math.isfinite(val):
                    run_vals.append(val)
            if run_vals:
                means.append(sum(run_vals) / len(run_vals))
        return means

    is_grid_sweep = isinstance(sweep_param, (list, tuple))

    def estimate_sweep_time():
        if not print_estimate:
            return

        try:
            if is_grid_sweep:
                if not isinstance(sweep_values, (list, tuple)):
                    return
                if len(sweep_param) == 2 and len(sweep_values) == 2:
                    combinations = len(sweep_values[0]) * len(sweep_values[1])
                elif len(sweep_param) == 3 and len(sweep_values) == 3:
                    combinations = len(sweep_values[0]) * len(sweep_values[1]) * len(sweep_values[2])
                else:
                    return
            else:
                combinations = len(sweep_values)
        except Exception:
            return

        if combinations <= 0:
            return

        images_count = len(image_name) if isinstance(image_name, (list, tuple)) else 1
        seconds_per_run = MAX_SECONDS if MAX_SECONDS is not None else 0
        if seconds_per_run <= 0:
            return

        total_seconds = seconds_per_run * n_train * combinations * images_count
        hours = total_seconds / 3600.0
        print(
            f"Estimated sweep time: ~{hours:.3g} hours "
            f"({images_count} image(s) × {combinations} combo(s) × n_train={n_train}, ~{seconds_per_run:.3g}s/run)"
        )

    estimate_sweep_time()

    def build_lookup(single_result):
        lookup = {}
        params = single_result.get("params")
        if not params:
            for res in single_result.get("results", []):
                lookup[(res["value"],)] = res
            return lookup
        if len(params) == 2:
            for row in single_result.get("results", []):
                for res in row:
                    lookup[tuple(res["values"])] = res
            return lookup
        for panel in single_result.get("results", []):
            for row in panel["results"]:
                for res in row:
                    lookup[tuple(res["values"])] = res
        return lookup

    def combine_image_results(per_image_results):
        if not per_image_results:
            return None
        lookups = [build_lookup(res) for res in per_image_results]
        n_images = len(per_image_results)

        if not is_grid_sweep:
            mean_losses = []
            std_errors = []
            mean_psnrs = []
            std_error_psnrs = []
            combined_results = []
            for val in sweep_values:
                key = (val,)
                per_image_losses = [
                    lookup[key].get("losses", [])
                    for lookup in lookups
                    if key in lookup
                ]
                per_image_psnrs = [
                    lookup[key].get("psnrs", [])
                    for lookup in lookups
                    if key in lookup
                ]

                run_avg_losses = run_means(per_image_losses, n_train)
                run_avg_psnrs = run_means(per_image_psnrs, n_train)

                mean_loss, std_error = mean_and_stderr(run_avg_losses)
                mean_psnr_val, std_err_psnr = finite_mean_and_stderr(run_avg_psnrs)
                combined_results.append(
                    {
                        "value": val,
                        "mean_loss": mean_loss,
                        "std_error": std_error,
                        "mean_psnr": mean_psnr_val,
                        "std_error_psnr": std_err_psnr,
                    }
                )
                mean_losses.append(mean_loss)
                std_errors.append(std_error)
                mean_psnrs.append(mean_psnr_val)
                std_error_psnrs.append(std_err_psnr)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            x_pos = list(range(len(sweep_values)))
            ax.bar(x_pos, mean_losses, yerr=std_errors, capsize=5, color="#4C72B0")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(v) for v in sweep_values])
            ax.set_xlabel(sweep_param)
            ax.set_ylabel("Mean best validation loss (MSE)")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_title(
                f"Sweep over {sweep_param} (mean across {n_images} images, stderr over n={n_train})"
            )
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            fig.tight_layout()
            bar_plot_path = sweep_media_path / f"sweep_{sweep_param}_mse_combined.png"
            fig.savefig(bar_plot_path, dpi=200)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            psnr_heights = [val if math.isfinite(val) else float("nan") for val in mean_psnrs]
            psnr_err = [err if math.isfinite(err) else 0.0 for err in std_error_psnrs]
            ax.bar(x_pos, psnr_heights, yerr=psnr_err, capsize=5, color="#D55E00")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(v) for v in sweep_values])
            ax.set_xlabel(sweep_param)
            ax.set_ylabel("Mean PSNR across images (dB)")
            ax.set_title(
                f"PSNR over {sweep_param} (mean across {n_images} images, stderr over n={n_train})"
            )
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            fig.tight_layout()
            psnr_bar_plot_path = sweep_media_path / f"sweep_{sweep_param}_psnr_combined.png"
            fig.savefig(psnr_bar_plot_path, dpi=200)
            plt.close(fig)

            return {
                "param": sweep_param,
                "values": list(sweep_values),
                "results": combined_results,
                "bar_plot_path": bar_plot_path,
                "psnr_bar_plot_path": psnr_bar_plot_path,
                "loss_curve_plot_path": None,
                "psnr_curve_plot_path": None,
                "image_name": "combined",
            }

        if len(sweep_param) == 2:
            param_x, param_y = sweep_param
            values_x, values_y = sweep_values
            mean_grid = []
            psnr_grid = []
            result_grid = []
            for val_x in values_x:
                row_means = []
                row_psnrs = []
                row_results = []
                for val_y in values_y:
                    key = (val_x, val_y)
                    per_image_losses = [
                        lookup[key].get("losses", [])
                        for lookup in lookups
                        if key in lookup
                    ]
                    per_image_psnrs = [
                        lookup[key].get("psnrs", [])
                        for lookup in lookups
                        if key in lookup
                    ]
                    run_avg_losses = run_means(per_image_losses, n_train)
                    run_avg_psnrs = run_means(per_image_psnrs, n_train)
                    mean_loss, std_error = mean_and_stderr(run_avg_losses)
                    mean_psnr_val, std_err_psnr = finite_mean_and_stderr(run_avg_psnrs)
                    row_means.append(mean_loss)
                    row_psnrs.append(mean_psnr_val)
                    row_results.append(
                        {
                            "values": (val_x, val_y),
                            "mean_loss": mean_loss,
                            "std_error": std_error,
                            "mean_psnr": mean_psnr_val,
                            "std_error_psnr": std_err_psnr,
                        }
                    )
                mean_grid.append(row_means)
                psnr_grid.append(row_psnrs)
                result_grid.append(row_results)

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
            ax.set_title(
                f"Grid sweep: {param_x} vs {param_y} (mean across {n_images} images, n={n_train})"
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mean best validation loss (MSE)")
            cbar.formatter = ticker.FormatStrFormatter("%.1e")
            cbar.update_ticks()
            bar_plot_path = sweep_media_path / f"sweep_{param_x}_vs_{param_y}_mse_combined.png"
            fig.savefig(bar_plot_path, dpi=200)
            plt.close(fig)

            psnr_bar_plot_path = None
            psnr_values = [val for row in psnr_grid for val in row if math.isfinite(val)]
            if psnr_values:
                fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
                im = ax.imshow(
                    psnr_grid,
                    origin="lower",
                    cmap="magma",
                    aspect="auto",
                    vmin=min(psnr_values),
                    vmax=max(psnr_values),
                )
                ax.set_xticks(range(len(values_y)))
                ax.set_xticklabels([format_tick(v) for v in values_y], rotation=45, ha="right")
                ax.set_yticks(range(len(values_x)))
                ax.set_yticklabels([format_tick(v) for v in values_x])
                ax.set_xlabel(param_y)
                ax.set_ylabel(param_x)
                ax.set_title(
                    f"Grid sweep PSNR: {param_x} vs {param_y} (mean across {n_images} images)"
                )
                cbar_psnr = fig.colorbar(im, ax=ax)
                cbar_psnr.set_label("Mean PSNR (dB)")
                psnr_bar_plot_path = sweep_media_path / f"sweep_{param_x}_vs_{param_y}_psnr_combined.png"
                fig.savefig(psnr_bar_plot_path, dpi=200)
                plt.close(fig)

            return {
                "param": sweep_param,
                "params": [param_x, param_y],
                "values": [list(values_x), list(values_y)],
                "results": result_grid,
                "plot_path": bar_plot_path,
                "bar_plot_path": bar_plot_path,
                "psnr_bar_plot_path": psnr_bar_plot_path,
                "loss_curve_plot_path": None,
                "psnr_curve_plot_path": None,
                "image_name": "combined",
            }

        param_a, param_b, param_c = sweep_param
        values_a, values_b, values_c = sweep_values
        panel_results = []
        all_mean_losses = []
        all_mean_psnrs = []
        for val_a in values_a:
            mean_grid = []
            psnr_grid = []
            result_grid = []
            for val_b in values_b:
                row_means = []
                row_psnrs = []
                row_results = []
                for val_c in values_c:
                    key = (val_a, val_b, val_c)
                    per_image_losses = [
                        lookup[key].get("losses", [])
                        for lookup in lookups
                        if key in lookup
                    ]
                    per_image_psnrs = [
                        lookup[key].get("psnrs", [])
                        for lookup in lookups
                        if key in lookup
                    ]
                    run_avg_losses = run_means(per_image_losses, n_train)
                    run_avg_psnrs = run_means(per_image_psnrs, n_train)
                    mean_loss, std_error = mean_and_stderr(run_avg_losses)
                    mean_psnr_val, std_err_psnr = finite_mean_and_stderr(run_avg_psnrs)
                    if math.isfinite(mean_loss):
                        all_mean_losses.append(mean_loss)
                    if math.isfinite(mean_psnr_val):
                        all_mean_psnrs.append(mean_psnr_val)
                    row_means.append(mean_loss)
                    row_psnrs.append(mean_psnr_val)
                    row_results.append(
                        {
                            "values": (val_a, val_b, val_c),
                            "mean_loss": mean_loss,
                            "std_error": std_error,
                            "mean_psnr": mean_psnr_val,
                            "std_error_psnr": std_err_psnr,
                        }
                    )
                mean_grid.append(row_means)
                psnr_grid.append(row_psnrs)
                result_grid.append(row_results)

            panel_results.append(
                {
                    "param_value": val_a,
                    "mean_grid": mean_grid,
                    "psnr_grid": psnr_grid,
                    "results": result_grid,
                }
            )

        vmin = min(all_mean_losses) if all_mean_losses else None
        vmax = max(all_mean_losses) if all_mean_losses else None
        psnr_vmin = min(all_mean_psnrs) if all_mean_psnrs else None
        psnr_vmax = max(all_mean_psnrs) if all_mean_psnrs else None
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

        fig.suptitle(
            f"Grid sweep: {param_a} vs {param_b} vs {param_c} (mean across {n_images} images)"
        )
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
        bar_plot_path = sweep_media_path / f"sweep_{param_a}_vs_{param_b}_vs_{param_c}_mse_combined.png"
        fig.savefig(bar_plot_path, dpi=200)
        plt.close(fig)

        psnr_bar_plot_path = None
        if psnr_vmin is not None and psnr_vmax is not None:
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(width, height),
                squeeze=False,
                layout="constrained",
            )
            axes_list = [ax for ax in axes.flatten()]
            norm_psnr = colors.Normalize(vmin=psnr_vmin, vmax=psnr_vmax)
            for idx, (val_a, panel) in enumerate(zip(values_a, panel_results)):
                ax = axes_list[idx]
                im = ax.imshow(
                    panel["psnr_grid"],
                    origin="lower",
                    cmap="magma",
                    norm=norm_psnr,
                    aspect="auto",
                )
                ax.set_xticks(range(len(values_c)))
                ax.set_xticklabels([format_tick(v) for v in values_c], rotation=45, ha="right")
                ax.set_yticks(range(len(values_b)))
                ax.set_yticklabels([format_tick(v) for v in values_b])
                ax.set_xlabel(param_c)
                ax.set_ylabel(param_b)
                ax.set_title(f"{param_a} = {format_tick(val_a)}")

            for ax in axes_list[len(values_a) :]:
                ax.axis("off")

            fig.suptitle(
                f"Grid PSNR: {param_a} vs {param_b} vs {param_c} (mean across {n_images} images)"
            )
            used_axes = axes_list[: len(values_a)]
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm_psnr, cmap="magma"),
                ax=used_axes,
                fraction=0.035,
                pad=0.02,
            )
            cbar.set_label("Mean PSNR (dB)")
            psnr_bar_plot_path = sweep_media_path / f"sweep_{param_a}_vs_{param_b}_vs_{param_c}_psnr_combined.png"
            fig.savefig(psnr_bar_plot_path, dpi=200)
            plt.close(fig)

        return {
            "param": sweep_param,
            "params": [param_a, param_b, param_c],
            "values": [list(values_a), list(values_b), list(values_c)],
            "results": panel_results,
            "plot_path": bar_plot_path,
            "bar_plot_path": bar_plot_path,
            "psnr_bar_plot_path": psnr_bar_plot_path,
            "loss_curve_plot_path": None,
            "psnr_curve_plot_path": None,
            "image_name": "combined",
        }

    if isinstance(image_name, (list, tuple)):
        image_list = list(image_name)
        if not image_list:
            raise ValueError("At least one image_name must be provided.")
        per_image_results = [
            sweep_model(
                image_name=img_name,
                base_model_args=base_model_args,
                sweep_param=sweep_param,
                sweep_values=sweep_values,
                n_train=n_train,
                default_lr=default_lr,
                default_batch_size=default_batch_size,
                default_max_epochs=default_max_epochs,
                print_estimate=False,
            )
            for img_name in image_list
        ]
        combined_result = combine_image_results(per_image_results)
        return {
            "param": sweep_param,
            "values": combined_result.get("values") if combined_result else list(sweep_values),
            "results": combined_result.get("results") if combined_result else [],
            "bar_plot_path": combined_result.get("bar_plot_path") if combined_result else None,
            "psnr_bar_plot_path": combined_result.get("psnr_bar_plot_path") if combined_result else None,
            "loss_curve_plot_path": combined_result.get("loss_curve_plot_path") if combined_result else None,
            "psnr_curve_plot_path": combined_result.get("psnr_curve_plot_path") if combined_result else None,
            "per_image_results": per_image_results,
            "combined_result": combined_result,
        }

    image_tag = safe_image_tag(image_name)

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
            psnr_grid = []
            result_grid = []

            total_runs = len(values_x) * len(values_y) * n_train
            desc = f"Sweeping {param_x} vs {param_y} ({image_tag})"
            with tqdm(total=total_runs, desc=desc) as progress:
                for val_x in values_x:
                    row_means = []
                    row_psnrs = []
                    row_results = []
                    for val_y in values_y:
                        losses = []
                        psnrs = []
                        for run_idx in range(n_train):
                            torch.manual_seed(run_idx)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(run_idx)

                            run_model_args = dict(base_model_args)
                            run_lr, run_bs, run_epochs = (
                                default_lr,
                                default_batch_size,
                                default_max_epochs,
                            )

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
                            psnr_list = run_result.get("val_psnrs", [])
                            if psnr_list:
                                psnrs.append(psnr_list[-1])
                            progress.set_postfix(
                                {
                                    str(param_x): format_tick(val_x),
                                    str(param_y): format_tick(val_y),
                                    "run": f"{run_idx + 1}/{n_train}",
                                    "img": image_tag,
                                }
                            )
                            progress.update(1)

                        mean_loss = sum(losses) / len(losses)
                        if len(losses) > 1:
                            variance = sum((loss - mean_loss) ** 2 for loss in losses) / (len(losses) - 1)
                            std_error = math.sqrt(variance) / math.sqrt(len(losses))
                        else:
                            std_error = 0.0

                        if psnrs:
                            mean_psnr_val = sum(psnrs) / len(psnrs)
                            if len(psnrs) > 1:
                                var_psnr = sum((p - mean_psnr_val) ** 2 for p in psnrs) / (len(psnrs) - 1)
                                std_err_psnr = math.sqrt(var_psnr) / math.sqrt(len(psnrs))
                            else:
                                std_err_psnr = 0.0
                        else:
                            mean_psnr_val = float("nan")
                            std_err_psnr = float("nan")

                        row_means.append(mean_loss)
                        row_psnrs.append(mean_psnr_val)
                        row_results.append(
                            {
                                "values": (val_x, val_y),
                                "losses": losses,
                                "mean_loss": mean_loss,
                                "std_error": std_error,
                                "psnrs": psnrs,
                                "mean_psnr": mean_psnr_val,
                                "std_error_psnr": std_err_psnr,
                            }
                        )

                    mean_grid.append(row_means)
                    psnr_grid.append(row_psnrs)
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
            ax.set_title(f"Grid sweep: {param_x} vs {param_y} (n={n_train}, img={image_tag})")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mean best validation loss (MSE)")
            cbar.formatter = ticker.FormatStrFormatter("%.1e")
            cbar.update_ticks()
            plot_path = sweep_media_path / f"sweep_{param_x}_vs_{param_y}_mse_{image_tag}.png"
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)

            psnr_plot_path = None
            psnr_values = [val for row in psnr_grid for val in row if math.isfinite(val)]
            if psnr_values:
                fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
                im = ax.imshow(
                    psnr_grid,
                    origin="lower",
                    cmap="magma",
                    aspect="auto",
                    vmin=min(psnr_values),
                    vmax=max(psnr_values),
                )
                ax.set_xticks(range(len(values_y)))
                ax.set_xticklabels([format_tick(v) for v in values_y], rotation=45, ha="right")
                ax.set_yticks(range(len(values_x)))
                ax.set_yticklabels([format_tick(v) for v in values_x])
                ax.set_xlabel(param_y)
                ax.set_ylabel(param_x)
                ax.set_title(f"Grid sweep PSNR: {param_x} vs {param_y} (img={image_tag})")
                cbar_psnr = fig.colorbar(im, ax=ax)
                cbar_psnr.set_label("Mean PSNR (dB)")
                psnr_plot_path = sweep_media_path / f"sweep_{param_x}_vs_{param_y}_psnr_{image_tag}.png"
                fig.savefig(psnr_plot_path, dpi=200)
                plt.close(fig)

            return {
                "param": sweep_param,
                "params": [param_x, param_y],
                "values": [list(values_x), list(values_y)],
                "results": result_grid,
                "plot_path": plot_path,
                "bar_plot_path": plot_path,
                "psnr_bar_plot_path": psnr_plot_path,
                "loss_curve_plot_path": None,
                "image_name": image_name,
            }

        param_a, param_b, param_c = sweep_param
        values_a, values_b, values_c = sweep_values

        panel_results = []
        all_mean_losses = []
        all_mean_psnrs = []

        total_runs = len(values_a) * len(values_b) * len(values_c) * n_train
        progress = tqdm(
            total=total_runs,
            desc=f"Sweeping {param_a} vs {param_b} vs {param_c} ({image_tag})",
        )

        for val_a in values_a:
            mean_grid = []
            psnr_grid = []
            result_grid = []
            for val_b in values_b:
                row_means = []
                row_psnrs = []
                row_results = []
                for val_c in values_c:
                    losses = []
                    psnrs = []
                    for run_idx in range(n_train):
                        torch.manual_seed(run_idx)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(run_idx)

                        run_model_args = dict(base_model_args)
                        run_lr, run_bs, run_epochs = (
                            default_lr,
                            default_batch_size,
                            default_max_epochs,
                        )

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
                        psnr_list = run_result.get("val_psnrs", [])
                        if psnr_list:
                            psnrs.append(psnr_list[-1])
                        progress.update(1)

                    mean_loss = sum(losses) / len(losses)
                    if len(losses) > 1:
                        variance = sum((loss - mean_loss) ** 2 for loss in losses) / (len(losses) - 1)
                        std_error = math.sqrt(variance) / math.sqrt(len(losses))
                    else:
                        std_error = 0.0

                    if psnrs:
                        mean_psnr_val = sum(psnrs) / len(psnrs)
                        if len(psnrs) > 1:
                            var_psnr = sum((p - mean_psnr_val) ** 2 for p in psnrs) / (len(psnrs) - 1)
                            std_err_psnr = math.sqrt(var_psnr) / math.sqrt(len(psnrs))
                        else:
                            std_err_psnr = 0.0
                    else:
                        mean_psnr_val = float("nan")
                        std_err_psnr = float("nan")

                    row_means.append(mean_loss)
                    row_psnrs.append(mean_psnr_val)
                    all_mean_losses.append(mean_loss)
                    if math.isfinite(mean_psnr_val):
                        all_mean_psnrs.append(mean_psnr_val)
                    row_results.append(
                        {
                            "values": (val_a, val_b, val_c),
                            "losses": losses,
                            "mean_loss": mean_loss,
                            "std_error": std_error,
                            "psnrs": psnrs,
                            "mean_psnr": mean_psnr_val,
                            "std_error_psnr": std_err_psnr,
                        }
                    )
                mean_grid.append(row_means)
                psnr_grid.append(row_psnrs)
                result_grid.append(row_results)

            panel_results.append(
                {
                    "param_value": val_a,
                    "mean_grid": mean_grid,
                    "psnr_grid": psnr_grid,
                    "results": result_grid,
                }
            )

        progress.close()

        vmin = min(all_mean_losses) if all_mean_losses else None
        vmax = max(all_mean_losses) if all_mean_losses else None
        psnr_vmin = min(all_mean_psnrs) if all_mean_psnrs else None
        psnr_vmax = max(all_mean_psnrs) if all_mean_psnrs else None
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

        suptitle = f"Grid sweep: {param_a} vs {param_b} vs {param_c} (n={n_train}, img={image_tag})"
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
        plot_path = sweep_media_path / f"sweep_{param_a}_vs_{param_b}_vs_{param_c}_mse_{image_tag}.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        psnr_plot_path = None
        if psnr_vmin is not None and psnr_vmax is not None:
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(width, height),
                squeeze=False,
                layout="constrained",
            )
            axes_list = [ax for ax in axes.flatten()]
            norm_psnr = colors.Normalize(vmin=psnr_vmin, vmax=psnr_vmax)
            for idx, (val_a, panel) in enumerate(zip(values_a, panel_results)):
                ax = axes_list[idx]
                im = ax.imshow(
                    panel["psnr_grid"],
                    origin="lower",
                    cmap="magma",
                    norm=norm_psnr,
                    aspect="auto",
                )
                ax.set_xticks(range(len(values_c)))
                ax.set_xticklabels([format_tick(v) for v in values_c], rotation=45, ha="right")
                ax.set_yticks(range(len(values_b)))
                ax.set_yticklabels([format_tick(v) for v in values_b])
                ax.set_xlabel(param_c)
                ax.set_ylabel(param_b)
                ax.set_title(f"{param_a} = {format_tick(val_a)}")

            for ax in axes_list[len(values_a) :]:
                ax.axis("off")

            fig.suptitle(f"Grid PSNR: {param_a} vs {param_b} vs {param_c} (img={image_tag})")
            used_axes = axes_list[: len(values_a)]
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm_psnr, cmap="magma"),
                ax=used_axes,
                fraction=0.035,
                pad=0.02,
            )
            cbar.set_label("Mean PSNR (dB)")
            psnr_plot_path = sweep_media_path / f"sweep_{param_a}_vs_{param_b}_vs_{param_c}_psnr_{image_tag}.png"
            fig.savefig(psnr_plot_path, dpi=200)
            plt.close(fig)

        return {
            "param": sweep_param,
            "params": [param_a, param_b, param_c],
            "values": [list(values_a), list(values_b), list(values_c)],
            "results": panel_results,
            "plot_path": plot_path,
            "bar_plot_path": plot_path,
            "psnr_bar_plot_path": psnr_plot_path,
            "loss_curve_plot_path": None,
            "image_name": image_name,
        }

    results = []
    mean_losses = []
    std_errors = []
    mean_psnrs = []
    std_error_psnrs = []
    loss_curves_by_value = {}
    psnr_curves_by_value = {}

    total_runs = len(sweep_values) * n_train
    with tqdm(total=total_runs, desc=f"Sweeping {sweep_param} ({image_tag})") as progress:
        for val in sweep_values:
            losses = []
            psnrs = []
            loss_curves = []
            psnr_curves = []
            for run_idx in range(n_train):
                torch.manual_seed(run_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(run_idx)

                model_args = dict(base_model_args)
                run_lr, run_bs, run_epochs = (
                    default_lr,
                    default_batch_size,
                    default_max_epochs,
                )
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
                psnr_list = run_result.get("val_psnrs", [])
                psnr_curves.append(psnr_list)
                if psnr_list:
                    psnrs.append(psnr_list[-1])
                progress.set_postfix(
                    {
                        str(sweep_param): format_tick(val),
                        "run": f"{run_idx + 1}/{n_train}",
                        "img": image_tag,
                    }
                )
                progress.update(1)

            mean_loss = sum(losses) / len(losses)
            if len(losses) > 1:
                variance = sum((loss - mean_loss) ** 2 for loss in losses) / (len(losses) - 1)
                std_error = math.sqrt(variance) / math.sqrt(len(losses))
            else:
                std_error = 0.0

            finite_psnrs = [p for p in psnrs if math.isfinite(p)]
            if finite_psnrs:
                mean_psnr_val = sum(finite_psnrs) / len(finite_psnrs)
                if len(finite_psnrs) > 1:
                    var_psnr = sum((p - mean_psnr_val) ** 2 for p in finite_psnrs) / (len(finite_psnrs) - 1)
                    std_err_psnr = math.sqrt(var_psnr) / math.sqrt(len(finite_psnrs))
                else:
                    std_err_psnr = 0.0
            else:
                mean_psnr_val = float("nan")
                std_err_psnr = float("nan")

            results.append(
                {
                    "value": val,
                    "losses": losses,
                    "mean_loss": mean_loss,
                    "std_error": std_error,
                    "psnrs": psnrs,
                    "mean_psnr": mean_psnr_val,
                    "std_error_psnr": std_err_psnr,
                }
            )
            mean_losses.append(mean_loss)
            std_errors.append(std_error)
            mean_psnrs.append(mean_psnr_val)
            std_error_psnrs.append(std_err_psnr)
            loss_curves_by_value[val] = loss_curves
            psnr_curves_by_value[val] = psnr_curves

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_pos = list(range(len(sweep_values)))
    ax.bar(x_pos, mean_losses, yerr=std_errors, capsize=5, color="#4C72B0")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in sweep_values])
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("Best validation loss (MSE)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title(f"Sweep over {sweep_param} (n={n_train}, img={image_tag})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    bar_plot_path = sweep_media_path / f"sweep_{sweep_param}_mse_{image_tag}.png"
    fig.savefig(bar_plot_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    psnr_heights = [val if math.isfinite(val) else float("nan") for val in mean_psnrs]
    psnr_err = [err if math.isfinite(err) else 0.0 for err in std_error_psnrs]
    ax.bar(x_pos, psnr_heights, yerr=psnr_err, capsize=5, color="#D55E00")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in sweep_values])
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title(f"PSNR over {sweep_param} (n={n_train}, img={image_tag})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    psnr_bar_plot_path = sweep_media_path / f"sweep_{sweep_param}_psnr_{image_tag}.png"
    fig.savefig(psnr_bar_plot_path, dpi=200)
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

    y_min = float("inf")
    y_max = 0.0

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
        lower = [max(m - s, 1e-12) for m, s in zip(mean_curve, std_err_curve)]
        y_min = min(y_min, min(lower))
        y_max = max(y_max, max(upper))
        ax.fill_between(
            epochs,
            lower,
            upper,
            color=color_map[val],
            alpha=0.2,
            linewidth=0,
        )

    ax.set_yscale("log")
    if y_min < float("inf") and y_max > 0:
        log_span = math.log10(y_max) - math.log10(y_min)
        # Use denser major ticks when the range is narrow, otherwise stick to decades
        if log_span <= 2.0:
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1.0, 2.0, 5.0]))
        else:
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs="auto"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (MSE)")
    ax.set_title(f"Loss for different {sweep_param} (img={image_tag})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title=str(sweep_param))
    fig.tight_layout()
    loss_curve_plot_path = sweep_media_path / f"sweep_{sweep_param}_loss_curves_{image_tag}.png"
    fig.savefig(loss_curve_plot_path, dpi=200)
    plt.close(fig)

    # Plot aggregated PSNR curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for val in sweep_values:
        curves = psnr_curves_by_value.get(val, [])
        if not curves:
            continue
        non_empty = [c for c in curves if c]
        if not non_empty:
            continue
        min_len = min(len(c) for c in non_empty)
        if min_len == 0:
            continue
        trimmed = [c[:min_len] for c in non_empty]

        mean_curve = []
        std_err_curve = []
        for epoch_psnrs in zip(*trimmed):
            n = len(epoch_psnrs)
            mean_epoch = sum(epoch_psnrs) / n
            if n > 1:
                variance = sum((p - mean_epoch) ** 2 for p in epoch_psnrs) / (n - 1)
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
            color=color_map.get(val, None),
            linewidth=2,
        )
        upper = [m + s for m, s in zip(mean_curve, std_err_curve)]
        lower = [m - s for m, s in zip(mean_curve, std_err_curve)]
        ax.fill_between(
            epochs,
            lower,
            upper,
            color=color_map.get(val, None),
            alpha=0.15,
            linewidth=0,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"PSNR for different {sweep_param} (img={image_tag})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title=str(sweep_param))
    fig.tight_layout()
    psnr_curve_plot_path = sweep_media_path / f"sweep_{sweep_param}_psnr_curves_{image_tag}.png"
    fig.savefig(psnr_curve_plot_path, dpi=200)
    plt.close(fig)

    return {
        "param": sweep_param,
        "values": list(sweep_values),
        "results": results,
        "bar_plot_path": bar_plot_path,
        "psnr_bar_plot_path": psnr_bar_plot_path,
        "loss_curve_plot_path": loss_curve_plot_path,
        "psnr_curve_plot_path": psnr_curve_plot_path,
        "image_name": image_name,
    }

if __name__ == "__main__":

    """
    CONCEPTS:
    - dont use layernorm and siren together
    """
    
    base_model_args = dict(
        position_encoding="fourier",
        trainable_encodings=False,
        freq_scale=80,
        encoding_dim=256,
        n_layers=6,
        n_experts=1,
        hidden_dim=1024,
        skip_connections=False,
        layernorm=False,
        layer_type="relu",
    )

    #sweep_param = ("hidden_dim", "lr")
    #sweep_values = ([256, 384, 512, 768, 1024], [2e-4, 3e-4, 4e-4, 5e-4])

    #sweep_param="batch_size"
    #sweep_values=[512, 1024, 2048, 4096, 8192]

    #sweep_param="lr"
    #sweep_values=[1e-4, 2e-4, 3e-4, 4e-4, 5e-4]

    #sweep_param="layernorm"
    #sweep_values=[True, False]

    #sweep_param="skip_connections"
    #sweep_values=[True, False]

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

    #sweep_param=("freq_scale", "encoding_dim")
    #sweep_values=([50, 80, 110, 140, 170], [64, 128, 256])

    #----------------------------
    
    #sweep_param=("hidden_dim", "n_layers")
    #sweep_values=([512, 768, 1024], [3, 4, 5, 6])

    # 0: train only, 1: sweep only
    mode = 0

    if mode == 0:
        train_model(
            image_name=["station.jpg", "farming.jpg", "stairs.jpg", "thing.jpg", "waves.jpg"],
            model_args=base_model_args,
            lr=DEFAULT_LR,
            batch_size=DEFAULT_BATCH_SIZE,
            max_epochs=DEFAULT_MAX_EPOCHS,
            report_progress=True,
            save_video=True
        )
    
    if mode == 1:
        sweep_result = sweep_model(
            image_name=["station.jpg", "farming.jpg", "stairs.jpg", "thing.jpg", "waves.jpg"],
            base_model_args=base_model_args,
            sweep_param=sweep_param,
            sweep_values=sweep_values,
            n_train=1,
            default_lr=DEFAULT_LR,
            default_batch_size=DEFAULT_BATCH_SIZE,
            default_max_epochs=DEFAULT_MAX_EPOCHS,
        )

        print_sweep_summary(sweep_result)

# gabor 7e-5
