
import torch
import torch.nn.functional as F
from torchvision.io import read_image
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import math


def load_target_image(image_path, image_size):
    image = read_image(str(image_path)).float().unsqueeze(0) / 255.0
    image = F.interpolate(
        image,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    return image.squeeze(0).permute(1, 2, 0)

def save_loss_curve(losses, media_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses, linewidth=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(1, 10), numticks=100))
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title("Training Loss Curve")
    fig.tight_layout()
    fig.savefig(media_path / "loss_curve.png", dpi=200)
    plt.close(fig)

def _mse_to_psnr(mse):
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)

def evaluate_and_save_output(model, criterion, batch_size, epoch_idx, num_samples, indices, coords, pixels, media_path, target_cpu, plot=False):
    model.eval()
    predictions = []
    val_loss = 0.0
    val_samples = 0
    height, width = target_cpu.shape[:2]

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]

            batch_coords = coords[batch_idx]
            batch_pixels = pixels[batch_idx]

            preds = model(batch_coords)
            predictions.append(preds)

            batch_size = batch_coords.size(0)
            val_loss += criterion(preds, batch_pixels).item() * batch_size
            val_samples += batch_size

    predictions_cat = torch.cat(predictions, dim=0)
    predicted_image = (
        predictions_cat.reshape(height, width, 3).clamp(0.0, 1.0).detach().cpu().numpy()
    )

    avg_val_loss = val_loss / max(val_samples, 1)
    psnr = _mse_to_psnr(avg_val_loss)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(target_cpu)
        axes[0].set_title("Target")
        axes[0].axis("off")

        axes[1].imshow(predicted_image)
        axes[1].set_title("INR Output")
        axes[1].axis("off")
        axes[1].text(
            0.98,
            0.97,
            f"Epoch {epoch_idx} | MSE={avg_val_loss:.2e} | PSNR={psnr:.2f} dB",
            ha="right",
            va="top",
            transform=axes[1].transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6, edgecolor="none"),
            fontsize=9,
        )

        plt.tight_layout()
        plt.savefig(media_path / "last_output.png", dpi=200)
        plt.close(fig)
    return avg_val_loss, psnr

def plot_moe_activations(model, coords, indices, height, width, batch_size, media_path):
    if not hasattr(model, "n_experts") or getattr(model, "n_experts", 1) <= 1:
        return
    if not hasattr(model, "compute_gate_weights"):
        return

    gate_weights = []
    with torch.no_grad():
        for start in range(0, indices.numel(), batch_size):
            end = min(start + batch_size, indices.numel())
            batch_idx = indices[start:end]
            batch_coords = coords[batch_idx]
            weights = model.compute_gate_weights(batch_coords)
            gate_weights.append(weights)

    if not gate_weights:
        return

    gate_weights_cat = torch.cat(gate_weights, dim=0)
    if gate_weights_cat.numel() != height * width * model.n_experts:
        return

    weight_maps = gate_weights_cat.reshape(height, width, model.n_experts).detach().cpu().numpy()
    n_experts = weight_maps.shape[-1]
    cols = min(4, n_experts)
    rows = (n_experts + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx in range(rows * cols):
        ax = axes[idx]
        if idx < n_experts:
            im = ax.imshow(weight_maps[:, :, idx], vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_title(f"Expert {idx + 1} weight")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")

    fig.suptitle("Mixture of Experts Activations", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(media_path / "moe_activations.png", dpi=200)
    plt.close(fig)


def print_sweep_summary(sweep_result):
    def fmt_psnr(res):
        mean_psnr = res.get("mean_psnr")
        stderr_psnr = res.get("std_error_psnr")
        if mean_psnr is None or not math.isfinite(mean_psnr):
            return "psnr=n/a"
        if stderr_psnr is None or not math.isfinite(stderr_psnr):
            return f"psnr_mean={mean_psnr:.2f}"
        return f"psnr_mean={mean_psnr:.2f} psnr_stderr={stderr_psnr:.2f}"

    def print_paths(res, prefix=""):
        if not res:
            return
        if res.get("bar_plot_path"):
            print(f"{prefix}Bar plot: {res['bar_plot_path']}")
        if res.get("psnr_bar_plot_path"):
            print(f"{prefix}PSNR plot: {res['psnr_bar_plot_path']}")
        if res.get("loss_curve_plot_path"):
            print(f"{prefix}Loss curves plot: {res['loss_curve_plot_path']}")
        if res.get("psnr_curve_plot_path"):
            print(f"{prefix}PSNR curves plot: {res['psnr_curve_plot_path']}")

    def print_results(res, prefix=""):
        if not res:
            return
        print_paths(res, prefix)
        params = res.get("params")
        if params:
            if len(params) == 2:
                p_x, p_y = params
                flat_results = [item for row in res["results"] for item in row]
                for item in flat_results:
                    vx, vy = item["values"]
                    print(
                        f"{prefix}{p_x}={vx}, {p_y}={vy} "
                        f"mean={item['mean_loss']:.3e} stderr={item['std_error']:.3e} "
                        f"{fmt_psnr(item)}"
                    )
            elif len(params) == 3:
                p_a, p_b, p_c = params
                for panel in res["results"]:
                    for row in panel["results"]:
                        for item in row:
                            va, vb, vc = item["values"]
                            print(
                                f"{prefix}{p_a}={va}, {p_b}={vb}, {p_c}={vc} "
                                f"mean={item['mean_loss']:.3e} stderr={item['std_error']:.3e} "
                                f"{fmt_psnr(item)}"
                            )
            return

        for item in res.get("results", []):
            print(
                f"{prefix}{res['param']}={item['value']} "
                f"mean={item['mean_loss']:.3e} stderr={item['std_error']:.3e} "
                f"{fmt_psnr(item)}"
            )

    per_image = sweep_result.get("per_image_results")
    if per_image:
        for res in per_image:
            prefix = f"[{res.get('image_name')}] " if res.get("image_name") else ""
            print_results(res, prefix=prefix)
        combined = sweep_result.get("combined_result")
        if combined:
            print_results(combined, prefix="[combined] ")
        return

    print_results(sweep_result)
