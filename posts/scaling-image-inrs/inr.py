import torch
import torch.nn.functional as F
from torch import nn
from torchvision.io import read_image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from models import GeneralModel, MOEMLP, SIREN

BATCH_SIZE = 4096
IMAGE_SIZE = 1024
MAX_EPOCHS = 40
LR = 3e-4
PLOT_SNAPSHOTS = True

# mlp gets val loss 0.01

def load_target_image(image_path):
    image = read_image(str(image_path)).float().unsqueeze(0) / 255.0
    image = F.interpolate(
        image,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return image.squeeze(0).permute(1, 2, 0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_path = Path(__file__).resolve().parent
    image_path = base_path / "image5.jpg"
    media_path = base_path / "media"
    media_path.mkdir(exist_ok=True)

    # Target to GPU
    target = load_target_image(image_path).to(device)  # (H, W, 3)
    height, width, _ = target.shape
    target_cpu = target.detach().cpu().numpy()

    # Build coords directly on GPU
    y_coords = torch.linspace(0.0, 1.0, steps=height, device=device)
    x_coords = torch.linspace(0.0, 1.0, steps=width, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    coords = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)  # (N, 2)

    pixels = target.reshape(-1, 3)  # (N, 3),

    num_samples = coords.shape[0]
    indices = torch.arange(num_samples, device=device)

    model = GeneralModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses = []

    def save_loss_curve(losses):
        if not PLOT_SNAPSHOTS:
            return
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

    def evaluate_and_save_output(epoch_idx: int):
        if not PLOT_SNAPSHOTS:
            return
        model.eval()
        predictions = []
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for start in range(0, num_samples, BATCH_SIZE):
                end = min(start + BATCH_SIZE, num_samples)
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
            predictions_cat.reshape(height, width, 3)
            .clamp(0.0, 1.0)
            .detach()
            .cpu()
            .numpy()
        )

        avg_val_loss = val_loss / max(val_samples, 1)

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
            f"Epoch {epoch_idx} | MSE={avg_val_loss:.2e}",
            ha="right",
            va="top",
            transform=axes[1].transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6, edgecolor="none"),
            fontsize=9,
        )

        plt.tight_layout()
        plt.savefig(media_path / "last_output.png", dpi=200)
        plt.close(fig)

    # Training loop
    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        seen_samples = 0

        # Shuffle indices on GPU
        perm = indices[torch.randperm(num_samples, device=device)]

        batch_iter = range(0, num_samples, BATCH_SIZE)
        progress = tqdm(
            batch_iter,
            desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}",
            leave=False,
        )

        for start in progress:
            end = min(start + BATCH_SIZE, num_samples)
            batch_idx = perm[start:end]

            batch_coords = coords[batch_idx]
            batch_pixels = pixels[batch_idx]

            optimizer.zero_grad()
            preds = model(batch_coords)
            loss = criterion(preds, batch_pixels)
            loss.backward()
            optimizer.step()

            batch_size = batch_coords.size(0)
            running_loss += loss.item() * batch_size
            seen_samples += batch_size
            progress.set_postfix(
                loss=f"{running_loss / max(seen_samples, 1):.2e}"
            )

        train_losses.append(running_loss / max(seen_samples, 1))
        if (epoch + 1) % 5 == 0:
            save_loss_curve(train_losses)
            evaluate_and_save_output(epoch + 1)

    if MAX_EPOCHS % 5 != 0:
        save_loss_curve(train_losses)
        evaluate_and_save_output(MAX_EPOCHS)


if __name__ == "__main__":
    main()
