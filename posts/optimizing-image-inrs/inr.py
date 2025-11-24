import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import read_image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from models import Transformer, MLP, MOEMLP, SIREN

BATCH_SIZE = 512
IMAGE_SIZE = 256
MAX_EPOCHS = 100
LR = 5e-4

#mlp gets val loss 0.01

def load_target_image(image_path):
    image = read_image(str(image_path)).float().unsqueeze(0) / 255.0
    image = F.interpolate(image, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return image.squeeze(0).permute(1, 2, 0)


def main():
    base_path = Path(__file__).resolve().parent
    image_path = base_path / "image.jpg"
    target = load_target_image(image_path)
    height, width, _ = target.shape

    y_coords = torch.linspace(0.0, 1.0, steps=height, dtype=torch.float32)
    x_coords = torch.linspace(0.0, 1.0, steps=width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    coords = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
    pixels = target.reshape(-1, 3)

    dataset = TensorDataset(coords, pixels)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #model = MOEMLP().to(device)
    model = MOEMLP(num_experts=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}", leave=False)
        for batch_coords, batch_pixels in progress:
            batch_coords = batch_coords.to(device)
            batch_pixels = batch_pixels.to(device)

            optimizer.zero_grad()
            preds = model(batch_coords)
            loss = criterion(preds, batch_pixels)
            loss.backward()
            optimizer.step()

            batch_size = batch_coords.size(0)
            running_loss += loss.item() * batch_size
            seen_samples += batch_size
            progress.set_postfix(loss=100*running_loss / max(seen_samples, 1))

    model.eval()
    predictions = []
    val_loss = 0.0
    val_samples = 0
    with torch.no_grad():
        for batch_coords, batch_pixels in eval_loader:
            batch_coords = batch_coords.to(device)
            batch_pixels = batch_pixels.to(device)

            preds = model(batch_coords)
            predictions.append(preds.cpu())

            batch_size = batch_coords.size(0)
            val_loss += criterion(preds, batch_pixels).item() * batch_size
            val_samples += batch_size

    avg_val_loss = val_loss / max(val_samples, 1)
    print(f"Validation MSE: {avg_val_loss:.6f}")

    predicted_image = torch.cat(predictions, dim=0).reshape(height, width, 3).clamp(0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(target.numpy())
    axes[0].set_title("Target")
    axes[0].axis("off")

    axes[1].imshow(predicted_image.numpy())
    axes[1].set_title("Transformer Output")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
