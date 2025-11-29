import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
from models import GeneralModel
from utils import load_target_image, save_loss_curve, evaluate_and_save_output, plot_moe_activations

BATCH_SIZE = 4096
IMAGE_SIZE = 512
MAX_EPOCHS = 200
LR = 3e-4
PLOT_INTERVAL = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(image_name, model_args, report_progress=False):

    base_path = Path(__file__).resolve().parent
    image_path = base_path / image_name
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    eval_batch_size = BATCH_SIZE

    train_losses = []

    epoch_progress = tqdm(range(1, MAX_EPOCHS + 1), desc="Epochs", leave=False)

    for epoch in epoch_progress:
        model.train()
        running_loss = 0.0
        seen_samples = 0

        perm = indices[torch.randperm(num_samples, device=device)]

        for start_idx in range(0, num_samples, BATCH_SIZE):
            end = min(start_idx + BATCH_SIZE, num_samples)
            batch_idx = perm[start_idx:end]

            batch_coords = coords[batch_idx]
            batch_pixels = pixels[batch_idx]

            optimizer.zero_grad()
            preds = model(batch_coords)
            loss = criterion(preds, batch_pixels)
            loss.backward()
            optimizer.step()

            batch_size = batch_coords.size(0)
            running_loss += loss.detach() * batch_size
            seen_samples += batch_size

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
                batch_size = batch_coords.size(0)
                full_loss += criterion(preds, batch_pixels).item() * batch_size
                full_samples += batch_size

        avg_full_loss = full_loss / max(full_samples, 1)
        epoch_progress.set_postfix(loss=f"{avg_full_loss:.2e}")

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
    
    evaluate_and_save_output(
        model,
        criterion,
        eval_batch_size,
        MAX_EPOCHS,
        num_samples,
        indices,
        coords,
        pixels,
        media_path,
        target_cpu,
        report_progress,
    )
    
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


if __name__ == "__main__":
    model_args = dict(
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
    train_model("image3.jpg", model_args, report_progress=True)

#gabor 7e-5