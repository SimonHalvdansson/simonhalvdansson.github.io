import math, time, os, random
from contextlib import nullcontext
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np


# --------------------------
# Device / AMP
# --------------------------
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
is_cuda = (device == "cuda")
use_amp = is_cuda
amp_dtype = torch.bfloat16


# --------------------------
# Vision Transformer (ViT)
# --------------------------
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_positions: int, d_model: int):
        super().__init__()
        self.num_positions = num_positions
        self.d_model = d_model
        self.register_buffer("pos_embedding", self._build())

    def _build(self) -> torch.Tensor:
        position = torch.arange(0, self.num_positions).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.num_positions, self.d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe.unsqueeze(0)  # (1, T, D)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        T = positions.size(-1)
        return self.pos_embedding[:, :T, :]


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding using a Conv2d with kernel_size=stride=patch_size.
    Input:  (B, C, H, W)
    Output: (B, N, D) where N = (H/ps)*(W/ps)
    """
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=224):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x = self.proj(x)                                # (B, D, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)               # (B, N, D)
        return x


class ViTBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout, norm="layer", ffn="mlp", prepost="pre"):
        super().__init__()
        if norm == "layer":
            self.n1 = nn.LayerNorm(d_model)
            self.n2 = nn.LayerNorm(d_model)
        else:
            self.n1 = nn.RMSNorm(d_model)
            self.n2 = nn.RMSNorm(d_model)

        self.prepost = prepost
        self.resid_drop = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=n_head,
                                          dropout=dropout,
                                          batch_first=True,
                                          bias=False)

        if ffn == "mlp":
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )
        else:
            hidden = int(8 * d_model / 3)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 2 * hidden),
                nn.SiLU(),
                nn.Linear(hidden, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        if self.prepost == "pre":
            xn = self.n1(x)
            y, _ = self.attn(xn, xn, xn, need_weights=False)
            x = x + self.resid_drop(y)
            y = self.ffn(self.n2(x))
            x = x + self.resid_drop(y)
        else:
            y, _ = self.attn(x, x, x, need_weights=False)
            x = self.n1(x + self.resid_drop(y))
            y = self.ffn(x)
            x = self.n2(x + self.resid_drop(y))
        return x


class ViT(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=14,
                 in_chans=3,
                 num_classes=1000,
                 n_layer=2,
                 n_head=4,
                 d_model=224,
                 dropout=0.0,
                 norm="layer",
                 ffn="mlp",
                 prepost="pre",
                 pos_emb="learned"):
        super().__init__()
        assert d_model % n_head == 0
        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid  # 16*16=256

        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=d_model)
        # No CLS token for exact 256 positions; use mean pooling for classification
        self.pos_emb = nn.Embedding(self.num_patches, d_model) if pos_emb == "learned" else SinusoidalPositionalEmbedding(self.num_patches, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ViTBlock(d_model, n_head, dropout, norm, ffn, prepost) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model) if norm == "layer" else nn.RMSNorm(d_model)
        self.prepost = prepost
        self.head = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, imgs):
        # imgs: (B, 3, 224, 224)
        B = imgs.size(0)
        x = self.patch_embed(imgs)                   # (B, N=256, D)
        pos = torch.arange(0, x.size(1), device=imgs.device).unsqueeze(0)
        x = x + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        if self.prepost == "pre":
            x = self.ln_f(x)
        # mean pool over patch tokens (no CLS)
        x = x.mean(dim=1)
        logits = self.head(x)
        return logits


# --------------------------
# Train/Eval helpers (images)
# --------------------------
def make_autocast_and_scaler():
    autocast_ctx = (lambda: torch.amp.autocast(device_type=device, dtype=amp_dtype)) if use_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    return autocast_ctx, scaler


def run_epoch(model, loader, args, optimizer, train=True, autocast_ctx=nullcontext, scaler=None, selfsim=None):
    model.train(train)
    total_loss_sum, total_count = 0.0, 0

    if args.get("optimizer") == "schedulefree" and optimizer is not None:
        try:
            if train: optimizer.train()
            else: optimizer.eval()
        except Exception:
            pass

    pbar = tqdm(
        loader, leave=False, desc="train" if train else "eval",
        dynamic_ncols=True, mininterval=0.1, maxinterval=1.0, miniters=1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    first_draw = True

    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)

        with autocast_ctx():
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.get("grad_clip") is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.get("grad_clip") is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                optimizer.step()

        if is_cuda:
            torch.cuda.synchronize()

        bs = imgs.size(0)
        total_loss_sum += loss.item() * bs
        total_count += bs

        pbar.set_postfix(loss=f"{loss.item():.3f}")
        if first_draw:
            pbar.refresh(); first_draw = False

        # Periodic self-similarity snapshots during training (step-based)
        if train and selfsim is not None and "interval_steps" in selfsim:
            try:
                selfsim["step"] = selfsim.get("step", 0) + 1
                while (
                    selfsim["step"] >= selfsim.get("next_step", selfsim["interval_steps"]) and
                    selfsim.get("next_step", selfsim["interval_steps"]) <= selfsim.get("max_steps", float("inf"))
                ):
                    step_for_name = selfsim["next_step"]
                    fname = f"{selfsim['prefix']}_step{step_for_name:05d}.png"
                    fpath = os.path.join(selfsim["save_dir"], fname)
                    os.makedirs(selfsim["save_dir"], exist_ok=True)
                    plot_positional_self_similarity(
                        model,
                        save_path=fpath,
                        dpi=300,
                        figsize=(5, 4.5),
                        show=False,
                        title=f"Self-similarity, step: {step_for_name}"
                    )
                    selfsim["next_step"] = step_for_name + selfsim["interval_steps"]
            except Exception:
                pass

    avg_loss = total_loss_sum / max(total_count, 1)
    return avg_loss


def train_one_epoch(model, train_loader, val_loader, args, selfsim=None):
    autocast_ctx, scaler = make_autocast_and_scaler()
    if args.get("optimizer") == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])
    else:
        # fallback; schedulefree optional
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])  # simple default

    train_loss = run_epoch(model, train_loader, args=args, optimizer=optimizer, train=True,
                           autocast_ctx=autocast_ctx, scaler=scaler, selfsim=selfsim)
    val_loss = run_epoch(model, val_loader, args=args, optimizer=None, train=False,
                         autocast_ctx=autocast_ctx, scaler=None)

    print(f"Train_ce={train_loss:.3f} | Val_ce={val_loss:.3f}")
    return val_loss


# --------------------------
# Visualization helpers (shared style with LLM pe.py)
# --------------------------
def _get_positional_weights(model: nn.Module) -> np.ndarray:
    """
    Returns positional embedding weights as a numpy array of shape (T, D).
    Supports both learned (nn.Embedding) and sinusoidal buffer variants.
    """
    if isinstance(getattr(model, "pos_emb", None), nn.Embedding):
        return model.pos_emb.weight.detach().cpu().numpy()

    pe = getattr(getattr(model, "pos_emb", object()), "pos_embedding", None)
    if pe is not None:
        arr = pe.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        return arr

    raise ValueError("Model does not expose recognizable positional embeddings under model.pos_emb.")


def plot_positional_embeddings_and_fft(model: nn.Module, dims=(0, 1, 2), save_path: str = None, *, figsize=(10, 8), dpi=240, show=True):
    import matplotlib.pyplot as plt

    pe = _get_positional_weights(model)  # (T, D)
    T = pe.shape[0]

    fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)
    for r, d in enumerate(dims):
        d = int(d)
        d = max(0, min(d, pe.shape[1] - 1))
        curve = pe[:, d]
        ax_time = axes[r, 0]
        ax_time.plot(curve, lw=1.2)
        ax_time.set_title(f"Pos Emb dim {d}")
        ax_time.set_xlabel("Position")
        ax_time.set_ylabel("Value")

        ax_freq = axes[r, 1]
        spec = np.fft.rfft(curve)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(T, d=1.0)
        ax_freq.plot(freqs, mag, lw=1.0)
        ax_freq.set_title(f"FFT magnitude dim {d}")
        ax_freq.set_xlabel("Frequency (cycles/pos)")
        ax_freq.set_ylabel("|FFT|")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


def plot_positional_self_similarity(model: nn.Module, save_path: str = None, *, figsize=(5, 4.5), dpi=300, show=True, title=None):
    import matplotlib.pyplot as plt

    pe = _get_positional_weights(model)  # (T, D)
    norms = np.linalg.norm(pe, axis=1, keepdims=True) + 1e-12
    pe_norm = pe / norms
    sim = pe_norm @ pe_norm.T

    plt.figure(figsize=figsize)
    im = plt.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1, origin="upper", interpolation="nearest")
    plt.title(title if title is not None else "Positional Self-Similarity (cosine)")
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


# --------------------------
# Data
# --------------------------
def build_imagenet_loaders(train_root: str,
                           img_size: int = 224,
                           batch_size: int = 64,
                           num_workers: int = 8,
                           val_split: float = 0.05,
                           seed: int = 42):
    """
    Builds ImageFolder loaders from a single train directory by splitting off a validation set.
    train_root: path to folder with subfolders per class (e.g., E:\\ML Data\\train)
    Returns: train_loader, val_loader, num_classes
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    full = datasets.ImageFolder(train_root, transform=tfm)

    n = len(full)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_val = int(val_split * n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx) if n_val > 0 else Subset(full, [])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=is_cuda, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=is_cuda, drop_last=False)

    num_classes = len(full.classes)
    return train_loader, val_loader, num_classes


if __name__ == "__main__":
    # Hyperparameters for ViT comparable in spirit to LLM defaults
    args = {
        "lr":        1.0e-3,
        "optimizer": "adamw",
        "n_layer":   2,
        "n_head":    4,      # 224 / 4 = 56 dim per head
        "d_model":   224,
        "dropout":   0.0,
        "grad_clip": None,
        "norm":      "layer",
        "ffn":       "mlp",
        "prepost":   "pre",
        "pos_emb":   "learned",
        "batch_size": 64,
        "epochs":     6,
        "num_workers": 8,
    }

    # Paths and media outputs
    base_dir = os.path.dirname(__file__)
    media_dir = os.path.join(base_dir, "media")
    os.makedirs(media_dir, exist_ok=True)

    # Plots for sinusoidal baseline (untrained) and learned after training
    grid_path_sinus = os.path.join(media_dir, "vit_positional_embeddings_fft_sinusoidal_d0_63_127.png")
    sim_path_sinus = os.path.join(media_dir, "vit_positional_self_similarity_sinusoidal.png")
    grid_path_learned = os.path.join(media_dir, "vit_positional_embeddings_fft_learned_d0_63_127.png")
    sim_during_dir = os.path.join(media_dir, "vit_self_similarity_during_training")

    # Data path (Windows-style path with space)
    imagenet_train_root = r"E:\\ML Data\\train"

    # Build dataloaders to discover num_classes
    try:
        train_loader, val_loader, num_classes = build_imagenet_loaders(
            imagenet_train_root,
            img_size=224,
            batch_size=args["batch_size"],
            num_workers=args["num_workers"],
            val_split=0.05,
        )
    except Exception as e:
        print(f"Failed to build loaders from '{imagenet_train_root}': {e}")
        print("You may need to adjust the path or install torchvision.")
        raise

    # Sinusoidal baseline model for comparison plots (no training)
    sinus_model = ViT(image_size=224, patch_size=14, in_chans=3, num_classes=num_classes,
                      n_layer=args["n_layer"], n_head=args["n_head"], d_model=args["d_model"],
                      dropout=args["dropout"], norm=args["norm"], ffn=args["ffn"], prepost=args["prepost"],
                      pos_emb="sinusoidal").to(device)
    plot_positional_self_similarity(sinus_model, save_path=sim_path_sinus, dpi=300, figsize=(5, 4.5))
    plot_positional_embeddings_and_fft(sinus_model, dims=(0, 63, 127), save_path=grid_path_sinus, dpi=300, figsize=(10, 8))

    # Learned positional embeddings model (to train)
    model = ViT(image_size=224, patch_size=14, in_chans=3, num_classes=num_classes,
                n_layer=args["n_layer"], n_head=args["n_head"], d_model=args["d_model"],
                dropout=args["dropout"], norm=args["norm"], ffn=args["ffn"], prepost=args["prepost"],
                pos_emb=args["pos_emb"]).to(device)

    # Configure periodic self-similarity snapshots: every 50 steps, up to 1000
    selfsim_cfg = {
        "interval_steps": 50,
        "next_step": 50,
        "max_steps": 1000,
        "step": 0,
        "save_dir": sim_during_dir,
        "prefix": "vit_positional_self_similarity_learned",
    }

    # Train for a few epochs with snapshots, saving a per-epoch self-sim figure too
    best_val = float('inf')
    for epoch in range(1, args["epochs"] + 1):
        val_loss = train_one_epoch(model, train_loader, val_loader, args=args, selfsim=selfsim_cfg)
        print(f"Finished epoch {epoch}. Val CE: {val_loss:.3f}")
        best_val = min(best_val, val_loss)

        # Save per-epoch self-sim figure
        epoch_sim_path = os.path.join(media_dir, f"vit_positional_self_similarity_learned_epoch{epoch}.png")
        plot_positional_self_similarity(
            model,
            save_path=epoch_sim_path,
            dpi=300,
            figsize=(5, 4.5),
            title=f"Self-similarity, step: {selfsim_cfg['step']}"
        )

    # After all epochs, save the FFT grid for the learned model
    plot_positional_embeddings_and_fft(model, dims=(0, 63, 127), save_path=grid_path_learned, dpi=300, figsize=(10, 8))

