import csv
import random
from pathlib import Path
import os

import torch
from tqdm import tqdm

from models import SpectrogramLoss, SpectrogramMaskUNet, SpectrogramUNet
from utils import (
    load_audio,
    normalize_waveform,
    pad_collate,
    plot_denoise_comparison,
    plot_loss_curve,
    save_wav,
)


class CommonVoiceRandomSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, root="data", split="train", segment_seconds=2.0, seed=None):
        self.root = Path(root)
        self.clips_dir = self.root / "clips"
        self.segment_seconds = float(segment_seconds)
        self.last_sample_rate = None

        split_map = {
            "train": "train.tsv",
            "test": "test.tsv",
        }
        if split not in split_map:
            raise ValueError(f"Unknown split '{split}'. Expected one of: {', '.join(split_map)}")

        tsv_path = self.root / split_map[split]
        self._paths = self._load_paths(tsv_path)
        if not self._paths:
            raise RuntimeError(f"No audio paths found in {tsv_path}")

        self._rng = random.Random(seed)

    def _load_paths(self, tsv_path: Path):
        paths = []
        with tsv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not reader.fieldnames or "path" not in reader.fieldnames:
                raise RuntimeError(f"Missing 'path' column in {tsv_path}")
            for row in reader:
                filename = row.get("path")
                if filename:
                    paths.append(self.clips_dir / filename)
        return paths

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, _idx):
        last_error = None
        waveform = None
        sample_rate = None
        for _ in range(10):
            path = self._paths[self._rng.randrange(len(self._paths))]
            try:
                waveform, sample_rate = load_audio(path)
                self.last_sample_rate = sample_rate
                break
            except Exception as exc:
                last_error = exc
                continue
        if waveform is None:
            raise RuntimeError("Failed to load audio after multiple attempts") from last_error

        target_samples = int(self.segment_seconds * sample_rate)

        num_samples = waveform.shape[1]
        if num_samples >= target_samples:
            start = self._rng.randrange(0, num_samples - target_samples + 1)
            segment = waveform[:, start : start + target_samples]
        else:
            pad_amount = target_samples - num_samples
            segment = torch.nn.functional.pad(waveform, (0, pad_amount))


        #want the noisy (input) to always have energy 1
        # then the clean version has less energy than that but it might not be normalized
        segment = normalize_waveform(segment)
        noise = normalize_waveform(torch.randn_like(segment)) * 0.6*torch.rand(1).item()

        noisy = segment + noise

        normalization_factor = noisy.pow(2).mean(dim=1, keepdim=True).sqrt()

        clean = segment/normalization_factor
        noisy = noisy/normalization_factor

        return clean, noisy, sample_rate



if __name__ == "__main__":
    use_complex_unet = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("media/complex_output" if use_complex_unet else "media/mask_output")
    LOSS_PLOT_PATH = output_dir / "loss_curve.png"
    RESULT_PLOT_PATH = output_dir / "denoise_compare.png"
    CLEAN_WAV_PATH = output_dir / "clean.wav"
    NOISY_WAV_PATH = output_dir / "noisy.wav"
    DENOISED_WAV_PATH = output_dir / "denoised.wav"
    dataset = CommonVoiceRandomSegmentDataset(split="train")
    num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_collate,
    )

    model = (SpectrogramUNet() if use_complex_unet else SpectrogramMaskUNet()).to(device)
    loss_fn = SpectrogramLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    step = 0
    loss_steps = []
    loss_values = []

    model.train()
    for epoch in range(200):
        progress = tqdm(total=len(dataset), desc=f"Epoch ({epoch + 1}/20)", unit="clip")
        epoch_loss_total = 0.0
        epoch_loss_count = 0
        for clean, noisy, sample_rates in loader:
            clean = clean.to(device)
            noisy = noisy.to(device)

            denoised, model_out = model(noisy)

            loss = loss_fn(clean, denoised)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            loss_steps.append(step)
            loss_values.append(loss.item())
            epoch_loss_total += loss.item()
            epoch_loss_count += 1
            running_loss = epoch_loss_total / epoch_loss_count
            progress.set_postfix(loss=f"{running_loss:.4f}")
            progress.update(clean.shape[0])

            if step % 300 == 0:
                plot_loss_curve(loss_steps, loss_values, LOSS_PLOT_PATH)
                sample_noisy = noisy.mean(dim=1, keepdim=True)[0, 0].detach().cpu()
                sample_denoised = denoised.mean(dim=1, keepdim=True)[0, 0].detach().cpu()
                sample_clean = clean.mean(dim=1, keepdim=True)[0, 0].detach().cpu()
                if use_complex_unet:
                    sample_mag = model_out.abs()[0].detach().cpu()
                    mag_min = sample_mag.min()
                    mag_max = sample_mag.max()
                    if (mag_max - mag_min) > 1e-8:
                        sample_mag = (sample_mag - mag_min) / (mag_max - mag_min)
                    else:
                        sample_mag = torch.zeros_like(sample_mag)
                    mask_panel = sample_mag
                    mask_title = "Output magnitude (norm)"
                else:
                    mask_panel = model_out[0, 0].detach().cpu()
                    mask_title = "Mask (0-1)"
                sample_rate = int(sample_rates[0])
                plot_denoise_comparison(
                    sample_noisy,
                    sample_denoised,
                    sample_clean,
                    mask_panel,
                    model.window.detach().cpu(),
                    sample_rate,
                    RESULT_PLOT_PATH,
                    mask_title=mask_title,
                )
                save_wav(sample_noisy, sample_rate, NOISY_WAV_PATH)
                save_wav(sample_clean, sample_rate, CLEAN_WAV_PATH)
                save_wav(sample_denoised, sample_rate, DENOISED_WAV_PATH)
        progress.close()

    if loss_steps:
        plot_loss_curve(loss_steps, loss_values, LOSS_PLOT_PATH)

    model.eval()
    with torch.no_grad():
        clean, noisy, sample_rates = next(iter(loader))
        clean = clean.to(device)
        noisy = noisy.to(device)
        denoised, model_out = model(noisy)

        clean_mono = clean.mean(dim=1, keepdim=True)[0, 0].detach().cpu()
        noisy_mono = noisy.mean(dim=1, keepdim=True)[0, 0].detach().cpu()
        denoised_mono = denoised.mean(dim=1, keepdim=True)[0, 0].detach().cpu()
        if use_complex_unet:
            mask_mono = model_out.abs()[0].detach().cpu()
            mag_min = mask_mono.min()
            mag_max = mask_mono.max()
            if (mag_max - mag_min) > 1e-8:
                mask_mono = (mask_mono - mag_min) / (mag_max - mag_min)
            else:
                mask_mono = torch.zeros_like(mask_mono)
            mask_title = "Output magnitude (norm)"
        else:
            mask_mono = model_out[0, 0].detach().cpu()
            mask_title = "Mask (0-1)"
        sample_rate = int(sample_rates[0])

    plot_denoise_comparison(
        noisy_mono,
        denoised_mono,
        clean_mono,
        mask_mono,
        model.window.detach().cpu(),
        sample_rate,
        RESULT_PLOT_PATH,
        mask_title=mask_title,
    )
    save_wav(noisy_mono, sample_rate, NOISY_WAV_PATH)
    save_wav(clean_mono, sample_rate, CLEAN_WAV_PATH)
    save_wav(denoised_mono, sample_rate, DENOISED_WAV_PATH)
