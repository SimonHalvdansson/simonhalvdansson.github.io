import csv
import random
from pathlib import Path
import os
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from models import (
    PhaseGradientLoss,
    SpectrogramLoss,
    SpectrogramMaskUNet,
    SpectrogramPhaseGradientUNet,
    Spectrogram2ChannelUNet,
    wrap_phase,
)
from utils import (
    HOP_LENGTH,
    N_FFT,
    WINDOW_LENGTH,
    load_audio,
    normalize_waveform,
    pad_collate,
    plot_denoise_comparison,
    plot_loss_curve,
    plot_simple_loss_curve,
    phase_hsv_rgb,
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

        waveform = waveform.mean(dim=0, keepdim=True)
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



def minimize_phase_from_gradients(
    phi_x: torch.Tensor,
    phi_y: torch.Tensor,
    num_iters: int = 120,
    lr: float = 0.2,
    return_loss: bool = False,
) -> Tuple[torch.Tensor, Optional[List[float]]]:
    phi_x = phi_x.detach()
    phi_y = phi_y.detach()

    loss_history: Optional[List[float]] = [] if return_loss else None
    with torch.enable_grad():
        phase = torch.zeros_like(phi_x, requires_grad=True)
        optimizer = torch.optim.Adam([phase], lr=lr)
        for _ in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            diff_t = phase[:, 1:] - phase[:, :-1]
            diff_f = phase[1:, :] - phase[:-1, :]
            loss_t = wrap_phase(diff_t - phi_x[:, :-1])
            loss_f = wrap_phase(diff_f - phi_y[:-1, :])
            loss = loss_t.pow(2).mean() + loss_f.pow(2).mean()
            loss.backward()
            optimizer.step()
            if loss_history is not None:
                loss_history.append(loss.item())

    return phase.detach(), loss_history


def reconstruct_phase_gradient_denoised(
    pred_mag: torch.Tensor,
    pred_phi_x: torch.Tensor,
    pred_phi_y: torch.Tensor,
    window: torch.Tensor,
    length: int,
    num_iters: int = 480,
    lr: float = 0.2,
    return_loss: bool = False,
):
    pred_mag = pred_mag.detach().cpu()
    pred_phi_x = pred_phi_x.detach().cpu()
    pred_phi_y = pred_phi_y.detach().cpu()

    if pred_mag.dim() == 3:
        pred_mag = pred_mag.squeeze(0)
    if pred_phi_x.dim() == 3:
        pred_phi_x = pred_phi_x.squeeze(0)
    if pred_phi_y.dim() == 3:
        pred_phi_y = pred_phi_y.squeeze(0)

    phase, loss_history = minimize_phase_from_gradients(
        pred_phi_x,
        pred_phi_y,
        num_iters=num_iters,
        lr=lr,
        return_loss=return_loss,
    )
    complex_stft = torch.polar(pred_mag, phase)
    denoised = torch.istft(
        complex_stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        window=window.detach().cpu(),
        length=length,
    )
    return denoised, loss_history, phase


def prepare_example_outputs(
    model_type: str,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    denoised: Optional[torch.Tensor],
    model_out,
    model,
    gradient_loss_path: Optional[Path] = None,
):
    sample_noisy = noisy[0, 0].detach().cpu()
    sample_clean = clean[0, 0].detach().cpu()

    if model_type == "phase_gradient":
        pred_mag = model_out["mag"][0]
        pred_phi_x = model_out["phi_x"][0]
        pred_phi_y = model_out["phi_y"][0]
        sample_denoised, loss_history, phase = reconstruct_phase_gradient_denoised(
            pred_mag,
            pred_phi_x,
            pred_phi_y,
            model.window,
            length=sample_noisy.shape[-1],
            return_loss=gradient_loss_path is not None,
        )
        phase_pred_rgb = phase_hsv_rgb(phase, pred_mag)
        gabor_stft = torch.stft(
            sample_denoised,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=model.window.detach().cpu(),
            return_complex=True,
        )
        phase_gabor_rgb = phase_hsv_rgb(torch.angle(gabor_stft), gabor_stft.abs())
        if gradient_loss_path is not None and loss_history:
            plot_simple_loss_curve(loss_history, gradient_loss_path)
        output_mag_db = 20.0 * torch.log10(pred_mag.detach().cpu().clamp_min(1e-8))
        panel_data = output_mag_db
        mask_title = "Output magnitude (dB)"
    elif model_type == "2channel":
        sample_denoised = denoised[0, 0].detach().cpu()
        output_mag = model_out.abs()[0].detach().cpu()
        output_mag_db = 20.0 * torch.log10(output_mag.clamp_min(1e-8))
        panel_data = output_mag_db
        mask_title = "Output magnitude (dB)"
    else:
        sample_denoised = denoised[0, 0].detach().cpu()
        panel_data = model_out[0, 0].detach().cpu()
        mask_title = "Mask (0-1)"

    if model_type == "phase_gradient":
        return (
            sample_noisy,
            sample_denoised,
            sample_clean,
            panel_data,
            mask_title,
            phase_pred_rgb,
            phase_gabor_rgb,
        )
    return sample_noisy, sample_denoised, sample_clean, panel_data, mask_title, None, None


def log_training_snapshot(
    step: int,
    loss_steps: List[int],
    loss_values: List[float],
    clean: torch.Tensor,
    noisy: torch.Tensor,
    denoised: Optional[torch.Tensor],
    model_out,
    model,
    sample_rates,
    loss_plot_path: Path,
    result_plot_path: Path,
    clean_wav_path: Path,
    noisy_wav_path: Path,
    denoised_wav_path: Path,
    gradient_loss_path: Path,
    model_type: str,
):
    if step % 300 != 0:
        return

    plot_loss_curve(loss_steps, loss_values, loss_plot_path)
    sample_noisy, sample_denoised, sample_clean, panel_data, mask_title, phase_pred_rgb, phase_gabor_rgb = prepare_example_outputs(
        model_type,
        clean,
        noisy,
        denoised,
        model_out,
        model,
        gradient_loss_path=gradient_loss_path if model_type == "phase_gradient" else None,
    )
    sample_rate = int(sample_rates[0])
    plot_denoise_comparison(
        sample_noisy,
        sample_denoised,
        sample_clean,
        panel_data,
        model.window.detach().cpu(),
        sample_rate,
        result_plot_path,
        mask_title=mask_title,
        model_type=model_type,
        phase_pred_rgb=phase_pred_rgb,
        phase_gabor_rgb=phase_gabor_rgb,
    )
    save_wav(sample_noisy, sample_rate, noisy_wav_path)
    save_wav(sample_clean, sample_rate, clean_wav_path)
    save_wav(sample_denoised, sample_rate, denoised_wav_path)


if __name__ == "__main__":
    model_type = "phase_gradient"

    max_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_slug = model_type.replace(" ", "_")
    output_dir = Path(f"media/{model_slug}_output")

    LOSS_PLOT_PATH = output_dir / "loss_curve.png"
    RESULT_PLOT_PATH = output_dir / "denoise_compare.png"
    GRADIENT_LOSS_PLOT_PATH = output_dir / "gradient_integration_loss.png"
    CLEAN_WAV_PATH = output_dir / "clean.wav"
    NOISY_WAV_PATH = output_dir / "noisy.wav"
    DENOISED_WAV_PATH = output_dir / "denoised.wav"
    dataset = CommonVoiceRandomSegmentDataset(split="train")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=min(8, max(2, (os.cpu_count() or 4) // 2)),
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_collate,
    )

    if model_type == "phase_gradient":
        model = SpectrogramPhaseGradientUNet().to(device)
        loss_fn = PhaseGradientLoss().to(device)
    elif model_type == "2channel":
        model = Spectrogram2ChannelUNet().to(device)
        loss_fn = SpectrogramLoss().to(device)
    else:
        model = SpectrogramMaskUNet().to(device)
        loss_fn = SpectrogramLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)

    step = 0
    loss_steps = []
    loss_values = []

    total_params = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {total_params:,}")

    model.train()
    for epoch in range(max_epochs):
        progress = tqdm(total=len(dataset), desc=f"Epoch ({epoch + 1}/{max_epochs})", unit="clip")
        epoch_loss_total = 0.0
        epoch_loss_count = 0
        for clean, noisy, sample_rates in loader:
            clean = clean.to(device)
            noisy = noisy.to(device)

            if model_type == "phase_gradient":
                model_out = model(noisy)
                denoised = None
                loss = loss_fn(clean, model_out)
            else:
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

            log_training_snapshot(
                step,
                loss_steps,
                loss_values,
                clean,
                noisy,
                denoised,
                model_out,
                model,
                sample_rates,
                LOSS_PLOT_PATH,
                RESULT_PLOT_PATH,
                CLEAN_WAV_PATH,
                NOISY_WAV_PATH,
                DENOISED_WAV_PATH,
                GRADIENT_LOSS_PLOT_PATH,
                model_type,
            )
        progress.close()

    if loss_steps:
        plot_loss_curve(loss_steps, loss_values, LOSS_PLOT_PATH)

    model.eval()
    with torch.no_grad():
        clean, noisy, sample_rates = next(iter(loader))
        clean = clean.to(device)
        noisy = noisy.to(device)
        if model_type == "phase_gradient":
            model_out = model(noisy)
            denoised = None
        else:
            denoised, model_out = model(noisy)

    sample_noisy, sample_denoised, sample_clean, panel_data, mask_title, phase_pred_rgb, phase_gabor_rgb = prepare_example_outputs(
        model_type,
        clean,
        noisy,
        denoised,
        model_out,
        model,
        gradient_loss_path=GRADIENT_LOSS_PLOT_PATH if model_type == "phase_gradient" else None,
    )
    sample_rate = int(sample_rates[0])

    plot_denoise_comparison(
        sample_noisy,
        sample_denoised,
        sample_clean,
        panel_data,
        model.window.detach().cpu(),
        sample_rate,
        RESULT_PLOT_PATH,
        mask_title=mask_title,
        model_type=model_type,
        phase_pred_rgb=phase_pred_rgb,
        phase_gabor_rgb=phase_gabor_rgb,
    )
    save_wav(sample_noisy, sample_rate, NOISY_WAV_PATH)
    save_wav(sample_clean, sample_rate, CLEAN_WAV_PATH)
    save_wav(sample_denoised, sample_rate, DENOISED_WAV_PATH)
