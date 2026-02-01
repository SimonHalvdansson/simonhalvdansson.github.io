import wave
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_FFT = 1024
HOP_LENGTH = 256
WINDOW_LENGTH = N_FFT


def load_audio(path: Path):
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)
    return waveform, sample_rate


def normalize_waveform(waveform):
    return waveform / waveform.pow(2).mean(dim=1, keepdim=True).sqrt()


def save_wav(waveform, sample_rate, wav_path: Path):
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    channels = waveform.shape[0]
    data = waveform.transpose(0, 1).contiguous()
    if data.is_floating_point():
        data = data.clamp(-1.0, 1.0)
        data = (data * 32767.0).to(torch.int16)

    data_np = data.cpu().numpy().tobytes()
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data_np)


def save_spectrogram_image(waveform, spec_path: Path, sample_rate: int | None = None):
    spec_path = Path(spec_path)
    spec_path.parent.mkdir(parents=True, exist_ok=True)

    mono = waveform.squeeze(0)

    window = torch.hann_window(WINDOW_LENGTH)
    spec = torch.stft(
        mono,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True,
    )
    spec_mag = spec.abs()
    power = spec_mag.pow(2)
    spec_db = 10 * torch.log10(power + 1e-10)
    shape_text = f"{tuple(spec_db.shape)}"
    window_text = f"window: {WINDOW_LENGTH}"

    fig = plt.figure(figsize=(12, 4), layout="constrained")
    subfigs = fig.subfigures(1, 2, wspace=0.05)

    subfigs[0].suptitle(f"dB Spectrogram (shape: {shape_text}, {window_text})", fontsize=10)
    ax_db = subfigs[0].subplots(1, 1)
    if sample_rate is None:
        extent = None
        y_label = "Frequency (bins)"
    else:
        extent = [0.0, float(spec_db.shape[-1]), 0.0, float(sample_rate) / 2000.0]
        y_label = "Frequency (kHz)"
    im_db = ax_db.imshow(spec_db.numpy(), origin="lower", aspect="auto", cmap="magma", extent=extent)
    ax_db.set_ylabel(y_label)
    ax_db.set_xlabel("Frame")
    subfigs[0].colorbar(im_db, ax=ax_db, label="dB")

    subfigs[1].suptitle(
        f"Linear Spectrogram (abs) (shape: {shape_text}, {window_text})",
        fontsize=10,
    )
    ax_lin = subfigs[1].subplots(1, 1)
    im_lin = ax_lin.imshow(spec_mag.numpy(), origin="lower", aspect="auto", cmap="magma", extent=extent)
    ax_lin.set_ylabel(y_label)
    ax_lin.set_xlabel("Frame")
    subfigs[1].colorbar(im_lin, ax=ax_lin, label="abs")

    fig.savefig(spec_path, dpi=150)
    plt.close(fig)


def save_example(waveform, sample_rate, wav_path="media/example.wav", spec_path="media/example_spec.png"):
    save_wav(waveform, sample_rate, Path(wav_path))
    save_spectrogram_image(waveform, Path(spec_path), sample_rate=sample_rate)


def plot_loss_curve(steps, losses, path: Path, ma_window: int = 50, long_ma_window: int = 500, extra_long_ma_window: int = 5000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 4), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    if len(losses) > 500:
        steps = steps[500:]
        losses = losses[500:]
    if len(losses) >= ma_window:
        ma_values = [
            sum(losses[i - ma_window + 1 : i + 1]) / ma_window
            for i in range(ma_window - 1, len(losses))
        ]
        ax.plot(
            steps[ma_window - 1 :],
            ma_values,
            color="#ff7f0e",
            linewidth=1.5,
            label=f"MA {ma_window}",
        )
    if len(losses) >= long_ma_window:
        long_ma_values = [
            sum(losses[i - long_ma_window + 1 : i + 1]) / long_ma_window
            for i in range(long_ma_window - 1, len(losses))
        ]
        ax.plot(
            steps[long_ma_window - 1 :],
            long_ma_values,
            color="#2ca02c",
            linewidth=1.5,
            label=f"MA {long_ma_window}",
        )
    if len(losses) >= extra_long_ma_window:
        extra_long_ma_values = [
            sum(losses[i - extra_long_ma_window + 1 : i + 1]) / extra_long_ma_window
            for i in range(extra_long_ma_window - 1, len(losses))
        ]
        ax.plot(
            steps[extra_long_ma_window - 1 :],
            extra_long_ma_values,
            color="#1f77b4",
            linewidth=1.5,
            label=f"MA {extra_long_ma_window}",
        )
    ax.set_yscale("log")
    ax.yaxis.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Step")
    ax.set_ylabel("L1 Spectrogram Loss")
    ax.set_title("Training Loss (log scale)")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_spec_db(waveform: torch.Tensor, window: torch.Tensor):
    spec = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        window=window,
        return_complex=True,
    )
    power = spec.abs().pow(2)
    return 10 * torch.log10(power + 1e-10)


def plot_denoise_comparison(
    noisy,
    denoised,
    clean,
    mask,
    window: torch.Tensor,
    sample_rate: int,
    path: Path,
    mask_title: str = "Mask (0-1)",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    noisy_spec = compute_spec_db(noisy, window).cpu().numpy()
    denoised_spec = compute_spec_db(denoised, window).cpu().numpy()
    clean_spec = compute_spec_db(clean, window).cpu().numpy()

    duration = float(noisy.shape[-1]) / float(sample_rate)
    x_max = min(2.0, duration)
    extent = [0.0, duration, 0.0, float(sample_rate) / 2000.0]

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    axes = fig.subplots(2, 2, sharex=True)

    im0 = axes[0, 0].imshow(noisy_spec, origin="lower", aspect="auto", cmap="magma", extent=extent)
    axes[0, 0].set_title("Noisy Spectrogram (dB)")
    im1 = axes[0, 1].imshow(denoised_spec, origin="lower", aspect="auto", cmap="magma", extent=extent)
    axes[0, 1].set_title("Denoised Spectrogram (dB)")
    im2 = axes[1, 0].imshow(clean_spec, origin="lower", aspect="auto", cmap="magma", extent=extent)
    axes[1, 0].set_title("Clean Spectrogram (dB)")

    mask_np = mask.cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    axes[1, 1].imshow(
        mask_np,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )
    axes[1, 1].set_title(mask_title)

    axes[0, 1].sharey(axes[0, 0])
    axes[1, 0].sharey(axes[0, 0])

    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0.0, x_max)
    for ax in axes[0, :]:
        ax.set_xlim(0.0, x_max)

    axes[0, 0].set_ylabel("Frequency (kHz)")
    axes[1, 0].set_ylabel("Frequency (kHz)")

    fig.savefig(path, dpi=150)
    plt.close(fig)


def pad_collate(batch):
    clean_list, noisy_list, sample_rates = zip(*batch)
    max_len = max(item.shape[-1] for item in clean_list)

    def _pad(t):
        pad_amount = max_len - t.shape[-1]
        if pad_amount <= 0:
            return t
        return F.pad(t, (0, pad_amount))

    clean = torch.stack([_pad(item) for item in clean_list], dim=0)
    noisy = torch.stack([_pad(item) for item in noisy_list], dim=0)
    return clean, noisy, sample_rates
