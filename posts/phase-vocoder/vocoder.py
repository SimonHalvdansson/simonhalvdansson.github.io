#!/usr/bin/env python3
# vocoder.py — 2× speed phase-vocoder with two phase reconstruction methods
# - Reads audio.mp3 (same folder)
# - Builds half-time STFT target (magnitudes & phase gradients averaged pairwise)
# - Method 1: integrate phases along time using target dφ/dt
# - Method 2: optimize phases to match target gradients (time & freq) with magnitude-weighted loss
# - Robust iSTFT (manual overlap-add fallback)
# - Simple Spyder-friendly tqdm + plots:
#     * loss curve
#     * phase-gradient HSV maps (direction=hue, magnitude=lightness)
#     * phase maps (cyclic colormap)

import math
import sys
from pathlib import Path

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from tqdm import tqdm  # simple mode; tends to refresh better in Spyder

# =========================
# User options (top-level)
# =========================
INPUT_FILE = "audio.mp3"    # Input audio in this folder
METHOD = "both"             # "both", "method1", or "method2"
N_FFT = 2048                # STFT size
HOP = 512                   # Hop length (Hann + manual iSTFT makes this safe)
ITERS = 200                 # Method-2 optimizer steps
LR = 0.1                   # Method-2 learning rate
DEVICE_PREF = "auto"        # "auto", "cuda", "mps", or "cpu"
SHOW_TQDM = True            # Show tqdm progress bars
PLOT_GRAD_PERCENTILE = 95.0 # Clip gradient magnitude at this percentile for HSV map
FIG_DPI = 200               # a bit higher DPI for clarity

# tqdm defaults that work better in Spyder
_TQDM_KW = dict(ascii=True, ncols=80, mininterval=0.08, file=sys.stdout, leave=True)

PI = math.pi
TWOPI = 2 * math.pi

# ---------------------------- Utilities ---------------------------- #

def choose_device(prefer: str = "auto") -> torch.device:
    prefer = (prefer or "auto").lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        return wav
    return wav.mean(dim=0)


def principal_value(x: torch.Tensor) -> torch.Tensor:
    # map angles to (-pi, pi]
    return torch.atan2(torch.sin(x), torch.cos(x))


def circdiff(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    # circular difference b - a in (-pi, pi]
    return principal_value(b - a)


def compute_phase_grads(phase: torch.Tensor):
    """
    Central-difference circular gradients:
    dx: along time (x), dy: along frequency (y).
    Edges: forward/backward difference.
    phase: [F, T]
    """
    F, T = phase.shape
    dx = torch.zeros_like(phase)
    dy = torch.zeros_like(phase)

    # time diffs
    if T >= 3:
        dx[:, 1:-1] = circdiff(phase[:, 2:], phase[:, :-2]) / 2.0
    if T >= 2:
        dx[:, 0]  = circdiff(phase[:, 1],  phase[:, 0])
        dx[:, -1] = circdiff(phase[:, -1], phase[:, -2])

    # freq diffs
    if F >= 3:
        dy[1:-1, :] = circdiff(phase[2:, :], phase[:-2, :]) / 2.0
    if F >= 2:
        dy[0, :]  = circdiff(phase[1, :],  phase[0, :])
        dy[-1, :] = circdiff(phase[-1, :], phase[-2, :])

    return dx, dy


def pairwise_halve_time(x: torch.Tensor):
    """
    Average pairs of frames along time:
    x: [F, T] -> [F, T2] with T2 = T//2
    """
    F, T = x.shape
    T2 = T // 2
    if T2 == 0:
        raise ValueError("Not enough time frames to halve; increase hop or audio length.")
    x = x[:, : 2 * T2]
    return x.reshape(F, T2, 2).mean(dim=-1)


def pairwise_circular_mean_phase(phase: torch.Tensor) -> torch.Tensor:
    """
    Circular mean of phase over pairs of time frames (2t, 2t+1).
    Returns [F, T2], where angle(mean(exp(i*phase))) is taken.
    """
    F, T = phase.shape
    T2 = T // 2
    if T2 == 0:
        raise ValueError("Not enough time frames to halve for circular mean.")
    p = phase[:, :2*T2]
    u = torch.exp(1j * p)               # [F, 2*T2]
    u_pairs = u.reshape(F, T2, 2)
    u_mean = u_pairs.mean(dim=-1)       # complex mean over the pair
    return torch.angle(u_mean)          # [F, T2]


def complex_from_mag_phase(mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    return mag * torch.exp(1j * phase)


def estimate_length_from_frames(n_frames: int, n_fft: int, hop: int) -> int:
    # Center=True convention
    return max(0, (n_frames - 1) * hop + n_fft)


def weighted_phase_gradient_loss(phase: torch.Tensor,
                                 target_dx: torch.Tensor,
                                 target_dy: torch.Tensor,
                                 weight: torch.Tensor,
                                 eps: float = 1e-8) -> torch.Tensor:
    dx, dy = compute_phase_grads(phase)
    err_x = circdiff(dx, target_dx)
    err_y = circdiff(dy, target_dy)
    w = weight
    num = (err_x**2 + err_y**2) * w
    denom = w.sum().clamp_min(eps)
    return num.sum() / denom

# ------------------ Manual iSTFT (overlap-add) fallback ------------------ #

def _onesided_to_full_spectrum(spec_onesided: torch.Tensor, n_fft: int) -> torch.Tensor:
    """
    spec_onesided: [F, T] with F = n_fft//2 + 1
    Returns full spectrum [n_fft, T] with Hermitian symmetry.
    """
    F, T = spec_onesided.shape
    assert F == n_fft // 2 + 1, "One-sided spec F must be n_fft//2 + 1"
    dtype = spec_onesided.dtype
    device = spec_onesided.device

    full = torch.zeros((n_fft, T), dtype=dtype, device=device)
    full[:F, :] = spec_onesided
    if n_fft % 2 == 0:
        # exclude DC (0) and Nyquist (F-1) from mirror
        mirror = torch.conj(spec_onesided[1:F-1, :]).flip(0)
    else:
        mirror = torch.conj(spec_onesided[1:F, :]).flip(0)
    full[F:, :] = mirror
    return full


def istft_overlap_add_manual(spec_c: torch.Tensor,
                             n_fft: int,
                             hop: int,
                             win: torch.Tensor,
                             length: int | None):
    """
    Manual iSTFT via overlap-add with window-envelope normalization.
    """
    F, T = spec_c.shape
    device = spec_c.device
    full_spec = _onesided_to_full_spectrum(spec_c, n_fft)  # [n_fft, T]

    frames = torch.fft.ifft(full_spec, n=n_fft, dim=0).real
    frames = frames * win.view(-1, 1)

    out_len = (T - 1) * hop + n_fft
    y = torch.zeros(out_len, device=device)
    wsum = torch.zeros(out_len, device=device)
    w2 = (win ** 2)

    for t in range(T):
        start = t * hop
        y[start:start + n_fft] += frames[:, t]
        wsum[start:start + n_fft] += w2

    y = y / (wsum + 1e-12)

    if length is not None:
        if y.numel() >= length:
            y = y[:length]
        else:
            y = torch.nn.functional.pad(y, (0, length - y.numel()))

    return y.detach().cpu().to(torch.float32)


def istft_from_spec_safe(spec_c: torch.Tensor,
                         n_fft: int,
                         hop: int,
                         win: torch.Tensor,
                         length: int):
    """
    Try torch.istft; if COLA fails, use manual OA iSTFT.
    """
    def _istft(spec, window):
        return torch.istft(
            spec, n_fft=n_fft, hop_length=hop, win_length=window.numel(),
            window=window, center=True, length=length, return_complex=False
        )

    try:
        return _istft(spec_c, win).detach().cpu().to(torch.float32)
    except RuntimeError:
        return istft_overlap_add_manual(spec_c, n_fft, hop, win, length)

# ---------------------- Reconstruction Methods ---------------------- #

def reconstruct_phase_method1(seed_phase0: torch.Tensor,
                              target_dx: torch.Tensor) -> torch.Tensor:
    F, T2 = target_dx.shape
    phase = torch.zeros((F, T2), device=target_dx.device, dtype=target_dx.dtype)
    phase[:, 0] = seed_phase0
    for t in range(1, T2):
        phase[:, t] = principal_value(phase[:, t - 1] + target_dx[:, t - 1])
    return phase


def reconstruct_phase_method2(target_dx: torch.Tensor,
                              target_dy: torch.Tensor,
                              weight: torch.Tensor,
                              init_phase: torch.Tensor,
                              iters: int = 800,
                              lr: float = 0.1,
                              show_bar: bool = True):
    """
    Returns (optimized_phase, loss_history_list)
    """
    phase = init_phase.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([phase], lr=lr)
    losses = []

    bar = tqdm(range(iters), disable=not show_bar, desc="Optimizing phase (method2)", **_TQDM_KW)
    for _ in bar:
        opt.zero_grad(set_to_none=True)
        loss = weighted_phase_gradient_loss(phase, target_dx, target_dy, weight)
        loss.backward()
        opt.step()
        with torch.no_grad():
            phase[:] = principal_value(phase)
        lval = float(loss.detach().cpu())
        losses.append(lval)
        bar.set_postfix({"loss": f"{lval:.6g}"})
    return phase.detach(), losses

# --------------------------- Plotting --------------------------- #

def _grads_to_hsv_rgb(dx: np.ndarray, dy: np.ndarray, clip_percentile=95.0, sat=1.0):
    """
    Map (dx, dy) -> HSV image with hue=direction, value=|grad| normalized by percentile,
    saturation fixed. Returns RGB image (float 0..1).
    """
    angle = np.arctan2(dy, dx)  # (-pi, pi]
    hue = (angle + np.pi) / (2.0 * np.pi)  # 0..1

    mag = np.sqrt(dx * dx + dy * dy)
    thresh = np.percentile(mag, clip_percentile)
    thresh = max(thresh, 1e-9)
    val = np.clip(mag / thresh, 0.0, 1.0)

    hsv = np.stack([hue, np.full_like(hue, sat), val], axis=-1)  # [F, T, 3]
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb


def plot_phase_gradient_maps(panels, title="Phase gradient: hue=direction, value=magnitude"):
    """
    panels: list of (name, dx_tensor, dy_tensor)
    """
    n = len(panels)
    plt.figure(figsize=(4.5 * n, 4.0), dpi=FIG_DPI)
    for i, (name, dx_t, dy_t) in enumerate(panels, 1):
        dx = dx_t.detach().cpu().numpy()
        dy = dy_t.detach().cpu().numpy()
        rgb = _grads_to_hsv_rgb(dx, dy, clip_percentile=PLOT_GRAD_PERCENTILE, sat=1.0)
        ax = plt.subplot(1, n, i)
        ax.imshow(rgb, origin="lower", aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("time frames")
        ax.set_ylabel("freq bins")
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(title, y=1.02)
    plt.tight_layout()


def plot_loss_curve(losses_m2, loss_m1=None):
    if not losses_m2:
        return
    plt.figure(figsize=(6, 3.5), dpi=FIG_DPI)
    xs = np.arange(1, len(losses_m2) + 1)
    plt.plot(xs, losses_m2, label="method2 loss")
    if loss_m1 is not None:
        plt.axhline(loss_m1, color="k", linestyle="--", linewidth=1.0, label="method1 loss")
    plt.xlabel("iteration")
    plt.ylabel("weighted gradient loss")
    plt.title("Method-2 optimization loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_phase_maps(panels, vmin=-math.pi, vmax=math.pi, cmap="twilight",
                    title="Phase (radians)"):
    """
    panels: list of (name, phase_tensor [F,T])
    """
    n = len(panels)
    if n == 0:
        return
    plt.figure(figsize=(4.5 * n, 4.0), dpi=FIG_DPI)
    for i, (name, phase_t) in enumerate(panels, 1):
        P = phase_t.detach().cpu().numpy()
        ax = plt.subplot(1, n, i)
        im = ax.imshow(P, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(name)
        ax.set_xlabel("time frames")
        ax.set_ylabel("freq bins")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(title + f" | colormap={cmap}", y=1.02)
    plt.tight_layout()

# ------------------------------ Main ------------------------------ #

def run_pipeline(input_path: Path,
                 method: str,
                 n_fft: int,
                 hop: int,
                 iters: int,
                 lr: float,
                 device: torch.device,
                 show_tqdm: bool = True):
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find {input_path.resolve()}")

    print(f"[info] device: {device.type}")
    print(f"[info] input:  {input_path.name}")
    print(f"[info] method: {method} | n_fft={n_fft} hop={hop} iters={iters} lr={lr}")

    # Load
    wav, sr = torchaudio.load(str(input_path))
    wav = to_mono(wav).to(device)

    # STFT
    win = torch.hann_window(n_fft, device=device)
    spec = torch.stft(
        wav, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=win, center=True, return_complex=True
    )  # [F, T]
    F, T = spec.shape
    mag = spec.abs()
    phase = torch.angle(spec)

    # Phase gradients on original
    dx, dy = compute_phase_grads(phase)

    # Target (half frames)
    T2 = T // 2
    if T2 < 2:
        raise RuntimeError("Not enough frames to time-compress by 2×; choose smaller hop or longer audio.")

    mag_tgt = pairwise_halve_time(mag)   # [F, T2]
    dx_tgt  = pairwise_halve_time(dx)    # [F, T2]
    dy_tgt  = pairwise_halve_time(dy)    # [F, T2]
    phase_tgt_circ = pairwise_circular_mean_phase(phase)  # reference target phase

    # Seed phase t=0 from original
    seed_phase0 = phase[:, 0]

    # Output length derived from new number of frames
    out_len = estimate_length_from_frames(T2, n_fft, hop)

    # Method 1
    run_m1 = method in ("both", "method1")
    out1 = None
    loss_m1 = None
    phase_m1 = None
    if run_m1:
        phase_m1 = reconstruct_phase_method1(seed_phase0, dx_tgt)
        loss_m1 = weighted_phase_gradient_loss(phase_m1, dx_tgt, dy_tgt, mag_tgt).detach().cpu().item()
        print(f"[method1] weighted gradient loss: {loss_m1:.6f}")

        spec_m1 = complex_from_mag_phase(mag_tgt, phase_m1)
        audio_m1 = istft_from_spec_safe(spec_m1, n_fft, hop, win, length=out_len)
        out1 = input_path.with_name(input_path.stem + "_vocoder2x_method1.wav")
        torchaudio.save(str(out1), audio_m1.unsqueeze(0), sample_rate=sr)
        print(f"[method1] saved: {out1.name}")

    # Method 2
    run_m2 = method in ("both", "method2")
    out2 = None
    losses_m2 = []
    phase_m2 = None
    if run_m2:
        init = phase_m1 if run_m1 else reconstruct_phase_method1(seed_phase0, dx_tgt)
        phase_m2, losses_m2 = reconstruct_phase_method2(
            target_dx=dx_tgt, target_dy=dy_tgt, weight=mag_tgt,
            init_phase=init, iters=iters, lr=lr, show_bar=show_tqdm
        )
        loss_m2_final = weighted_phase_gradient_loss(phase_m2, dx_tgt, dy_tgt, mag_tgt).detach().cpu().item()
        print(f"[method2] final weighted gradient loss: {loss_m2_final:.6f}")

        spec_m2 = complex_from_mag_phase(mag_tgt, phase_m2)
        audio_m2 = istft_from_spec_safe(spec_m2, n_fft, hop, win, length=out_len)
        out2 = input_path.with_name(input_path.stem + "_vocoder2x_method2.wav")
        torchaudio.save(str(out2), audio_m2.unsqueeze(0), sample_rate=sr)
        print(f"[method2] saved: {out2.name}")

    # -------- Plots --------
    plt.rcParams["figure.dpi"] = FIG_DPI

    # Loss curve (if method2 ran)
    if run_m2 and losses_m2:
        plot_loss_curve(losses_m2, loss_m1=loss_m1)

    # Phase gradient HSV maps: target vs reconstructions
    grad_panels = [("Target (2×)", dx_tgt, dy_tgt)]
    if run_m1:
        dx_m1, dy_m1 = compute_phase_grads(phase_m1)
        grad_panels.append(("Method 1", dx_m1, dy_m1))
    if run_m2:
        dx_m2, dy_m2 = compute_phase_grads(phase_m2)
        grad_panels.append(("Method 2", dx_m2, dy_m2))
    plot_phase_gradient_maps(grad_panels)

    # Phase maps: target (circular-mean of pairs) vs reconstructions
    phase_panels = [("Target phase (circ-mean)", phase_tgt_circ)]
    if run_m1:
        phase_panels.append(("Method 1 phase", phase_m1))
    if run_m2:
        phase_panels.append(("Method 2 phase", phase_m2))
    plot_phase_maps(phase_panels, cmap="twilight")

    plt.show()

    print("Done.")
    return out1, out2


def main():
    device = choose_device(DEVICE_PREF)
    run_pipeline(
        input_path=Path(INPUT_FILE),
        method=METHOD,
        n_fft=N_FFT,
        hop=HOP,
        iters=ITERS,
        lr=LR,
        device=device,
        show_tqdm=SHOW_TQDM,
    )


if __name__ == "__main__":
    main()
