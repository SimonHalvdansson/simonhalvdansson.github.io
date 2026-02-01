import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.io.wavfile import write as write_wav
from scipy.signal import stft, istft


def phase_gradient(phi, axis):
    """Compute wrapped phase gradient along an axis."""
    d = np.diff(phi, axis=axis)
    return np.angle(np.exp(1j * d))


def plot_stft(f, tt, Zxx, *, title_prefix, out_path, ylim=(0, 1000)):
    mag = np.abs(Zxx)
    phi = np.angle(Zxx)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-8))

    # Normalize magnitude for brightness
    mag_norm = mag / (mag.max() + 1e-12)

    # Phase as HSV (hue=phase, value=mag)
    hue = (phi + np.pi) / (2 * np.pi)
    hsv = np.stack([hue, np.ones_like(hue), mag_norm], axis=-1)
    rgb = hsv_to_rgb(hsv)

    # Phase gradients with wrapping
    grad_t = phase_gradient(phi, axis=1)
    grad_f = phase_gradient(phi, axis=0)

    # STFT figure (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # Magnitude
    im0 = axs[0, 0].imshow(mag_db, origin='lower', aspect='auto',
                           extent=[tt[0], tt[-1], f[0], f[-1]], cmap='magma')
    axs[0, 0].set_title(f'{title_prefix} STFT Magnitude (dB)')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Frequency [Hz]')
    axs[0, 0].set_ylim(*ylim)
    fig.colorbar(im0, ax=axs[0, 0], shrink=0.8)

    # Phase (color wheel with brightness from magnitude)
    axs[0, 1].imshow(rgb, origin='lower', aspect='auto', extent=[tt[0], tt[-1], f[0], f[-1]])
    axs[0, 1].set_title(f'{title_prefix} STFT Phase (HSV hue, mag as brightness)')
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Frequency [Hz]')
    axs[0, 1].set_ylim(*ylim)

    # Phase gradients
    im1 = axs[1, 0].imshow(grad_t, origin='lower', aspect='auto',
                           extent=[tt[0], tt[-1], f[0], f[-1]], cmap='twilight')
    axs[1, 0].set_title(f'{title_prefix} Phase Gradient (time axis)')
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Frequency [Hz]')
    axs[1, 0].set_ylim(*ylim)
    fig.colorbar(im1, ax=axs[1, 0], shrink=0.8)

    im2 = axs[1, 1].imshow(grad_f, origin='lower', aspect='auto',
                           extent=[tt[0], tt[-1], f[0], f[-1]], cmap='twilight')
    axs[1, 1].set_title(f'{title_prefix} Phase Gradient (frequency axis)')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Frequency [Hz]')
    axs[1, 1].set_ylim(*ylim)
    fig.colorbar(im2, ax=axs[1, 1], shrink=0.8)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    fs = 16000
    duration = 1.0
    freq = 440.0
    t = np.arange(int(fs * duration)) / fs
    x = np.sin(2 * np.pi * freq * t)

    nperseg = 1024
    noverlap = 768
    f, tt, Zxx = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann', boundary=None, padded=False)

    # Randomize phase, keep magnitude, and synthesize waveform
    mag = np.abs(Zxx)
    rand_phase = np.random.uniform(-np.pi, np.pi, size=Zxx.shape)
    Zxx_rand = mag * np.exp(1j * rand_phase)
    _, x_mod = istft(Zxx_rand, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann', input_onesided=True)

    # Save waveforms
    write_wav('media/pure_sine.wav', fs, x.astype(np.float32))
    write_wav('media/modified_sine.wav', fs, x_mod.astype(np.float32))

    # Plots
    plot_stft(f, tt, Zxx, title_prefix='Original', out_path='media/sine.png', ylim=(0, 1000))
    f2, tt2, Zxx2 = stft(x_mod, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann', boundary=None, padded=False)
    plot_stft(f2, tt2, Zxx2, title_prefix='Modified', out_path='media/modified_sine.png', ylim=(0, 1000))


if __name__ == '__main__':
    main()
