#!/usr/bin/env python3
"""
Compare normalized L², DTW, and OST distances between various signal pairs:
  • Sinusoids (100 Hz vs 100–200 Hz)
  • Speed‐warped modulated Gaussian (3 cases)
  • Center‐shifted modulated Gaussian (3 cases)
  • Chirp alignment (5 warp factors)
  • Composite tones with a parametrized gap (3 cases)
  • Noise robustness on a modulated Gaussian (3 cases)

Each figure has three rows:
  1) Raw time‐domain signals
  2) Spectrograms (with consistent y‐limits per experiment)
  3) Normalized distance curves

All text is small and plots are high‐res (dpi=300).

Dependencies:
    pip install numpy scipy matplotlib fastdtw pot tqdm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, chirp
from fastdtw import fastdtw
import ot
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D



# --- Configuration ---
# High resolution and smaller fonts
plt.rcParams.update({'font.size': 8})
DPI = 200
SAVE_DIR = "media" # Directory to save figures

# --- Helper Functions ---
def l2_distance(a, b, t):
    return np.sqrt(np.trapz((a - b)**2, t))

def dtw_distance(a, b):
    dist, _ = fastdtw(a, b, dist=lambda x, y: abs(x - y))
    return dist

def ost_distance(x, y, fs):
    nperseg, noverlap = 128, 64
    _, _, S1 = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, _, S2 = spectrogram(y, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # Handle silent signals to avoid division by zero
    if S1.sum() == 0 or S2.sum() == 0:
        return 0.0 if S1.sum() == S2.sum() else np.inf
    P1 = S1.flatten(); P1 /= P1.sum()
    P2 = S2.flatten(); P2 /= P2.sum()
    fb, tb = S1.shape
    coords = np.array([[i, j] for i in range(fb) for j in range(tb)])
    M = ot.dist(coords, coords)
    return ot.emd2(P1, P2, M)

# --- Experiment Functions ---
def experiment_sinusoids():
    fs, duration = 1000.0, 1.0
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    omegas = np.array([100, 150, 200])
    base = np.sin(2*np.pi*100*t)

    d_l2, d_dtw, d_ost = [], [], []
    sweep = np.linspace(100, 200, 50)
    for ω in tqdm(sweep, desc="1. Sinusoids"):
        sig = np.sin(2*np.pi*ω*t)
        d_l2.append(l2_distance(base, sig, t))
        d_dtw.append(dtw_distance(base, sig))
        d_ost.append(ost_distance(base, sig, fs))
    d_l2 = np.array(d_l2)/np.max(d_l2)
    d_dtw = np.array(d_dtw)/np.max(d_dtw)
    d_ost = np.array(d_ost)/np.max(d_ost)

    ylimit = 2*omegas.max()  # consistent across all three = 400 Hz

    fig = plt.figure(constrained_layout=True, dpi=DPI)
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # Row 0: time-domain
    for i, freq in enumerate(omegas):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(t, np.sin(2*np.pi*freq*t))
        ax.set_title(f"{freq} Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amp")

    # Row 1: spectrograms
    for i, freq in enumerate(omegas):
        sig = np.sin(2*np.pi*freq*t)
        ax = fig.add_subplot(gs[1, i])
        f_spec, t_spec, S = spectrogram(sig, fs=fs, nperseg=128, noverlap=64)
        ax.pcolormesh(t_spec, f_spec, S, shading='gouraud') # Restored to original
        ax.set_ylim(0, ylimit)
        ax.set_title(f"{freq} Hz (spec)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")

    # Row 2: curves
    ax = fig.add_subplot(gs[2, :])
    ax.plot(sweep, d_l2,  label='L²')
    ax.plot(sweep, d_dtw, label='DTW')
    ax.plot(sweep, d_ost, label='OST')
    ax.set_xlabel("ω (Hz)")
    ax.set_ylabel("Norm Dist")
    ax.set_title("Sinusoid Distances")
    ax.legend()
    
    # Save, then show, then close
    filename = os.path.join(SAVE_DIR, "01_sinusoid_comparison.png")
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def experiment_gaussian_speed():
    fs, duration = 1000.0, 1.0
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    center, sigma = duration/2, 0.1
    raw = np.exp(-0.5*((t-center)/sigma)**2)
    mod = np.sin(10*2*np.pi*t)
    g0  = raw*mod

    speed_factors = np.linspace(0.4, 2.0, 50)
    d_l2, d_dtw, d_ost = [], [], []
    for sf in tqdm(speed_factors, desc="2. Gauss Speed"):
        tau = sf*(t-center)+center
        gw  = np.exp(-0.5*((tau-center)/sigma)**2)*np.sin(10*2*np.pi*tau)
        d_l2.append(l2_distance(g0, gw, t))
        d_dtw.append(dtw_distance(g0, gw))
        d_ost.append(ost_distance(g0, gw, fs))
    d_l2 = np.array(d_l2)/np.max(d_l2)
    d_dtw = np.array(d_dtw)/np.max(d_dtw)
    d_ost = np.array(d_ost)/np.max(d_ost)

    sf_vals = [speed_factors[0], 1.0, speed_factors[-1]]
    sigs = []
    for sf in sf_vals:
        tau = sf*(t-center)+center
        sigs.append(np.exp(-0.5*((tau-center)/sigma)**2)*np.sin(10*2*np.pi*tau))

    fig = plt.figure(constrained_layout=True, dpi=DPI)
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # Row 0
    for i, (sig, sf) in enumerate(zip(sigs, sf_vals)):
        ax = fig.add_subplot(gs[0,i])
        ax.plot(t, sig)
        ax.set_title(f"sf={sf:.2f}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amp")

    # Row 1: spectrograms (0–50 Hz)
    for i, sig in enumerate(sigs):
        ax = fig.add_subplot(gs[1,i])
        f_spec, t_spec, S = spectrogram(sig, fs=fs, nperseg=128, noverlap=64)
        ax.pcolormesh(t_spec, f_spec, S, shading='gouraud') # Restored to original
        ax.set_ylim(0, 50)
        ax.set_title("spec")
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")

    # Row 2
    ax = fig.add_subplot(gs[2,:])
    ax.plot(speed_factors, d_l2,  label='L²')
    ax.plot(speed_factors, d_dtw, label='DTW')
    ax.plot(speed_factors, d_ost, label='OST')
    ax.set_xlabel("sf")
    ax.set_ylabel("Norm Dist")
    ax.set_title("Gaussian Speed Distances")
    ax.legend()
    
    # Save, then show, then close
    filename = os.path.join(SAVE_DIR, "02_gaussian_speed_warp.png")
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def experiment_gaussian_shift():
    fs, duration = 1000.0, 1.0
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    center, sigma = duration/2, 0.1
    raw = np.exp(-0.5*((t-center)/sigma)**2)
    mod = np.sin(10*2*np.pi*t)
    g0  = raw*mod

    shifts = np.linspace(-0.4, 0.4, 50)
    d_l2, d_dtw, d_ost = [], [], []
    for Δ in tqdm(shifts, desc="3. Gauss Shift"):
        tau = t - Δ
        gw  = np.exp(-0.5*((tau-center)/sigma)**2)*np.sin(10*2*np.pi*tau)
        d_l2.append(l2_distance(g0, gw, t))
        d_dtw.append(dtw_distance(g0, gw))
        d_ost.append(ost_distance(g0, gw, fs))
    d_l2 = np.array(d_l2)/np.max(d_l2)
    d_dtw = np.array(d_dtw)/np.max(d_dtw)
    d_ost = np.array(d_ost)/np.max(d_ost)

    sh_vals = [shifts[0], 0.0, shifts[-1]]
    sigs = []
    for Δ in sh_vals:
        tau = t - Δ
        sigs.append(np.exp(-0.5*((tau-center)/sigma)**2)*np.sin(10*2*np.pi*tau))

    fig = plt.figure(constrained_layout=True, dpi=DPI)
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # Row 0
    for i, (sig, Δ) in enumerate(zip(sigs, sh_vals)):
        ax = fig.add_subplot(gs[0,i])
        ax.plot(t, sig)
        ax.set_title(f"Δ={Δ:.2f}s")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amp")

    # Row 1: spectrograms (0–50 Hz)
    for i, sig in enumerate(sigs):
        ax = fig.add_subplot(gs[1,i])
        f_spec, t_spec, S = spectrogram(sig, fs=fs, nperseg=128, noverlap=64)
        ax.pcolormesh(t_spec, f_spec, S, shading='gouraud') # Restored to original
        ax.set_ylim(0, 50)
        ax.set_title("spec")
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")

    # Row 2
    ax = fig.add_subplot(gs[2,:])
    ax.plot(shifts, d_l2,  label='L²')
    ax.plot(shifts, d_dtw, label='DTW')
    ax.plot(shifts, d_ost, label='OST')
    ax.set_xlabel("Δ (s)")
    ax.set_ylabel("Norm Dist")
    ax.set_title("Gaussian Shift Distances")
    ax.legend()

    # Save, then show, then close
    filename = os.path.join(SAVE_DIR, "03_gaussian_time_shift.png")
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def experiment_chirp_alignment():
    fs, duration = 1000.0, 1.0
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    base = chirp(t, f0=50, t1=duration, f1=250, method='linear')

    alphas = np.linspace(0.5, 1.5, 50)
    d_l2, d_dtw, d_ost = [], [], []
    for α in tqdm(alphas, desc="4. Chirp Align"):
        tau = np.where(t<0.5, α*t, 0.5*α + (t-0.5)*(2-α))
        warped = chirp(tau, f0=50, t1=duration, f1=250, method='linear')
        d_l2.append(l2_distance(base, warped, t))
        d_dtw.append(dtw_distance(base, warped))
        d_ost.append(ost_distance(base, warped, fs))
    d_l2  = np.array(d_l2)/np.max(d_l2)
    d_dtw = np.array(d_dtw)/np.max(d_dtw)
    d_ost = np.array(d_ost)/np.max(d_ost)

    α_disp = np.linspace(0.5, 1.5, 5)
    sigs = [(chirp(np.where(t<0.5, α*t, 0.5*α + (t-0.5)*(2-α)),
                   f0=50, t1=duration, f1=250, method='linear'), α)
            for α in α_disp]

    fig = plt.figure(constrained_layout=True, dpi=DPI)
    gs = gridspec.GridSpec(3, 5, figure=fig)

    # Row 0
    for i, (sig, α) in enumerate(sigs):
        ax = fig.add_subplot(gs[0,i])
        ax.plot(t, sig)
        ax.set_title(f"α={α:.2f}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amp")

    # Row 1: spectrograms (0–500 Hz)
    for i, (sig, α) in enumerate(sigs):
        ax = fig.add_subplot(gs[1,i])
        f_spec, t_spec, S = spectrogram(sig, fs=fs, nperseg=128, noverlap=64)
        ax.pcolormesh(t_spec, f_spec, S, shading='gouraud') # Restored to original
        ax.set_ylim(0, fs/2)
        ax.set_title(f"α={α:.2f} (spec)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")

    # Row 2
    ax = fig.add_subplot(gs[2,:])
    ax.plot(alphas, d_l2,  label='L²')
    ax.plot(alphas, d_dtw, label='DTW')
    ax.plot(alphas, d_ost, label='OST')
    ax.set_xlabel("α")
    ax.set_ylabel("Norm Dist")
    ax.set_title("Chirp Alignment Distances")
    ax.legend()
    
    # Save, then show, then close
    filename = os.path.join(SAVE_DIR, "04_chirp_alignment.png")
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def experiment_composite_gap():
    fs, duration = 1000.0, 1.0
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    seg = int(0.2*fs)
    max_gap = duration - 3*(seg/fs)  # 0.4
    gaps = np.linspace(0, max_gap, 50)

    d_l2, d_dtw, d_ost = [], [], []
    note1_base = np.sin(2*np.pi*100*t[:seg])
    note2_base = np.sin(2*np.pi*150*t[:seg])
    note3_base = np.sin(2*np.pi*200*t[:seg])
    base = np.concatenate([note1_base, note2_base, note3_base, np.zeros(len(t)-3*seg)])

    for Δ in tqdm(gaps, desc="5. Composite Gap"):
        sil = int(Δ*fs)
        variant = np.concatenate([note1_base, note2_base, np.zeros(sil), note3_base,
                                  np.zeros(len(t)-3*seg-sil)])
        d_l2.append(l2_distance(base, variant, t))
        d_dtw.append(dtw_distance(base, variant))
        d_ost.append(ost_distance(base, variant, fs))
    d_l2  = np.array(d_l2)/np.max(d_l2)
    d_dtw = np.array(d_dtw)/np.max(d_dtw)
    d_ost = np.array(d_ost)/np.max(d_ost)

    # pick 3: no gap, mid gap, max gap
    disp_gap_vals = [0.0, max_gap/2, max_gap]
    sigs = []
    for Δ in disp_gap_vals:
        sil = int(Δ*fs)
        variant = np.concatenate([note1_base, note2_base, np.zeros(sil), note3_base,
                                  np.zeros(len(t)-3*seg-sil)])
        sigs.append((variant, Δ))

    fig = plt.figure(constrained_layout=True, dpi=DPI)
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # Row 0
    for i, (sig, Δ) in enumerate(sigs):
        ax = fig.add_subplot(gs[0,i])
        ax.plot(t, sig)
        ax.set_title(f"Δ={Δ:.2f}s")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amp")

    # Row 1: spectrograms (0–300 Hz)
    for i, (sig, Δ) in enumerate(sigs):
        ax = fig.add_subplot(gs[1,i])
        f_spec, t_spec, S = spectrogram(sig, fs=fs, nperseg=128, noverlap=64)
        ax.pcolormesh(t_spec, f_spec, S, shading='gouraud') # Restored to original
        ax.set_ylim(0, 300)
        ax.set_title(f"Δ={Δ:.2f}s (spec)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")

    # Row 2
    ax = fig.add_subplot(gs[2,:])
    ax.plot(gaps, d_l2,  label='L²')
    ax.plot(gaps, d_dtw, label='DTW')
    ax.plot(gaps, d_ost, label='OST')
    ax.set_xlabel("Gap Δ (s)")
    ax.set_ylabel("Norm Dist")
    ax.set_title("Composite Gap Distances")
    ax.legend()
    
    # Save, then show, then close
    filename = os.path.join(SAVE_DIR, "05_composite_tone_gap.png")
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def experiment_noise_robustness():
    fs, duration = 1000.0, 1.0
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    raw = np.exp(-0.5*((t-0.5)/0.1)**2)
    mod = np.sin(10*2*np.pi*t)
    base = raw*mod

    levels = np.linspace(0, 1.0, 50)
    d_l2, d_dtw, d_ost = [], [], []
    rng = np.random.default_rng(42)
    for nl in tqdm(levels, desc="6. Noise Robust"):
        noisy = base + nl*rng.standard_normal(len(t))
        d_l2.append(l2_distance(base, noisy, t))
        d_dtw.append(dtw_distance(base, noisy))
        d_ost.append(ost_distance(base, noisy, fs))
    d_l2  = np.array(d_l2)/np.max(d_l2)
    d_dtw = np.array(d_dtw)/np.max(d_dtw)
    d_ost = np.array(d_ost)/np.max(d_ost)

    # Re-generate noise for display examples for consistency
    rng_disp = np.random.default_rng(42)
    sigs = [
        (base, "Original"),
        (base + (levels[-1]/2)*rng_disp.standard_normal(len(t)), "Medium Noise"),
        (base + levels[-1]*rng_disp.standard_normal(len(t)), "High Noise")
    ]

    fig = plt.figure(constrained_layout=True, dpi=DPI)
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # Row 0
    for i, (sig, lbl) in enumerate(sigs):
        ax = fig.add_subplot(gs[0,i])
        ax.plot(t, sig)
        ax.set_title(lbl)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amp")

    # Row 1: spectrograms (0–300 Hz)
    for i, (sig, lbl) in enumerate(sigs):
        ax = fig.add_subplot(gs[1,i])
        f_spec, t_spec, S = spectrogram(sig, fs=fs, nperseg=128, noverlap=64)
        ax.pcolormesh(t_spec, f_spec, S, shading='gouraud') # Restored to original
        ax.set_ylim(0, 300)
        ax.set_title(lbl+" (spec)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")

    # Row 2
    ax = fig.add_subplot(gs[2,:])
    ax.plot(levels, d_l2,  label='L²')
    ax.plot(levels, d_dtw, label='DTW')
    ax.plot(levels, d_ost, label='OST')
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Norm Dist")
    ax.set_title("Noise Robustness")
    ax.legend()
    
    # Save, then show, then close
    filename = os.path.join(SAVE_DIR, "06_noise_robustness.png")
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def experiment_sinusoid_distance_gif():
    """
    GIF of sinusoid distance (10–40 Hz sweep):
      - Left 2/3: waveform (top, ref in red) + spectrogram (bottom, current image + ref red contours)
      - Right 1/3: bar chart with normalized L², DTW, OST in [0,1]
    Output: media/07_sinusoid_distance.gif
    """
    fs, duration = 1000.0, 1.0
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)

    # Reference and sweep
    f_ref = 10.0
    base = np.sin(2*np.pi*f_ref*t)
    freqs = np.linspace(10.0, 40.0, 60)  # 10 → 40 Hz

    # Spectrogram settings (higher-res, less blocky)
    nperseg, noverlap, nfft = 512, 448, 2048  # heavy overlap + zero-padding

    # Precompute signals, distances, spectrograms
    sigs, d_l2, d_dtw, d_ost, S_list = [], [], [], [], []
    for f in tqdm(freqs, desc="GIF: Sinusoid sweep (precompute)"):
        sig = np.sin(2*np.pi*f*t)
        sigs.append(sig)
        d_l2.append(l2_distance(base, sig, t))
        d_dtw.append(dtw_distance(base, sig))
        d_ost.append(ost_distance(base, sig, fs))
        f_spec, t_spec, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        S_list.append((f_spec, t_spec, Sxx))

    # Reference spectrogram (for contours)
    f_spec_ref, t_spec_ref, S_ref = spectrogram(base, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # Normalize distances once (fix bar axis across frames)
    d_l2  = np.array(d_l2);  d_l2  = d_l2  / (d_l2.max()  if d_l2.max()  > 0 else 1.0)
    d_dtw = np.array(d_dtw); d_dtw = d_dtw / (d_dtw.max() if d_dtw.max() > 0 else 1.0)
    d_ost = np.array(d_ost); d_ost = d_ost / (d_ost.max() if d_ost.max() > 0 else 1.0)

    # Fixed waveform y-limits across frames
    max_abs = max(np.max(np.abs(s)) for s in sigs + [base])
    ypad = 0.05 * max_abs
    ylo, yhi = -max_abs - ypad, max_abs + ypad

    # Fixed spectrogram frequency band & color scale
    fmax = 60.0
    vmax = max(Sxx[f_spec <= fmax, :].max() for (f_spec, _, Sxx) in S_list)
    vmin = 0.0
    fmask_ref = f_spec_ref <= fmax
    S_ref_band = S_ref[fmask_ref, :]
    ref_max = S_ref_band.max() if S_ref_band.size else 1.0
    ref_levels = [0.2*ref_max, 0.4*ref_max, 0.6*ref_max, 0.8*ref_max]
    extent = [t_spec_ref.min(), t_spec_ref.max(), 0.0, fmax]

    # Wider figure; make waveform row less tall
    fig = plt.figure(constrained_layout=True, dpi=DPI, figsize=(8, 3.5))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           width_ratios=[1, 1, 1],
                           height_ratios=[1, 3])  # waveform shorter than spectrogram

    # Left 2/3
    ax_wave = fig.add_subplot(gs[0, :2])
    ax_spec = fig.add_subplot(gs[1, :2])

    # Right 1/3 (bars)
    ax_bar = fig.add_subplot(gs[:, 2])

    # Waveform init: plot reference (red) and current (default color)
    ax_wave.plot(t, base, lw=1.25, color='red', label='Ref 10 Hz')
    cur_line, = ax_wave.plot(t, sigs[0], lw=1.25, label='Current f')
    ax_wave.set_xlim(0, duration)
    ax_wave.set_ylim(ylo, yhi)
    ax_wave.set_title("Sinusoid waveform (ref=10 Hz, sweep 10–40 Hz)")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.legend(loc='upper right', fontsize=7)

    # Spectrogram init: current as image, reference as red contours on top
    f_spec0, t_spec0, S0_full = S_list[0]
    fmask0 = f_spec0 <= fmax
    S0 = S0_full[fmask0, :]
    im = ax_spec.imshow(
        S0, origin="lower", aspect="auto", extent=extent,
        vmin=vmin, vmax=vmax, interpolation="bilinear"
    )
    ax_spec.contour(t_spec_ref, f_spec_ref[fmask_ref], S_ref_band,
                    levels=ref_levels, colors='red', linewidths=0.6)
    ax_spec.set_xlim(extent[0], extent[1])
    ax_spec.set_ylim(0.0, fmax)
    ax_spec.set_title("Spectrogram (current image + ref contours)")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")
    proxy = Line2D([0], [0], color='red', lw=1.0)
    ax_spec.legend([proxy], ['Ref spec (contours)'], loc='upper right', fontsize=7)

    # Bars init
    labels = ["L²", "DTW", "OST"]
    vals0 = [d_l2[0], d_dtw[0], d_ost[0]]
    bars = ax_bar.bar(labels, vals0)
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_title("Normalized distances")
    ax_bar.set_ylabel("Value")

    def update(i):
        # Waveform: update current
        cur_line.set_ydata(sigs[i])
        # Spectrogram: update current image
        f_spec_i, t_spec_i, S_full = S_list[i]
        fmask_i = f_spec_i <= fmax
        Si = S_full[fmask_i, :]
        im.set_data(Si)
        # Bars
        for b, v in zip(bars, (d_l2[i], d_dtw[i], d_ost[i])):
            b.set_height(v)
        return (cur_line, im, *bars)

    # 7 fps
    anim = FuncAnimation(fig, update, frames=len(sigs), interval=1000/7, blit=False)
    out_path = os.path.join(SAVE_DIR, "07_sinusoid_distance.gif")
    writer = PillowWriter(fps=7)  # no loop parameter
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved GIF to: {out_path}")
    

def experiment_ot_morph_mp4():
    """
    Optimal Transport morph animation (MP4):
      Left  : source distribution morphs over frames via displacement interpolation
              (using the optimal transport plan) toward the target.
      Right : fixed target distribution (2D Gaussian centered in the middle).

    Output: media/08_ot_morph.mp4
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import ot
    import os

    SAVE_DIR = "media"
    DPI = 150
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ----- Grid (use cell centers in [0,1]x[0,1]) -----
    G = 150
    xs = (np.arange(G) + 0.5) / G
    ys = (np.arange(G) + 0.5) / G
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    coords = np.column_stack([X.ravel(), Y.ravel()])   # (G^2, 2)

    # ----- Source: "ring + two blobs" -----
    cx, cy = 0.35, 0.50
    r0, sig_r = 0.28, 0.05
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    ring = np.exp(-0.5 * ((R - r0) / sig_r) ** 2)

    blob1 = np.exp(-0.5 * (((X - 0.75) / 0.08) ** 2 + ((Y - 0.25) / 0.06) ** 2))
    blob2 = np.exp(-0.5 * (((X - 0.78) / 0.07) ** 2 + ((Y - 0.75) / 0.07) ** 2))
    skew = 0.3 + 1.0 / (1.0 + np.exp((X - 0.45) / 0.04))

    A = (0.6 * ring + 0.8 * blob1 + 0.7 * blob2) * skew
    A = np.clip(A, 0.0, None)
    a = A.ravel()
    a = a / a.sum()

    # ----- Target: centered 2D Gaussian -----
    B = np.exp(-0.5 * (((X - 0.5) / 0.14) ** 2 + ((Y - 0.5) / 0.14) ** 2))
    B = np.clip(B, 0.0, None)
    b = B.ravel()
    b = b / b.sum()

    # ----- Optimal transport plan -----
    M = ot.dist(coords, coords, metric="euclidean") ** 2
    Gamma = ot.emd(a, b, M)

    I, J = np.where(Gamma > 0)
    m_ij = Gamma[I, J]
    xi = coords[I]
    yj = coords[J]

    A_grid = a.reshape(G, G)
    B_grid = b.reshape(G, G)
    vmin, vmax = 0.0, max(A_grid.max(), B_grid.max())

    # ----- Figure -----
    fig = plt.figure(constrained_layout=True, dpi=DPI, figsize=(6.8, 3.2))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    im_right = ax_right.imshow(
        B_grid,
        origin="lower",
        extent=[0, 1, 0, 1],
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
    )
    ax_right.set_title("Target distribution (2D Gaussian)")
    ax_right.set_xlabel("x")
    ax_right.set_ylabel("y")

    im_left = ax_left.imshow(
        A_grid,
        origin="lower",
        extent=[0, 1, 0, 1],
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
    )
    ttl = ax_left.set_title("OT displacement interpolation: t = 0.00")
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("y")

    # ----- Animation -----
    frames = 120

    def _positions_at(tau):
        tau = tau**2 * (2 - tau**2)  # easing
        return (1.0 - tau) * xi + tau * yj

    def _grid_index_from_pos(pos):
        ix = np.rint(pos[:, 0] * G - 0.5).astype(int)
        iy = np.rint(pos[:, 1] * G - 0.5).astype(int)
        np.clip(ix, 0, G - 1, out=ix)
        np.clip(iy, 0, G - 1, out=iy)
        return iy, ix

    def update(f):
        tau = f / (frames - 1)
        pos = _positions_at(tau)
        iy, ix = _grid_index_from_pos(pos)

        Z = np.zeros((G, G), dtype=float)
        np.add.at(Z, (iy, ix), m_ij)

        im_left.set_data(Z)
        ttl.set_text(f"OT displacement interpolation: t = {tau:0.2f}")
        return (im_left, ttl)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / 20, blit=False)

    out_path = os.path.join(SAVE_DIR, "08_ot_morph.mp4")
    writer = FFMpegWriter(fps=20, codec="h264", bitrate=1800)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved MP4 to: {out_path}")

def experiment_dtw_alignment_mp4():
    """
    Dynamic Time Warping (DTW) — signals + warping map (no cost matrix).
    Produces:
      • media/09_dtw_illustration.png (static, two panels)
      • media/09_dtw_alignment.mp4    (animated, H.264)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    # ---------- helpers ----------
    def _build_time_warp(t):
        """
        Smooth monotone piecewise-linear warp on t∈[0,1]:
        compress early part, stretch later part, average slope ~1.
        """
        t0, t1 = 0.45, 0.55
        s0, s1 = 0.75, 1.35
        sm = (1.0 - s0 * t0 - s1 * (1.0 - t1)) / (t1 - t0)
        c0 = 0.0
        c1 = c0 + s0 * t0
        c2 = c1 + sm * (t1 - t0)
        tau = np.where(
            t < t0,
            c0 + s0 * t,
            np.where(t < t1, c1 + sm * (t - t0), c2 + s1 * (t - t1)),
        )
        return np.clip(tau + 0.02, 0.0, 1.0)

    def _dtw_full(a, b):
        """
        Classic O(nm) DTW with |a_i - b_j| local cost.
        Returns D (local), C (cum), and optimal path as [(i,j), ...].
        """
        n, m = len(a), len(b)
        D = np.abs(a[:, None] - b[None, :])
        C = np.full((n + 1, m + 1), np.inf, dtype=float)
        C[0, 0] = 0.0
        ptr = np.zeros((n, m), dtype=np.uint8)  # 0=diag, 1=up, 2=left
        for i in range(1, n + 1):
            ci = i - 1
            row_prev = C[i - 1]
            row_cur = C[i]
            for j in range(1, m + 1):
                cj = j - 1
                choices = (row_prev[j - 1], row_prev[j], row_cur[j - 1])
                k = int(np.argmin(choices))
                C[i, j] = D[ci, cj] + choices[k]
                ptr[ci, cj] = k
        # Backtrack
        i, j = n, m
        path = []
        while i > 0 and j > 0:
            ci, cj = i - 1, j - 1
            path.append((ci, cj))
            k = ptr[ci, cj]
            if k == 0:
                i -= 1; j -= 1
            elif k == 1:
                i -= 1
            else:
                j -= 1
        while i > 0:
            i -= 1; path.append((i, 0))
        while j > 0:
            j -= 1; path.append((0, j))
        path.reverse()
        return D, C[1:, 1:], np.asarray(path, dtype=int)

    def _warping_map_from_path(path, n, m):
        """Average j for each i along the path; linearly interpolate gaps."""
        j_lists = [[] for _ in range(n)]
        for i, j in path:
            j_lists[i].append(j)
        j_of_i = np.full(n, np.nan, dtype=float)
        for i in range(n):
            if j_lists[i]:
                j_of_i[i] = np.mean(j_lists[i])
        if np.isnan(j_of_i).any():
            idx = np.arange(n)
            good = ~np.isnan(j_of_i)
            j_of_i = np.interp(idx, idx[good], j_of_i[good])
        return j_of_i

    # ---------- signals ----------
    fs = 400.0
    duration = 1.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tn = (t - t.min()) / (t.max() - t.min())

    # Reference: Gaussian envelope × slow-FM sinusoid
    env = np.exp(-0.5 * ((tn - 0.5) / 0.18) ** 2)
    f0, f1 = 6.0, 12.0
    inst_f = f0 + (f1 - f0) * (0.5 * (1 + np.sin(2 * np.pi * (tn - 0.25))))
    phase = 2 * np.pi * np.cumsum(inst_f) / fs
    a = env * np.sin(phase)

    # Warped by resampling reference at warped times
    tau = _build_time_warp(tn)
    idx = np.linspace(0, len(tn) - 1, len(tn))
    b = np.interp(tau * (len(tn) - 1), idx, a)

    # Downsample for DTW matrix size (display still uses full-res curves)
    reduce = 2  # 1: none; 2: half; 4: quarter, etc.
    ai, bi, ti = a[::reduce], b[::reduce], t[::reduce]

    # ---------- DTW & warping map ----------
    _, _, path = _dtw_full(ai, bi)
    n, m = len(ai), len(bi)
    j_of_i = _warping_map_from_path(path, n, m)

    # ---------- Static figure (signals + warping map) ----------
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig = plt.figure(constrained_layout=True, dpi=DPI, figsize=(8.6, 4.6))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.15, 1.0])

    ax_sig = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[1, 0])

    ymax = max(np.max(np.abs(a)), np.max(np.abs(b)))
    yshift = 1.4 * ymax
    ax_sig.plot(t, a, lw=1.2, label="Reference")
    ax_sig.plot(t, b + yshift, lw=1.2, label="Warped (offset)")
    ax_sig.set_title("DTW Illustration: signals & warping map")
    ax_sig.set_xlabel("Time (s)")
    ax_sig.set_ylabel("Amplitude")
    ax_sig.legend(loc="upper right", fontsize=8)

    # Light subset of correspondence lines for context
    step_conn = max(1, len(path) // 120)
    for (ii, jj) in path[::step_conn]:
        ax_sig.plot([ti[ii], ti[jj]], [ai[ii], bi[jj] + yshift], lw=0.5, alpha=0.35)

    ax_map.plot(np.arange(n), j_of_i, lw=1.4, label="j(i) from DTW")
    ax_map.plot([0, n - 1], [0, m - 1], ls="--", lw=1.0, label="identity")
    ax_map.set_xlim(0, n - 1)
    ax_map.set_ylim(0, m - 1)
    ax_map.set_xlabel("i (reference index)")
    ax_map.set_ylabel("j (warped index)")
    ax_map.set_title("Warping function")
    ax_map.legend(loc="upper left", fontsize=8)

    out_png = os.path.join(SAVE_DIR, "09_dtw_illustration.png")
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # ---------- MP4 animation (signals + growing j(i)) ----------
    fig2 = plt.figure(constrained_layout=True, dpi=DPI, figsize=(8.6, 4.6))
    gs2 = gridspec.GridSpec(2, 1, figure=fig2, height_ratios=[1.15, 1.0])
    ax_sig2 = fig2.add_subplot(gs2[0, 0])
    ax_map2 = fig2.add_subplot(gs2[1, 0])

    ax_sig2.plot(t, a, lw=1.0)
    ax_sig2.plot(t, b + yshift, lw=1.0)
    ax_sig2.set_title("DTW alignment (animated)")
    ax_sig2.set_xlabel("Time (s)")
    ax_sig2.set_ylabel("Amplitude")

    # moving connector between matched samples (uses downsampled times)
    conn_line, = ax_sig2.plot([], [], lw=1.8)

    ax_map2.plot([0, n - 1], [0, m - 1], ls="--", lw=1.0)
    ax_map2.set_xlim(0, n - 1)
    ax_map2.set_ylim(0, m - 1)
    ax_map2.set_xlabel("i (reference index)")
    ax_map2.set_ylabel("j (warped index)")
    ax_map2.set_title("Warping map prefix")
    line_map, = ax_map2.plot([], [], lw=1.6)

    # Thin the path for frame count
    step_anim = max(1, len(path) // 300)
    path_anim = path[::step_anim]
    i_prefix = []
    j_prefix = []

    def update(k):
        ii, jj = path_anim[k]
        # update connector on signals
        conn_line.set_data([ti[ii], ti[jj]], [ai[ii], bi[jj] + yshift])
        # update warping map prefix
        i_prefix.append(ii)
        j_prefix.append(jj)
        line_map.set_data(i_prefix, j_prefix)
        return (conn_line, line_map)

    anim = FuncAnimation(fig2, update, frames=len(path_anim), interval=1000 / 12, blit=False)
    out_mp4 = os.path.join(SAVE_DIR, "09_dtw_alignment.mp4")
    writer = FFMpegWriter(fps=12, codec="h264", bitrate=2000)
    anim.save(out_mp4, writer=writer)
    plt.close(fig2)

    print(f"Saved: {out_png} and {out_mp4}")

def experiment_spectrogram_showcase():
    """
    Single PNG:
      • Top: composite waveform
      • Bottom: single spectrogram titled "Spectrogram"
    Labels are white; figure is wide. X-axes align exactly (no left/right padding).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.signal import spectrogram
    import os

    fs, duration = 1000.0, 4.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # ---- Build composite signal ----
    x = np.zeros_like(t)

    # 1) Steady 60 Hz tone (0.0–1.2 s)
    m1 = (t >= 0.0) & (t < 1.2)
    x[m1] += 0.9 * np.sin(2 * np.pi * 60 * t[m1])

    # 2) Linear chirp 40→220 Hz (1.0–2.6 s), amplitude ×2
    m2 = (t >= 1.0) & (t < 2.6)
    w2 = np.hanning(m2.sum()) if m2.sum() > 4 else np.ones(m2.sum())
    tt2 = t[m2]
    f_start, f_end = 40.0, 220.0
    f_inst = f_start + (f_end - f_start) * (tt2 - tt2[0]) / (tt2[-1] - tt2[0])
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    x[m2] += 1.6 * np.sin(phase) * w2

    # 3) Silence (2.6–3.2 s)

    # 4) Two close tones 120 & 140 Hz (3.2–4.0 s), amplitude ×2
    m4 = (t >= 3.2) & (t <= 4.0)
    w4 = np.hanning(m4.sum()) if m4.sum() > 4 else np.ones(m4.sum())
    x[m4] += 1.5 * 0.5 * (
        np.sin(2 * np.pi * 120 * t[m4]) + np.sin(2 * np.pi * 140 * t[m4])
    ) * w4

    # ---- Spectrogram (long window) ----
    f_l, ts_l, S_l = spectrogram(x, fs=fs, nperseg=256, noverlap=255)
    fmax = 300.0
    mask = f_l <= fmax
    Sv = S_l[mask, :]
    nf, nt = Sv.shape

    # Use explicit bin EDGES so the image spans exactly [0, duration] × [0, fmax]
    t_edges = np.linspace(0.0, duration, nt + 1)
    f_top = f_l[mask][-1]
    f_edges = np.linspace(0.0, min(fmax, f_top), nf + 1)

    vmin, vmax = 0.0, Sv.max() if Sv.size else 1.0

    # ---- Figure (wider), perfectly aligned x-lims ----
    fig = plt.figure(constrained_layout=True, dpi=DPI, figsize=(12.5, 5.2))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.1, 1.0])

    # Top: waveform
    ax_w = fig.add_subplot(gs[0, 0])
    ax_w.plot(t, x, lw=1.0)
    ax_w.set_title("Composite signal: steady tone → chirp → silence → two close tones")
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("Amplitude")
    ax_w.grid(True, alpha=0.3)
    ax_w.set_xlim(0.0, duration)     # no extra padding
    ax_w.margins(x=0)

    # Bottom: spectrogram (edges ensure exact span)
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_spec.pcolormesh(t_edges, f_edges, Sv, shading="auto", vmin=vmin, vmax=vmax)
    ax_spec.set_ylim(0, f_edges[-1])
    ax_spec.set_xlim(0.0, duration)  # align with waveform
    ax_spec.margins(x=0)
    ax_spec.set_title("Spectrogram")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")

    # White labels (slightly larger), positions as before
    ax_spec.text(0.45, 75,  "steady tone ~60 Hz",            fontsize=11, color="white")
    ax_spec.text(1.50, 190, "linear chirp 40→220 Hz",        fontsize=11, color="white")
    ax_spec.text(3.55, 155, "two close tones\n120 & 140 Hz", fontsize=11, color="white", ha="center")
    ax_spec.text(2.85, 155, "(silence)", fontsize=11, color="white", ha="center")


    # Save & show
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, "10_spectrogram_showcase.png")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved spectrogram showcase to: {out_path}")

def experiment_ot_spectrogram_morph_mp4():
    """
    Optimal Transport morph between two spectrograms (MP4).
    Layout:
      • Top   : morphing spectrogram (source → target via OT)
      • Bottom: static target spectrogram

    Target modifications vs source:
      1) Steady tone at higher frequency (e.g., 90 Hz instead of 60 Hz)
      2) Chirp reversed: 220 → 40 Hz
      3) Two tones moved earlier (same length), farther apart (e.g., 100 & 180 Hz)

    Output: media/10_ot_spectrogram_morph.mp4
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from scipy.signal import spectrogram
    import ot

    # ---------- common plotting / signal settings ----------
    fs, duration = 1000.0, 4.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    fmax = 300.0

    # Spectrogram params (match your showcase)
    nperseg, noverlap = 256, 224

    # OT grid resolution (memory saver)
    G = 170

    # ---------- helper: build "showcase-style" source signal ----------
    def build_source_signal(t):
        x = np.zeros_like(t)

        # steady ~60 Hz (0.0–1.2 s)
        m1 = (t >= 0.0) & (t < 1.2)
        x[m1] += 0.9 * np.sin(2 * np.pi * 60 * t[m1])

        # short click at 0.8 s
        x += 0.5 * np.exp(-0.5 * ((t - 0.8) / 0.01) ** 2)

        # chirp 40 → 220 Hz (1.0–2.6 s), amplitude ×2 (1.6 vs 0.8)
        m2 = (t >= 1.0) & (t < 2.6)
        w2 = np.hanning(m2.sum()) if m2.sum() > 4 else np.ones(m2.sum())
        tt2 = t[m2]
        f0, f1 = 40.0, 220.0
        f_inst = f0 + (f1 - f0) * (tt2 - tt2[0]) / (tt2[-1] - tt2[0])
        phase = 2 * np.pi * np.cumsum(f_inst) / fs
        x[m2] += 1.6 * np.sin(phase) * w2

        # two close tones 120 & 140 Hz (3.2–4.0 s), amplitude ×2
        m4 = (t >= 3.2) & (t <= 4.0)
        w4 = np.hanning(m4.sum()) if m4.sum() > 4 else np.ones(m4.sum())
        x[m4] += 1.5 * 0.5 * (
            np.sin(2 * np.pi * 120 * t[m4]) + np.sin(2 * np.pi * 140 * t[m4])
        ) * w4

        return x

    # ---------- helper: build modified target signal ----------
    def build_target_signal(t):
        x = np.zeros_like(t)

        # (1) steady tone at higher freq: ~90 Hz (0.0–1.2 s)
        m1 = (t >= 0.0) & (t < 1.2)
        x[m1] += 0.9 * np.sin(2 * np.pi * 90 * t[m1])

        # same click to keep a shared feature (helps the OT)
        x += 0.5 * np.exp(-0.5 * ((t - 0.8) / 0.01) ** 2)

        # (2) reversed chirp 220 → 40 Hz (1.0–2.6 s), amplitude ×2
        m2 = (t >= 1.0) & (t < 2.6)
        w2 = np.hanning(m2.sum()) if m2.sum() > 4 else np.ones(m2.sum())
        tt2 = t[m2]
        f0, f1 = 220.0, 40.0
        f_inst = f0 + (f1 - f0) * (tt2 - tt2[0]) / (tt2[-1] - tt2[0])
        phase = 2 * np.pi * np.cumsum(f_inst) / fs
        x[m2] += 1.6 * np.sin(phase) * w2

        # (3) two tones earlier (same length 0.8 s), farther apart in freq: 100 & 180 Hz (2.9–3.7 s)
        start, end = 2.9, 3.7
        m4 = (t >= start) & (t <= end)
        w4 = np.hanning(m4.sum()) if m4.sum() > 4 else np.ones(m4.sum())
        x[m4] += 1.5 * 0.5 * (
            np.sin(2 * np.pi * 100 * t[m4]) + np.sin(2 * np.pi * 180 * t[m4])
        ) * w4

        return x

    # ---------- helper: spectrogram → masked array + edges ----------
    def spec_masked(x):
        f, ts, S = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mask = f <= fmax
        Sv = S[mask, :]
        nf, nt = Sv.shape
        # exact edges so plot spans [0, duration] × [0, fmax']
        t_edges = np.linspace(0.0, duration, nt + 1)
        f_top = f[mask][-1]
        f_edges = np.linspace(0.0, min(fmax, f_top), nf + 1)
        return f, ts, Sv, f_edges, t_edges

    # ---------- helper: reduce to G×G with simple bin-averaging ----------
    def reduce_to_grid(S, Gf, Gt):
        nf, nt = S.shape
        f_edges = np.linspace(0, nf, Gf + 1).astype(int)
        t_edges = np.linspace(0, nt, Gt + 1).astype(int)
        R = np.zeros((Gf, Gt), dtype=float)
        for i in range(Gf):
            f0, f1 = f_edges[i], max(f_edges[i + 1], f_edges[i] + 1)
            for j in range(Gt):
                t0, t1 = t_edges[j], max(t_edges[j + 1], t_edges[j] + 1)
                block = S[f0:f1, t0:t1]
                R[i, j] = block.mean() if block.size else 0.0
        return R

    # ---------- build signals & spectrograms ----------
    x_src = build_source_signal(t)
    x_tgt = build_target_signal(t)

    _, _, S_src, f_edges_src, t_edges_src = spec_masked(x_src)
    _, _, S_tgt, f_edges_tgt, t_edges_tgt = spec_masked(x_tgt)

    # common color scale
    vmin, vmax = 0.0, max(S_src.max(), S_tgt.max())

    # ---------- OT setup on downsampled grids ----------
    A = reduce_to_grid(S_src, G, G)  # (G×G)
    B = reduce_to_grid(S_tgt, G, G)

    # keep intensity scales for display; normalize copies for OT mass
    a = A.ravel().copy()
    b = B.ravel().copy()
    a_mass = a / (a.sum() if a.sum() > 0 else 1.0)
    b_mass = b / (b.sum() if b.sum() > 0 else 1.0)

    # coords on unit square (cell centers)
    xs = (np.arange(G) + 0.5) / G
    ys = (np.arange(G) + 0.5) / G
    Xc, Yc = np.meshgrid(xs, ys, indexing="xy")  # (G,G)
    coords = np.column_stack([Xc.ravel(), Yc.ravel()])

    # EMD plan with squared Euclidean ground cost
    M = ot.dist(coords, coords, metric="euclidean") ** 2
    Gamma = ot.emd(a_mass, b_mass, M)

    # sparse representation
    I, J = np.where(Gamma > 0)
    m_ij = Gamma[I, J]
    xi = coords[I]
    yj = coords[J]

    # per-frame intensity scaling (so the morph doesn't look "washed out")
    alpha_src = (A.max() / a_mass.max()) if a_mass.max() > 0 else 1.0
    alpha_tgt = (B.max() / b_mass.max()) if b_mass.max() > 0 else 1.0

    # ---------- figure & animation ----------
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig = plt.figure(constrained_layout=True, dpi=DPI, figsize=(12.5, 5.6))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0])

    # Top (morphing) — use imshow with exact extent to align axes
    ax_top = fig.add_subplot(gs[0, 0])
    Z0 = A  # start from source (G×G)
    im_top = ax_top.imshow(
        Z0,
        origin="lower",
        aspect="auto",
        extent=[0.0, duration, 0.0, f_edges_src[-1]],
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax_top.set_xlim(0.0, duration)
    ax_top.set_ylim(0.0, f_edges_src[-1])
    ax_top.margins(x=0)
    ttl = ax_top.set_title("OT morph between spectrograms: t = 0.00")
    ax_top.set_xlabel("Time (s)")
    ax_top.set_ylabel("Frequency (Hz)")

    # Bottom (static target) — high-res pcolormesh with edges
    ax_bot = fig.add_subplot(gs[1, 0])
    ax_bot.pcolormesh(t_edges_tgt, f_edges_tgt, S_tgt, shading="auto", vmin=vmin, vmax=vmax)
    ax_bot.set_xlim(0.0, duration)
    ax_bot.set_ylim(0.0, f_edges_tgt[-1])
    ax_bot.margins(x=0)
    ax_bot.set_title("Target spectrogram")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Frequency (Hz)")

    frames = 120

    def _easing(tau):
        # smooth ease-in/out; stays in [0,1]
        return tau * tau * (3 - 2 * tau)

    def _grid_index_from_pos(pos):
        # pos in unit square; map to grid indices 0..G-1
        ix = np.rint(pos[:, 0] * G - 0.5).astype(int)
        iy = np.rint(pos[:, 1] * G - 0.5).astype(int)
        np.clip(ix, 0, G - 1, out=ix)
        np.clip(iy, 0, G - 1, out=iy)
        return iy, ix  # row, col

    def update(fnum):
        tau = _easing(fnum / (frames - 1))
        pos = (1.0 - tau) * xi + tau * yj
        iy, ix = _grid_index_from_pos(pos)

        Zm = np.zeros((G, G), dtype=float)
        np.add.at(Zm, (iy, ix), m_ij)  # mass histogram on the grid

        # per-frame intensity scaling (blend source/target scale)
        alpha = (1.0 - tau) * alpha_src + tau * alpha_tgt
        Z_disp = Zm * alpha

        im_top.set_data(Z_disp)
        ttl.set_text(f"OT morph between spectrograms: t = {tau:0.2f}")
        return (im_top, ttl)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / 20, blit=False)

    out_path = os.path.join(SAVE_DIR, "10_ot_spectrogram_morph.mp4")
    writer = FFMpegWriter(fps=20, codec="h264", bitrate=1800)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved MP4 to: {out_path}")

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Running experiments. Figures will be saved to '{SAVE_DIR}/' and displayed.")

    experiment_ot_spectrogram_morph_mp4()
    #experiment_spectrogram_showcase()
    #experiment_dtw_alignment_mp4()
    #experiment_ot_morph_mp4()
    #experiment_ot_morph_gif()
    #experiment_sinusoid_distance_gif()
    #experiment_sinusoids()
    #experiment_gaussian_speed()
    #experiment_gaussian_shift()
    #experiment_chirp_alignment()
    #experiment_composite_gap()
    #experiment_noise_robustness()

    print("\nAll experiments complete.")