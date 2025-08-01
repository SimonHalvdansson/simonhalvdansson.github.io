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
import os # Added for directory and file path operations

# --- Configuration ---
# High resolution and smaller fonts
plt.rcParams.update({'font.size': 8})
DPI = 300
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

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Running experiments. Figures will be saved to '{SAVE_DIR}/' and displayed.")

    experiment_sinusoids()
    experiment_gaussian_speed()
    experiment_gaussian_shift()
    experiment_chirp_alignment()
    experiment_composite_gap()
    experiment_noise_robustness()

    print("\nAll experiments complete.")