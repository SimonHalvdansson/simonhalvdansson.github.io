# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import imageio
import io
from PIL import Image
from tqdm import tqdm

# --- Helper Functions ------------------------------------------------------
def save_audio(name, audio, sample_rate):
    # Ensure output directory exists
    os.makedirs('media', exist_ok=True)
    # audio: Tensor of shape [samples] or [channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    # Normalize to max amplitude of 0.9
    audio = audio / audio.abs().max() * 0.9
    # Save as MP3
    path = f"media/{audio_name}_{name}.mp3"
    torchaudio.save(path, audio.cpu(), sample_rate=sample_rate, format='mp3')
    print(f"Saved MP3: {path}")

# --- Configuration ---------------------------------------------------------
# Audio file and method toggles
audio_name =        "jazz"
audio_filetype =    ".mp3"
audio_file =        audio_name + audio_filetype    # Path to input audio
seconds_to_include = 1
freq_bins_max_plot = 40
include_siren    = True            # Toggle SIREN model
include_griffin  = True            # Toggle Griffin-Lim reconstruction
include_phase    = True             # Toggle phase gradient descent

# GIF generation toggles
generate_spectrogram_gif = False  # Toggle spectrogram grid GIF

# Model/training settings
HIDDEN_DIM      = 256      # Hidden dimension for MLP/SIREN
NUM_LAYERS      = 3        # Number of hidden layers
NUM_EPOCHS      = 2000
LEARNING_RATE   = 1e-4
SNAPSHOT_FREQ   = 20       # Epochs between GIF snapshots

NUM_EPOCHS_PHASE = 20000
LEARNING_RATE_PHASE = 5e-4

# Spectrogram/Griffin-Lim configuration
SPECTROGRAM_CONFIG = {
    "n_fft": 512,
    "win_length": 512,
    "hop_length": 256,
    "power": 1,
    "window_fn": torch.hann_window
}

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

# Prepare spectrogram transform
spec_transform = torchaudio.transforms.Spectrogram(**SPECTROGRAM_CONFIG).to(device)

# Color map for plotting
LOSS_COLORS = {
    "siren":   "red",
    "phase":   "blue"
}

# --- SIREN (Sitzmann et al.) -----------------------------------------------
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/in_features, 1/in_features)
            else:
                bound = np.sqrt(6/in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, first_omega_0=3000, hidden_omega_0=30.):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features,
                    is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                        is_first=False, omega_0=hidden_omega_0))
        
        lin = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6/hidden_features) / hidden_omega_0
            lin.weight.uniform_(-bound, bound)
        layers.append(lin)
       
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- Loss ------------------------------------------------------------------
class SpectrogramPhaseRetrievalLoss(nn.Module):
    def __init__(self, spec_transform):
        super().__init__()
        self.spec = spec_transform

    def forward(self, pred, target):
        sp_pred   = self.spec(pred.squeeze(1))
        sp_target = self.spec(target.squeeze(1))
        return torch.mean(torch.abs(sp_pred - sp_target))

# --- Prepare Data ---------------------------------------------------------
ys, sr = torchaudio.load(audio_file)

audio_len = sr*seconds_to_include
ys = ys[0][:audio_len]
x = torch.linspace(0, 1, audio_len).unsqueeze(1)
x_t = x.to(device)
y_t = ys.unsqueeze(1).to(device)

# Save ground truth first-second audio for comparison
save_audio('ground_truth', ys, sr)

# --- Instantiate networks ------------------------------------------------
methods = {}
if include_siren:
    siren_net = Siren(1, HIDDEN_DIM, NUM_LAYERS, 1,
                      first_omega_0=3000,
                      hidden_omega_0=30.).to(device)
    siren_opt = optim.Adam(siren_net.parameters(), lr=LEARNING_RATE)
    methods['siren'] = {'model': siren_net,
                       'opt': siren_opt,
                       'loss_curve': []}

# --- Training & GIF snapshots ---------------------------------------------
criterion = SpectrogramPhaseRetrievalLoss(spec_transform).to(device)
spec_frames = []
pbar = tqdm(range(1, NUM_EPOCHS+1), desc="Training", ncols=85)
for ep in pbar:
    # train each network
    for name, m in methods.items():
        m['opt'].zero_grad()
        pred = m['model'](x_t)
        loss = criterion(pred, y_t)
        loss.backward()
        m['opt'].step()
        m['loss_curve'].append(loss.item())
    pbar.set_postfix({n: f"{m['loss_curve'][-1]:.6f}" for n,m in methods.items()})
    # snapshots
    if ep % SNAPSHOT_FREQ == 0:
        if generate_spectrogram_gif:
            fig, axs = plt.subplots(1, len(methods)+1, figsize=(8,4), dpi=100)
            target_sp = spec_transform(y_t.squeeze(1)).abs().cpu().numpy()
            axs[0].imshow(target_sp[:30, :], aspect='auto', origin='lower')
            axs[0].set_title('Target')
            for i, (n, m) in enumerate(methods.items()):
                with torch.no_grad():
                    sp = spec_transform(m['model'](x_t).squeeze(1)).abs().cpu().numpy()[:30, :]
                axs[i+1].imshow(sp, aspect='auto', origin='lower')
                axs[i+1].set_title(n)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            spec_frames.append(np.array(Image.open(buf)))
            plt.close()
            
if generate_spectrogram_gif and spec_frames:
    os.makedirs('media', exist_ok=True)
    gif_path = f'media/{audio_name}_spectrogram_progress.gif'
    imageio.mimsave(gif_path, spec_frames, fps=10, loop=0)   # or duration=0.5, etc.
    print(f"Saved spectrogram GIF: {gif_path}")

# --- Save reconstructions as MP3 -----------------------------------------
for name, m in methods.items():
    with torch.no_grad():
        audio = m['model'](x_t).cpu().squeeze(1)
    save_audio(name, audio, sr)

# --- Griffin-Lim ----------------------------------------------------------
if include_griffin:
    mag_spec = spec_transform(y_t.squeeze(1)).abs().to(device)
    griffin = torchaudio.transforms.GriffinLim(
        n_fft=SPECTROGRAM_CONFIG["n_fft"],
        win_length=SPECTROGRAM_CONFIG["win_length"],
        hop_length=SPECTROGRAM_CONFIG["hop_length"],
        window_fn=SPECTROGRAM_CONFIG["window_fn"],
        power=1,
    ).to(device)
    with torch.no_grad():
        recon = griffin(mag_spec).cpu()
    recon = recon.unsqueeze(0)
    save_audio('griffin_lim', recon, sr)
    
    # --- Plot Ground Truth & Griffin-Lim Spectrograms + Print Griffin-Lim Loss ---
    with torch.no_grad():
        # Move recon onto the same device as spec_transform
        recon_cuda = recon.to(device)
    
        # Ground truth magnitude spectrogram (already on device)
        gt_spec = spec_transform(y_t.squeeze(1)).abs().cpu().numpy()
    
        # Griffin–Lim spectrogram (now on device)
        gl_spec = spec_transform(recon_cuda.squeeze(0)).abs().cpu().numpy()
    
    # Compute Griffin–Lim spectral loss (ensure recon is on device and has right shape)
    gl_loss = SpectrogramPhaseRetrievalLoss(spec_transform).to(device)(
        recon_cuda.unsqueeze(1), y_t
    ).item()
    print(f"Griffin–Lim Final Loss: {gl_loss:.6f}")
    
    # Plot only the first 40 frequency bins for each
    plt.figure(figsize=(8, 2), dpi=150)
    plt.imshow(gt_spec[:freq_bins_max_plot, :], aspect='auto', origin='lower')
    plt.title('Ground Truth Spectrogram (MSE = 0)')
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"media/{audio_name}_ground_truth_spec.png")
    plt.show()
    
    plt.figure(figsize=(8, 2), dpi=150)
    plt.imshow(gl_spec[:freq_bins_max_plot, :], aspect='auto', origin='lower')
    plt.title(f'Griffin–Lim Spectrogram (MSE = {gl_loss:.5f})')
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"media/{audio_name}_griffin_lim_spec.png")
    plt.show()

# --- Plot loss curve and final spec ------------------------------------------
if include_siren:
    plt.figure(figsize=(8,6), dpi=150)
    for n, m in methods.items():
        plt.plot(np.arange(1, len(m['loss_curve'])+1), m['loss_curve'],
                 label=n, color=LOSS_COLORS.get(n, None))
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (SIREN)')
    plt.tight_layout()
    plt.savefig("media/siren_loss.png")
    plt.show()
    
    with torch.no_grad():
        # Get the trained SIREN output (shape [1, samples])
        siren_out = siren_net(x_t).squeeze(1)
        # Compute its magnitude spectrogram
        siren_spec = spec_transform(siren_out.unsqueeze(0).squeeze(0)).abs().cpu().numpy()

    plt.figure(figsize=(8, 2), dpi=150)
    plt.imshow(siren_spec[:freq_bins_max_plot, :], aspect='auto', origin='lower')
    
    final_siren_loss = methods["siren"]["loss_curve"][-1]
    
    plt.title(f'SIREN Reconstructed Spectrogram (MSE = {final_siren_loss:.5f})')
    plt.xticks([])   # drop x‐ticks
    plt.yticks([])   # drop y‐ticks
    plt.tight_layout()
    plt.savefig(f"media/{audio_name}_siren_mag.png")
    plt.show()

# --- Phase Gradient Descent on Full STFT --------------------------------
if include_phase:
    # prepare window for ISTFT
    win = SPECTROGRAM_CONFIG["window_fn"](SPECTROGRAM_CONFIG["n_fft"]).to(device)

    # compute target magnitude spectrogram (no grad)
    mag = spec_transform(y_t.squeeze(1)).abs().detach()

    # initialize learnable real & imaginary parts of the STFT
    shape = mag.shape
    
    # random phase in [0, 2π)
    phi = torch.rand(shape, device=device) * 2 * np.pi
    
    # build real & imaginary parts with correct magnitudes
    real_init = mag * torch.cos(phi)
    imag_init = mag * torch.sin(phi)
    
    # make them leaf tensors requiring gradients
    real_part = real_init.clone().detach().requires_grad_(True)
    imag_part = imag_init.clone().detach().requires_grad_(True)
    
    optimizer_complex = optim.Adam([real_part, imag_part], lr=LEARNING_RATE_PHASE)
    complex_loss_curve = []

    # gradient‐descent loop on full complex spectrogram
    pbar = tqdm(range(1, NUM_EPOCHS_PHASE+1),
                desc="Complex Spec GD",
                ncols=90)
    for ep in pbar:
        optimizer_complex.zero_grad()

        # assemble complex spectrogram
        complex_spec = torch.complex(real_part, imag_part)

        # inverse STFT back to time‐domain waveform
        audio_pred = torch.istft(
            complex_spec,
            n_fft=SPECTROGRAM_CONFIG["n_fft"],
            hop_length=SPECTROGRAM_CONFIG["hop_length"],
            win_length=SPECTROGRAM_CONFIG["win_length"],
            window=win,
            length=audio_len
        )

        # compute loss against target spectrogram
        loss = criterion(audio_pred.unsqueeze(1), y_t)
        loss.backward()
        optimizer_complex.step()

        complex_loss_curve.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    # save final reconstruction
    with torch.no_grad():
        final_spec = torch.complex(real_part, imag_part)
        recon = torch.istft(
            final_spec,
            n_fft=SPECTROGRAM_CONFIG["n_fft"],
            hop_length=SPECTROGRAM_CONFIG["hop_length"],
            win_length=SPECTROGRAM_CONFIG["win_length"],
            window=win,
            length=audio_len
        )
        recon = recon.unsqueeze(0)
        save_audio('complex_spec', recon, sr)

    # plot complex‐spec GD loss curve
    plt.figure(figsize=(8,6), dpi=150)
    plt.plot(np.arange(1, len(complex_loss_curve)+1), complex_loss_curve,
             label='complex_spec', color=LOSS_COLORS.get('phase', 'blue'))
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Complex Spectrogram GD Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"media/{audio_name}spec_gd_loss.png")
    plt.show()

    final_mag = final_spec.abs().cpu().numpy()
    plt.figure(figsize=(8, 2), dpi=150)
    plt.imshow(final_mag[:freq_bins_max_plot, :], aspect='auto', origin='lower')
    plt.title(f'Complex Gradient Descent Spectrogram (MSE={complex_loss_curve[-1]:.5f})')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"media/{audio_name}_spec_gd_mag.png")
    plt.show()
