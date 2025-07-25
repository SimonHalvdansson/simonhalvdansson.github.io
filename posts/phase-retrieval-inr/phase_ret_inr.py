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

# --- Configuration ---------------------------------------------------------
HIDDEN_DIM      = 64      # MLP hidden size
NUM_LAYERS      = 2       # Number of MLP layers
NUM_FEATURES    = 24      # Number of Fourier/Gabor features
NUM_EPOCHS      = 400
LEARNING_RATE   = 5e-3
SNAPSHOT_FREQ   = 20

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

# Color map for loss curves and waveform overlays
LOSS_COLORS = {
	"naive":          "blue",
	"fixed_fourier":  "red",
	"fixed_gabor":    "green",
	"opt_fourier":    "orange",
	"opt_gabor":      "purple",
}

# --- Loss ---------------------------------------------------------------
class SpectrogramPhaseRetrievalLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.spec = torchaudio.transforms.Spectrogram(power=1).to(device)

	def forward(self, pred, target):
		# compute spectrograms and measure L1 difference
		sp_pred = self.spec(pred.squeeze(1))
		sp_target = self.spec(target.squeeze(1))
		return torch.sum(torch.abs(sp_pred - sp_target))

# --- Models -------------------------------------------------------------
class MLP(nn.Module):
	def __init__(self, in_dim, hid_dim, num_layers, out_dim, activation=nn.ReLU(), final_act=None):
		super().__init__()
		layers = []
		cur_dim = in_dim
		for _ in range(num_layers):
			layers.append(nn.Linear(cur_dim, hid_dim))
			layers.append(activation)
			cur_dim = hid_dim
		layers.append(nn.Linear(cur_dim, out_dim))
		self.net = nn.Sequential(*layers)
		self.final_act = final_act

	def forward(self, x):
		out = self.net(x)
		return self.final_act(out) if self.final_act is not None else out

class FourierFeatureMapping(nn.Module):
	def __init__(self, in_dim, n_feats, sigma=20.0, trainable=False):
		super().__init__()
		freqs = torch.randn(in_dim, n_feats, device=device) * sigma
		if trainable:
			self.freqs = nn.Parameter(freqs)
		else:
			self.register_buffer('freqs', freqs)

	def forward(self, x):
		proj = 2 * np.pi * (x @ self.freqs)
		return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

class GaborFeatureMapping(nn.Module):
	def __init__(self, in_dim, n_feats, sigma=0.3, sigma_freq=20.0, trainable=False):
		super().__init__()
		centers     = torch.rand(n_feats, in_dim, device=device)
		frequencies = torch.randn(n_feats, in_dim, device=device) * sigma_freq
		sigmas       = torch.full((n_feats, in_dim), sigma, device=device)
		if trainable:
			self.centers     = nn.Parameter(centers)
			self.frequencies = nn.Parameter(frequencies)
			self.sigma       = nn.Parameter(sigmas)
		else:
			self.register_buffer('centers', centers)
			self.register_buffer('frequencies', frequencies)
			self.register_buffer('sigma', sigmas)

	def forward(self, x):
		diff     = x.unsqueeze(1) - self.centers.unsqueeze(0)
		env      = torch.exp(-0.5 * (diff**2 / self.sigma.unsqueeze(0)**2).sum(dim=2))
		phase    = 2 * np.pi * (diff * self.frequencies.unsqueeze(0)).sum(dim=2)
		return torch.cat([env * torch.cos(phase), env * torch.sin(phase)], dim=-1)

# --- Prepare Data ------------------------------------------------------
"""
audio_len = 512
xs  = torch.linspace(0, 1, audio_len)
ys  = torch.sin(xs * xs * 200)
"""
ys = torchaudio.load("audio.wav")[0][0]
audio_len = ys.shape[0]
xs = torch.linspace(0, 100, audio_len)


x_t  = xs.unsqueeze(1).to(device)
y_t  = ys.unsqueeze(1).to(device)

# --- Instantiate models, mappings, and optimizers ----------------------
methods = {}
for name, fmap_class, trainable in [
	("naive",        None,                  False),
	("fixed_fourier",  FourierFeatureMapping, False),
	("fixed_gabor",    GaborFeatureMapping,   False),
	("opt_fourier",    FourierFeatureMapping, True),
	("opt_gabor",      GaborFeatureMapping,   True),
]:
	# instantiate mapping if needed
	fmap = fmap_class(1, NUM_FEATURES, trainable=trainable) if fmap_class else None
	if fmap is not None:
		fmap = fmap.to(device)

	# create model
	in_dim = 1 if fmap is None else fmap(torch.zeros(1,1, device=device)).shape[-1]
	model = MLP(in_dim, HIDDEN_DIM, NUM_LAYERS, 1, activation=nn.ReLU(), final_act=nn.Tanh()).to(device)

	# optimizer over model and mapping params
	params = list(model.parameters()) + (list(fmap.parameters()) if (fmap is not None and trainable) else [])
	opt    = optim.Adam(params, lr=LEARNING_RATE)

	methods[name] = {"model": model, "map": fmap, "opt": opt, "loss_curve": []}

# --- Training & GIF creation ------------------------------------------
criterion = SpectrogramPhaseRetrievalLoss().to(device)
waveform_frames    = []
spectrogram_frames = []
spec_tf            = torchaudio.transforms.Spectrogram(power=1).to(device)
target_spec       = spec_tf(y_t.squeeze(1)).abs().cpu().numpy()

for ep in tqdm(range(1, NUM_EPOCHS+1), desc="Training waveform & spectrograms"):
	# training step
	for m in methods.values():
		m['opt'].zero_grad()
		inp = x_t if m['map'] is None else m['map'](x_t)
		pred = m['model'](inp)
		loss = criterion(pred, y_t)
		loss.backward()
		m['opt'].step()
		m['loss_curve'].append(loss.item())

	# snapshots
	if ep % SNAPSHOT_FREQ == 0:
		# waveform plot
		plt.figure(figsize=(15,4), dpi=150)
		plt.plot(xs, ys, 'k--', label="Target")
		for nm, m in methods.items():
			with torch.no_grad():
				inp = x_t if m['map'] is None else m['map'](x_t)
				p = m['model'](inp).cpu().numpy().flatten()
				plt.plot(xs, p, label=nm, color=LOSS_COLORS[nm])
		plt.title(f"Waveform Approximation | Epoch {ep}")
		plt.legend(); plt.tight_layout()
		buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
		waveform_frames.append(np.array(Image.open(buf))); plt.close()

		# spectrogram grid
		fig, axes = plt.subplots(1, len(methods)+1, figsize=(15,4), dpi=100)
		axes[0].imshow(target_spec, aspect='auto', origin='lower')
		axes[0].set_title("Target Spec")
		for i, (nm, m) in enumerate(methods.items()):
			with torch.no_grad():
				inp = x_t if m['map'] is None else m['map'](x_t)
				spec = spec_tf(m['model'](inp).squeeze(1)).abs().cpu().numpy()
			axes[i+1].imshow(spec, aspect='auto', origin='lower')
			axes[i+1].set_title(nm)
		plt.tight_layout()
		buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
		spectrogram_frames.append(np.array(Image.open(buf))); plt.close()

# ensure output directory
os.makedirs('media', exist_ok=True)
# save GIFs
wf_gif = 'media/waveform_progress.gif'
sp_gif = 'media/spectrogram_progress.gif'
imageio.mimsave(wf_gif, waveform_frames, duration=0.2, loop=0)
imageio.mimsave(sp_gif, spectrogram_frames, duration=0.2, loop=0)
print(f"Saved waveform GIF: {wf_gif}")
print(f"Saved spectrogram GIF: {sp_gif}")

# --- Plot loss curves --------------------------------------------------
plt.figure(figsize=(8,6), dpi=150)
epochs = np.arange(1, NUM_EPOCHS+1)
for nm, m in methods.items():
	plt.plot(epochs, m['loss_curve'], label=nm, color=LOSS_COLORS[nm])
plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Loss (L1 spec)")
plt.title("Training Loss Curves"); plt.legend(); plt.tight_layout(); plt.show()
