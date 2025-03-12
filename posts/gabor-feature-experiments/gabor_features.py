import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import librosa
import imageio
import io
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm

# =============================================================================
# Configuration Parameters
# =============================================================================
DO_AUDIO_EXPERIMENT = False  # Set to False to skip the audio experiment
#SELECTED_IMAGE_FILES = ["image1.jpg", "image2.jpg", "image3.jpg"]  # Images to process
SELECTED_IMAGE_FILES = ["image2.jpg"]  # Images to process


# Model configuration parameters
HIDDEN_DIM = 256     # Hidden dimension for MLPs
NUM_LAYERS = 3       # Number of layers for MLPs
NUM_FEATURES = 32    # Number of Fourier/Gabor features

# =============================================================================
# 1. Define a generic MLP with an optional final activation.
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, output_dim=1, activation=nn.ReLU(), final_activation=None):
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation)
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.final_activation = final_activation
        
    def forward(self, x):
        out = self.net(x)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out

# =============================================================================
# 2. Define ND Feature Mapping classes (works for 1D and 2D inputs)
#    (a) FourierFeatureMappingND – applies cos & sin to (2π x @ freqs)
#    (b) GaborFeatureMappingND – applies a Gaussian envelope with a phase
# =============================================================================
class FourierFeatureMappingND(nn.Module):
    def __init__(self, input_dim, num_features=16, sigma_freq=20.0, trainable=False):
        super(FourierFeatureMappingND, self).__init__()
        # Initialize frequency matrix of shape (num_features, input_dim) from Gaussian
        freqs = torch.randn(input_dim, num_features) * sigma_freq
        if trainable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)
        self.input_dim = input_dim

    def forward(self, x):
        projection = 2 * np.pi * (x @ self.freqs)
        return torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)

class GaborFeatureMappingND(nn.Module):
    def __init__(self, input_dim, num_features=16, sigma=0.1, sigma_freq=20.0, trainable=False):
        super(GaborFeatureMappingND, self).__init__()
        
        centers = torch.rand(num_features, input_dim)
        frequencies = torch.randn(num_features, input_dim) * sigma_freq
        sigmas = torch.full((num_features,), sigma)

        if trainable:
            self.centers = nn.Parameter(centers)
            self.frequencies = nn.Parameter(frequencies)
            self.sigma = nn.Parameter(sigmas)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('frequencies', frequencies)
            self.register_buffer('sigma', sigmas)

    def forward(self, x):
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        envelope = torch.exp(- (diff ** 2).sum(dim=2) / (2 * (self.sigma.unsqueeze(0) ** 2)))
        phase = 2 * np.pi * (diff * self.frequencies.unsqueeze(0)).sum(dim=2)
        cos_part = envelope * torch.cos(phase)
        sin_part = envelope * torch.sin(phase)
        return torch.cat([cos_part, sin_part], dim=-1)

# =============================================================================
# 3. Define snapshot epochs and training parameters
# =============================================================================
num_epochs = 3000
lr = 1e-3

loss_colors = {
    "naive": "blue",
    "fixed_fourier": "red",
    "fixed_gabor": "green",
    "opt_fourier": "orange",
    "opt_gabor": "purple"
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 4. AUDIO EXPERIMENT
# =============================================================================
if DO_AUDIO_EXPERIMENT:
    print("Starting Audio Experiment...")

    # Load audio.wav (assumes file is in same folder)
    audio, sr = librosa.load("audio.wav", sr=None)
    total_samples = len(audio)
    print(f"Audio loaded: {total_samples} samples, {sr} Hz")

    # Extract 1 second from the middle
    start_sample = total_samples // 2 - sr // 2
    end_sample = start_sample + sr
    audio_segment = audio[start_sample:end_sample]
    print(f"Using samples {start_sample} to {end_sample}.")

    # Create time coordinates normalized to [0,1]
    t = np.linspace(0, 1, sr)
    x_audio = torch.tensor(t, dtype=torch.float32).unsqueeze(1).to(device)
    y_audio = torch.tensor(audio_segment, dtype=torch.float32).unsqueeze(1).to(device)

    # Set up five methods for audio with configurable parameters
    naive_audio = MLP(input_dim=1, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1, 
                      activation=nn.ReLU(), final_activation=nn.Tanh()).to(device)
    fixed_fourier_audio_map = FourierFeatureMappingND(input_dim=1, num_features=NUM_FEATURES, sigma_freq=100, trainable=False).to(device)
    fixed_fourier_audio = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1, 
                              activation=nn.ReLU(), final_activation=nn.Tanh()).to(device)
    fixed_gabor_audio_map = GaborFeatureMappingND(input_dim=1, num_features=NUM_FEATURES, sigma=0.1, sigma_freq=100, trainable=False).to(device)
    fixed_gabor_audio = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1, 
                            activation=nn.ReLU(), final_activation=nn.Tanh()).to(device)
    opt_fourier_audio_map = FourierFeatureMappingND(input_dim=1, num_features=NUM_FEATURES, sigma_freq=100, trainable=True).to(device)
    opt_fourier_audio = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1, 
                            activation=nn.ReLU(), final_activation=nn.Tanh()).to(device)
    opt_gabor_audio_map = GaborFeatureMappingND(input_dim=1, num_features=NUM_FEATURES, sigma=0.1, sigma_freq=100, trainable=True).to(device)
    opt_gabor_audio = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1, 
                          activation=nn.ReLU(), final_activation=nn.Tanh()).to(device)

    optimizers_audio = {
        "naive": optim.Adam(naive_audio.parameters(), lr=lr),
        "fixed_fourier": optim.Adam(fixed_fourier_audio.parameters(), lr=lr),
        "fixed_gabor": optim.Adam(fixed_gabor_audio.parameters(), lr=lr),
        "opt_fourier": optim.Adam(list(opt_fourier_audio.parameters()) + list(opt_fourier_audio_map.parameters()), lr=lr),
        "opt_gabor": optim.Adam(list(opt_gabor_audio.parameters()) + list(opt_gabor_audio_map.parameters()), lr=lr)
    }

    loss_curves_audio = {key: [] for key in optimizers_audio.keys()}
    criterion = nn.MSELoss()

    for epoch in tqdm(range(1, num_epochs+1), desc="Audio Training", leave=False):
        for opt in optimizers_audio.values():
            opt.zero_grad()
            
        pred_naive = naive_audio(x_audio)
        pred_fixed_fourier = fixed_fourier_audio(fixed_fourier_audio_map(x_audio))
        pred_fixed_gabor = fixed_gabor_audio(fixed_gabor_audio_map(x_audio))
        pred_opt_fourier = opt_fourier_audio(opt_fourier_audio_map(x_audio))
        pred_opt_gabor = opt_gabor_audio(opt_gabor_audio_map(x_audio))
        
        loss_naive = criterion(pred_naive, y_audio)
        loss_fixed_fourier = criterion(pred_fixed_fourier, y_audio)
        loss_fixed_gabor = criterion(pred_fixed_gabor, y_audio)
        loss_opt_fourier = criterion(pred_opt_fourier, y_audio)
        loss_opt_gabor = criterion(pred_opt_gabor, y_audio)
        
        loss_naive.backward()
        loss_fixed_fourier.backward()
        loss_fixed_gabor.backward()
        loss_opt_fourier.backward()
        loss_opt_gabor.backward()
        
        optimizers_audio["naive"].step()
        optimizers_audio["fixed_fourier"].step()
        optimizers_audio["fixed_gabor"].step()
        optimizers_audio["opt_fourier"].step()
        optimizers_audio["opt_gabor"].step()
        
        loss_curves_audio["naive"].append(loss_naive.item())
        loss_curves_audio["fixed_fourier"].append(loss_fixed_fourier.item())
        loss_curves_audio["fixed_gabor"].append(loss_fixed_gabor.item())
        loss_curves_audio["opt_fourier"].append(loss_opt_fourier.item())
        loss_curves_audio["opt_gabor"].append(loss_opt_gabor.item())

    # Print sorted learned parameters for audio mappings
    opt_fourier_audio_freq = np.sort(opt_fourier_audio_map.freqs.detach().cpu().numpy().flatten())
    print("\nLearned parameters for Audio Optimizable Fourier Mapping (sorted freqs):")
    print(opt_fourier_audio_freq)

    opt_gabor_audio_centers = np.sort(opt_gabor_audio_map.centers.detach().cpu().numpy().flatten())
    opt_gabor_audio_freq = np.sort(opt_gabor_audio_map.frequencies.detach().cpu().numpy().flatten())
    print("\nLearned parameters for Audio Optimizable Gabor Mapping (sorted centers):")
    print(opt_gabor_audio_centers)
    print("Learned parameters for Audio Optimizable Gabor Mapping (sorted frequencies):")
    print(opt_gabor_audio_freq)
else:
    print("Skipping Audio Experiment.")

# =============================================================================
# 5. IMAGE EXPERIMENT
# =============================================================================
print("Starting Image Experiment...")

# Use the selected image files from configuration
image_files = SELECTED_IMAGE_FILES

# Helper function to create a frame with overlaid text on top of the image.
def create_frame(pred_img, epoch, loss):
    img_uint8 = (pred_img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("cour.ttf", 22)
    text = f"Epoch: {epoch} | Loss: {loss:.4e}"
    x_text, y_text = 10, 10
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x_text+dx, y_text+dy), text, font=font, fill="white")
    draw.text((x_text, y_text), text, font=font, fill="black")
    return np.array(pil_img)

# For each image file, create a looping GIF for each method.
for img_idx, img_file in enumerate(image_files, start=1):
    if not os.path.exists(img_file):
        print(f"Image file {img_file} not found; skipping.")
        continue
    print(f"\nProcessing {img_file} ...")
    img = Image.open(img_file).convert("RGB")
    img = img.resize((512, 512))
    img_np = np.array(img).astype(np.float32) / 255.0
    H, W, C = img_np.shape
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.stack([grid_x, grid_y], axis=-1)
    coords = coords.reshape(-1, 2)
    x_img = torch.tensor(coords, dtype=torch.float32).to(device)
    y_img = torch.tensor(img_np.reshape(-1, 3), dtype=torch.float32).to(device)
    
    # Set up five models for images with configurable parameters
    naive_img = MLP(input_dim=2, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3, 
                    activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)
    fixed_fourier_img_map = FourierFeatureMappingND(input_dim=2, num_features=NUM_FEATURES, sigma_freq=20.0, trainable=False).to(device)
    fixed_fourier_img = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3, 
                            activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)
    fixed_gabor_img_map = GaborFeatureMappingND(input_dim=2, num_features=NUM_FEATURES, sigma=0.1, sigma_freq=20.0, trainable=False).to(device)
    fixed_gabor_img = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3, 
                          activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)
    opt_fourier_img_map = FourierFeatureMappingND(input_dim=2, num_features=NUM_FEATURES, sigma_freq=20.0, trainable=True).to(device)
    opt_fourier_img = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3, 
                          activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)
    opt_gabor_img_map = GaborFeatureMappingND(input_dim=2, num_features=NUM_FEATURES, sigma=0.1, sigma_freq=20.0, trainable=True).to(device)
    opt_gabor_img = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3, 
                        activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)
    
    optimizers_img = {
        "naive": optim.Adam(naive_img.parameters(), lr=lr),
        "fixed_fourier": optim.Adam(fixed_fourier_img.parameters(), lr=lr),
        "fixed_gabor": optim.Adam(fixed_gabor_img.parameters(), lr=lr),
        "opt_fourier": optim.Adam(list(opt_fourier_img.parameters()) + list(opt_fourier_img_map.parameters()), lr=lr),
        "opt_gabor": optim.Adam(list(opt_gabor_img.parameters()) + list(opt_gabor_img_map.parameters()), lr=lr)
    }
    
    loss_curves_img = {key: [] for key in optimizers_img.keys()}
    gif_frames = {key: [] for key in optimizers_img.keys()}
    
    gif_snapshot_freq = 40
    criterion = nn.MSELoss()
    methods = ["naive", "fixed_fourier", "fixed_gabor", "opt_fourier", "opt_gabor"]
    
    for epoch in tqdm(range(1, num_epochs+1), desc=f"Training {img_file}", leave=False):
        for opt in optimizers_img.values():
            opt.zero_grad()
            
        pred_naive = naive_img(x_img)
        pred_fixed_fourier = fixed_fourier_img(fixed_fourier_img_map(x_img))
        pred_fixed_gabor = fixed_gabor_img(fixed_gabor_img_map(x_img))
        pred_opt_fourier = opt_fourier_img(opt_fourier_img_map(x_img))
        pred_opt_gabor = opt_gabor_img(opt_gabor_img_map(x_img))
        
        loss_naive = criterion(pred_naive, y_img)
        loss_fixed_fourier = criterion(pred_fixed_fourier, y_img)
        loss_fixed_gabor = criterion(pred_fixed_gabor, y_img)
        loss_opt_fourier = criterion(pred_opt_fourier, y_img)
        loss_opt_gabor = criterion(pred_opt_gabor, y_img)
        
        loss_naive.backward()
        loss_fixed_fourier.backward()
        loss_fixed_gabor.backward()
        loss_opt_fourier.backward()
        loss_opt_gabor.backward()
        
        optimizers_img["naive"].step()
        optimizers_img["fixed_fourier"].step()
        optimizers_img["fixed_gabor"].step()
        optimizers_img["opt_fourier"].step()
        optimizers_img["opt_gabor"].step()
        
        loss_curves_img["naive"].append(loss_naive.item())
        loss_curves_img["fixed_fourier"].append(loss_fixed_fourier.item())
        loss_curves_img["fixed_gabor"].append(loss_fixed_gabor.item())
        loss_curves_img["opt_fourier"].append(loss_opt_fourier.item())
        loss_curves_img["opt_gabor"].append(loss_opt_gabor.item())
        
        if epoch % gif_snapshot_freq == 0:
            pred_naive_img = pred_naive.detach().cpu().numpy().reshape(H, W, 3)
            pred_fixed_fourier_img = pred_fixed_fourier.detach().cpu().numpy().reshape(H, W, 3)
            pred_fixed_gabor_img = pred_fixed_gabor.detach().cpu().numpy().reshape(H, W, 3)
            pred_opt_fourier_img = pred_opt_fourier.detach().cpu().numpy().reshape(H, W, 3)
            pred_opt_gabor_img = pred_opt_gabor.detach().cpu().numpy().reshape(H, W, 3)
            
            gif_frames["naive"].append(create_frame(pred_naive_img, epoch, loss_naive.item()))
            gif_frames["fixed_fourier"].append(create_frame(pred_fixed_fourier_img, epoch, loss_fixed_fourier.item()))
            gif_frames["fixed_gabor"].append(create_frame(pred_fixed_gabor_img, epoch, loss_fixed_gabor.item()))
            gif_frames["opt_fourier"].append(create_frame(pred_opt_fourier_img, epoch, loss_opt_fourier.item()))
            gif_frames["opt_gabor"].append(create_frame(pred_opt_gabor_img, epoch, loss_opt_gabor.item()))
    
    # Print sorted learned parameters for image mappings
    opt_fourier_img_freq = np.sort(opt_fourier_img_map.freqs.detach().cpu().numpy().flatten())
    print(f"\nLearned parameters for Image Optimizable Fourier Mapping (sorted freqs) for {img_file}:")
    print(opt_fourier_img_freq)
    
    opt_gabor_img_centers = np.sort(opt_gabor_img_map.centers.detach().cpu().numpy().flatten())
    opt_gabor_img_freq = np.sort(opt_gabor_img_map.frequencies.detach().cpu().numpy().flatten())
    print(f"\nLearned parameters for Image Optimizable Gabor Mapping (sorted centers) for {img_file}:")
    print(opt_gabor_img_centers)
    print(f"Learned parameters for Image Optimizable Gabor Mapping (sorted frequencies) for {img_file}:")
    print(opt_gabor_img_freq)
    
    # Save a looping GIF for each method
    for m in methods:
        gif_filename = f"media/{img_file}_{m}_progress.gif"
        imageio.mimsave(gif_filename, gif_frames[m], duration=0.2, loop=0)
        print(f"Saved GIF: {gif_filename}")
    
    # Plot loss curves for this image experiment
    plt.figure(figsize=(8,6), dpi=150)
    epochs_arr = np.arange(1, num_epochs+1)
    for m in methods:
        plt.plot(epochs_arr, loss_curves_img[m], label=m, color=loss_colors[m])
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{img_file} Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()
