import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio
import io
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import cv2

# =============================================================================
# Experiment Toggles
# =============================================================================
DO_IMAGE_EXPERIMENT = True
DO_CHIRP_EXPERIMENT = False
DO_VIDEO_EXPERIMENT = False

# =============================================================================
# Configuration Parameters
# =============================================================================
# For image experiment
SELECTED_IMAGE_FILES = ["image2.jpg", "image4.jpg"]  # Images to process

# Toggle AMP usage (fp16 training)
USE_AMP = True

# Model configuration parameters
HIDDEN_DIM = 128     # Hidden dimension for MLPs
NUM_LAYERS = 3      # Number of layers for MLPs
NUM_FEATURES = 48   # Number of Fourier/Gabor features

# Device and scaler setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if USE_AMP and device.type == 'cuda':
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None

criterion = nn.MSELoss()

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
#    (a) FourierFeatureMapping – applies cos & sin to (2π x @ freqs)
#    (b) GaborFeatureMapping – applies a Gaussian envelope with a phase
# =============================================================================
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, num_features=16, sigma_freq=20.0, trainable=False):
        super(FourierFeatureMapping, self).__init__()
        freqs = torch.randn(input_dim, num_features) * sigma_freq
        if trainable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)
        self.input_dim = input_dim

    def forward(self, x):
        projection = 2 * np.pi * (x @ self.freqs)
        return torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)

class GaborFeatureMapping(nn.Module):
    def __init__(self, input_dim, num_features=16, sigma=0.1, sigma_freq=20.0, trainable=False):
        super(GaborFeatureMapping, self).__init__()
        
        centers = torch.rand(num_features, input_dim)
        frequencies = torch.randn(num_features, input_dim) * sigma_freq
        sigmas = torch.full((num_features, input_dim), sigma)

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
        envelope = torch.exp(- (diff ** 2 / (self.sigma.unsqueeze(0) ** 2)).sum(dim=2) / 2)
        phase = 2 * np.pi * (diff * self.frequencies.unsqueeze(0)).sum(dim=2)
        cos_part = envelope * torch.cos(phase)
        sin_part = envelope * torch.sin(phase)
        return torch.cat([cos_part, sin_part], dim=-1)

# =============================================================================
# 3. IMAGE EXPERIMENT (Sequential Training)
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

if DO_IMAGE_EXPERIMENT:
    print("\nStarting Image Experiment...")
    image_files = SELECTED_IMAGE_FILES

    # Helper function to create a frame with overlaid text on top of the image.
    def create_frame(pred_img, epoch, loss):
        img_uint8 = (pred_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        draw = ImageDraw.Draw(pil_img)
        # Adjust the font path as needed
        font = ImageFont.truetype("cour.ttf", 22)
        text = f"Epoch: {epoch} | Loss: {loss:.4e}"
        x_text, y_text = 10, 10
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((x_text+dx, y_text+dy), text, font=font, fill="white")
        draw.text((x_text, y_text), text, font=font, fill="black")
        return np.array(pil_img)

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

        # Set up models for image experiment
        naive_img = MLP(input_dim=2, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3, 
                        activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)
        fixed_fourier_img_map = FourierFeatureMapping(input_dim=2, num_features=NUM_FEATURES, sigma_freq=20.0, trainable=False).to(device)
        fixed_fourier_img = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3, 
                                activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)
        fixed_gabor_img_map = GaborFeatureMapping(input_dim=2, num_features=NUM_FEATURES, sigma=0.3, sigma_freq=20.0, trainable=False).to(device)
        fixed_gabor_img = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3, 
                              activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)
        opt_fourier_img_map = FourierFeatureMapping(input_dim=2, num_features=NUM_FEATURES, sigma_freq=20.0, trainable=True).to(device)
        opt_fourier_img = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3, 
                              activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)
        opt_gabor_img_map = GaborFeatureMapping(input_dim=2, num_features=NUM_FEATURES, sigma=0.3, sigma_freq=20.0, trainable=True).to(device)
        opt_gabor_img = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3, 
                            activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)

        # Set up optimizers for each image method
        img_methods = {
            "naive": {
                "model": naive_img,
                "mapping": None,
                "optimizer": optim.Adam(naive_img.parameters(), lr=lr),
                "loss_curve": [],
                "gif_frames": []
            },
            "fixed_fourier": {
                "model": fixed_fourier_img,
                "mapping": fixed_fourier_img_map,
                "optimizer": optim.Adam(fixed_fourier_img.parameters(), lr=lr),
                "loss_curve": [],
                "gif_frames": []
            },
            "fixed_gabor": {
                "model": fixed_gabor_img,
                "mapping": fixed_gabor_img_map,
                "optimizer": optim.Adam(fixed_gabor_img.parameters(), lr=lr),
                "loss_curve": [],
                "gif_frames": []
            },
            "opt_fourier": {
                "model": opt_fourier_img,
                "mapping": opt_fourier_img_map,
                "optimizer": optim.Adam(list(opt_fourier_img.parameters()) + list(opt_fourier_img_map.parameters()), lr=lr),
                "loss_curve": [],
                "gif_frames": []
            },
            "opt_gabor": {
                "model": opt_gabor_img,
                "mapping": opt_gabor_img_map,
                "optimizer": optim.Adam(list(opt_gabor_img.parameters()) + list(opt_gabor_img_map.parameters()), lr=lr),
                "loss_curve": [],
                "gif_frames": []
            }
        }
        
        gif_snapshot_freq = 40  # Save a snapshot every 40 epochs

        # Train each image method sequentially
        for method_name, method in img_methods.items():
            print(f"\nTraining {img_file} method: {method_name}")
            for epoch in tqdm(range(1, num_epochs+1), desc=f"Training {img_file}: {method_name}"):
                method["optimizer"].zero_grad()
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    if method["mapping"] is not None:
                        pred = method["model"](method["mapping"](x_img))
                    else:
                        pred = method["model"](x_img)
                    loss = criterion(pred, y_img)
                if USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.step(method["optimizer"])
                    scaler.update()
                else:
                    loss.backward()
                    method["optimizer"].step()
                method["loss_curve"].append(loss.item())
                
                if epoch % gif_snapshot_freq == 0:
                    pred_img = pred.detach().cpu().numpy().reshape(H, W, 3)
                    frame = create_frame(pred_img, epoch, loss.item())
                    method["gif_frames"].append(frame)

            # After training, print learned parameters (if applicable)
            if method_name == "opt_fourier":
                learned_freqs = np.sort(method["mapping"].freqs.detach().cpu().numpy().flatten())
                print(f"\nLearned parameters for Image Optimizable Fourier Mapping (sorted freqs) for {img_file}:")
                print(learned_freqs)
            elif method_name == "opt_gabor":
                learned_centers = np.sort(method["mapping"].centers.detach().cpu().numpy().flatten())
                learned_freqs = np.sort(method["mapping"].frequencies.detach().cpu().numpy().flatten())
                print(f"\nLearned parameters for Image Optimizable Gabor Mapping (sorted centers) for {img_file}:")
                print(learned_centers)
                print(f"Learned parameters for Image Optimizable Gabor Mapping (sorted frequencies) for {img_file}:")
                print(learned_freqs)
            
            # Save a looping GIF for the method
            gif_filename = f"media/{img_file[0:-4]}_{method_name}_progress.gif"
            imageio.mimsave(gif_filename, method["gif_frames"], duration=0.2, loop=0)
            print(f"Saved GIF: {gif_filename}")
        
        # Plot loss curves for this image experiment (all methods in one plot)
        plt.figure(figsize=(8,6), dpi=150)
        epochs_arr = np.arange(1, num_epochs+1)
        for m in img_methods.keys():
             plt.plot(epochs_arr, img_methods[m]["loss_curve"], label=m, color=loss_colors[m])
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"{img_file} Training Loss Curves")
        plt.legend()
        plt.tight_layout()
        plt.show()

# =============================================================================
# 4. CHIRP EXPERIMENT (Function Approximation)
# =============================================================================
if DO_CHIRP_EXPERIMENT:
    print("\nStarting Chirp Experiment...")

    # Toggle which methods to use for the chirp experiment
    CHIRP_USE_NAIVE = True
    CHIRP_USE_FIXED_FOURIER = True
    CHIRP_USE_FIXED_GABOR = False
    CHIRP_USE_OPT_FOURIER = True
    CHIRP_USE_OPT_GABOR = False

    # Define the target chirp function (a linear chirp from 5 Hz to 50 Hz)
    def chirp(x, f0=5, f1=50):
        return np.sin(2 * np.pi * (f0 * x + 0.5 * (f1 - f0) * x**2))

    num_points = 1000
    t_chirp = np.linspace(0, 1, num_points)
    y_target = chirp(t_chirp)
    # Convert to torch tensors
    x_chirp = torch.tensor(t_chirp, dtype=torch.float32).unsqueeze(1).to(device)
    y_chirp = torch.tensor(y_target, dtype=torch.float32).unsqueeze(1).to(device)

    # Training parameters for chirp experiment
    num_epochs_chirp = 500
    lr_chirp = 1e-3

    # Set up models for chirp experiment based on toggles
    chirp_methods = {}

    if CHIRP_USE_NAIVE:
        naive_chirp = MLP(input_dim=1, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1, 
                          activation=nn.ReLU(), final_activation=nn.Tanh()).to(device)
        chirp_methods["naive"] = {
            "model": naive_chirp,
            "mapping": None,
            "optimizer": optim.Adam(naive_chirp.parameters(), lr=lr_chirp),
            "loss_curve": []
        }

    if CHIRP_USE_FIXED_FOURIER:
        fixed_fourier_chirp_map = FourierFeatureMapping(input_dim=1, num_features=NUM_FEATURES, sigma_freq=100, trainable=False).to(device)
        fixed_fourier_chirp = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1, 
                                  activation=nn.ReLU(), final_activation=nn.Tanh()).to(device)
        chirp_methods["fixed_fourier"] = {
            "model": fixed_fourier_chirp,
            "mapping": fixed_fourier_chirp_map,
            "optimizer": optim.Adam(fixed_fourier_chirp.parameters(), lr=lr_chirp),
            "loss_curve": []
        }

    if CHIRP_USE_FIXED_GABOR:
        fixed_gabor_chirp_map = GaborFeatureMapping(input_dim=1, num_features=NUM_FEATURES, sigma=0.1, sigma_freq=100, trainable=False).to(device)
        fixed_gabor_chirp = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1, 
                                activation=nn.ReLU(), final_activation=nn.Tanh()).to(device)
        chirp_methods["fixed_gabor"] = {
            "model": fixed_gabor_chirp,
            "mapping": fixed_gabor_chirp_map,
            "optimizer": optim.Adam(fixed_gabor_chirp.parameters(), lr=lr_chirp),
            "loss_curve": []
        }

    if CHIRP_USE_OPT_FOURIER:
        opt_fourier_chirp_map = FourierFeatureMapping(input_dim=1, num_features=NUM_FEATURES, sigma_freq=100, trainable=True).to(device)
        opt_fourier_chirp = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1, 
                                activation=nn.ReLU(), final_activation=nn.Tanh()).to(device)
        chirp_methods["opt_fourier"] = {
            "model": opt_fourier_chirp,
            "mapping": opt_fourier_chirp_map,
            "optimizer": optim.Adam(list(opt_fourier_chirp.parameters()) + list(opt_fourier_chirp_map.parameters()), lr=lr_chirp),
            "loss_curve": []
        }

    if CHIRP_USE_OPT_GABOR:
        opt_gabor_chirp_map = GaborFeatureMapping(input_dim=1, num_features=NUM_FEATURES, sigma=0.1, sigma_freq=100, trainable=True).to(device)
        opt_gabor_chirp = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1, 
                              activation=nn.ReLU(), final_activation=nn.Tanh()).to(device)
        chirp_methods["opt_gabor"] = {
            "model": opt_gabor_chirp,
            "mapping": opt_gabor_chirp_map,
            "optimizer": optim.Adam(list(opt_gabor_chirp.parameters()) + list(opt_gabor_chirp_map.parameters()), lr=lr_chirp),
            "loss_curve": []
        }

    # List to store GIF frames (one combined GIF for all methods)
    chirp_gif_frames = []
    gif_snapshot_freq_chirp = 10

    # Training loop for chirp experiment with tqdm progress bar
    print("\nTraining Chirp Experiment...")
    for epoch in tqdm(range(1, num_epochs_chirp+1), desc="Training Chirp Experiment"):
        for method_name, method in chirp_methods.items():
            method["optimizer"].zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                if method["mapping"] is not None:
                    pred = method["model"](method["mapping"](x_chirp))
                else:
                    pred = method["model"](x_chirp)
                loss = criterion(pred, y_chirp)
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(method["optimizer"])
                scaler.update()
            else:
                loss.backward()
                method["optimizer"].step()
            method["loss_curve"].append(loss.item())
        
        # Every gif_snapshot_freq_chirp epochs, create a plot frame combining all methods
        if epoch % gif_snapshot_freq_chirp == 0:
            plt.figure(figsize=(15,4), dpi=150)  # Wider figure
            plt.plot(t_chirp, y_target, 'k--', label="Target")
            for m_name, method in chirp_methods.items():
                with torch.no_grad():
                    if method["mapping"] is not None:
                        pred = method["model"](method["mapping"](x_chirp))
                    else:
                        pred = method["model"](x_chirp)
                pred_np = pred.detach().cpu().numpy().flatten()
                plt.plot(t_chirp, pred_np, label=m_name, color=loss_colors[m_name])
            # Remove axis labels
            # plt.xlabel("t")
            # plt.ylabel("Amplitude")
            plt.title(f"Chirp Approximation | Epoch: {epoch}")  # Title without methods info
            plt.legend()
            plt.tight_layout()
            
            # Save the current frame to a buffer and append to the list
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frame = np.array(Image.open(buf))
            chirp_gif_frames.append(frame)
            plt.close()

    # Save the combined GIF for chirp experiment
    gif_filename = "media/chirp_progress.gif"
    imageio.mimsave(gif_filename, chirp_gif_frames, duration=0.2, loop=0)
    print(f"Saved chirp progress GIF: {gif_filename}")

    # Plot loss curves for chirp experiment and save as chirp_loss.png
    plt.figure(figsize=(8,6), dpi=230)
    epochs_arr_chirp = np.arange(1, num_epochs_chirp+1)
    for m_name, method in chirp_methods.items():
        plt.plot(epochs_arr_chirp, method["loss_curve"], label=m_name, color=loss_colors[m_name])
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Chirp Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    loss_plot_filename = "media/chirp_loss.png"
    plt.savefig(loss_plot_filename)
    print(f"Saved chirp loss plot: {loss_plot_filename}")
    plt.show()

# =============================================================================
# 5. VIDEO EXPERIMENT (Function Approximation for Video)
# =============================================================================
lr = 1e-3
NUM_FEATURES = 64
HIDDEN_DIM = 256
num_epochs_video = 15000
gif_snapshot_freq_video = 1000  # Save a GIF every 1000 epochs

if DO_VIDEO_EXPERIMENT:
    print("\nStarting Video Experiment...")

    # --- Video Preprocessing Parameters ---
    video_file = "video.mp4"
    target_width = 128
    target_height = 128
    target_fps = 24
    video_duration = 2  # seconds
    num_frames = target_fps * video_duration

    if not os.path.exists(video_file):
        print(f"Video file {video_file} not found; skipping video experiment.")
    else:
        print(f"Processing video file {video_file} ...")
        # Use OpenCV to read the video file
        import cv2
        cap = cv2.VideoCapture(video_file)
        video_frames = []
        frame_count = 0
        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert from BGR (OpenCV default) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)
            pil_frame = pil_frame.resize((target_width, target_height))
            video_frames.append(np.array(pil_frame).astype(np.float32) / 255.0)
            frame_count += 1
        cap.release()
        
        if len(video_frames) == 0:
            print(f"No frames were read from {video_file}. Please verify the file and its format.")
        else:
            video_np = np.array(video_frames)  # Shape: (num_frames, target_height, target_width, 3)

            # Create coordinate grid for video: (x, y, t)
            xs = np.linspace(0, 1, target_width)
            ys = np.linspace(0, 1, target_height)
            ts = np.linspace(0, 1, num_frames)
            grid_t, grid_y, grid_x = np.meshgrid(ts, ys, xs, indexing='ij')
            coords = np.stack([grid_x, grid_y, grid_t], axis=-1)  # (num_frames, target_height, target_width, 3)
            coords = coords.reshape(-1, 3)
            video_gt = video_np.reshape(-1, 3)

            # Convert ground truth data to torch tensors
            x_video = torch.tensor(coords, dtype=torch.float32).to(device)
            y_video = torch.tensor(video_gt, dtype=torch.float32).to(device)

            # --- Setup Models for Video Experiment (Input dim = 3) ---
            naive_video = MLP(input_dim=3, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3,
                               activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)
            opt_fourier_video_map = FourierFeatureMapping(input_dim=3, num_features=NUM_FEATURES, sigma_freq=20.0, trainable=True).to(device)
            opt_fourier_video = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3,
                                    activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)
            opt_gabor_video_map = GaborFeatureMapping(input_dim=3, num_features=NUM_FEATURES, sigma=0.3, sigma_freq=20.0, trainable=True).to(device)
            opt_gabor_video = MLP(input_dim=2*NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3,
                                  activation=nn.ReLU(), final_activation=nn.Sigmoid()).to(device)

            # Set up optimizers for each video method
            video_methods = {
                "naive": {
                    "model": naive_video,
                    "mapping": None,
                    "optimizer": optim.Adam(naive_video.parameters(), lr=lr),
                    "loss_curve": []
                },
                "opt_fourier": {
                    "model": opt_fourier_video,
                    "mapping": opt_fourier_video_map,
                    "optimizer": optim.Adam(list(opt_fourier_video.parameters()) + list(opt_fourier_video_map.parameters()), lr=lr),
                    "loss_curve": []
                },
                "opt_gabor": {
                    "model": opt_gabor_video,
                    "mapping": opt_gabor_video_map,
                    "optimizer": optim.Adam(list(opt_gabor_video.parameters()) + list(opt_gabor_video_map.parameters()), lr=lr),
                    "loss_curve": []
                }
            }

            # Create a folder to store video GIFs
            video_gif_folder = "media/video_gifs"
            if not os.path.exists(video_gif_folder):
                os.makedirs(video_gif_folder)

            # Helper function to overlay text on a single video frame
            def create_video_frame(frame_img, epoch, loss, method_name):
                img_uint8 = (frame_img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8)
                draw = ImageDraw.Draw(pil_img)
                # Use a smaller font size and adjust the text position
                font = ImageFont.truetype("cour.ttf", 10)
                text = f"{method_name}\n{epoch}\n{loss:.3e}"
                x_text, y_text = 5, 5
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((x_text + dx, y_text + dy), text, font=font, fill="white")
                draw.text((x_text, y_text), text, font=font, fill="black")
                return np.array(pil_img)

            # --- Sequential Training Loop for Video Experiment ---
            for method_name, method in video_methods.items():
                print(f"\nTraining Video Experiment method: {method_name}")
                pbar = tqdm(range(1, num_epochs_video + 1), desc=f"Training {method_name}")
                for epoch in pbar:
                    method["optimizer"].zero_grad()
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        if method["mapping"] is not None:
                            pred = method["model"](method["mapping"](x_video))
                        else:
                            pred = method["model"](x_video)
                        loss = criterion(pred, y_video)
                    if USE_AMP:
                        scaler.scale(loss).backward()
                        scaler.step(method["optimizer"])
                        scaler.update()
                    else:
                        loss.backward()
                        method["optimizer"].step()
                    method["loss_curve"].append(loss.item())
                    
                    # Update tqdm progress bar with current loss
                    pbar.set_postfix(loss=f"{loss.item():.4e}")

                    # Every 100 epochs, generate a GIF for the current method
                    if epoch % gif_snapshot_freq_video == 0:
                        with torch.no_grad():
                            if method["mapping"] is not None:
                                pred = method["model"](method["mapping"](x_video))
                            else:
                                pred = method["model"](x_video)
                        pred_np = pred.detach().cpu().numpy().reshape(num_frames, target_height, target_width, 3)
                        video_frames_pred = []
                        for i in range(num_frames):
                            frame_with_overlay = create_video_frame(pred_np[i], epoch, method["loss_curve"][-1], method_name)
                            video_frames_pred.append(frame_with_overlay)
                        gif_filename = os.path.join(video_gif_folder, f"{method_name}_epoch{epoch}.gif")
                        imageio.mimsave(gif_filename, video_frames_pred, duration=1/target_fps, loop=0)
                
                # Clear GPU cache after finishing training one method
                torch.cuda.empty_cache()

            # --- Create Ground Truth GIF for Reference ---
            ground_truth_frames = []
            for i in range(num_frames):
                img_uint8 = (video_np[i] * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8)
                draw = ImageDraw.Draw(pil_img)
                font = ImageFont.truetype("cour.ttf", 14)
                text = "Ground Truth"
                x_text, y_text = 5, 5
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((x_text + dx, y_text + dy), text, font=font, fill="white")
                draw.text((x_text, y_text), text, font=font, fill="black")
                ground_truth_frames.append(np.array(pil_img))
            ground_truth_gif_filename = os.path.join(video_gif_folder, "ground_truth.gif")
            imageio.mimsave(ground_truth_gif_filename, ground_truth_frames, duration=1/target_fps, loop=0)
            print(f"Saved ground truth video GIF: {ground_truth_gif_filename}")

            # --- Plot Loss Curves ---
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6), dpi=150)
            for method_name, method in video_methods.items():
                plt.plot(np.arange(1, num_epochs_video + 1), method["loss_curve"], label=method_name)
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title("Video Training Loss Curves")
            plt.legend()
            plt.tight_layout()
            plt.show()
