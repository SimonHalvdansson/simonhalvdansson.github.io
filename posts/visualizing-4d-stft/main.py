import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram
from scipy.fft import fft, fftshift, fft2
from PIL import Image
import os
import sys

# Ensure the output directory exists
output_dir = 'media'
os.makedirs(output_dir, exist_ok=True)

def generate_chirp_image(filename='chirp.png', width=2400, dpi=200):
    """
    Generates a chirp signal waveform and its spectrogram side by side.
    Saves the image as a PNG file with higher resolution and adjusted dimensions.
    Removes the colorbar and makes the image wider and less tall.
    """
    # Parameters
    fs = 1000  # Sampling frequency in Hz
    T = 2      # Duration in seconds
    t = np.linspace(0, T, int(fs*T), endpoint=False)
    
    # Generate chirp signal with lower frequencies
    f0 = 3     # Start frequency in Hz
    f1 = 40    # End frequency in Hz
    signal = chirp(t, f0=f0, f1=f1, t1=T, method='linear')
    
    # Create figure with wider and less tall aspect ratio
    fig, axs = plt.subplots(1, 2, figsize=(width / dpi, (width / dpi) * 0.3), dpi=dpi)
    
    # Plot waveform
    axs[0].plot(t, signal, color='blue')
    axs[0].set_title('Chirp Waveform', fontsize=16)
    
    # Remove axis numbers and labels
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    # Remove grid lines
    axs[0].grid(False)
    
    # Plot spectrogram with increased frequency resolution
    # Increase nperseg for better frequency resolution
    nperseg = 512  # You can experiment with 256, 512, 1024, etc.
    noverlap = nperseg // 2  # 50% overlap
    f, tau, Sxx = spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    
    # Convert power spectrogram to dB scale for better visibility
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Adding a small value to avoid log(0)
    
    mesh = axs[1].pcolormesh(tau, f, Sxx_dB, shading='gouraud', cmap='viridis')
    axs[1].set_title('Spectrogram (dB)', fontsize=16)
    
    # Remove axis numbers and labels
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    # Remove grid lines
    axs[1].grid(False)
    
    # Limit frequency axis to avoid Nyquist issues
    axs[1].set_ylim(0, f1 + 10)
    
    # Remove colorbar as per your request
    # Commented out the colorbar lines
    # cbar = fig.colorbar(mesh, ax=axs[1], format='%+2.0f dB')
    # cbar.set_label('Intensity [dB]')
    
    # Adjust layout to minimize whitespace and make the figure wider and less tall
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.05)
    
    # Save the figure with higher DPI for better resolution
    plt.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def generate_modulated_gaussians_image(filename='modulated_gaussians.png', width=1600, dpi=100):
    """
    Generates a 2x2 subplot with modulated Gaussians and their spectrums.
    Saves the image as a PNG file.
    """
    # Parameters
    x = np.linspace(-10, 10, 5000)  # Increased sampling density
    modulation = 0.2  # Low modulation depth
    modulation_freq = 20  # Higher modulation frequency
    shift = 4  # Spatial shift for the second Gaussian
    
    # Generate two modulated Gaussians with high modulation frequency
    gaussian1 = np.exp(-x**2) * (modulation * np.cos(modulation_freq * x))
    gaussian2 = np.exp(-(x - shift)**2) * (modulation * np.cos(modulation_freq * (x - shift)))
    
    # Compute FFTs and power spectra
    def compute_spectrum(signal, dx):
        N = len(signal)
        fft_vals = fft(signal)
        fft_power = np.abs(fft_vals)**2
        freq = np.fft.fftfreq(N, d=dx)
        # Only take positive frequencies
        idx = freq >= 0
        return freq[idx], fft_power[idx]
    
    dx = x[1] - x[0]
    freq1, spectrum1 = compute_spectrum(gaussian1, dx)
    freq2, spectrum2 = compute_spectrum(gaussian2, dx)
    
    # Create figure with wider aspect ratio
    fig, axs = plt.subplots(2, 2, figsize=(width / dpi, (width / dpi) * 0.6), dpi=dpi)
    
    # Plot Gaussians
    axs[0, 0].plot(x, gaussian1, color='green')
    axs[0, 0].set_title('Modulated Gaussian 1', fontsize=16)
    
    axs[1, 0].plot(x, gaussian2, color='red')
    axs[1, 0].set_title('Modulated Gaussian 2', fontsize=16)
    
    # Remove axis numbers and labels
    for ax in [axs[0, 0], axs[1, 0]]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    
    # Plot Spectrums
    axs[0, 1].plot(freq1[:200], spectrum1[:200], color='green')
    axs[0, 1].set_title('Spectrum 1', fontsize=16)
    
    axs[1, 1].plot(freq2[:200], spectrum2[:200], color='red')
    axs[1, 1].set_title('Spectrum 2', fontsize=16)
    
    # Remove axis numbers and labels
    for ax in [axs[0, 1], axs[1, 1]]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    
    # Adjust layout to minimize whitespace
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)
    
    plt.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def generate_fft_image(input_filename, output_filename, example_title, width=800, dpi=100):
    """
    Generates a subplot with the original image and its 2D FFT.
    Saves the image as a PNG file.
    """
    img_path = os.path.join(os.getcwd(), input_filename)
    if not os.path.exists(img_path):
        print(f"Error: {input_filename} not found in the current directory.")
        sys.exit(1)
    
    # Open image and convert to grayscale
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)
    
    # Compute 2D FFT
    fft_vals = fft2(img_array)
    fft_shifted = fftshift(fft_vals)
    magnitude = np.abs(fft_shifted)
    magnitude_log = np.log1p(magnitude)  # Logarithmic scaling
    
    # Normalize for display
    magnitude_log_normalized = (magnitude_log / np.max(magnitude_log)) * 255
    magnitude_image = Image.fromarray(magnitude_log_normalized.astype(np.uint8))
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(width / dpi, (width / dpi) * 0.75), dpi=dpi)
    
    # Set individual titles directly above each image
    axs[0].set_title(example_title, fontsize=14, pad=10)  # Title above the original image
    axs[1].set_title('2D FFT', fontsize=14, pad=10)  # Title above the FFT image
    
    # Original image
    axs[0].imshow(img, cmap='gray')
    axs[0].axis('off')
    
    # FFT image
    axs[1].imshow(magnitude_image, cmap='gray')
    axs[1].axis('off')
    
    # Adjust layout to remove top and bottom margins
    plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.15, wspace=0.05)
    
    # Remove unnecessary whitespace
    plt.tight_layout(pad=0.5)
    
    plt.savefig(os.path.join(output_dir, output_filename), dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_filename}")

def generate_all_images():
    # 1st Image: Chirp Signal with higher resolution and adjusted dimensions
    generate_chirp_image()
    
    # 2nd Image: Modulated Gaussians and Spectrums
    generate_modulated_gaussians_image()
    
    # 3rd and 4th Images: Read existing images and generate FFT plots
    input_images = [('image1.png', 'image1_fft.png', 'Example Image 1'),
                    ('image2.png', 'image2_fft.png', 'Example Image 2')]
    
    for input_file, output_file, title in input_images:
        generate_fft_image(input_file, output_file, title)

if __name__ == "__main__":
    generate_all_images()
