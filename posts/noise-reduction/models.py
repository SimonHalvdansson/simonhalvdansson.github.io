import math
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import torchaudio

from utils import HOP_LENGTH, N_FFT, WINDOW_LENGTH


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(torch.nn.Module):
    def __init__(self, in_channels=1, base_channels=16, out_channels=1, use_sigmoid=True):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        self.pool = torch.nn.MaxPool2d(2)

        self.dec4 = ConvBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)

        self.out_conv = torch.nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.use_sigmoid = use_sigmoid

    def _upsample_to(self, x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self._upsample_to(b, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self._upsample_to(d4, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self._upsample_to(d3, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self._upsample_to(d2, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        output = self.out_conv(d1)
        if self.use_sigmoid:
            output = torch.sigmoid(output)
        return output

class SpectrogramMaskUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = N_FFT
        self.unet = UNet2D(in_channels=1, base_channels=16)
        self.register_buffer("window", torch.hann_window(WINDOW_LENGTH))

    def forward(self, waveform):
        stft = torch.stft(
            waveform.squeeze(1),
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=self.window,
            return_complex=True,
        )
        mag = stft.abs()
        mag_in = mag.unsqueeze(1)
        mask = self.unet(mag_in)
        masked_stft = stft * mask.squeeze(1)

        denoised = torch.istft(
            masked_stft,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=self.window,
            length=waveform.shape[-1],
        )

        return denoised.unsqueeze(1), mask


class Spectrogram2ChannelUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = N_FFT
        self.unet = UNet2D(
            in_channels=2,
            base_channels=16,
            out_channels=2,
            use_sigmoid=False,
        )
        self.register_buffer("window", torch.hann_window(WINDOW_LENGTH))

    def forward(self, waveform):
        stft = torch.stft(
            waveform.squeeze(1),
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=self.window,
            return_complex=True,
        )
        stft_in = torch.stack([stft.real, stft.imag], dim=1)
        stft_out = self.unet(stft_in)
        out_complex = torch.complex(stft_out[:, 0], stft_out[:, 1])

        denoised = torch.istft(
            out_complex,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=self.window,
            length=waveform.shape[-1],
        )

        return denoised.unsqueeze(1), out_complex


def wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def compute_phase_gradients(phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_t = torch.zeros_like(phase)
    grad_f = torch.zeros_like(phase)

    diff_t = phase[..., 1:] - phase[..., :-1]
    diff_f = phase[:, 1:, :] - phase[:, :-1, :]

    grad_t[..., :-1] = wrap_phase(diff_t)
    grad_f[:, :-1, :] = wrap_phase(diff_f)
    return grad_t, grad_f


class SpectrogramPhaseGradientUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = N_FFT
        self.unet = UNet2D(
            in_channels=5,
            base_channels=32,
            out_channels=5,
            use_sigmoid=False,
        )
        self.register_buffer("window", torch.hann_window(WINDOW_LENGTH))

    def forward(self, waveform):
        stft = torch.stft(
            waveform.squeeze(1),
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=self.window,
            return_complex=True,
        )
        mag_in = stft.abs()
        phase = torch.angle(stft)
        phi_x_in, phi_y_in = compute_phase_gradients(phase)
        stft_in = torch.stack([mag_in,
                               torch.cos(phi_x_in),
                               torch.sin(phi_x_in),
                               torch.cos(phi_y_in),
                               torch.sin(phi_y_in)], dim=1)

        stft_out = self.unet(stft_in)

        mag = F.softplus(stft_out[:, 0])
        phi_x = torch.atan2(stft_out[:, 1], stft_out[:, 2])
        phi_y = torch.atan2(stft_out[:, 3], stft_out[:, 4])

        return {"mag": mag, "phi_x": phi_x, "phi_y": phi_y}


class SpectrogramLoss(torch.nn.Module):
    def __init__(self, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.register_buffer("window", torch.hann_window(self.win_length))

    def _stft_mag(self, waveform: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        return stft.abs()

    def forward(self, clean: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        clean_mono = clean.squeeze(1)
        denoised_mono = denoised.squeeze(1)
        clean_mag = self._stft_mag(clean_mono)
        denoised_mag = self._stft_mag(denoised_mono)
        return F.l1_loss(denoised_mag, clean_mag)


class PhaseGradientLoss(torch.nn.Module):
    def __init__(self, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, clean: torch.Tensor, pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        clean_mono = clean.squeeze(1)
        stft = torch.stft(
            clean_mono,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        target_mag = stft.abs()
        target_phase = torch.angle(stft)
        target_phi_x, target_phi_y = compute_phase_gradients(target_phase)

        pred_mag = pred["mag"]
        pred_phi_x = pred["phi_x"]
        pred_phi_y = pred["phi_y"]

        mag_loss = F.mse_loss(pred_mag, target_mag)

        delta_x = wrap_phase(pred_phi_x - target_phi_x)
        delta_y = wrap_phase(pred_phi_y - target_phi_y)
        weight = target_mag
        grad_loss = (weight * (delta_x.abs()**2 + delta_y.abs()**2)).mean()

        return mag_loss + grad_loss
