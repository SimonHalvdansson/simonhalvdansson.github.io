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
        self.use_sigmoid = bool(use_sigmoid)

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
        self.unet = UNet2D(
            in_channels=1,
            base_channels=16,
        )
        self.register_buffer("window", torch.hann_window(WINDOW_LENGTH))

    def forward(self, waveform):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        mono = waveform.mean(dim=1, keepdim=True)
        stft = torch.stft(
            mono.squeeze(1),
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=self.window,
            return_complex=True,
        )
        mag = stft.abs()
        mag_in = mag.unsqueeze(1)
        mask = self.unet(mag_in)
        masked_mag = mag * mask.squeeze(1)
        phase = torch.angle(stft)
        masked_stft = torch.polar(masked_mag, phase)

        denoised = torch.istft(
            masked_stft,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=self.window,
            length=mono.shape[-1],
        )

        return denoised.unsqueeze(1), mask


class SpectrogramUNet(torch.nn.Module):
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
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        mono = waveform.mean(dim=1, keepdim=True)
        stft = torch.stft(
            mono.squeeze(1),
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
            length=mono.shape[-1],
        )

        return denoised.unsqueeze(1), out_complex


class SpectrogramLoss(torch.nn.Module):
    def __init__(self, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.register_buffer("window", torch.hann_window(self.win_length))

    def _to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            return waveform.mean(dim=1)
        if waveform.dim() == 2:
            return waveform
        if waveform.dim() == 1:
            return waveform.unsqueeze(0)
        raise ValueError(f"Expected waveform with 1-3 dims, got shape {tuple(waveform.shape)}")

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
        clean_mono = self._to_mono(clean)
        denoised_mono = self._to_mono(denoised)
        clean_mag = self._stft_mag(clean_mono)
        denoised_mag = self._stft_mag(denoised_mono)
        return F.l1_loss(denoised_mag, clean_mag)
