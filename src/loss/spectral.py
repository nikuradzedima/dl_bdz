import math
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks


class MultiScaleSpectralLoss(nn.Module):

    def __init__(self, fft_sizes=None, sample_rate: int = 16000, n_mels: int = 64):
        super().__init__()
        self.fft_sizes = fft_sizes or [64, 128, 256, 512, 1024, 2048]
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        for n_fft in self.fft_sizes:
            self.register_buffer(
                f"window_{n_fft}", torch.hann_window(n_fft), persistent=False
            )
            self.register_buffer(
                f"mel_{n_fft}", self._build_mel_filterbank(n_fft), persistent=False
            )

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real = real.squeeze(1)
        fake = fake.squeeze(1)
        losses = []
        for n_fft in self.fft_sizes:
            hop_length = n_fft // 4
            window = getattr(self, f"window_{n_fft}").to(dtype=real.dtype)
            real_mag = self._magnitude(real, n_fft, hop_length, window)
            fake_mag = self._magnitude(fake, n_fft, hop_length, window)
            real_mel = self._mel(real_mag, n_fft)
            fake_mel = self._mel(fake_mag, n_fft)
            mag_loss = F.l1_loss(fake_mel, real_mel)
            log_loss = F.mse_loss((fake_mel + 1e-07).log(), (real_mel + 1e-07).log())
            losses.append(mag_loss + math.sqrt(n_fft / 2.0) * log_loss)
        return torch.stack(losses).mean()

    @staticmethod
    def _magnitude(
        audio: torch.Tensor, n_fft: int, hop_length: int, window: torch.Tensor
    ) -> torch.Tensor:
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        return spec.abs()

    def _mel(self, magnitude: torch.Tensor, n_fft: int) -> torch.Tensor:
        fbanks = getattr(self, f"mel_{n_fft}").to(dtype=magnitude.dtype)
        return torch.matmul(magnitude.transpose(1, 2), fbanks).transpose(1, 2)

    def _build_mel_filterbank(self, n_fft: int) -> torch.Tensor:
        n_freqs = n_fft // 2 + 1
        n_mels = min(self.n_mels, max(1, (n_freqs - 1) * 3 // 8))
        return melscale_fbanks(
            n_freqs=n_freqs,
            f_min=0.0,
            f_max=float(self.sample_rate // 2),
            n_mels=n_mels,
            sample_rate=self.sample_rate,
            norm=None,
            mel_scale="htk",
        )
