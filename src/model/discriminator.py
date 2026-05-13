import torch
from torch import nn
import torch.nn.functional as F


class ScaleDiscriminator(nn.Module):

    def __init__(self, base_channels: int = 32):
        super().__init__()
        channels = [
            base_channels,
            base_channels * 4,
            base_channels * 16,
            min(base_channels * 64, 1024),
            min(base_channels * 64, 1024),
        ]
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(1, channels[0], kernel_size=15, padding=7),
                nn.Conv1d(
                    channels[0],
                    channels[1],
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=4,
                ),
                nn.Conv1d(
                    channels[1],
                    channels[2],
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=16,
                ),
                nn.Conv1d(
                    channels[2],
                    channels[3],
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=64,
                ),
                nn.Conv1d(channels[3], channels[4], kernel_size=5, padding=2),
                nn.Conv1d(channels[4], 1, kernel_size=3, padding=1),
            ]
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            features.append(x)
            if layer_idx != len(self.layers) - 1:
                x = self.activation(x)
        return features


class MultiScaleWaveDiscriminator(nn.Module):

    def __init__(self, base_channels: int = 32, scales: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [ScaleDiscriminator(base_channels) for _ in range(scales)]
        )

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        outputs = []
        current = x
        for idx, discriminator in enumerate(self.discriminators):
            if idx > 0:
                current = F.avg_pool1d(current, kernel_size=4, stride=2, padding=1)
            outputs.append(discriminator(current))
        return outputs


class STFTDiscriminator(nn.Module):

    def __init__(
        self, base_channels: int = 32, n_fft: int = 1024, hop_length: int = 256
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        channels = [
            base_channels,
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8,
        ]
        layers = [nn.Conv2d(2, channels[0], kernel_size=7, padding=3)]
        strides = [(1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)]
        for i, stride in enumerate(strides):
            layers.append(
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=(3, 4),
                    stride=stride,
                    padding=(1, 1),
                )
            )
        layers.append(nn.Conv2d(channels[-1], 1, kernel_size=(1, 8)))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = x.squeeze(1)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )
        spec = spec.transpose(1, 2)
        features = torch.stack([spec.real, spec.imag], dim=1)
        outputs = []
        for layer_idx, layer in enumerate(self.layers):
            features = layer(features)
            outputs.append(features)
            if layer_idx != len(self.layers) - 1:
                features = self.activation(features)
        return outputs


class SoundStreamDiscriminator(nn.Module):

    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.wave = MultiScaleWaveDiscriminator(base_channels)
        self.stft = STFTDiscriminator(base_channels)

    def forward(self, audio: torch.Tensor) -> list[list[torch.Tensor]]:
        return self.wave(audio) + [self.stft(audio)]
