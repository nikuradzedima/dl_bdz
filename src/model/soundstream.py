import torch
from torch import nn
import torch.nn.functional as F
from src.model.causal import CausalConv1d, DecoderBlock, EncoderBlock
from src.model.rvq import ResidualVectorQuantizer


class Encoder(nn.Module):

    def __init__(self, channels: int = 32, latent_dim: int = 128, strides=None):
        super().__init__()
        strides = strides or [2, 4, 5, 5]
        layers = [CausalConv1d(1, channels, kernel_size=7)]
        in_channels = channels
        for i, stride in enumerate(strides, start=1):
            out_channels = channels * 2**i
            layers.append(EncoderBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        layers.append(CausalConv1d(in_channels, latent_dim, kernel_size=3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self, channels: int = 32, latent_dim: int = 128, strides=None):
        super().__init__()
        strides = strides or [2, 4, 5, 5]
        self.initial = CausalConv1d(latent_dim, channels * 16, kernel_size=7)
        blocks = []
        in_channels = channels * 16
        for i, stride in enumerate(reversed(strides)):
            out_channels = channels * 2 ** (len(strides) - i - 1)
            blocks.append(DecoderBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.final = CausalConv1d(channels, 1, kernel_size=7)

    def forward(self, z: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        x = self.initial(z)
        x = self.blocks(x)
        x = self.final(x)
        if target_len is not None:
            x = match_length(x, target_len)
        return x


def match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    current_len = x.shape[-1]
    if current_len > target_len:
        return x[..., :target_len]
    if current_len < target_len:
        return F.pad(x, (0, target_len - current_len))
    return x


class SoundStream(nn.Module):

    def __init__(
        self,
        channels: int = 32,
        latent_dim: int = 128,
        strides=None,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 0.001,
    ):
        super().__init__()
        strides = strides or [2, 4, 5, 5]
        self.encoder = Encoder(channels, latent_dim, strides)
        self.quantizer = ResidualVectorQuantizer(
            latent_dim=latent_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            ema_decay=ema_decay,
            dead_code_threshold=dead_code_threshold,
        )
        self.decoder = Decoder(channels, latent_dim, strides)

    def forward(self, audio: torch.Tensor, **_) -> dict:
        target_len = audio.shape[-1]
        z = self.encoder(audio)
        q = self.quantizer(z)
        audio_hat = self.decoder(q["quantized"], target_len=target_len)
        return {
            "audio_hat": audio_hat,
            "latents": z,
            "quantized": q["quantized"],
            "indices": q["indices"],
            "commitment_loss": q["commitment_loss"],
            "codebook_perplexity": q["codebook_perplexity"],
        }

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self.quantizer(self.encoder(audio))["indices"]

    @torch.no_grad()
    def reconstruct(self, audio: torch.Tensor) -> torch.Tensor:
        return self.forward(audio)["audio_hat"]
