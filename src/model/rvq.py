import math
import torch
from torch import nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 0.001,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        scale = 1.0 / math.sqrt(latent_dim)
        self.register_buffer(
            "codebooks",
            torch.empty(num_quantizers, codebook_size, latent_dim).uniform_(
                -scale, scale
            ),
        )
        self.register_buffer("ema_counts", torch.zeros(num_quantizers, codebook_size))
        self.register_buffer(
            "ema_sums", torch.zeros(num_quantizers, codebook_size, latent_dim)
        )
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, z: torch.Tensor) -> dict:
        z_bsd = z.transpose(1, 2).contiguous()
        residual = z_bsd
        quantized = torch.zeros_like(z_bsd)
        all_indices = []
        if self.training and (not bool(self.initialized.item())):
            self._initialize_codebooks(z_bsd)
        for quantizer_idx in range(self.num_quantizers):
            codebook = self.codebooks[quantizer_idx]
            residual_before = residual
            indices = self._nearest_codebook_indices(residual_before, codebook)
            selected = F.embedding(indices, codebook)
            quantized = quantized + selected
            residual = residual - selected.detach()
            all_indices.append(indices)
            if self.training:
                self._ema_update(quantizer_idx, residual_before.detach(), indices)
        indices = torch.stack(all_indices, dim=-1)
        commitment_loss = F.mse_loss(z_bsd, quantized.detach())
        quantized_st = z_bsd + (quantized - z_bsd).detach()
        return {
            "quantized": quantized_st.transpose(1, 2).contiguous(),
            "indices": indices,
            "commitment_loss": commitment_loss,
            "codebook_perplexity": self.perplexity(indices),
        }

    @staticmethod
    def _nearest_codebook_indices(
        x: torch.Tensor, codebook: torch.Tensor
    ) -> torch.Tensor:
        flat_x = x.reshape(-1, x.shape[-1])
        distances = (
            flat_x.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_x @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        return distances.argmin(dim=1).view(*x.shape[:-1])

    @torch.no_grad()
    def _initialize_codebooks(self, z_bsd: torch.Tensor) -> None:
        residual = z_bsd.detach()
        for quantizer_idx in range(self.num_quantizers):
            vectors = self._sample_vectors(residual, self.codebook_size)
            noise = 0.0001 * torch.randn_like(vectors)
            vectors = vectors + noise
            self.codebooks[quantizer_idx].copy_(vectors)
            self.ema_sums[quantizer_idx].copy_(vectors)
            self.ema_counts[quantizer_idx].fill_(1.0)
            indices = self._nearest_codebook_indices(
                residual, self.codebooks[quantizer_idx]
            )
            selected = F.embedding(indices, self.codebooks[quantizer_idx])
            residual = residual - selected
        self.initialized.fill_(True)

    @torch.no_grad()
    def _ema_update(
        self, quantizer_idx: int, residual: torch.Tensor, indices: torch.Tensor
    ) -> None:
        flat_residual = residual.reshape(-1, residual.shape[-1])
        flat_indices = indices.reshape(-1)
        one_hot = F.one_hot(flat_indices, self.codebook_size).type_as(flat_residual)
        counts = one_hot.sum(dim=0)
        sums = one_hot.transpose(0, 1) @ flat_residual
        self.ema_counts[quantizer_idx].mul_(self.ema_decay).add_(
            counts, alpha=1.0 - self.ema_decay
        )
        self.ema_sums[quantizer_idx].mul_(self.ema_decay).add_(
            sums, alpha=1.0 - self.ema_decay
        )
        counts = self.ema_counts[quantizer_idx].clamp_min(1e-05)
        updated = self.ema_sums[quantizer_idx] / counts.unsqueeze(1)
        self.codebooks[quantizer_idx].copy_(updated)
        self._replace_dead_codes(quantizer_idx, flat_residual)

    @torch.no_grad()
    def _replace_dead_codes(
        self, quantizer_idx: int, flat_residual: torch.Tensor
    ) -> None:
        dead = self.ema_counts[quantizer_idx] < self.dead_code_threshold
        dead_indices = dead.nonzero(as_tuple=False).flatten()
        if dead_indices.numel() == 0 or flat_residual.numel() == 0:
            return
        max_replacements = max(1, self.codebook_size // 100)
        dead_indices = dead_indices[:max_replacements]
        replacement = self._sample_vectors(
            flat_residual.unsqueeze(0), dead_indices.numel()
        )
        self.codebooks[quantizer_idx, dead_indices] = replacement
        self.ema_sums[quantizer_idx, dead_indices] = replacement
        self.ema_counts[quantizer_idx, dead_indices] = 1.0

    @staticmethod
    @torch.no_grad()
    def _sample_vectors(x: torch.Tensor, num_vectors: int) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] >= num_vectors:
            indices = torch.randperm(flat.shape[0], device=flat.device)[:num_vectors]
        else:
            indices = torch.randint(flat.shape[0], (num_vectors,), device=flat.device)
        return flat[indices]

    @torch.no_grad()
    def perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        perplexities = []
        for quantizer_idx in range(indices.shape[-1]):
            counts = torch.bincount(
                indices[..., quantizer_idx].reshape(-1), minlength=self.codebook_size
            ).float()
            probs = counts / counts.sum().clamp_min(1.0)
            entropy = -(probs * (probs + 1e-10).log()).sum()
            perplexities.append(entropy.exp())
        return torch.stack(perplexities).mean()
