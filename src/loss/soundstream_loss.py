import torch
from torch import nn
import torch.nn.functional as F
from src.loss.spectral import MultiScaleSpectralLoss


class SoundStreamLoss(nn.Module):

    def __init__(
        self,
        sample_rate: int = 16000,
        spectral_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        feature_matching_weight: float = 100.0,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spectral = MultiScaleSpectralLoss(sample_rate=sample_rate)
        self.spectral_weight = spectral_weight
        self.adversarial_weight = adversarial_weight
        self.feature_matching_weight = feature_matching_weight
        self.commitment_weight = commitment_weight

    def discriminator_loss(
        self,
        real_outputs: list[list[torch.Tensor]],
        fake_outputs: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        losses = []
        for real_features, fake_features in zip(real_outputs, fake_outputs):
            real_logits = real_features[-1]
            fake_logits = fake_features[-1]
            losses.append(F.relu(1.0 - real_logits).mean())
            losses.append(F.relu(1.0 + fake_logits).mean())
        return torch.stack(losses).mean()

    def generator_loss(
        self,
        real_audio: torch.Tensor,
        fake_audio: torch.Tensor,
        real_outputs: list[list[torch.Tensor]],
        fake_outputs: list[list[torch.Tensor]],
        commitment_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        spectral_loss = self.spectral(real_audio, fake_audio)
        adversarial_loss = self.adversarial_loss(fake_outputs)
        feature_matching_loss = self.feature_matching_loss(real_outputs, fake_outputs)
        total = (
            self.spectral_weight * spectral_loss
            + self.adversarial_weight * adversarial_loss
            + self.feature_matching_weight * feature_matching_loss
            + self.commitment_weight * commitment_loss
        )
        return {
            "generator_loss": total,
            "spectral_loss": spectral_loss,
            "adversarial_loss": adversarial_loss,
            "feature_matching_loss": feature_matching_loss,
            "commitment_loss": commitment_loss,
        }

    @staticmethod
    def adversarial_loss(fake_outputs: list[list[torch.Tensor]]) -> torch.Tensor:
        losses = []
        for fake_features in fake_outputs:
            fake_logits = fake_features[-1]
            losses.append(F.relu(1.0 - fake_logits).mean())
        return torch.stack(losses).mean()

    @staticmethod
    def feature_matching_loss(
        real_outputs: list[list[torch.Tensor]], fake_outputs: list[list[torch.Tensor]]
    ) -> torch.Tensor:
        losses = []
        for real_features, fake_features in zip(real_outputs, fake_outputs):
            for real_feature, fake_feature in zip(
                real_features[:-1], fake_features[:-1]
            ):
                losses.append(F.l1_loss(fake_feature, real_feature.detach()))
        return torch.stack(losses).mean()
