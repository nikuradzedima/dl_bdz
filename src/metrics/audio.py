import numpy as np
import torch
from pystoi import stoi
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment


class STOIMetric:
    name = "STOI"

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def __call__(
        self, audio: torch.Tensor, audio_hat: torch.Tensor, lengths=None, **_
    ) -> float:
        scores = []
        audio = audio.detach().cpu()
        audio_hat = audio_hat.detach().cpu()
        if lengths is None:
            lengths = torch.full((audio.shape[0],), audio.shape[-1], dtype=torch.long)
        for real, fake, length in zip(audio, audio_hat, lengths.cpu()):
            length = int(length.item())
            real_np = real[0, :length].numpy()
            fake_np = fake[0, :length].numpy()
            scores.append(
                float(stoi(real_np, fake_np, self.sample_rate, extended=False))
            )
        return float(np.mean(scores))


class NISQAMetric:
    name = "NISQA"

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.metric = NonIntrusiveSpeechQualityAssessment(fs=sample_rate)

    def __call__(self, audio_hat: torch.Tensor, lengths=None, **_) -> float:
        scores = []
        preds = audio_hat.detach().cpu()
        if lengths is None:
            lengths = torch.full((preds.shape[0],), preds.shape[-1], dtype=torch.long)
        for pred, length in zip(preds, lengths.cpu()):
            length = int(length.item())
            score = self.metric(pred[0, :length].unsqueeze(0))
            scores.append(float(score.detach().cpu().mean().item()))
        return float(np.mean(scores))
