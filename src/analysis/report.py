from pathlib import Path
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from src.utils.audio_io import load_audio
from src.utils.io_utils import resolve_path


def _load_model(checkpoint_path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(
        resolve_path(checkpoint_path), map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    model = instantiate(config.model).to(device)
    sample_rate = int(config.trainer.sample_rate)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return (model, sample_rate)


def _load_audio(path: str | Path, sample_rate: int):
    return load_audio(resolve_path(path), sample_rate)


@torch.no_grad()
def resynthesize_file(
    audio_path: str | Path,
    checkpoint_path: str | Path = "checkpoints/model_best.pth",
    device: str = "cpu",
):
    model, sample_rate = _load_model(checkpoint_path, device=device)
    audio = _load_audio(audio_path, sample_rate).unsqueeze(0).to(device)
    reconstructed = model(audio)["audio_hat"].squeeze(0).cpu()
    return (audio.squeeze(0).cpu(), reconstructed, sample_rate)


def plot_waveform_and_spectrogram(
    audio_path: str | Path,
    checkpoint_path: str | Path = "checkpoints/model_best.pth",
    device: str = "cpu",
):
    original, reconstructed, sample_rate = resynthesize_file(
        audio_path, checkpoint_path, device
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    titles = ["Оригинал", "Восстановленное аудио"]
    for column, audio in enumerate([original, reconstructed]):
        waveform = audio[0].numpy()
        axes[0, column].plot(waveform)
        axes[0, column].set_title(f"{titles[column]} waveform")
        spec = torch.stft(
            audio[0],
            n_fft=512,
            hop_length=128,
            window=torch.hann_window(512),
            return_complex=True,
        ).abs()
        axes[1, column].imshow(
            (spec + 1e-07).log().numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        axes[1, column].set_title(f"{titles[column]} log STFT")
    fig.suptitle(f"Sample rate: {sample_rate} Hz")
    fig.tight_layout()
    return fig
