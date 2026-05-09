from pathlib import Path
import soundfile as sf
import torch
import torchaudio


def load_audio(path, sample_rate):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = torch.from_numpy(data.T)
    audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio


def save_audio(path, audio, sample_rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().cpu().float()
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio[0]
    audio = audio.clamp(-1.0, 1.0).numpy()
    sf.write(path, audio, sample_rate)
