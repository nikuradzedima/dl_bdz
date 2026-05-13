from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.utils.audio_io import load_audio
from src.utils.io_utils import resolve_path


class LibriSpeechCodecDataset(Dataset):
    AUDIO_EXTENSIONS = ("*.flac", "*.wav", "*.ogg", "*.mp3")

    def __init__(
        self,
        root: str,
        split: str,
        sample_rate: int = 16000,
        crop_length: int | None = 8000,
        training: bool = True,
        max_items: int | None = None,
    ):
        self.root = resolve_path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.crop_length = crop_length
        self.training = training
        split_dir = self.root / self.split
        files = []
        for pattern in self.AUDIO_EXTENSIONS:
            files.extend(split_dir.rglob(pattern))
        self.files = sorted(files)
        if max_items is not None:
            self.files = self.files[:max_items]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        waveform = load_audio(path, self.sample_rate)
        if self.training and self.crop_length is not None:
            waveform = self._random_crop_or_pad(waveform, self.crop_length)
        return {
            "audio": waveform,
            "length": torch.tensor(waveform.shape[-1], dtype=torch.long),
            "path": str(path),
        }

    @staticmethod
    def _random_crop_or_pad(waveform: torch.Tensor, crop_length: int) -> torch.Tensor:
        length = waveform.shape[-1]
        if length == crop_length:
            return waveform
        if length > crop_length:
            start = torch.randint(0, length - crop_length + 1, (1,)).item()
            return waveform[:, start : start + crop_length]
        pad = crop_length - length
        if length == 0:
            return waveform.new_zeros(1, crop_length)
        return F.pad(waveform.unsqueeze(0), (0, pad), mode="replicate").squeeze(0)
