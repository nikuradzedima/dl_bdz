import torch
import torch.nn.functional as F


def collate_fn(batch: list[dict]) -> dict:
    max_len = max((item["audio"].shape[-1] for item in batch))
    audios = []
    lengths = []
    paths = []
    for item in batch:
        audio = item["audio"]
        length = audio.shape[-1]
        if length < max_len:
            audio = F.pad(audio, (0, max_len - length))
        audios.append(audio)
        lengths.append(item["length"])
        paths.append(item["path"])
    return {
        "audio": torch.stack(audios, dim=0),
        "lengths": torch.stack(lengths, dim=0),
        "paths": paths,
    }
