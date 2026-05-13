from pathlib import Path
from urllib.request import urlretrieve
import hydra
import torch
from hydra.utils import instantiate
from src.model.soundstream import match_length
from src.utils.audio_io import load_audio, save_audio
from src.utils.io_utils import ROOT_PATH, resolve_path


def load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    checkpoint_path = resolve_path(config.checkpoint_path)
    output_path = resolve_path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.input_url is not None:
        input_path = ROOT_PATH / "outputs" / "input_from_url.wav"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(config.input_url, input_path)
    elif config.input_path is not None:
        input_path = resolve_path(config.input_path)
    model = instantiate(config.model).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    audio = load_audio(input_path, config.sample_rate).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed = model(audio)["audio_hat"]
        reconstructed = match_length(reconstructed, audio.shape[-1])
    save_audio(output_path, reconstructed.squeeze(0), config.sample_rate)
    print(f"Восстановленное аудио сохранено в {output_path}")


if __name__ == "__main__":
    main()
