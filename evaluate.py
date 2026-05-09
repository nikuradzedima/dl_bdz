import hydra
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.datasets.collate import collate_fn
from src.utils.io_utils import resolve_path


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)


@hydra.main(version_base=None, config_path="src/configs", config_name="soundstream")
def main(config):
    if config.trainer.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.trainer.device)
    dataset = instantiate(config.datasets.test)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataloader.get(
            "eval_batch_size", config.dataloader.batch_size
        ),
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        collate_fn=collate_fn,
    )
    model = instantiate(config.model).to(device)
    load_checkpoint(model, resolve_path(config.evaluation.checkpoint_path), device)
    model.eval()
    metrics = instantiate(config.metrics)["inference"]
    totals = {metric.name: 0.0 for metric in metrics}
    counts = {metric.name: 0 for metric in metrics}
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc="evaluate", dynamic_ncols=True, mininterval=10.0
        ):
            audio = batch["audio"].to(device)
            lengths = batch["lengths"].to(device)
            outputs = model(audio)
            for metric in metrics:
                value = metric(audio=audio, lengths=lengths, **outputs)
                if value == value:
                    totals[metric.name] += value
                    counts[metric.name] += 1
    for name, total in totals.items():
        value = total / max(counts[name], 1)
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
