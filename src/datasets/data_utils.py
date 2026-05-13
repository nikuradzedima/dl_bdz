from itertools import repeat
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    for loader in repeat(dataloader):
        yield from loader


def get_dataloaders(config) -> dict[str, DataLoader]:
    datasets = instantiate(config.datasets)
    dataloaders = {}
    for part, dataset in datasets.items():
        is_train = part == "train"
        dataloader_config = OmegaConf.create(
            OmegaConf.to_container(config.dataloader, resolve=True)
        )
        eval_batch_size = dataloader_config.pop("eval_batch_size", None)
        if not is_train and eval_batch_size is not None:
            dataloader_config.batch_size = eval_batch_size
        dataloaders[part] = instantiate(
            dataloader_config,
            dataset=dataset,
            collate_fn=collate_fn,
            shuffle=is_train,
            drop_last=is_train,
            worker_init_fn=set_worker_seed,
        )
    return dataloaders
