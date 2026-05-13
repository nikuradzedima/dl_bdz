import warnings
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.datasets.data_utils import get_dataloaders
from src.trainer import SoundStreamTrainer
from src.utils.init_utils import set_random_seed, setup_experiment_dir, setup_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="soundstream")
def main(config):
    set_random_seed(config.trainer.seed)
    save_dir, _ = setup_experiment_dir(config)
    logger = setup_logging(save_dir)
    project_config = OmegaConf.to_container(config, resolve=True)
    writer = instantiate(
        config.writer, logger=logger, project_config=project_config, _recursive_=False
    )
    if config.trainer.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.trainer.device)
    logger.info(f"используем device: {device}")
    dataloaders = get_dataloaders(config)
    model = instantiate(config.model).to(device)
    discriminator = instantiate(config.discriminator).to(device)
    criterion = instantiate(config.loss).to(device)
    metrics = instantiate(config.metrics)
    optimizer_g = instantiate(config.optimizer_g, params=model.parameters())
    optimizer_d = instantiate(config.optimizer_d, params=discriminator.parameters())
    trainer = SoundStreamTrainer(
        model=model,
        discriminator=discriminator,
        criterion=criterion,
        metrics=metrics,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        save_dir=save_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
