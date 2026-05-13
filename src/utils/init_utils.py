import logging
import os
import random
import secrets
import shutil
import string
from pathlib import Path
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from src.utils.io_utils import ROOT_PATH


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_id(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join((secrets.choice(alphabet) for _ in range(length)))


def setup_logging(save_dir: Path) -> logging.Logger:
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("soundstream")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(save_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def setup_experiment_dir(config) -> tuple[Path, str]:
    load_dotenv(ROOT_PATH / ".env")
    run_id = config.writer.get("run_id")
    if run_id is None:
        run_id = generate_id(config.writer.get("id_length", 16))
    save_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    if save_dir.exists() and config.trainer.get("override", False):
        for child in save_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.set_struct(config, False)
    config.writer.run_id = run_id
    OmegaConf.set_struct(config, True)
    OmegaConf.save(config, save_dir / "config.yaml")
    return (save_dir, run_id)
