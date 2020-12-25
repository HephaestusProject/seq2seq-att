import functools
from typing import Tuple

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data import MNIST


class Config:
    """Config: yaml parser using OmegaConf"""

    def __init__(self) -> None:
        self.configs = OmegaConf.create()

    def add_dataset(self, config_path: str) -> None:
        self.configs.update({"dataset": OmegaConf.load(config_path)})

    def add_model(self, config_path: str) -> None:
        self.configs.update({"model": OmegaConf.load(config_path)})

    def add_api(self, config_path: str) -> None:
        self.configs.update({"api": OmegaConf.load(config_path)})

    def add_trainer(self, config_path: str) -> None:
        self.configs.update({"trainer": OmegaConf.load(config_path)})

    def merge(self: str) -> DictConfig:
        return OmegaConf.merge(self.configs)

    @property
    def dataset(self):
        return self.configs.dataset

    @property
    def model(self):
        return self.configs.model

    @property
    def api(self):
        return self.configs.api

    @property
    def trainer(self):
        return self.configs.trainer


def get_dataloader(conf: str) -> Tuple[DataLoader, DataLoader]:
    if conf.dataset.name == "mnist":
        data = MNIST(conf)
        return data.train_dataloader(), data.val_dataloader()
    else:
        raise Exception(f"Invalid dataset name: {conf.dataset.name}")


def get_exp_name(params: dict) -> str:
    param_str_list = [f"{k}_{v}" for k, v in params.items()]
    name = functools.reduce(lambda first, second: first + "-" + second, param_str_list)
    return name
