import functools
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data import MNIST


def get_config(dataset: str, model: str) -> DictConfig:
    config_dir = Path("conf")
    dataset_config_filename = f"{config_dir}/dataset/{dataset}.yml"
    model_config_filename = f"{config_dir}/model/{model}.yml"

    return OmegaConf.merge(
        {"dataset": OmegaConf.load(dataset_config_filename)},
        {"model": OmegaConf.load(model_config_filename)},
    )


def get_dataloader(conf: str) -> (DataLoader, DataLoader):
    if conf.dataset.name == "mnist":
        data = MNIST(conf)
        return data.train_dataloader(), data.val_dataloader()
    else:
        raise Exception(f"Invalid dataset name: {conf.dataset.name}")


def get_exp_name(params: dict) -> str:
    param_str_list = [f"{k}_{v}" for k, v in params.items()]
    name = functools.reduce(lambda first, second: first + "-" + second, param_str_list)
    return name
