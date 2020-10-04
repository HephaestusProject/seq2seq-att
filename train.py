"""
    This script was made by Nick at 19/07/20.
    To implement code for training your model.
"""
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

from src.utils import get_config, get_exp_name

pl.seed_everything(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args() -> Namespace:
    # configurations
    parser = ArgumentParser(description="Run Autoencoders")
    parser.add_argument("--model", default="AE", type=str, help="select model")

    return parser.parse_args()


def run(conf: dict):
    exp_name = get_exp_name(conf.model.params)
    wandb_logger = WandbLogger(
        name=exp_name,
        project="hephaestusproject-pytorch-AE",
        log_model=True,
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=conf.model.params.max_epochs,
        deterministic=True,
    )


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args)
    run(config)
