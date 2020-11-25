"""
    This script was made by Nick at 19/07/20.
    To implement code for training your model.
"""
from argparse import ArgumentParser, Namespace
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch

from src.ae_module import AE
from src.utils import Config, get_dataloader, get_exp_name

pl.seed_everything(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args() -> Namespace:
    # configurations
    parser = ArgumentParser(description="Run Autoencoders")
    parser.add_argument(
        "--cfg-dataset",
        default="./configs/dataset/mnist.yml",
        type=str,
        help="select dataset",
    )
    parser.add_argument(
        "--cfg-model", default="./configs/model/AE.yml", type=str, help="select model"
    )
    parser.add_argument("--wandb", action="store_true", help="use wandb logger")

    return parser.parse_args()


def run(cfg: dict, use_wandb: bool):
    # Set logger
    exp_name = get_exp_name(cfg.model.params)

    if use_wandb:
        wandb_logger = WandbLogger(
            name=exp_name,
            project="hephaestusproject-pytorch-AE",
            log_model=True,
        )
    else:
        wandb_logger = None

    # Create dataloader
    train_dataloader, val_dataloader = get_dataloader(cfg)

    # Create model
    runner = AE(cfg.model.params)

    # Set trainer (pytorch lightening)
    os.makedirs(cfg.model.ckpt.path, exist_ok=True)
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=cfg.model.params.max_epochs,
        deterministic=True,
        checkpoint_callback=ModelCheckpoint(cfg.model.ckpt.path),
    )

    # Train
    trainer.fit(
        runner, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    args = parse_args()

    cfg = Config()
    cfg.add_dataset(args.cfg_dataset)
    cfg.add_model(args.cfg_model)

    run(cfg, args.wandb)
