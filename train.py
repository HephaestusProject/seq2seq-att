"""
    This script was made by Nick at 19/07/20.
    To implement code for training your model.
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

from src.ae_module import AE, SaveCheckpointEveryNEpoch
from src.utils import get_config, get_dataloader, get_exp_name

pl.seed_everything(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args() -> Namespace:
    # configurations
    parser = ArgumentParser(description="Run Autoencoders")
    parser.add_argument("--dataset", default="mnist", type=str, help="select dataset")
    parser.add_argument("--model", default="AE", type=str, help="select model")
    parser.add_argument("--wandb", action="store_true", help="use wandb logger")

    return parser.parse_args()


def run(conf: dict, use_wandb: bool):
    # Set logger
    exp_name = get_exp_name(conf.model.params)

    if use_wandb:
        wandb_logger = WandbLogger(
            name=exp_name,
            project="hephaestusproject-pytorch-AE",
            log_model=True,
        )
    else:
        wandb_logger = None

    # Create dataloader
    train_dataloader, val_dataloader = get_dataloader(conf)

    # Create model
    runner = AE(conf.model.params)

    # Set trainer (pytorch lightening)
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=conf.model.params.max_epochs,
        deterministic=True,
        callbacks=[
            SaveCheckpointEveryNEpoch(
                n=5, file_path=Path("checkpoints"), filename_prefix=exp_name
            )
        ],
    )

    # Train
    trainer.fit(
        runner, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.dataset, args.model)
    run(config, args.wandb)
