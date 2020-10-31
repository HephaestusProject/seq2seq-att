"""
    This script was made by Nick at 19/07/20.
    To implement code for inference with your model.
"""
from argparse import ArgumentParser, Namespace
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from src.ae_module import AE
from src.utils import get_config, get_dataloader

pl.seed_everything(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args() -> Namespace:
    # configurations
    parser = ArgumentParser(description="Run Autoencoders")
    parser.add_argument("--dataset", default="mnist", type=str, help="select dataset")
    parser.add_argument("--model", default="AE", type=str, help="select model")

    return parser.parse_args()


def show_result(input_img, output_img):
    fig = plt.figure()
    rows = 1
    cols = 2

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(input_img)
    ax1.set_title("Input")
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(output_img)
    ax2.set_title("Ouput")
    ax2.axis("off")

    plt.show()


def run(conf: dict):
    # Load checkpoint
    checkpoint_path = os.path.join(conf.model.ckpt.path, conf.model.ckpt.filename)
    model = AE.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams=conf.model.params,
    )

    # Select test image
    _, val_dataloader = get_dataloader(conf)
    test_image = None
    for data in val_dataloader:
        images, _ = data
        test_image = images[0, :, :, :].unsqueeze(0)
        break

    # Inference
    x = torch.Tensor(test_image)
    y = model(x)
    output = np.transpose(y[0].cpu().detach().numpy(), [1, 2, 0])
    test_image = np.transpose(test_image[0, :, :, :].cpu().numpy(), [1, 2, 0])

    show_result(test_image, output)


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.dataset, args.model)
    run(config)
