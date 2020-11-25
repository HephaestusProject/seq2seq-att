"""
    This script was made by Nick at 19/07/20.
    To implement code of your network using operation from ops.py.
"""

from typing import Tuple

import torch
from torch import nn


class AEEncoder(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super(AEEncoder, self).__init__()
        self.hparams = hparams
        self.layers = self._build_model()

    def _build_model(self) -> nn.Sequential:
        model = nn.Sequential(
            self._conv_block(self.hparams.num_channels, 32, 4, 2),
            self._conv_block(32, 32, 4, 2),
            self._conv_block(32, 64, 3, 2),
            # self._conv_block(64, 64, 4, 2),
            nn.Flatten(),
            self._linear_block(),
        )
        return model

    def _conv_block(
        self, c_in: int, c_out: int, kernel_size: int = 4, stride: int = 2
    ) -> nn.Sequential:
        block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )
        return block

    def _linear_block(self) -> nn.Sequential:
        model = nn.Sequential(
            nn.Linear(self.hparams.encoder_output_size, self.hparams.latent_dim)
        )
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AEDecoder(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super(AEDecoder, self).__init__()
        self.hparams = hparams
        self.fc = nn.Linear(self.hparams.latent_dim, 64 * 4 * 4)
        self.layers = self._build_model()

    def _build_model(self) -> nn.Sequential:
        model = nn.Sequential(
            # self._transpose_conv_block(64, 64, 4, 2),
            self._transpose_conv_block(64, 32, 3, 2),
            self._transpose_conv_block(32, 32, 4, 2),
            self._transpose_conv_block(32, self.hparams.num_channels, 4, 2),
        )
        return model

    def _transpose_conv_block(
        self, c_in: int, c_out: int, kernel_size: int = 4, stride: int = 2
    ) -> nn.Sequential:
        block = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(-1, 64, 4, 4)
        return self.layers(x)


def get_AE_models(hparams: dict) -> Tuple[AEEncoder, AEDecoder]:
    return AEEncoder(hparams), AEDecoder(hparams)
