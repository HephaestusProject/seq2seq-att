"""
    This script was made by Nick at 19/07/20.
    To implement code of your network using operation from ops.py.
"""

from typing import List, Tuple

import torch
from torch import nn


def conv_block(
    c_in: int, c_out: int, kernel_size: int = 4, stride: int = 2
) -> nn.Sequential:
    block = nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size, stride, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )
    return block


def transpose_conv_block(
    c_in: int, c_out: int, kernel_size: int = 4, stride: int = 2
) -> nn.Sequential:
    block = nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )
    return block


def linear_block(input_dim: int, output_dim: int) -> nn.Sequential:
    block = nn.Sequential(nn.Linear(input_dim, output_dim))
    return block


class AEEncoder(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super(AEEncoder, self).__init__()
        self.hparams = hparams
        self.conv = self._build_conv_block()
        self.fc = linear_block(hparams.encoder_output_size, hparams.latent_dim)

    def _build_conv_block(self) -> nn.Sequential:
        block = nn.Sequential(
            conv_block(self.hparams.num_channels, 32, 4, 2),
            conv_block(32, 32, 4, 2),
            conv_block(32, 64, 3, 2),
            # self._conv_block(64, 64, 4, 2),
            nn.Flatten(),
        )
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return x


class VAEEncoder(AEEncoder):
    def __init__(self, hparams: dict) -> None:
        super(VAEEncoder, self).__init__()
        self.hparams = hparams
        self.conv = self._build_conv_block()
        self.fc_mu = linear_block(hparams.encoder_output_size, hparams.latent_dim)
        self.fc_var = linear_block(hparams.encoder_output_size, hparams.latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv(x)
        mu, log_var = self.fc_mu(x), self.fc_var(x)
        z = self.reparameterize(mu, log_var)
        return [z, mu, log_var]


class Decoder(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super(Decoder, self).__init__()
        self.hparams = hparams
        self.fc = nn.Linear(self.hparams.latent_dim, 64 * 4 * 4)
        self.transpose_conv = self._build_transpose_conv_block()

    def _build_transpose_conv_block(self) -> nn.Sequential:
        model = nn.Sequential(
            # self._transpose_conv_block(64, 64, 4, 2),
            transpose_conv_block(64, 32, 3, 2),
            transpose_conv_block(32, 32, 4, 2),
            transpose_conv_block(32, self.hparams.num_channels, 4, 2),
        )
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(-1, 64, 4, 4)
        return self.transpose_conv(x)


def get_AE_models(hparams: dict) -> Tuple[AEEncoder, Decoder]:
    return AEEncoder(hparams), Decoder(hparams)


def get_VAE_models(hparams: dict) -> Tuple[VAEEncoder, Decoder]:
    return VAEEncoder(hparams), Decoder(hparams)
