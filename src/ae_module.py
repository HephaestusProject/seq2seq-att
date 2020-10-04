from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.net import cnn_decoder, cnn_encoder


class AE(pl.LighteningModule):
    def __init__(self, hparams: dict) -> None:
        super(AE, self).__init__()
        self.hparams = hparams

        self.encoder = cnn_encoder(self.hparams)
        self.decoder = cnn_decoder(self.hparams)

        self.fc = nn.Linear(self.hparams.encoder_output_size, self.hparams.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        z = self.fc(features)
        x_hat = self.decoder(z)
        return x_hat

    def _step(self, batch) -> Tuple[torch.Tensor, dict]:
        x, _ = batch
        x_hat = self.forward(x)

        loss = F.mse_loss(x_hat, x, reduction="mean")

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self._step(batch)

        result = pl.TrainResult(minimize=loss)
        result.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return result

    def validation_step(self, batch, batch_idx):
        pass
