from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.net import get_AE_models


class AE(pl.LightningModule):
    def __init__(self, hparams: dict) -> None:
        super(AE, self).__init__()
        self.hparams = hparams

        self.encoder, self.decoder = get_AE_models(self.hparams)

    def configure_optimizers(self) -> torch.optim:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
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
        loss, logs = self._step(batch)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({f"val_{k}": v for k, v in logs.items()})
        return result
