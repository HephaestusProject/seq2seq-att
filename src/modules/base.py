from abc import abstractmethod
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModule(pl.LightningModule):
    def __init__(self, hparams: dict) -> None:
        super(BaseModule, self).__init__()
        self.hparams = hparams

        self.encoder, self.decoder = self.get_models()

    @abstractmethod
    def get_models(self, hparams: dict):
        pass

    @abstractmethod
    def get_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        pass

    def _step(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        x, _ = batch

        loss, log = self.get_loss(x)

        return loss, log

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

    def configure_optimizers(self) -> torch.optim:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat

    def inference(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)

        return out
