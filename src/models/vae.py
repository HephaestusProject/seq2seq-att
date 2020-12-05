from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.net import get_VAE_models


class VAE(pl.LightningModule):
    def __init__(self, hparams: dict) -> None:
        super(VAE, self).__init__()
        self.hparams = hparams

        self.encoder, self.decoder = get_VAE_models(self.hparams)

    def configure_optimizers(self) -> torch.optim:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _, _ = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat

    def get_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        z, mu, log_var = self.encoder(x)
        x_hat = self.decoder(z)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
        loss = recon_loss + kld_loss

        log = {
            "recon_loss": recon_loss,
            "kld_loss": kld_loss,
            "loss": loss,
        }
        return loss, log

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
