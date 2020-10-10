from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from src.model.net import cnn_decoder, cnn_encoder


class SaveCheckpointEveryNEpoch(pl.Callback):
    def __init__(self, file_path: str, n: int = 1, filename_prefix: str = "") -> None:
        self.n = n
        self.file_path = file_path
        self.filename_prefix = filename_prefix

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if epoch % self.n == 0:
            # save models
            filename = f"{self.filename_prefix}_epoch_{epoch}.ckpt"
            ckpt_path = f"{self.file_path}/{filename}"
            torch.save(
                {
                    "encoder": pl_module.encoder.state_dict(),
                    "encoder_to_decoder_fc": pl_module.fc.state_dict(),
                    "decoder": pl_module.decoder.state_dict(),
                },
                ckpt_path,
            )


class AE(pl.LightningModule):
    def __init__(self, hparams: dict) -> None:
        super(AE, self).__init__()
        self.hparams = hparams

        self.encoder = cnn_encoder(self.hparams)
        self.decoder = cnn_decoder(self.hparams)

        self.fc = nn.Linear(self.hparams.encoder_output_size, self.hparams.latent_dim)

    def configure_optimizers(self) -> torch.optim:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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
        loss, logs = self._step(batch)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({f"val_{k}": v for k, v in logs.items()})
        return result
