from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.model.net import get_AE_models
from src.modules.base import BaseModule


class AE(BaseModule):
    def get_models(self):
        return get_AE_models(self.hparams)

    def get_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction="mean")

        return loss, {"loss": loss}
