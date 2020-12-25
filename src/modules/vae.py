from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.model.net import get_VAE_models
from src.modules.base import BaseModule


class VAE(BaseModule):
    def get_models(self):
        return get_VAE_models(self.hparams)

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
