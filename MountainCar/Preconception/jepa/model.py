import copy
from typing import Literal, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            if d_out != out_dim:
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JEPA(pl.LightningModule):
    """
    Simple JEPA-style model for:
        (obs_{t-1}, obs_t, action_t) -> obs_{t+1}

    The model works in representation space:
        - context_encoder encodes [obs_{t-1}, obs_t]
        - target_encoder encodes obs_{t+1} (EMA copy of context_encoder)
        - predictor maps (context_embedding, action_t) to target_embedding

    Loss is MSE between predicted and target embeddings.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        ema_tau: float = 0.99,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        lr_scheduler: Literal["cosine", "plateau"] = "plateau",
        cosine_T_max: int = 1_000_000,
        cosine_eta_min: float = 1e-6,
        plateau_factor: float = 0.5,
        plateau_patience: int = 10,
        plateau_min_lr: float = 1e-6,
    ) -> None:
        """
        Args:
            obs_dim: Dimension of a single observation vector.
            action_dim: Dimension of the action vector at time t.
            embed_dim: Size of latent representation.
            hidden_dim: Hidden size for MLPs.
            lr: Adam learning rate.
            ema_tau: EMA coefficient for target encoder update.
            weight_decay: AdamW weight decay (stronger helps against overfitting).
            dropout: Dropout probability in encoder/predictor MLPs (0 = disabled).
            lr_scheduler: "cosine" for CosineAnnealingLR, "plateau" for ReduceLROnPlateau.
            cosine_T_max: Max steps for cosine schedule (only if lr_scheduler=="cosine").
            cosine_eta_min: Min LR for cosine schedule (only if lr_scheduler=="cosine").
            plateau_factor: Factor by which LR is reduced (only if lr_scheduler=="plateau").
            plateau_patience: Epochs without improvement before reducing LR (only if lr_scheduler=="plateau").
            plateau_min_lr: Minimum LR for plateau schedule (only if lr_scheduler=="plateau").
        """
        super().__init__()
        self.save_hyperparameters()

        in_ctx_dim = obs_dim * 2  # [obs_{t-1}, obs_t]

        # Context encoder: [o_{t-1}, o_t] -> z_ctx
        self.context_encoder = MLP(
            in_dim=in_ctx_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            num_layers=3,
            dropout=dropout,
        )

        # Predictor: concat(z_ctx, a_t) -> z_pred
        self.predictor = MLP(
            in_dim=embed_dim + action_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            num_layers=3,
            dropout=dropout,
        )

        # Target encoder: [o_t, o_{t+1}] -> z_target (EMA copy of context encoder)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.ema_tau = ema_tau
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.cosine_T_max = cosine_T_max
        self.cosine_eta_min = cosine_eta_min
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience
        self.plateau_min_lr = plateau_min_lr

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        tau = self.ema_tau
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_tgt.data.lerp_(p_ctx.data, 1.0 - tau)

    def encode_context(self, obs_tm1: torch.Tensor, obs_t: torch.Tensor) -> torch.Tensor:
        """
        Encode (obs_{t-1}, obs_t) into latent context representation.

        Shapes:
            obs_tm1: (B, obs_dim)
            obs_t:   (B, obs_dim)
            returns: (B, embed_dim)
        """
        x = torch.cat([obs_tm1, obs_t], dim=-1)
        return self.context_encoder(x)

    def encode_target(self, obs_t: torch.Tensor, obs_tp1: torch.Tensor) -> torch.Tensor:
        """
        Encode (obs_t, obs_{t+1}) into latent target representation.

        Shapes:
            obs_t:   (B, obs_dim)
            obs_tp1: (B, obs_dim)
            returns: (B, embed_dim)
        """
        x = torch.cat([obs_t, obs_tp1], dim=-1)
        return self.target_encoder(x)

    def forward(
        self,
        obs_tm1: torch.Tensor,
        obs_t: torch.Tensor,
        action_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returning predicted target embedding.

        Inputs (all shape (B, dim)):
            obs_tm1: observation at t-1
            obs_t:   observation at t
            action_t: action at t

        Returns:
            z_pred: predicted embedding for obs_{t+1} (B, embed_dim)
        """
        z_ctx = self.encode_context(obs_tm1, obs_t)
        z_in = torch.cat([z_ctx, action_t], dim=-1)
        z_pred = self.predictor(z_in)
        return z_pred

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Expects batch = (obs_tm1, obs_t, action_t, obs_tp1).

        Each tensor has shape (B, dim).
        """
        obs_tm1, obs_t, action_t, obs_tp1 = batch

        # Predicted embedding for obs_{t+1}
        z_pred = self(obs_tm1, obs_t, action_t)

        # Target embedding from EMA encoder
        with torch.no_grad():
            z_target = self.encode_target(obs_t, obs_tp1)

        loss = F.mse_loss(z_pred, z_target)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False, on_step=True, on_epoch=False)

        # EMA update after computing gradients
        self._update_target_encoder()

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        obs_tm1, obs_t, action_t, obs_tp1 = batch
        z_pred = self(obs_tm1, obs_t, action_t)
        with torch.no_grad():
            z_target = self.encode_target(obs_t, obs_tp1)
        loss = F.mse_loss(z_pred, z_target)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cosine_T_max, eta_min=self.cosine_eta_min
            )
            lr_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        else:  # "plateau"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.plateau_factor,
                patience=self.plateau_patience,
                min_lr=self.plateau_min_lr,
            )
            lr_config = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        return {"optimizer": optimizer, "lr_scheduler": lr_config}