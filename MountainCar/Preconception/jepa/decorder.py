"""Decoder: maps JEPA features (z_pred) to observations."""

import torch
import torch.nn as nn


class ObsDecoder(nn.Module):
    """
    Decoder that maps predicted JEPA embedding z_pred to full observation.
    z_pred has shape (B, embed_dim), output has shape (B, obs_dim).
    """

    def __init__(
        self,
        embed_dim: int,
        obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = [embed_dim] + [hidden_dim] * (num_layers - 1) + [obs_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
