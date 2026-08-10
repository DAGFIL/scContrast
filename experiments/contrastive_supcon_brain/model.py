"""Encoder + projection head for contrastive learning on scRNA expression vectors."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], embed_dim: int, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            # LayerNorm (not BatchNorm): BatchNorm running stats would be estimated only on
            # augmented views during training, then mismatch clean eval-time inputs.
            layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)


class SupConModel(nn.Module):
    """Encoder backbone + projection head. Use `encoder(x)` embeddings for downstream eval."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] = (512, 256),
        embed_dim: int = 128,
        proj_hidden_dim: int = 128,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, list(hidden_dims), embed_dim, dropout)
        self.projection = ProjectionHead(embed_dim, proj_hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projection(h)
        return h, z
