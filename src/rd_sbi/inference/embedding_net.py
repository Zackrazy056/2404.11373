"""Embedding + NSF density-estimator builder for SNPE/TSNPE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sbi.neural_nets import posterior_nn
from torch import nn


@dataclass(frozen=True)
class EmbeddingConfig:
    """FC embedding architecture."""

    input_dim: int = 408
    num_hidden_layers: int = 2
    hidden_dim: int = 150
    output_dim: int = 128


class EmbeddingFCNet(nn.Module):
    """Fully-connected embedding network used before the NSF posterior."""

    def __init__(
        self,
        input_dim: int = 408,
        num_hidden_layers: int = 2,
        hidden_dim: int = 150,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, hidden_dim and output_dim must be positive")
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.network(x)


def build_nsf_density_estimator(
    *,
    embedding_config: EmbeddingConfig | None = None,
    hidden_features: int = 150,
    num_transforms: int = 5,
    num_bins: int = 10,
    num_blocks: int = 2,
    z_score_theta: str | None = "independent",
    z_score_x: str | None = "independent",
    batch_norm: bool = True,
):
    """Return a density-estimator builder callable for `sbi.inference.SNPE`.

    Uses an NSF posterior with paper-aligned hyperparameters.
    """
    cfg = embedding_config or EmbeddingConfig()
    embedding_net = EmbeddingFCNet(
        input_dim=cfg.input_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
    )

    return posterior_nn(
        model="nsf",
        z_score_theta=z_score_theta,
        z_score_x=z_score_x,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_bins=num_bins,
        embedding_net=embedding_net,
        num_blocks=num_blocks,
        batch_norm=batch_norm,
    )
