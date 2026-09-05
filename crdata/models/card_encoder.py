"""Trainable card identity embeddings for the battle-token encoder."""
from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn


class CardEmbedding(nn.Module):
    """Look up card vectors from vocabulary indices, including unknown at zero.

    Input shape is (...); output shape is (..., embedding_dim).
    Index zero is an ordinary trainable unknown row, not a padding row.
    """

    def __init__(self, embedding_rows: int, embedding_dim: int) -> None:
        super().__init__()
        if embedding_rows < 1 or embedding_dim < 1:
            raise ValueError("embedding_rows and embedding_dim must be positive")
        self.embedding = nn.Embedding(embedding_rows, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / sqrt(embedding_dim))

    def forward(self, card_indices: Tensor) -> Tensor:
        """Select one learned vector per card index without pooling cards."""
        return self.embedding(card_indices)


class CardFeatures(nn.Module):
    """Concatenate each card embedding with its standardized displayed level."""

    def __init__(
        self,
        embedding_rows: int,
        embedding_dim: int,
        level_mean: float,
        level_standard_deviation: float,
    ) -> None:
        super().__init__()
        if level_standard_deviation <= 0.0:
            raise ValueError("level_standard_deviation must be positive")
        self.card_embedding = CardEmbedding(embedding_rows, embedding_dim)
        self.register_buffer("level_mean", torch.tensor(float(level_mean)))
        self.register_buffer(
            "level_standard_deviation",
            torch.tensor(float(level_standard_deviation)),
        )

    def forward(self, card_indices: Tensor, displayed_levels: Tensor) -> Tensor:
        """Return one feature vector per card without pooling the card axis."""
        if card_indices.shape != displayed_levels.shape:
            raise ValueError("card_indices and displayed_levels must have the same shape")
        standardized_levels = (
            displayed_levels.float() - self.level_mean
        ) / self.level_standard_deviation
        return torch.cat(
            (self.card_embedding(card_indices), standardized_levels.unsqueeze(-1)),
            dim=-1,
        )


class CardMLP(nn.Module):
    """Apply one shared nonlinear transformation to every card feature vector."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, card_features: Tensor) -> Tensor:
        """Transform the final feature axis while preserving all leading axes."""
        return self.network(card_features)
