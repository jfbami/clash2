"""Assemble one model token for each battle in a player history."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from crdata.battle_features import MODEL_FEATURE_NAMES


class BattleTokenAssembler(nn.Module):
    """Concatenate each deck vector with its preprocessed battle features."""

    def __init__(self, battle_feature_dim: int = len(MODEL_FEATURE_NAMES)) -> None:
        super().__init__()
        if battle_feature_dim < 1:
            raise ValueError("battle_feature_dim must be positive")
        self.battle_feature_dim = battle_feature_dim

    def forward(self, deck_vectors: Tensor, battle_features: Tensor) -> Tensor:
        """Preserve leading axes and join the final feature axes."""
        if deck_vectors.shape[:-1] != battle_features.shape[:-1]:
            raise ValueError("deck vectors and battle features must share leading axes")
        if battle_features.shape[-1] != self.battle_feature_dim:
            raise ValueError(
                f"expected {self.battle_feature_dim} battle features, "
                f"received {battle_features.shape[-1]}"
            )
        if not torch.is_floating_point(deck_vectors) or not torch.is_floating_point(
            battle_features
        ):
            raise ValueError("deck vectors and battle features must be floating point")
        return torch.cat((deck_vectors, battle_features), dim=-1)
