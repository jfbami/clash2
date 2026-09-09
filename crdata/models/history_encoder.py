"""Encode recent battles and predict a player's next deck switch."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from crdata.battle_features import MODEL_FEATURE_NAMES
from crdata.sequences import SUMMARY_FEATURE_NAMES


DEFAULT_DECK_DIM = 48
DEFAULT_INPUT_DIM = DEFAULT_DECK_DIM + len(MODEL_FEATURE_NAMES)
DEFAULT_HIDDEN_DIM = 64
DEFAULT_HEAD_HIDDEN_DIM = 64
DEFAULT_DROPOUT = 0.1


class GRUHistoryEncoder(nn.Module):
    """Compress a forward sequence of battle tokens into its final hidden state."""

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use Xavier input weights and orthogonal recurrent weights per gate."""
        with torch.no_grad():
            for gate_weights in self.gru.weight_ih_l0.chunk(3, dim=0):
                nn.init.xavier_uniform_(gate_weights)
            for gate_weights in self.gru.weight_hh_l0.chunk(3, dim=0):
                nn.init.orthogonal_(gate_weights)
            nn.init.zeros_(self.gru.bias_ih_l0)
            nn.init.zeros_(self.gru.bias_hh_l0)

    def forward(self, battle_tokens: Tensor) -> Tensor:
        """Return one recent-player vector for every batch item."""
        if battle_tokens.ndim != 3:
            raise ValueError("battle_tokens must have batch, time, and feature axes")
        if battle_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} token features, "
                f"received {battle_tokens.shape[-1]}"
            )
        if not torch.is_floating_point(battle_tokens):
            raise ValueError("battle_tokens must be floating point")
        _, final_hidden = self.gru(battle_tokens)
        return final_hidden[0]


class NextSwitchHead(nn.Module):
    """Fuse recent and long-term player context into one switch logit."""

    def __init__(
        self,
        history_dim: int = DEFAULT_HIDDEN_DIM,
        summary_dim: int = len(SUMMARY_FEATURE_NAMES),
        hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if history_dim < 1 or summary_dim < 1 or hidden_dim < 1:
            raise ValueError("feature dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be at least zero and less than one")
        self.history_dim = history_dim
        self.summary_dim = summary_dim
        self.network = nn.Sequential(
            nn.Linear(history_dim + summary_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize both dense layers consistently with the existing model."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, history: Tensor, summary: Tensor) -> Tensor:
        """Return one unnormalized next-switch score per batch item."""
        self._validate_inputs(history, summary)
        return self.network(torch.cat((history, summary), dim=-1)).squeeze(-1)

    def _validate_inputs(self, history: Tensor, summary: Tensor) -> None:
        if history.ndim != 2 or summary.ndim != 2:
            raise ValueError("history and summary must have batch and feature axes")
        if history.shape[0] != summary.shape[0]:
            raise ValueError("history and summary must have the same batch size")
        if history.shape[1] != self.history_dim:
            raise ValueError(f"expected {self.history_dim} history features")
        if summary.shape[1] != self.summary_dim:
            raise ValueError(f"expected {self.summary_dim} summary features")
        if not torch.is_floating_point(history) or not torch.is_floating_point(summary):
            raise ValueError("history and summary must be floating point")


class NextSwitchModel(nn.Module):
    """Combine the causal GRU encoder with the next-switch prediction head."""

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        summary_dim: int = len(SUMMARY_FEATURE_NAMES),
        head_hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.history_encoder = GRUHistoryEncoder(input_dim, hidden_dim)
        self.switch_head = NextSwitchHead(
            history_dim=hidden_dim,
            summary_dim=summary_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    def forward(self, battle_tokens: Tensor, summary: Tensor) -> Tensor:
        """Return logits for binary cross-entropy with logits."""
        history = self.history_encoder(battle_tokens)
        return self.switch_head(history, summary)
