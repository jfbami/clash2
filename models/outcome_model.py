"""Action-conditioned model for next-battle win prediction."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from crdata.sequences import SUMMARY_FEATURE_NAMES
from models.battle_encoder import BattleTokenAssembler
from models.card_encoder import CardFeatures, CardMLP, DeckMLP, SumDeckPool
from models.history_encoder import DEFAULT_HIDDEN_DIM, DEFAULT_INPUT_DIM, GRUHistoryEncoder


DEFAULT_OUTCOME_HEAD_HIDDEN_DIM = 128
DEFAULT_OUTCOME_DROPOUT = 0.1


class ActionConditionedWinHead(nn.Module):
    """Predict a win from player context and one hypothetical deck action."""

    def __init__(
        self,
        history_dim: int = DEFAULT_HIDDEN_DIM,
        summary_dim: int = len(SUMMARY_FEATURE_NAMES),
        hidden_dim: int = DEFAULT_OUTCOME_HEAD_HIDDEN_DIM,
        dropout: float = DEFAULT_OUTCOME_DROPOUT,
    ) -> None:
        super().__init__()
        if history_dim < 1 or summary_dim < 1 or hidden_dim < 1:
            raise ValueError("feature dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be at least zero and less than one")
        self.history_dim = history_dim
        self.summary_dim = summary_dim
        self.network = nn.Sequential(
            nn.Linear(history_dim + summary_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize both dense layers consistently with the switch head."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, history: Tensor, summary: Tensor, actions: Tensor) -> Tensor:
        """Return one win logit for each supplied stay or switch action."""
        self._validate_inputs(history, summary, actions)
        context = torch.cat((history, summary, actions.unsqueeze(-1)), dim=-1)
        return self.network(context).squeeze(-1)

    def _validate_inputs(
        self, history: Tensor, summary: Tensor, actions: Tensor
    ) -> None:
        if history.ndim != 2 or summary.ndim != 2 or actions.ndim != 1:
            raise ValueError(
                "history and summary must be matrices and actions must be a vector"
            )
        if history.shape[0] != summary.shape[0] or history.shape[0] != actions.shape[0]:
            raise ValueError("history, summary, and actions must share a batch size")
        if history.shape[1] != self.history_dim:
            raise ValueError(f"expected {self.history_dim} history features")
        if summary.shape[1] != self.summary_dim:
            raise ValueError(f"expected {self.summary_dim} summary features")
        if not all(torch.is_floating_point(value) for value in (history, summary, actions)):
            raise ValueError("history, summary, and actions must be floating point")
        if not torch.all((actions == 0.0) | (actions == 1.0)):
            raise ValueError("actions must contain only stay 0 or switch 1")


class OutcomePredictionModel(nn.Module):
    """Encode battles 1 through 10 and predict a win under a supplied action."""

    def __init__(
        self,
        embedding_rows: int,
        level_mean: float,
        level_standard_deviation: float,
        head_hidden_dim: int = DEFAULT_OUTCOME_HEAD_HIDDEN_DIM,
        dropout: float = DEFAULT_OUTCOME_DROPOUT,
        summary_dim: int | None = None,
    ) -> None:
        super().__init__()
        if summary_dim is None:
            summary_dim = len(SUMMARY_FEATURE_NAMES)
        if summary_dim < 1:
            raise ValueError("summary_dim must be positive")
        self.card_features = CardFeatures(
            embedding_rows=embedding_rows,
            embedding_dim=24,
            level_mean=level_mean,
            level_standard_deviation=level_standard_deviation,
        )
        self.card_mlp = CardMLP(input_dim=25, hidden_dim=48)
        self.deck_pool = SumDeckPool()
        self.deck_mlp = DeckMLP(input_dim=48, hidden_dim=96, output_dim=48)
        self.token_assembler = BattleTokenAssembler()
        self.history_encoder = GRUHistoryEncoder(
            input_dim=DEFAULT_INPUT_DIM, hidden_dim=DEFAULT_HIDDEN_DIM
        )
        self.win_head = ActionConditionedWinHead(
            history_dim=DEFAULT_HIDDEN_DIM,
            summary_dim=summary_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        card_indices: Tensor,
        displayed_levels: Tensor,
        battle_features: Tensor,
        summary_features: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Return battle-11 win logits under the supplied hypothetical actions."""
        history = self._encode_history(
            card_indices, displayed_levels, battle_features
        )
        return self.win_head(history, summary_features, actions)

    def _encode_history(
        self,
        card_indices: Tensor,
        displayed_levels: Tensor,
        battle_features: Tensor,
    ) -> Tensor:
        """Encode the ten observed battles without using the target action."""
        cards = self.card_features(card_indices, displayed_levels)
        cards = self.card_mlp(cards)
        decks = self.deck_mlp(self.deck_pool(cards))
        tokens = self.token_assembler(decks, battle_features)
        return self.history_encoder(tokens)

    def potential_outcome_logits(
        self,
        card_indices: Tensor,
        displayed_levels: Tensor,
        battle_features: Tensor,
        summary_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate every history under stay and switch without observed actions."""
        history = self._encode_history(
            card_indices, displayed_levels, battle_features
        )
        batch_size = card_indices.shape[0]
        action_options = {
            "device": summary_features.device,
            "dtype": summary_features.dtype,
        }
        stay = torch.zeros(batch_size, **action_options)
        switch = torch.ones(batch_size, **action_options)
        stay_logits = self.win_head(
            history,
            summary_features,
            stay,
        )
        switch_logits = self.win_head(
            history,
            summary_features,
            switch,
        )
        return stay_logits, switch_logits
