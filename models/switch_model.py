"""End-to-end model for next-battle deck-switch prediction."""
from __future__ import annotations

from torch import Tensor, nn

from models.battle_encoder import BattleTokenAssembler
from models.card_encoder import CardFeatures, CardMLP, DeckMLP, SumDeckPool
from models.history_encoder import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_INPUT_DIM,
    GRUHistoryEncoder,
    LinearSwitchHead,
    NextSwitchHead,
)
from crdata.sequences import SUMMARY_FEATURE_NAMES


class SwitchPredictionModel(nn.Module):
    """Encode cards, decks, battles, and history before scoring a switch."""

    def __init__(
        self,
        embedding_rows: int,
        level_mean: float,
        level_standard_deviation: float,
        head_hidden_dim: int | None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
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
        self.switch_head: nn.Module
        if head_hidden_dim is None:
            self.switch_head = LinearSwitchHead(
                history_dim=DEFAULT_HIDDEN_DIM,
                summary_dim=len(SUMMARY_FEATURE_NAMES),
            )
        else:
            self.switch_head = NextSwitchHead(
                history_dim=DEFAULT_HIDDEN_DIM,
                summary_dim=len(SUMMARY_FEATURE_NAMES),
                hidden_dim=head_hidden_dim,
                dropout=dropout,
            )

    def forward(
        self,
        card_indices: Tensor,
        displayed_levels: Tensor,
        battle_features: Tensor,
        summary_features: Tensor,
    ) -> Tensor:
        """Return one raw next-switch logit for every history in the batch."""
        cards = self.card_features(card_indices, displayed_levels)
        cards = self.card_mlp(cards)
        decks = self.deck_mlp(self.deck_pool(cards))
        tokens = self.token_assembler(decks, battle_features)
        history = self.history_encoder(tokens)
        return self.switch_head(history, summary_features)
