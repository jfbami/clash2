"""Compare next-battle win probabilities under stay and switch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from crdata.battle_features import BattleFeatureStandardization
from crdata.card_levels import CardLevelConverter, load_card_level_converter
from crdata.sequences import (
    CURRENT_DECK_COUNT_FEATURE_SET,
    DEFAULT_HISTORY_LENGTH,
    PlayerBattle,
    SUMMARY_FEATURE_SETS,
    build_prebattle_context,
)
from crdata.summary_features import SummaryFeatureStandardization
from crdata.vocabulary import CardVocabulary, load_card_vocabulary
from models.outcome_model import OutcomePredictionModel


@dataclass(frozen=True)
class OutcomeComparison:
    """Numerical model outputs, without a recommendation or support judgment."""

    stay_win_probability: float
    switch_win_probability: float
    switch_minus_stay: float


class OutcomePredictor:
    """Apply a trained outcome checkpoint to a player's observed battle history."""

    def __init__(
        self,
        model: OutcomePredictionModel,
        vocabulary: CardVocabulary,
        level_converter: CardLevelConverter,
        battle_standardization: BattleFeatureStandardization,
        summary_standardization: SummaryFeatureStandardization,
        device: torch.device,
    ) -> None:
        if vocabulary.embedding_rows != model.card_features.card_embedding.embedding.num_embeddings:
            raise ValueError("card reference does not match the model embedding size")
        if len(summary_standardization.means) != model.win_head.summary_dim:
            raise ValueError("summary preprocessing does not match the outcome model")
        self.model = model.to(device).eval()
        self.vocabulary = vocabulary
        self.level_converter = level_converter
        self.battle_standardization = battle_standardization
        self.summary_standardization = summary_standardization
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        card_reference: Path,
        device: str | torch.device = "cpu",
    ) -> OutcomePredictor:
        """Load model weights and the preprocessing fitted during training."""
        selected_device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        metadata = checkpoint["data"]
        preprocessing = checkpoint["preprocessing"]
        training = checkpoint["training"]
        if metadata.get("continuity") != "same_collection":
            raise ValueError("checkpoint must use same_collection battle histories")
        if metadata.get("summary_feature_set") != CURRENT_DECK_COUNT_FEATURE_SET:
            raise ValueError("checkpoint must use the selected nine-feature summary")
        expected_summary_names = SUMMARY_FEATURE_SETS[CURRENT_DECK_COUNT_FEATURE_SET]
        if tuple(metadata["summary_feature_names"]) != expected_summary_names:
            raise ValueError("checkpoint summary features do not match inference order")
        model = OutcomePredictionModel(
            embedding_rows=int(metadata["embedding_rows"]),
            level_mean=float(preprocessing["level_mean"]),
            level_standard_deviation=float(preprocessing["level_standard_deviation"]),
            head_hidden_dim=int(training["head_hidden_dim"]),
            summary_dim=len(expected_summary_names),
        )
        model.load_state_dict(checkpoint["model_state"])
        vocabulary = load_card_vocabulary(card_reference)
        level_converter = load_card_level_converter(card_reference)
        battle_standardization = BattleFeatureStandardization(
            means=tuple(preprocessing["battle_feature_means"]),
            standard_deviations=tuple(
                preprocessing["battle_feature_standard_deviations"]
            ),
        )
        summary_standardization = SummaryFeatureStandardization(
            means=tuple(preprocessing["summary_feature_means"]),
            standard_deviations=tuple(
                preprocessing["summary_feature_standard_deviations"]
            ),
        )
        return cls(
            model,
            vocabulary,
            level_converter,
            battle_standardization,
            summary_standardization,
            selected_device,
        )

    def predict(self, battles: Iterable[PlayerBattle]) -> OutcomeComparison:
        """Compare actions using only battles observed before the next battle."""
        context = build_prebattle_context(
            battles,
            history_length=DEFAULT_HISTORY_LENGTH,
            summary_feature_set=CURRENT_DECK_COUNT_FEATURE_SET,
            continuity="same_collection",
        )
        card_indices = self.vocabulary.encode(context.deck_ids)
        displayed_levels = self.level_converter.convert(
            context.deck_ids, context.card_levels
        )
        battle_features = self.battle_standardization.transform(
            context.battle_features
        )
        summary_features = self.summary_standardization.transform(
            context.summary_features
        )
        inputs = (
            torch.as_tensor(card_indices[None], dtype=torch.long, device=self.device),
            torch.as_tensor(displayed_levels[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(battle_features[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(summary_features[None], dtype=torch.float32, device=self.device),
        )
        self.model.eval()
        with torch.inference_mode():
            stay_logit, switch_logit = self.model.potential_outcome_logits(*inputs)
            stay = float(torch.sigmoid(stay_logit).item())
            switch = float(torch.sigmoid(switch_logit).item())
        return OutcomeComparison(
            stay_win_probability=stay,
            switch_win_probability=switch,
            switch_minus_stay=switch - stay,
        )
