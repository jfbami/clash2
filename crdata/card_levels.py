"""Convert rarity-relative API card levels to one displayed-level scale."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RARITY_LEVEL_OFFSETS = {
    "common": 0,
    "rare": 2,
    "epic": 5,
    "legendary": 8,
    "champion": 10,
}


@dataclass(frozen=True)
class LevelStandardization:
    """Training-set statistics for z-score standardization."""

    mean: float
    standard_deviation: float

    def transform(self, levels: np.ndarray) -> np.ndarray:
        """Express displayed levels in training standard deviations."""
        return (np.asarray(levels, dtype=np.float32) - self.mean) / self.standard_deviation


def fit_level_standardization(training_levels: np.ndarray) -> LevelStandardization:
    """Fit population mean and standard deviation on displayed training levels."""
    levels = np.asarray(training_levels, dtype=np.float64)
    if levels.size == 0 or not np.isfinite(levels).all():
        raise ValueError("training_levels must contain finite values")
    standard_deviation = float(levels.std())
    if standard_deviation == 0.0:
        raise ValueError("training_levels must have nonzero variance")
    return LevelStandardization(float(levels.mean()), standard_deviation)


class CardLevelConverter:
    """Convert API levels using each card's rarity."""

    def __init__(self, rarity_by_card_id: Mapping[int, str]) -> None:
        self._offset_by_card_id = {
            int(card_id): _offset_for(rarity)
            for card_id, rarity in rarity_by_card_id.items()
        }

    def convert(self, card_ids: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """Return displayed levels without changing the input array shape."""
        card_ids = np.asarray(card_ids)
        levels = np.asarray(levels, dtype=np.float32)
        if card_ids.shape != levels.shape:
            raise ValueError("card_ids and levels must have the same shape")
        offsets = np.fromiter(
            (self._offset_for_card(card_id) for card_id in card_ids.flat),
            dtype=np.float32,
            count=card_ids.size,
        ).reshape(card_ids.shape)
        return levels + offsets

    def _offset_for_card(self, card_id: int) -> int:
        try:
            return self._offset_by_card_id[int(card_id)]
        except KeyError as error:
            raise ValueError(f"card id {card_id} has no known rarity") from error


def _offset_for(rarity: str) -> int:
    normalized = str(rarity).lower()
    try:
        return RARITY_LEVEL_OFFSETS[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported card rarity {rarity!r}") from error


def load_card_level_converter(path: Path | str) -> CardLevelConverter:
    """Load card rarities from the outcome-blind card reference table."""
    cards = pd.read_parquet(path, columns=["id", "rarity"])
    return CardLevelConverter(dict(zip(cards["id"], cards["rarity"])))
