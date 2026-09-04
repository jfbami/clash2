"""Deterministic mapping between Clash Royale card ids and model indices."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


UNKNOWN_INDEX = 0


class CardVocabulary:
    """Map sparse API card ids to contiguous embedding-table indices."""

    def __init__(self, card_ids: Iterable[int]) -> None:
        ordered_ids = tuple(sorted({int(card_id) for card_id in card_ids}))
        if not ordered_ids:
            raise ValueError("a card vocabulary cannot be empty")
        self._card_ids = ordered_ids
        self._index_by_card_id = {
            card_id: index for index, card_id in enumerate(ordered_ids, start=1)
        }

    @property
    def card_count(self) -> int:
        """Number of known cards, excluding the unknown index."""
        return len(self._card_ids)

    @property
    def embedding_rows(self) -> int:
        """Required embedding-table rows, including the unknown index."""
        return self.card_count + 1

    def index_for(self, card_id: int) -> int:
        """Return zero for an id absent from the reference vocabulary."""
        return self._index_by_card_id.get(int(card_id), UNKNOWN_INDEX)

    def card_id_for(self, index: int) -> int | None:
        """Reverse one known index, with zero represented by ``None``."""
        if index == UNKNOWN_INDEX:
            return None
        if index < 0 or index > self.card_count:
            raise IndexError(f"card index {index} is outside the vocabulary")
        return self._card_ids[index - 1]

    def encode(self, card_ids: np.ndarray) -> np.ndarray:
        """Replace card ids with indices without changing the array shape."""
        card_ids = np.asarray(card_ids)
        indices = np.fromiter(
            (self.index_for(card_id) for card_id in card_ids.flat),
            dtype=np.int64,
            count=card_ids.size,
        )
        return indices.reshape(card_ids.shape)


def load_card_vocabulary(path: Path | str) -> CardVocabulary:
    """Build the vocabulary from the outcome-blind card reference table."""
    card_ids = pd.read_parquet(path, columns=["id"])["id"]
    return CardVocabulary(card_ids)
