"""Outcome-blind card embeddings from deck co-occurrence.

The space is built only from which cards appear together, never from who won.
Keeping it outcome-blind matters: if archetypes were defined using win and loss
data, then measuring archetype strength would be circular.

`CardSpace` follows Levy and Goldberg (2014), who showed that skip-gram with
negative sampling implicitly factorises a shifted PMI matrix. Here the matrix is
factorised directly with SVD, which is deterministic and needs no tuning.

Two departures from the standard text recipe, both deliberate:

Negative PMI is kept rather than clamped to zero. In sparse text a negative value
is usually noise, but a 102 by 102 matrix estimated from two billion card pairs
is dense and precise, and a negative value carries real meaning: two cards
competing for the same deck slot.

There is no context window. A deck is a set of 8 cards, so the context of a card
is simply the other 7, which removes word2vec's most awkward hyperparameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DECK_SIZE = 8
LADDER_MODES = (72000006, 72000201, 72000044)  # Ladder, Crown Rush, Gold Rush
BATCH_ROWS = 200_000


@dataclass(frozen=True)
class CardSpace:
    card_ids: np.ndarray
    vectors: np.ndarray
    singular_values: np.ndarray
    pmi: np.ndarray
    cooccurrence: np.ndarray

    def deck_vectors(self, decks: np.ndarray) -> np.ndarray:
        """Sum the card vectors of each deck. Every deck holds exactly 8 cards,
        so summing and averaging differ only by a constant here."""
        position = {card: index for index, card in enumerate(self.card_ids)}
        rows = np.vectorize(position.get)(decks)
        return self.vectors[rows].sum(axis=1)


def _deck_columns() -> list[str]:
    return ([f"a_card{i}" for i in range(1, DECK_SIZE + 1)]
            + [f"b_card{i}" for i in range(1, DECK_SIZE + 1)])


def count_cooccurrence(parquet_files: list[Path],
                       modes: tuple[int, ...] = LADDER_MODES) -> tuple[np.ndarray, np.ndarray, int]:
    """Count how often each card pair shares a deck, streaming to bound memory.

    Only the 102 by 102 counts are held in memory, never the battles themselves.
    Returns the counts, the card ids indexing them, and the number of decks seen.
    """
    columns = _deck_columns() + ["game_mode"]
    card_ids = _distinct_cards(parquet_files, columns)
    position = np.full(int(card_ids.max()) + 1, -1, dtype=np.int32)
    position[card_ids] = np.arange(len(card_ids))

    size = len(card_ids)
    counts = np.zeros((size, size), dtype=np.int64)
    decks_seen = 0

    for path in parquet_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=columns):
            frame = batch.to_pandas()
            frame = frame[frame.game_mode.isin(modes)]
            if frame.empty:
                continue
            for side in ("a", "b"):
                decks = frame[[f"{side}_card{i}" for i in range(1, DECK_SIZE + 1)]].to_numpy()
                _accumulate(counts, position[decks], size)
                decks_seen += len(decks)

    return counts, card_ids, decks_seen


def _distinct_cards(parquet_files: list[Path], columns: list[str]) -> np.ndarray:
    seen: set[int] = set()
    for path in parquet_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=columns):
            frame = batch.to_pandas()
            for column in _deck_columns():
                seen.update(np.unique(frame[column].to_numpy()).tolist())
    return np.array(sorted(seen), dtype=np.int64)


def _accumulate(counts: np.ndarray, deck_positions: np.ndarray, size: int) -> None:
    """Add every unordered pair within each deck to the symmetric counts."""
    for first in range(DECK_SIZE):
        for second in range(first + 1, DECK_SIZE):
            flat = deck_positions[:, first] * size + deck_positions[:, second]
            counts.ravel()[:] += np.bincount(flat, minlength=size * size)
            flat_mirror = deck_positions[:, second] * size + deck_positions[:, first]
            counts.ravel()[:] += np.bincount(flat_mirror, minlength=size * size)


def pmi_matrix(counts: np.ndarray) -> np.ndarray:
    """Pointwise mutual information, with negative values retained.

    PMI(i,j) = log[ count(i,j) * total / (count(i) * count(j)) ]
    """
    total = counts.sum()
    marginal = counts.sum(axis=1)
    expected = np.outer(marginal, marginal) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(counts / expected)
    return np.nan_to_num(pmi, nan=0.0, neginf=0.0, posinf=0.0)


def factorise(pmi: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    """Truncated SVD of a symmetric PMI matrix.

    Vectors are scaled by the square root of the singular values, the standard
    choice that keeps dot products proportional to reconstructed PMI.
    """
    left, singular, _ = np.linalg.svd(pmi, full_matrices=False)
    vectors = left[:, :dimensions] * np.sqrt(singular[:dimensions])
    return vectors, singular


def build_card_space(parquet_files: list[Path], dimensions: int,
                     modes: tuple[int, ...] = LADDER_MODES) -> tuple[CardSpace, int]:
    counts, card_ids, decks_seen = count_cooccurrence(parquet_files, modes)
    pmi = pmi_matrix(counts)
    vectors, singular = factorise(pmi, dimensions)
    return CardSpace(card_ids=card_ids, vectors=vectors, singular_values=singular,
                     pmi=pmi, cooccurrence=counts), decks_seen
