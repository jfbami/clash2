"""Load Season 18 ladder battle CSV files into model-ready arrays.

A Season 18 file stores the winning player in `winner.*` columns and the losing
player in `loser.*` columns. Any model trained on that layout scores 100 percent
from column position alone, so `load_season18` assigns sides at random and
returns the resulting label.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

CARDS_PER_DECK = 8


@dataclass(frozen=True)
class BattleMatrix:
    """Model-ready view of a set of battles, with sides already randomised."""

    card_difference: sparse.csr_matrix
    level_difference: np.ndarray
    trophy_difference: np.ndarray
    side_a_won: np.ndarray
    card_ids: np.ndarray
    player_a: np.ndarray
    player_b: np.ndarray

    def __len__(self) -> int:
        return len(self.side_a_won)


def _deck_columns(side: str) -> list[str]:
    return [f"{side}.card{i}.id" for i in range(1, CARDS_PER_DECK + 1)]


def _required_columns() -> list[str]:
    return (_deck_columns("winner") + _deck_columns("loser") + [
        "battleTime", "winner.tag", "loser.tag",
        "winner.totalcard.level", "loser.totalcard.level",
        "winner.startingTrophies", "loser.startingTrophies"])


def _swap_where(flip: np.ndarray, winner_values, loser_values):
    """Return (side_a, side_b) after moving the loser to side A wherever flip."""
    return np.where(flip, loser_values, winner_values), np.where(flip, winner_values, loser_values)


def _card_difference_matrix(a_ids: np.ndarray, b_ids: np.ndarray,
                            card_ids: np.ndarray) -> sparse.csr_matrix:
    """Build a matrix holding +1 for a card on side A and -1 for a card on side B.

    The difference encoding makes any linear model antisymmetric by construction:
    swapping sides negates every feature and therefore negates the predicted logit.
    """
    column_of = {card: column for column, card in enumerate(card_ids)}
    lookup = np.vectorize(column_of.get)
    n_battles = len(a_ids)

    rows = np.repeat(np.arange(n_battles), 2 * CARDS_PER_DECK)
    columns = np.concatenate([lookup(a_ids), lookup(b_ids)], axis=1).ravel()
    values = np.tile(
        np.r_[np.ones(CARDS_PER_DECK), -np.ones(CARDS_PER_DECK)], n_battles)

    return sparse.csr_matrix(
        (values, (rows, columns)), shape=(n_battles, len(card_ids)), dtype=np.float32)


def load_season18(path: Path | str, subsample: int | None = None,
                  seed: int = 0) -> BattleMatrix:
    """Load one Season 18 CSV, randomise sides, and return a BattleMatrix.

    Raises FileNotFoundError when `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Season 18 file not found: {path}")

    frame = pd.read_csv(path, usecols=_required_columns(), low_memory=False).dropna()
    frame = frame.sort_values("battleTime").reset_index(drop=True)
    if subsample is not None:
        frame = frame.iloc[:subsample].reset_index(drop=True)

    n_battles = len(frame)
    flip = np.random.default_rng(seed).random(n_battles) < 0.5

    winner_ids = frame[_deck_columns("winner")].to_numpy(np.int64)
    loser_ids = frame[_deck_columns("loser")].to_numpy(np.int64)
    a_ids = np.where(flip[:, None], loser_ids, winner_ids)
    b_ids = np.where(flip[:, None], winner_ids, loser_ids)

    a_level, b_level = _swap_where(
        flip, frame["winner.totalcard.level"].to_numpy(float),
        frame["loser.totalcard.level"].to_numpy(float))
    a_trophies, b_trophies = _swap_where(
        flip, frame["winner.startingTrophies"].to_numpy(float),
        frame["loser.startingTrophies"].to_numpy(float))
    a_tag, b_tag = _swap_where(
        flip, frame["winner.tag"].to_numpy(object), frame["loser.tag"].to_numpy(object))

    card_ids = np.unique(np.concatenate([a_ids.ravel(), b_ids.ravel()]))

    return BattleMatrix(
        card_difference=_card_difference_matrix(a_ids, b_ids, card_ids),
        level_difference=a_level - b_level,
        trophy_difference=a_trophies - b_trophies,
        side_a_won=(~flip).astype(np.int8),
        card_ids=card_ids,
        player_a=a_tag,
        player_b=b_tag)
