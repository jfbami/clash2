"""Load Season 18 ladder battle CSV files into model-ready arrays.

A Season 18 file stores the winning player in `winner.*` columns and the losing
player in `loser.*` columns. Any model trained on that layout scores 100 percent
from column position alone, so every loader here assigns sides at random first
and derives the label from the assignment.

`load_randomised` does that work once. `as_difference_matrix` builds the sparse
view used by linear models, and `as_index_arrays` builds the integer view used by
embedding models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

CARDS_PER_DECK = 8


@dataclass(frozen=True)
class RandomisedBattles:
    """Battles with sides already assigned at random. Side A is not the winner."""

    cards_a: np.ndarray
    cards_b: np.ndarray
    level_a: np.ndarray
    level_b: np.ndarray
    trophies_a: np.ndarray
    trophies_b: np.ndarray
    tag_a: np.ndarray
    tag_b: np.ndarray
    side_a_won: np.ndarray

    def __len__(self) -> int:
        return len(self.side_a_won)


@dataclass(frozen=True)
class IndexArrays:
    """Integer-encoded view for embedding models."""

    cards_a: np.ndarray
    cards_b: np.ndarray
    player_a: np.ndarray
    player_b: np.ndarray
    level_a: np.ndarray
    level_b: np.ndarray
    side_a_won: np.ndarray
    card_ids: np.ndarray
    n_players: int

    def __len__(self) -> int:
        return len(self.side_a_won)


def _deck_columns(side: str) -> list[str]:
    return [f"{side}.card{i}.id" for i in range(1, CARDS_PER_DECK + 1)]


def _required_columns() -> list[str]:
    return (_deck_columns("winner") + _deck_columns("loser") + [
        "battleTime", "winner.tag", "loser.tag",
        "winner.totalcard.level", "loser.totalcard.level",
        "winner.startingTrophies", "loser.startingTrophies"])


def _swap_where(flip: np.ndarray, winner_values: np.ndarray, loser_values: np.ndarray):
    return np.where(flip, loser_values, winner_values), np.where(flip, winner_values, loser_values)


def load_randomised(path: Path | str, subsample: int | None = None,
                    seed: int = 0) -> RandomisedBattles:
    """Read one Season 18 CSV, sort by time, and assign sides at random.

    Raises FileNotFoundError when `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Season 18 file not found: {path}")

    frame = pd.read_csv(path, usecols=_required_columns(), low_memory=False).dropna()
    frame = frame.sort_values("battleTime").reset_index(drop=True)
    if subsample is not None:
        frame = frame.iloc[:subsample].reset_index(drop=True)

    flip = np.random.default_rng(seed).random(len(frame)) < 0.5
    winner_ids = frame[_deck_columns("winner")].to_numpy(np.int64)
    loser_ids = frame[_deck_columns("loser")].to_numpy(np.int64)

    level_a, level_b = _swap_where(
        flip, frame["winner.totalcard.level"].to_numpy(np.float32),
        frame["loser.totalcard.level"].to_numpy(np.float32))
    trophies_a, trophies_b = _swap_where(
        flip, frame["winner.startingTrophies"].to_numpy(np.float32),
        frame["loser.startingTrophies"].to_numpy(np.float32))
    tag_a, tag_b = _swap_where(
        flip, frame["winner.tag"].to_numpy(object), frame["loser.tag"].to_numpy(object))

    return RandomisedBattles(
        cards_a=np.where(flip[:, None], loser_ids, winner_ids),
        cards_b=np.where(flip[:, None], winner_ids, loser_ids),
        level_a=level_a, level_b=level_b,
        trophies_a=trophies_a, trophies_b=trophies_b,
        tag_a=tag_a, tag_b=tag_b,
        side_a_won=(~flip).astype(np.int8))


def as_difference_matrix(battles: RandomisedBattles) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build a matrix holding +1 for a card on side A and -1 for a card on side B.

    The difference encoding makes any linear model antisymmetric by construction.
    """
    card_ids = np.unique(np.concatenate([battles.cards_a.ravel(), battles.cards_b.ravel()]))
    column_of = {card: column for column, card in enumerate(card_ids)}
    lookup = np.vectorize(column_of.get)
    n_battles = len(battles)

    rows = np.repeat(np.arange(n_battles), 2 * CARDS_PER_DECK)
    columns = np.concatenate([lookup(battles.cards_a), lookup(battles.cards_b)], axis=1).ravel()
    values = np.tile(np.r_[np.ones(CARDS_PER_DECK), -np.ones(CARDS_PER_DECK)], n_battles)

    matrix = sparse.csr_matrix(
        (values, (rows, columns)), shape=(n_battles, len(card_ids)), dtype=np.float32)
    return matrix, card_ids


def as_index_arrays(battles: RandomisedBattles) -> IndexArrays:
    """Encode cards and players as contiguous integer indices for embedding lookup."""
    card_ids = np.unique(np.concatenate([battles.cards_a.ravel(), battles.cards_b.ravel()]))
    card_index = {card: position for position, card in enumerate(card_ids)}
    to_card_index = np.vectorize(card_index.get)

    players = np.unique(np.concatenate([battles.tag_a, battles.tag_b]))
    player_index = {tag: position for position, tag in enumerate(players)}
    to_player_index = np.vectorize(player_index.get)

    return IndexArrays(
        cards_a=to_card_index(battles.cards_a).astype(np.int64),
        cards_b=to_card_index(battles.cards_b).astype(np.int64),
        player_a=to_player_index(battles.tag_a).astype(np.int64),
        player_b=to_player_index(battles.tag_b).astype(np.int64),
        level_a=battles.level_a, level_b=battles.level_b,
        side_a_won=battles.side_a_won,
        card_ids=card_ids, n_players=len(players))
