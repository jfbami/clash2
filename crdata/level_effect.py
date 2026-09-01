"""Measure win rate against card level advantage, without fitting a model.

With 36.9 million battles the win rate at each level advantage can be counted
directly. Counting avoids the problem recorded in `RESULTS_NEURAL.md`, where
logistic regression credits card level with 79.7 percent of explained variance
and the neural model credits it with 29.7 percent on the same data. A counted
win rate has no model class, so it has nothing to disagree about.

Two curves come out of one pass.

`overall_win_rate` counts every battle. It credits card level with whatever
skill happens to travel alongside a stronger account, so it reads high.

`trophy_matched_win_rate` compares only players sitting at the same trophy
count, then averages those comparisons back to the season-wide trophy mix.
Matchmaking pairs players within 50 trophies in 99.86 percent of battles, so
holding trophies fixed also holds most of the skill gap fixed. It reads low,
because two players at the same trophy count with unequal cards imply the
weaker-carded player is the better player.

The true card level effect sits between the two curves.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CARDS_PER_DECK = 8
MAXED_DECK_LEVEL = 104
LADDER_MODES = (72000006, 72000201, 72000044)
BATCH_ROWS = 200_000

GAP_LIMIT = 40
TROPHY_BAND = 100
BAND_COUNT = 80


@dataclass(frozen=True)
class Curve:
    """A win rate read off counts, with the battles behind each point."""

    advantage: np.ndarray
    win_rate: np.ndarray
    battles: np.ndarray

    def standard_error(self) -> np.ndarray:
        return np.sqrt(self.win_rate * (1 - self.win_rate) / np.maximum(self.battles, 1))

    def at(self, advantage: int) -> float:
        return float(self.win_rate[np.searchsorted(self.advantage, advantage)])


@dataclass
class LevelCounts:
    """Battles and wins per trophy band and card level advantage."""

    battles: np.ndarray
    wins: np.ndarray

    @classmethod
    def empty(cls) -> "LevelCounts":
        shape = (BAND_COUNT, 2 * GAP_LIMIT + 1)
        return cls(battles=np.zeros(shape, np.int64), wins=np.zeros(shape, np.int64))

    @property
    def advantage_axis(self) -> np.ndarray:
        return np.arange(-GAP_LIMIT, GAP_LIMIT + 1)

    def total(self) -> int:
        return int(self.battles.sum())


def _level_columns(side: str) -> list[str]:
    return [f"{side}_level{index}" for index in range(1, CARDS_PER_DECK + 1)]


def _required_columns() -> list[str]:
    return (["game_mode", "a_won", "a_trophies", "b_trophies"]
            + _level_columns("a") + _level_columns("b"))


def season_files(directory: Path, exclude: tuple[str, ...]) -> list[Path]:
    """Parquet day files in `directory`, skipping any whose name contains an
    excluded token. Raises FileNotFoundError when nothing matches."""
    files = sorted(path for path in Path(directory).glob("*.parquet")
                   if not any(token in path.name for token in exclude))
    if not files:
        raise FileNotFoundError(f"no Parquet files in {directory} after excluding {exclude}")
    return files


def _accumulate(counts: LevelCounts, frame, generator: np.random.Generator) -> None:
    """Add one batch, assigning each battle's two players to sides by a fair coin.

    Storage puts the lexicographically smaller player tag on side A, and that
    side carries 0.4765 more card levels on average because Supercell issues
    tags roughly in creation order. Reading the win rate off stored order would
    therefore fold account age into the card level effect. D10 in `DECISIONS.md`
    requires the coin flip.
    """
    total_a = frame[_level_columns("a")].to_numpy(np.int64).sum(axis=1)
    total_b = frame[_level_columns("b")].to_numpy(np.int64).sum(axis=1)
    a_won = frame["a_won"].to_numpy(np.int64)

    flip = generator.random(len(frame)) < 0.5
    advantage = np.where(flip, total_b - total_a, total_a - total_b)
    won = np.where(flip, 1 - a_won, a_won)

    trophies = (frame["a_trophies"].to_numpy(np.int64)
                + frame["b_trophies"].to_numpy(np.int64)) // (2 * TROPHY_BAND)
    band = np.clip(trophies, 0, BAND_COUNT - 1)
    column = np.clip(advantage, -GAP_LIMIT, GAP_LIMIT) + GAP_LIMIT
    width = 2 * GAP_LIMIT + 1

    flat = band * width + column
    size = BAND_COUNT * width
    counts.battles.ravel()[:] += np.bincount(flat, minlength=size)
    counts.wins.ravel()[:] += np.bincount(flat, weights=won, minlength=size).astype(np.int64)


def count_season(files: list[Path], seed: int = 0,
                 modes: tuple[int, ...] = LADDER_MODES) -> LevelCounts:
    """Stream every day file, holding only the counts in memory."""
    counts = LevelCounts.empty()
    generator = np.random.default_rng(seed)
    columns = _required_columns()

    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=columns):
            frame = batch.to_pandas()
            frame = frame[frame.game_mode.isin(modes)]
            if frame.empty:
                continue
            _accumulate(counts, frame, generator)
        print(f"  counted {path.name}  running total {counts.total():,}", flush=True)

    return counts


def _trim(counts: LevelCounts, battles: np.ndarray, win_rate: np.ndarray,
          minimum: int) -> Curve:
    keep = battles >= minimum
    return Curve(advantage=counts.advantage_axis[keep],
                 win_rate=win_rate[keep], battles=battles[keep])


def overall_win_rate(counts: LevelCounts, minimum_battles: int = 10_000) -> Curve:
    """Win rate at each card level advantage, over every battle in the season."""
    battles = counts.battles.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        win_rate = counts.wins.sum(axis=0) / battles
    return _trim(counts, battles, win_rate, minimum_battles)


def trophy_matched_win_rate(counts: LevelCounts, minimum_battles: int = 10_000,
                            minimum_cell: int = 200) -> Curve:
    """Win rate at each card level advantage, comparing players at equal trophies.

    Each trophy band contributes its own win rate, and those are averaged using
    the season-wide share of battles per band. Averaging that way reports the
    win rate that would hold if every level advantage occurred at the same mix
    of trophy counts, which removes the trophy mix as an explanation.
    """
    band_weight = counts.battles.sum(axis=1).astype(float)
    width = 2 * GAP_LIMIT + 1
    win_rate = np.full(width, np.nan)
    battles = np.zeros(width, np.int64)

    for column in range(width):
        cell = counts.battles[:, column]
        usable = cell >= minimum_cell
        if not usable.any():
            continue
        weight = band_weight[usable]
        rate = counts.wins[usable, column] / cell[usable]
        win_rate[column] = float((weight * rate).sum() / weight.sum())
        battles[column] = int(cell[usable].sum())

    return _trim(counts, battles, win_rate, minimum_battles)
