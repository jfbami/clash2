"""Profile the data an investment dose-response curve would rest on.

Two facts decide whether that curve is buildable, and one streaming pass
answers both.

`card_level_histogram` answers whether the level columns share one scale.
The Clash Royale API historically reported a card's level on its own rarity
scale, capping Legendaries at 5 where Commons reach 13, and Season 18
predates the 2021 card level rework that unified them. Summing eight levels
is only meaningful if the source already normalised them.

`conditioning_cells` answers whether trophy conditioning is affordable.
The curve conditions on trophy band and reads win rate against card level
gap, so every cell it uses needs enough battles to estimate a rate.

`trophy_difference` sizes the band width. Matchmaking pairs on trophies, so
how tightly it pairs decides how narrow a band has to be to add anything.

No outcome column is read. This pass commits to no modelling decision.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CARDS_PER_DECK = 8
LADDER_MODES = (72000006, 72000201, 72000044)
BATCH_ROWS = 200_000

LEVEL_BINS = 16
TROPHY_BAND = 100
BAND_COUNT = 80
GAP_LIMIT = 40
TROPHY_DIFFERENCE_LIMIT = 200

PARQUET = Path(r"C:\Users\jfbaa\AppData\Local\Temp\claude"
               r"\C--Users-jfbaa-OneDrive-Documents-clash2"
               r"\d24c6794-c5fc-463a-925a-588dd12c92e6\scratchpad\season18_parquet")
EXCLUDED = ("01042021",)  # D9: the final day of the season is not representative.


@dataclass
class Profile:
    """Aggregates only. Nothing here grows with the number of battles."""

    card_ids: np.ndarray
    level_histogram: np.ndarray
    conditioning_cells: np.ndarray
    trophy_difference: np.ndarray
    battles: int = 0
    rows_read: int = 0

    @property
    def gap_axis(self) -> np.ndarray:
        return np.arange(-GAP_LIMIT, GAP_LIMIT + 1)


def _card_columns(side: str) -> list[str]:
    return [f"{side}_card{index}" for index in range(1, CARDS_PER_DECK + 1)]


def _level_columns(side: str) -> list[str]:
    return [f"{side}_level{index}" for index in range(1, CARDS_PER_DECK + 1)]


def _required_columns() -> list[str]:
    return (["game_mode", "a_trophies", "b_trophies"]
            + _card_columns("a") + _card_columns("b")
            + _level_columns("a") + _level_columns("b"))


def season_files(directory: Path, exclude: tuple[str, ...]) -> list[Path]:
    files = sorted(path for path in directory.glob("*.parquet")
                   if not any(token in path.name for token in exclude))
    if not files:
        raise FileNotFoundError(f"no Parquet files in {directory} after excluding {exclude}")
    return files


def _position_lookup(card_ids: np.ndarray) -> np.ndarray:
    lookup = np.full(int(card_ids.max()) + 1, -1, dtype=np.int32)
    lookup[card_ids] = np.arange(len(card_ids))
    return lookup


def _accumulate_levels(profile: Profile, lookup: np.ndarray, frame: pd.DataFrame) -> None:
    """Add every card-level observation to the per-card level histogram."""
    width = len(profile.card_ids)
    for side in ("a", "b"):
        cards = frame[_card_columns(side)].to_numpy(np.int64).ravel()
        levels = np.clip(frame[_level_columns(side)].to_numpy(np.int64).ravel(), 0, LEVEL_BINS - 1)
        positions = lookup[cards]
        if (positions < 0).any():
            raise ValueError("battle references a card absent from data/reference/cards.parquet")
        flat = positions * LEVEL_BINS + levels
        profile.level_histogram.ravel()[:] += np.bincount(flat, minlength=width * LEVEL_BINS)


def _accumulate_cells(profile: Profile, frame: pd.DataFrame) -> None:
    """Add every battle to the trophy band by card level gap contingency table."""
    trophies_a = frame["a_trophies"].to_numpy(np.int64)
    trophies_b = frame["b_trophies"].to_numpy(np.int64)
    total_a = frame[_level_columns("a")].to_numpy(np.int64).sum(axis=1)
    total_b = frame[_level_columns("b")].to_numpy(np.int64).sum(axis=1)

    band = np.clip((trophies_a + trophies_b) // (2 * TROPHY_BAND), 0, BAND_COUNT - 1)
    gap = np.clip(total_a - total_b, -GAP_LIMIT, GAP_LIMIT) + GAP_LIMIT
    width = 2 * GAP_LIMIT + 1
    profile.conditioning_cells.ravel()[:] += np.bincount(
        band * width + gap, minlength=BAND_COUNT * width)

    difference = np.clip(np.abs(trophies_a - trophies_b), 0, TROPHY_DIFFERENCE_LIMIT)
    profile.trophy_difference += np.bincount(difference, minlength=TROPHY_DIFFERENCE_LIMIT + 1)


def profile_season(files: list[Path], card_ids: np.ndarray,
                   modes: tuple[int, ...] = LADDER_MODES) -> Profile:
    lookup = _position_lookup(card_ids)
    width = 2 * GAP_LIMIT + 1
    profile = Profile(
        card_ids=card_ids,
        level_histogram=np.zeros((len(card_ids), LEVEL_BINS), dtype=np.int64),
        conditioning_cells=np.zeros((BAND_COUNT, width), dtype=np.int64),
        trophy_difference=np.zeros(TROPHY_DIFFERENCE_LIMIT + 1, dtype=np.int64))

    columns = _required_columns()
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=columns):
            frame = batch.to_pandas()
            profile.rows_read += len(frame)
            frame = frame[frame.game_mode.isin(modes)]
            if frame.empty:
                continue
            profile.battles += len(frame)
            _accumulate_levels(profile, lookup, frame)
            _accumulate_cells(profile, frame)
        print(f"  done {path.name}  running total {profile.battles:,}", flush=True)

    return profile


def main() -> int:
    cards = pd.read_parquet("data/reference/cards.parquet")
    files = season_files(PARQUET, EXCLUDED)
    print(f"{len(files)} day files, excluding {EXCLUDED}\n", flush=True)

    profile = profile_season(files, cards["id"].to_numpy(np.int64))

    output = Path(__file__).resolve().parents[1] / "data" / "profile_level_gap.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, card_ids=profile.card_ids, level_histogram=profile.level_histogram,
             conditioning_cells=profile.conditioning_cells,
             trophy_difference=profile.trophy_difference,
             battles=profile.battles, rows_read=profile.rows_read)

    print(f"\nrows read      {profile.rows_read:,}")
    print(f"ladder battles {profile.battles:,}")
    print(f"written        {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
