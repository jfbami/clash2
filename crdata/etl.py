"""Convert Season 18 ladder CSVs into compact Parquet.

The source CSVs hold 74 columns and 22 GB across the season, which does not fit
in memory. `convert_file` streams one CSV in chunks, keeps only the columns the
study needs, and writes Parquet.

Sides are stored in canonical order: side A is the lexicographically smaller
player tag. The source layout puts the winner first, so storing raw would let any
model read the label off column position. Canonical ordering destroys that layout
deterministically, which random side-swapping does not.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CARDS_PER_DECK = 8
CHUNK_ROWS = 500_000


@dataclass(frozen=True)
class ConversionReport:
    source: str
    rows_read: int
    rows_written: int
    side_a_win_rate: float


def _card_columns(side: str) -> list[str]:
    return [f"{side}.card{i}.id" for i in range(1, CARDS_PER_DECK + 1)]


def _level_columns(side: str) -> list[str]:
    return [f"{side}.card{i}.level" for i in range(1, CARDS_PER_DECK + 1)]


def essential_columns() -> list[str]:
    """Columns a battle cannot be used without.

    `kingTowerHitPoints` and `clan.tag` are deliberately excluded. A null king
    tower hit point value means the tower was destroyed, which happens in every
    three-crown win, and a null clan tag means the player has no clan. Dropping
    rows on either would delete a large non-random subset of battles.
    """
    return [column for column in _source_columns()
            if "kingTowerHitPoints" not in column and "clan.tag" not in column]


def _source_columns() -> list[str]:
    columns = ["battleTime", "gameMode.id", "arena.id"]
    for side in ("winner", "loser"):
        columns += _card_columns(side) + _level_columns(side)
        columns += [f"{side}.tag", f"{side}.startingTrophies", f"{side}.crowns",
                    f"{side}.kingTowerHitPoints", f"{side}.clan.tag"]
    return columns


def missing_columns(path: Path) -> list[str]:
    """Return required columns absent from `path`, so a schema change fails loudly."""
    header = pd.read_csv(path, nrows=0)
    return [column for column in _source_columns() if column not in header.columns]


def _canonicalise(chunk: pd.DataFrame) -> pd.DataFrame:
    """Emit one row per battle with side A as the smaller player tag."""
    winner_first = chunk["winner.tag"].to_numpy(str) < chunk["loser.tag"].to_numpy(str)

    def pick(column_suffix: str, dtype) -> np.ndarray:
        winner = chunk[f"winner.{column_suffix}"].to_numpy()
        loser = chunk[f"loser.{column_suffix}"].to_numpy()
        return np.where(winner_first, winner, loser).astype(dtype)

    def pick_other(column_suffix: str, dtype) -> np.ndarray:
        winner = chunk[f"winner.{column_suffix}"].to_numpy()
        loser = chunk[f"loser.{column_suffix}"].to_numpy()
        return np.where(winner_first, loser, winner).astype(dtype)

    # A destroyed king tower is reported as null. Zero is the correct value.
    for side in ("winner", "loser"):
        chunk[f"{side}.kingTowerHitPoints"] = chunk[f"{side}.kingTowerHitPoints"].fillna(0)
        chunk[f"{side}.clan.tag"] = chunk[f"{side}.clan.tag"].fillna("")

    output = {
        "battle_time": pd.to_datetime(chunk["battleTime"], errors="coerce", utc=True),
        "game_mode": chunk["gameMode.id"].astype("int32"),
        "arena": chunk["arena.id"].astype("int32"),
        "a_won": winner_first.astype("int8"),
    }
    for index in range(1, CARDS_PER_DECK + 1):
        output[f"a_card{index}"] = pick(f"card{index}.id", "int32")
        output[f"b_card{index}"] = pick_other(f"card{index}.id", "int32")
        output[f"a_level{index}"] = pick(f"card{index}.level", "int8")
        output[f"b_level{index}"] = pick_other(f"card{index}.level", "int8")
    for name, dtype in (("startingTrophies", "int16"), ("crowns", "int8"),
                        ("kingTowerHitPoints", "int32")):
        short = {"startingTrophies": "trophies", "crowns": "crowns",
                 "kingTowerHitPoints": "king_hp"}[name]
        output[f"a_{short}"] = pick(name, dtype)
        output[f"b_{short}"] = pick_other(name, dtype)
    output["a_tag"] = pick("tag", str)
    output["b_tag"] = pick_other("tag", str)
    output["a_clan"] = pick("clan.tag", str)
    output["b_clan"] = pick_other("clan.tag", str)

    return pd.DataFrame(output)


def season_files(directory: Path, exclude: tuple[str, ...] = ()) -> list[Path]:
    """Parquet day files in `directory`, skipping any whose name contains a
    string in `exclude`.

    Raises FileNotFoundError when nothing matches.
    """
    files = sorted(f for f in Path(directory).glob("*.parquet")
                   if not any(token in f.name for token in exclude))
    if not files:
        raise FileNotFoundError(f"no Parquet files in {directory} after excluding {exclude}")
    return files


_SWAP_SUFFIXES = (
    [f"card{i}" for i in range(1, CARDS_PER_DECK + 1)]
    + [f"level{i}" for i in range(1, CARDS_PER_DECK + 1)]
    + ["trophies", "crowns", "king_hp", "tag", "clan"])


def randomise_sides(battles: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Assign each battle's two players to side A and side B by a fair coin.

    Storage orders sides by player tag, which is not neutral: the
    alphabetically earlier tag wins about 50.6 percent of the time, because
    Supercell issues tags roughly in creation order and older accounts have had
    longer to level their cards. Slot A therefore leaks account age, which is
    correlated with the card level effect this study measures.

    A seeded fair coin removes that leak while staying reproducible.
    """
    flip = np.random.default_rng(seed).random(len(battles)) < 0.5
    swapped = battles.copy()

    for suffix in _SWAP_SUFFIXES:
        a_column, b_column = f"a_{suffix}", f"b_{suffix}"
        a_values = battles[a_column].to_numpy()
        b_values = battles[b_column].to_numpy()
        swapped[a_column] = np.where(flip, b_values, a_values)
        swapped[b_column] = np.where(flip, a_values, b_values)

    swapped["a_won"] = np.where(flip, 1 - battles["a_won"].to_numpy(),
                                battles["a_won"].to_numpy()).astype("int8")
    return swapped


def read_season(directory: Path, exclude: tuple[str, ...] = (),
                seed: int = 0, columns: list[str] | None = None) -> pd.DataFrame:
    """Load Parquet days from `directory` with sides assigned by a fair coin."""
    frames = (pd.read_parquet(path, columns=columns)
              for path in season_files(directory, exclude))
    return randomise_sides(pd.concat(frames, ignore_index=True), seed=seed)


def convert_file(source: Path, destination: Path) -> ConversionReport:
    """Stream one Season 18 CSV into a Parquet file, one chunk at a time.

    Chunks are written as they are read rather than accumulated, because the
    largest source file holds about 16 million rows and would not fit in memory.

    Raises ValueError when `source` lacks a required column.
    """
    absent = missing_columns(source)
    if absent:
        raise ValueError(f"{source.name} is missing required columns: {absent}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    
    reader = pd.read_csv(source, usecols=_source_columns(),
                         chunksize=CHUNK_ROWS, low_memory=False)

    writer, rows_read, rows_written, wins = None, 0, 0, 0
    try:
        for chunk in reader:
            rows_read += len(chunk)
            frame = _canonicalise(chunk.dropna(subset=essential_columns()))
            if frame.empty:
                continue
            batch = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, batch.schema, compression="zstd")
            writer.write_table(batch)
            rows_written += len(frame)
            wins += int(frame["a_won"].sum())
    finally:
        if writer is not None:
            writer.close()

    return ConversionReport(
        source=source.name, rows_read=rows_read, rows_written=rows_written,
        side_a_win_rate=wins / rows_written if rows_written else float("nan"))
