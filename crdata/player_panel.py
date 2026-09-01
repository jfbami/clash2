"""Build a one-row-per-player table from Season 18 battles.

A player archetype is a description of behaviour over time: how often someone
changes deck, whether losing triggers the change, how far they drift when they
do. None of that is visible in a single battle, so the season has to be turned
inside out. Battles are stored one row per pairing; behaviour needs one row per
player per battle, ordered in time.

That reshape doubles the row count to 73.7 million, which does not fit in
memory. `write_player_battles` melts each battle into its two sides and
hash-partitions the result by player tag, so every battle a player fought lands
in the same shard. `shard_features` then handles one shard at a time, and a
shard is small enough to sort.

Nothing is filtered. Players with one battle are emitted alongside players with
five hundred, and the `battles` column lets any threshold be applied later.
`CONVENTIONS.md` requires filtering at modelling time, not collection time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CARDS_PER_DECK = 8
LADDER_MODES = (72000006, 72000201, 72000044)
BATCH_ROWS = 200_000
SHARDS = 64
DECK_BASE = 128  # exceeds the 102 card indices, so packing a deck is collision-free


@dataclass(frozen=True)
class ShardReport:
    rows_written: int
    shards: int


def _card_columns(side: str) -> list[str]:
    return [f"{side}_card{index}" for index in range(1, CARDS_PER_DECK + 1)]


def _level_columns(side: str) -> list[str]:
    return [f"{side}_level{index}" for index in range(1, CARDS_PER_DECK + 1)]


def _source_columns() -> list[str]:
    return (["battle_time", "game_mode", "a_won", "a_tag", "b_tag",
             "a_trophies", "b_trophies"]
            + _card_columns("a") + _card_columns("b")
            + _level_columns("a") + _level_columns("b"))


def card_index(card_ids: np.ndarray) -> np.ndarray:
    """Map raw card ids onto 0..101 so a deck fits in eight bytes."""
    lookup = np.full(int(card_ids.max()) + 1, -1, dtype=np.int8)
    lookup[card_ids] = np.arange(len(card_ids), dtype=np.int8)
    return lookup


def pack_decks(cards: np.ndarray) -> np.ndarray:
    """Pack eight sorted card indices into one int64 key.

    Each index is below 102, so base 128 gives every distinct deck a distinct
    key with no collisions, unlike a hash.
    """
    weights = DECK_BASE ** np.arange(CARDS_PER_DECK, dtype=np.int64)
    return (cards.astype(np.int64) * weights).sum(axis=1)


def _one_side(frame: pd.DataFrame, side: str, won: np.ndarray,
              lookup: np.ndarray) -> pd.DataFrame:
    """Emit this side's view of every battle: who played what, when, and the result."""
    cards = np.sort(lookup[frame[_card_columns(side)].to_numpy(np.int64)], axis=1)
    output = {
        "tag": frame[f"{side}_tag"].to_numpy(),
        "minute": (frame["battle_time"].astype("int64") // 60_000_000_000).astype("int32"),
        "deck": pack_decks(cards),
        "total_level": frame[_level_columns(side)].to_numpy(np.int16).sum(axis=1).astype("int16"),
        "trophies": frame[f"{side}_trophies"].to_numpy("int16"),
        "won": won.astype("int8"),
    }
    for position in range(CARDS_PER_DECK):
        output[f"card{position + 1}"] = cards[:, position]
    return pd.DataFrame(output)


def _shard_of(tags: np.ndarray, shards: int) -> np.ndarray:
    """Deterministic across processes, unlike the built-in hash of a string."""
    return (pd.util.hash_array(tags) % shards).astype("int16")


def write_player_battles(files: list[Path], destination: Path, card_ids: np.ndarray,
                         shards: int = SHARDS,
                         modes: tuple[int, ...] = LADDER_MODES) -> ShardReport:
    """Melt every battle into two player rows and hash-partition them by tag."""
    destination.mkdir(parents=True, exist_ok=True)
    lookup = card_index(card_ids)
    writers: dict[int, pq.ParquetWriter] = {}
    rows = 0

    try:
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=BATCH_ROWS, columns=_source_columns()):
                frame = batch.to_pandas()
                frame = frame[frame.game_mode.isin(modes)]
                if frame.empty:
                    continue

                a_won = frame["a_won"].to_numpy(np.int8)
                melted = pd.concat(
                    [_one_side(frame, "a", a_won, lookup),
                     _one_side(frame, "b", 1 - a_won, lookup)], ignore_index=True)
                melted["shard"] = _shard_of(melted["tag"].to_numpy(), shards)

                for shard, part in melted.groupby("shard", sort=False):
                    table = pa.Table.from_pandas(
                        part.drop(columns="shard"), preserve_index=False)
                    if shard not in writers:
                        writers[shard] = pq.ParquetWriter(
                            destination / f"shard={shard:03d}.parquet",
                            table.schema, compression="zstd")
                    writers[shard].write_table(table)
                rows += len(melted)
            print(f"  melted {path.name}  running total {rows:,}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    return ShardReport(rows_written=rows, shards=len(writers))


def _shared_card_counts(cards: np.ndarray) -> np.ndarray:
    """How many cards each deck shares with the row before it."""
    matches = cards[1:, :, None] == cards[:-1, None, :]
    return matches.any(axis=2).sum(axis=1).astype(np.int8)


def shard_features(path: Path) -> pd.DataFrame:
    """Reduce one shard of player-battle rows to one row per player.

    `total_level` moves both when a player upgrades a card and when they switch
    to a deck of different levels, so `level_gain` is not a clean upgrade
    measure. It is reported as observed and interpreted downstream.
    """
    frame = pd.read_parquet(path).sort_values(["tag", "minute"], kind="stable")
    frame = frame.reset_index(drop=True)

    tag = frame["tag"].to_numpy()
    follows = np.zeros(len(frame), bool)
    follows[1:] = tag[1:] == tag[:-1]

    deck = frame["deck"].to_numpy()
    switched = np.zeros(len(frame), bool)
    switched[1:] = follows[1:] & (deck[1:] != deck[:-1])

    previous_won = np.zeros(len(frame), np.int8)
    previous_won[1:] = frame["won"].to_numpy()[:-1]

    cards = frame[[f"card{index}" for index in range(1, CARDS_PER_DECK + 1)]].to_numpy(np.int8)
    shared = np.zeros(len(frame), np.int8)
    shared[1:] = _shared_card_counts(cards)
    # Both decks hold 8 distinct cards, so the union is 16 minus the overlap.
    drift = np.where(switched, 1.0 - shared / (2 * CARDS_PER_DECK - shared), np.nan)

    frame = frame.assign(follows=follows, switched=switched,
                         previous_won=previous_won, drift=drift)
    consecutive = frame[frame.follows]

    features = frame.groupby("tag").agg(
        battles=("won", "size"),
        win_rate=("won", "mean"),
        distinct_decks=("deck", "nunique"),
        level_mean=("total_level", "mean"),
        level_first=("total_level", "first"),
        level_last=("total_level", "last"),
        level_max=("total_level", "max"),
        trophies_first=("trophies", "first"),
        trophies_last=("trophies", "last"))

    features["top_deck_share"] = (
        frame.groupby(["tag", "deck"]).size().groupby("tag").max() / features["battles"])
    features["switch_rate"] = consecutive.groupby("tag")["switched"].mean()
    features["switch_after_loss"] = (
        consecutive[consecutive.previous_won == 0].groupby("tag")["switched"].mean())
    features["switch_after_win"] = (
        consecutive[consecutive.previous_won == 1].groupby("tag")["switched"].mean())
    features["loss_reactivity"] = features.switch_after_loss - features.switch_after_win
    features["switch_drift"] = consecutive.groupby("tag")["drift"].mean()
    features["level_gain"] = features.level_last - features.level_first
    features["trophy_gain"] = features.trophies_last - features.trophies_first

    return features.reset_index()
