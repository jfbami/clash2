"""Print one real RoyaleAPI next-switch example for manual inspection.

Usage:
    python scripts/inspect_sequence_example.py
    python scripts/inspect_sequence_example.py --player-tag '#PLAYER_TAG'
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crdata.sequences import (
    DEFAULT_HISTORY_LENGTH,
    PlayerBattle,
    build_sequence_example,
    player_battle_from_live_row,
)
from crdata.vocabulary import UNKNOWN_INDEX, load_card_vocabulary


LIVE_COLUMNS = [
    "battle_key",
    "battle_time",
    "label_a_win",
    "a_tag",
    "b_tag",
    "a_crowns",
    "b_crowns",
    "a_trophies",
    "b_trophies",
    "a_card_ids",
    "b_card_ids",
    "a_card_levels",
    "b_card_levels",
    "is_clean_1v1",
]


def live_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("date=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no live battle files found under {root}")
    return paths


def iter_live_rows(paths: list[Path]) -> Iterator[dict]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=LIVE_COLUMNS, batch_size=32_768):
            yield from batch.to_pylist()


def find_player_with_history(paths: list[Path], required_battles: int) -> str:
    battle_keys_by_player: dict[str, set[str]] = defaultdict(set)
    for row in iter_live_rows(paths):
        if not row["is_clean_1v1"]:
            continue
        battle_key = str(row["battle_key"])
        for tag in (row["a_tag"], row["b_tag"]):
            seen = battle_keys_by_player[str(tag)]
            seen.add(battle_key)
            if len(seen) >= required_battles:
                return str(tag)
    raise ValueError(f"no player has {required_battles} unique clean battles")


def load_player_battles(paths: list[Path], player_tag: str) -> list[PlayerBattle]:
    rows_by_key: dict[str, dict] = {}
    for row in iter_live_rows(paths):
        if row["is_clean_1v1"] and player_tag in (row["a_tag"], row["b_tag"]):
            rows_by_key[str(row["battle_key"])] = row
    return [player_battle_from_live_row(row, player_tag) for row in rows_by_key.values()]


def print_example(example, vocabulary) -> None:
    deck_indices = vocabulary.encode(example.deck_ids)
    print(f"player: {example.player_tag}")
    print(f"known cards: {vocabulary.card_count}")
    print(f"embedding rows: {vocabulary.embedding_rows}")
    print(f"unknown card values: {int((deck_indices == UNKNOWN_INDEX).sum())}")
    print(f"deck_ids shape: {example.deck_ids.shape}")
    print(f"deck_indices shape: {deck_indices.shape}")
    print(f"card_levels shape: {example.card_levels.shape}")
    print(f"battle_features shape: {example.battle_features.shape}")
    print(f"feature order: {', '.join(example.feature_names)}")
    print()

    for position, battle_time in enumerate(example.history_times):
        named_features = ", ".join(
            f"{name}={example.battle_features[position, column]:.4g}"
            for column, name in enumerate(example.feature_names)
        )
        print(f"history battle {position + 1:02d} at {battle_time.isoformat()}")
        print(f"  deck ids: {example.deck_ids[position].tolist()}")
        print(f"  indices:  {deck_indices[position].tolist()}")
        print(f"  levels:   {example.card_levels[position].tolist()}")
        print(f"  features: {named_features}")

    print()
    print(f"target battle at {example.target_time.isoformat()}")
    print(f"target deck ids: {list(example.target_deck_ids)}")
    print(f"next_switch: {int(example.next_switch)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/battles"))
    parser.add_argument(
        "--card-reference", type=Path, default=Path("data/reference/cards.parquet")
    )
    parser.add_argument("--player-tag")
    parser.add_argument("--history-length", type=int, default=DEFAULT_HISTORY_LENGTH)
    parser.add_argument("--window-start", type=int, default=0)
    arguments = parser.parse_args()

    paths = live_paths(arguments.data_root)
    required = arguments.window_start + arguments.history_length + 1
    player_tag = arguments.player_tag or find_player_with_history(paths, required)
    battles = load_player_battles(paths, player_tag)
    example = build_sequence_example(
        battles,
        history_length=arguments.history_length,
        window_start=arguments.window_start,
    )
    vocabulary = load_card_vocabulary(arguments.card_reference)
    print_example(example, vocabulary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
