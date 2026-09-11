"""Build disk-backed next-switch arrays from deduplicated live battles."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from crdata.card_levels import load_card_level_converter
from crdata.sequences import (
    DEFAULT_HISTORY_LENGTH,
    PlayerBattle,
    iter_sequence_examples,
    player_battle_from_live_row,
)
from crdata.vocabulary import load_card_vocabulary


LIVE_COLUMNS = [
    "battle_key", "battle_time", "label_a_win", "a_tag", "b_tag",
    "a_crowns", "b_crowns", "a_trophies", "b_trophies",
    "a_card_ids", "b_card_ids", "a_card_levels", "b_card_levels",
    "is_clean_1v1",
]
ARRAY_NAMES = (
    "cards", "levels", "battle_features", "summary_features", "labels",
    "splits", "player_indices",
)
SPLIT_NAMES = ("train", "validation", "test")


def live_paths(root: Path) -> list[Path]:
    """Return all partitioned live-battle Parquet files."""
    paths = sorted(root.glob("date=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no live battle files found under {root}")
    return paths


def iter_live_rows(paths: list[Path]) -> Iterator[dict]:
    """Stream only fields needed by the sequential model."""
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            columns=LIVE_COLUMNS, batch_size=32_768
        ):
            yield from batch.to_pylist()


def qualified_player_counts(
    paths: list[Path], minimum_battles: int = DEFAULT_HISTORY_LENGTH + 1
) -> Counter[str]:
    """Count unique clean battles per player and retain eligible players."""
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for row in iter_live_rows(paths):
        key = str(row["battle_key"])
        if key in seen or not row["is_clean_1v1"] or row["label_a_win"] is None:
            continue
        seen.add(key)
        counts.update((str(row["a_tag"]), str(row["b_tag"])))
    return Counter({tag: count for tag, count in counts.items() if count >= minimum_battles})


def load_player_battles(
    paths: list[Path], player_tags: set[str]
) -> dict[str, list[PlayerBattle]]:
    """Load focal-player views for eligible players without duplicate battles."""
    battles: dict[str, list[PlayerBattle]] = defaultdict(list)
    seen: set[str] = set()
    for row in iter_live_rows(paths):
        key = str(row["battle_key"])
        if key in seen or not row["is_clean_1v1"] or row["label_a_win"] is None:
            continue
        seen.add(key)
        for tag in (str(row["a_tag"]), str(row["b_tag"])):
            if tag in player_tags:
                battles[tag].append(player_battle_from_live_row(row, tag))
    return battles


def assign_player_splits(player_tags: list[str], seed: int = 20260910) -> dict[str, int]:
    """Assign players reproducibly to 70/15/15 train, validation, and test splits."""
    ordered = np.asarray(sorted(player_tags), dtype=object)
    np.random.default_rng(seed).shuffle(ordered)
    train_end = int(0.70 * len(ordered))
    validation_end = int(0.85 * len(ordered))
    return {
        str(tag): (0 if index < train_end else 1 if index < validation_end else 2)
        for index, tag in enumerate(ordered)
    }


def build_switch_array_cache(
    data_root: Path,
    card_reference: Path,
    destination: Path,
    seed: int = 20260910,
) -> dict:
    """Build memory-mappable arrays and return their audit metadata."""
    paths = live_paths(data_root)
    counts = qualified_player_counts(paths)
    print(f"found {len(counts):,} eligible players", flush=True)
    players = load_player_battles(paths, set(counts))
    player_tags = sorted(players)
    split_by_player = assign_player_splits(player_tags, seed)
    example_count = sum(len(battles) - DEFAULT_HISTORY_LENGTH for battles in players.values())
    if example_count < 1:
        raise ValueError("no eligible next-switch examples were found")

    destination.mkdir(parents=True, exist_ok=True)
    shapes = {
        "cards": (example_count, DEFAULT_HISTORY_LENGTH, 8),
        "levels": (example_count, DEFAULT_HISTORY_LENGTH, 8),
        "battle_features": (example_count, DEFAULT_HISTORY_LENGTH, 6),
        "summary_features": (example_count, 8),
        "labels": (example_count,),
        "splits": (example_count,),
        "player_indices": (example_count,),
    }
    dtypes = {
        "cards": np.uint16,
        "levels": np.uint8,
        "battle_features": np.float32,
        "summary_features": np.float32,
        "labels": np.uint8,
        "splits": np.uint8,
        "player_indices": np.uint32,
    }
    arrays = {
        name: np.lib.format.open_memmap(
            destination / f"{name}.npy", mode="w+", dtype=dtypes[name], shape=shape
        )
        for name, shape in shapes.items()
    }
    vocabulary = load_card_vocabulary(card_reference)
    level_converter = load_card_level_converter(card_reference)

    position = 0
    for player_index, tag in enumerate(player_tags):
        for example in iter_sequence_examples(players[tag]):
            arrays["cards"][position] = vocabulary.encode(example.deck_ids)
            displayed = level_converter.convert(example.deck_ids, example.card_levels)
            arrays["levels"][position] = displayed.astype(np.uint8)
            arrays["battle_features"][position] = example.battle_features
            arrays["summary_features"][position] = example.summary_features
            arrays["labels"][position] = example.next_switch
            arrays["splits"][position] = split_by_player[tag]
            arrays["player_indices"][position] = player_index
            position += 1
        if (player_index + 1) % 500 == 0 or player_index + 1 == len(player_tags):
            print(
                f"  cached {player_index + 1:,}/{len(player_tags):,} players, "
                f"{position:,}/{example_count:,} examples",
                flush=True,
            )

    for array in arrays.values():
        array.flush()
    split_array = arrays["splits"]
    label_array = arrays["labels"]
    metadata = {
        "examples": example_count,
        "players": len(player_tags),
        "embedding_rows": vocabulary.embedding_rows,
        "seed": seed,
        "battle_files": len(paths),
        "splits": {
            name: {
                "players": sum(value == index for value in split_by_player.values()),
                "examples": int(np.sum(split_array == index)),
                "switch_rate": float(np.mean(label_array[split_array == index])),
            }
            for index, name in enumerate(SPLIT_NAMES)
        },
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def load_switch_arrays(cache: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Open an existing cache without loading every array into memory."""
    metadata_path = cache / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"switch array cache is missing under {cache}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = {
        name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in ARRAY_NAMES
    }
    return arrays, metadata
