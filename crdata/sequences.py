"""Build player-oriented sequential examples from live battle records.

The first modelling slice predicts whether a player changes decks in the next
battle.
Ten battles form the history and the following battle supplies only the target.
This module deliberately stops at the data contract: it does not embed cards,
normalise features, pad sequences, or construct a neural network.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import log1p
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CARDS_PER_DECK = 8
DEFAULT_HISTORY_LENGTH = 10
FEATURE_NAMES = (
    "result",
    "crown_difference",
    "changed_deck",
    "switch_magnitude",
    "log1p_time_gap_hours",
    "trophies",
)


@dataclass(frozen=True)
class PlayerBattle:
    """One live battle reoriented around the player whose history we model."""

    battle_key: str
    battle_time: datetime
    player_tag: str
    deck_ids: tuple[int, ...]
    card_levels: tuple[float, ...]
    result: int
    crown_difference: int
    trophies: float


@dataclass(frozen=True)
class SequenceExample:
    """One next-switch example with model inputs separated from metadata."""

    deck_ids: np.ndarray
    card_levels: np.ndarray
    battle_features: np.ndarray
    next_switch: np.int8
    player_tag: str
    history_times: tuple[datetime, ...]
    target_time: datetime
    target_deck_ids: tuple[int, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES


def jaccard_distance(deck_a: Sequence[int], deck_b: Sequence[int]) -> float:
    """Return set distance from zero for identical decks to one for disjoint decks."""
    cards_a, cards_b = set(deck_a), set(deck_b)
    union = cards_a | cards_b
    if not union:
        raise ValueError("a deck cannot be empty")
    return 1.0 - len(cards_a & cards_b) / len(union)


def _player_sides(row: Mapping[str, Any], player_tag: str) -> tuple[str, str, int]:
    if player_tag == row["a_tag"]:
        return "a", "b", 1 if int(row["label_a_win"]) == 1 else -1
    if player_tag == row["b_tag"]:
        return "b", "a", 1 if int(row["label_a_win"]) == 0 else -1
    raise ValueError(f"player {player_tag!r} is not in battle {row['battle_key']!r}")


def _ordered_card_levels(row: Mapping[str, Any], side: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
    cards = row[f"{side}_card_ids"]
    levels = row[f"{side}_card_levels"]
    if cards is None or levels is None or len(cards) != CARDS_PER_DECK or len(levels) != CARDS_PER_DECK:
        raise ValueError("a clean sequence battle must contain eight cards and eight levels")

    pairs = sorted(
        ((int(card), float(level)) for card, level in zip(cards, levels)),
        key=lambda pair: pair[0],
    )
    deck_ids = tuple(card for card, _ in pairs)
    if len(set(deck_ids)) != CARDS_PER_DECK:
        raise ValueError("a deck must contain eight distinct cards")
    return deck_ids, tuple(level for _, level in pairs)


def player_battle_from_live_row(
    row: Mapping[str, Any], player_tag: str
) -> PlayerBattle:
    """Orient one clean live row around ``player_tag``.

    Live storage uses canonical tag order for deduplication.
    Sequential examples instead use a focal-player orientation, so result and
    crown difference always describe the player whose history is encoded.
    """
    if not bool(row["is_clean_1v1"]):
        raise ValueError("sequence examples require a clean 1v1 battle")

    side, opponent, result = _player_sides(row, player_tag)
    deck_ids, card_levels = _ordered_card_levels(row, side)
    trophies = row[f"{side}_trophies"]
    return PlayerBattle(
        battle_key=str(row["battle_key"]),
        battle_time=row["battle_time"],
        player_tag=player_tag,
        deck_ids=deck_ids,
        card_levels=card_levels,
        result=result,
        crown_difference=int(row[f"{side}_crowns"]) - int(row[f"{opponent}_crowns"]),
        trophies=float(trophies) if trophies is not None else float("nan"),
    )


def _ordered_player_battles(battles: Iterable[PlayerBattle]) -> list[PlayerBattle]:
    ordered = sorted(battles, key=lambda battle: (battle.battle_time, battle.battle_key))
    keys = [battle.battle_key for battle in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate battle keys must be removed before building a sequence")
    if len({battle.player_tag for battle in ordered}) != 1:
        raise ValueError("all battles in one sequence must belong to the same player")
    return ordered


def _validate_window(history_length: int, window_start: int, battle_count: int) -> None:
    if history_length < 1:
        raise ValueError("history_length must be positive")
    if window_start < 0:
        raise ValueError("window_start cannot be negative")
    required = window_start + history_length + 1
    if battle_count < required:
        raise ValueError(f"need at least {required} battles, received {battle_count}")


def _transition_features(
    previous: PlayerBattle | None, battle: PlayerBattle
) -> tuple[float, float, float]:
    if previous is None:
        return 0.0, 0.0, 0.0
    seconds = (battle.battle_time - previous.battle_time).total_seconds()
    if seconds < 0:
        raise ValueError("battle times must be chronological")
    return (
        float(battle.deck_ids != previous.deck_ids),
        jaccard_distance(previous.deck_ids, battle.deck_ids),
        log1p(seconds / 3600.0),
    )


def _feature_matrix(history: Sequence[PlayerBattle]) -> np.ndarray:
    rows = []
    for position, battle in enumerate(history):
        previous = history[position - 1] if position else None
        changed_deck, switch_magnitude, time_gap = _transition_features(previous, battle)
        rows.append([
            float(battle.result), float(battle.crown_difference), changed_deck,
            switch_magnitude, time_gap, battle.trophies,
        ])
    return np.asarray(rows, dtype=np.float32)


def build_sequence_example(
    battles: Iterable[PlayerBattle],
    history_length: int = DEFAULT_HISTORY_LENGTH,
    window_start: int = 0,
) -> SequenceExample:
    """Build one chronological history and its next-switch target.

    Transition features on the first history row are zero because the model is
    not given the battle preceding the selected window.
    """
    ordered = _ordered_player_battles(battles)
    _validate_window(history_length, window_start, len(ordered))
    history = ordered[window_start:window_start + history_length]
    target = ordered[window_start + history_length]
    return SequenceExample(
        deck_ids=np.asarray([battle.deck_ids for battle in history], dtype=np.int64),
        card_levels=np.asarray([battle.card_levels for battle in history], dtype=np.float32),
        battle_features=_feature_matrix(history),
        next_switch=np.int8(target.deck_ids != history[-1].deck_ids),
        player_tag=history[0].player_tag,
        history_times=tuple(battle.battle_time for battle in history),
        target_time=target.battle_time,
        target_deck_ids=target.deck_ids,
    )
