from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from crdata.sequences import (
    FEATURE_NAMES,
    PlayerBattle,
    build_sequence_example,
    jaccard_distance,
    player_battle_from_live_row,
)


PLAYER = "#PLAYER"
BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def battle(position: int, deck: tuple[int, ...]) -> PlayerBattle:
    return PlayerBattle(
        battle_key=f"battle-{position}",
        battle_time=BASE_TIME + timedelta(hours=position),
        player_tag=PLAYER,
        deck_ids=deck,
        card_levels=tuple(float(position + 1) for _ in deck),
        result=1 if position % 2 == 0 else -1,
        crown_difference=1 if position % 2 == 0 else -1,
        trophies=float(5_000 + position),
    )


class SequenceExampleTests(unittest.TestCase):
    def test_next_battle_supplies_only_the_switch_target(self) -> None:
        original = tuple(range(1, 9))
        changed = tuple(range(2, 10))
        battles = [battle(position, original) for position in range(10)]
        battles.append(battle(10, changed))

        example = build_sequence_example(reversed(battles))

        self.assertEqual(example.deck_ids.shape, (10, 8))
        self.assertEqual(example.card_levels.shape, (10, 8))
        self.assertEqual(example.battle_features.shape, (10, len(FEATURE_NAMES)))
        self.assertEqual(int(example.next_switch), 1)
        np.testing.assert_array_equal(example.deck_ids[-1], original)
        self.assertEqual(example.target_deck_ids, changed)
        self.assertNotIn(changed[-1], example.deck_ids)

    def test_transition_features_use_only_history_battles(self) -> None:
        original = tuple(range(1, 9))
        changed = tuple(range(2, 10))
        battles = [battle(position, original if position < 5 else changed) for position in range(11)]

        example = build_sequence_example(battles)
        changed_column = FEATURE_NAMES.index("changed_deck")
        magnitude_column = FEATURE_NAMES.index("switch_magnitude")
        gap_column = FEATURE_NAMES.index("log1p_time_gap_hours")

        self.assertEqual(example.battle_features[0, changed_column], 0.0)
        self.assertEqual(example.battle_features[5, changed_column], 1.0)
        self.assertAlmostEqual(
            float(example.battle_features[5, magnitude_column]),
            jaccard_distance(original, changed),
        )
        self.assertAlmostEqual(float(example.battle_features[1, gap_column]), np.log(2.0))

    def test_live_row_is_oriented_around_the_requested_player(self) -> None:
        row = {
            "battle_key": "key",
            "battle_time": BASE_TIME,
            "label_a_win": 1,
            "a_tag": "#A",
            "b_tag": "#B",
            "a_crowns": 3,
            "b_crowns": 1,
            "a_trophies": 5_000,
            "b_trophies": 4_990,
            "a_card_ids": [8, 7, 6, 5, 4, 3, 2, 1],
            "b_card_ids": [18, 17, 16, 15, 14, 13, 12, 11],
            "a_card_levels": [8, 7, 6, 5, 4, 3, 2, 1],
            "b_card_levels": [18, 17, 16, 15, 14, 13, 12, 11],
            "is_clean_1v1": True,
        }

        oriented = player_battle_from_live_row(row, "#B")

        self.assertEqual(oriented.result, -1)
        self.assertEqual(oriented.crown_difference, -2)
        self.assertEqual(oriented.deck_ids, tuple(range(11, 19)))
        self.assertEqual(oriented.card_levels, tuple(float(x) for x in range(11, 19)))

    def test_duplicate_battle_keys_are_rejected(self) -> None:
        decks = tuple(range(1, 9))
        battles = [battle(position, decks) for position in range(11)]
        battles[-1] = PlayerBattle(
            battle_key=battles[0].battle_key,
            battle_time=battles[-1].battle_time,
            player_tag=PLAYER,
            deck_ids=decks,
            card_levels=battles[-1].card_levels,
            result=battles[-1].result,
            crown_difference=battles[-1].crown_difference,
            trophies=battles[-1].trophies,
        )

        with self.assertRaisesRegex(ValueError, "duplicate battle keys"):
            build_sequence_example(battles)


if __name__ == "__main__":
    unittest.main()
