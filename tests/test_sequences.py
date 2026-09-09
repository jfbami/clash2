from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from crdata.sequences import (
    FEATURE_NAMES,
    SUMMARY_FEATURE_NAMES,
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
        self.assertEqual(example.summary_features.shape, (len(SUMMARY_FEATURE_NAMES),))
        self.assertEqual(int(example.next_switch), 1)
        np.testing.assert_array_equal(example.deck_ids[-1], original)
        self.assertEqual(example.target_deck_ids, changed)
        self.assertNotIn(changed[-1], example.deck_ids)

    def test_summary_features_use_the_expanding_pre_target_prefix(self) -> None:
        deck_a = tuple(range(1, 9))
        deck_b = tuple(range(2, 10))
        deck_c = tuple(range(3, 11))
        decks = [deck_a, deck_a, deck_b, deck_b] + [deck_c] * 7
        battles = [battle(position, deck) for position, deck in enumerate(decks)]

        example = build_sequence_example(battles, history_length=5, window_start=5)
        summary = dict(zip(SUMMARY_FEATURE_NAMES, example.summary_features))

        self.assertAlmostEqual(float(summary["historical_switch_rate"]), 2.0 / 9.0)
        self.assertAlmostEqual(float(summary["post_loss_switch_rate"]), 0.5)
        self.assertEqual(float(summary["post_win_switch_rate"]), 0.0)
        self.assertAlmostEqual(
            float(summary["mean_switch_magnitude"]),
            jaccard_distance(deck_a, deck_b),
        )
        self.assertAlmostEqual(
            float(summary["log1p_current_deck_tenure"]), np.log(7.0), places=6
        )
        self.assertAlmostEqual(
            float(summary["log1p_prior_battle_count"]), np.log(11.0), places=6
        )
        self.assertAlmostEqual(
            float(summary["log1p_post_loss_opportunities"]), np.log(5.0), places=6
        )
        self.assertAlmostEqual(
            float(summary["log1p_post_win_opportunities"]), np.log(6.0), places=6
        )

    def test_undefined_summary_rates_remain_missing_until_preprocessing(self) -> None:
        deck = tuple(range(1, 9))
        battles = [battle(position, deck) for position in range(11)]
        battles = [
            PlayerBattle(
                battle_key=item.battle_key,
                battle_time=item.battle_time,
                player_tag=item.player_tag,
                deck_ids=item.deck_ids,
                card_levels=item.card_levels,
                result=1,
                crown_difference=item.crown_difference,
                trophies=item.trophies,
            )
            for item in battles
        ]

        example = build_sequence_example(battles)
        summary = dict(zip(SUMMARY_FEATURE_NAMES, example.summary_features))

        self.assertTrue(np.isnan(summary["post_loss_switch_rate"]))
        self.assertTrue(np.isnan(summary["mean_switch_magnitude"]))
        self.assertEqual(float(summary["log1p_post_loss_opportunities"]), 0.0)

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

    def test_invalid_result_is_rejected(self) -> None:
        deck = tuple(range(1, 9))
        battles = [battle(position, deck) for position in range(11)]
        battles[3] = PlayerBattle(
            battle_key=battles[3].battle_key,
            battle_time=battles[3].battle_time,
            player_tag=PLAYER,
            deck_ids=deck,
            card_levels=battles[3].card_levels,
            result=0,
            crown_difference=0,
            trophies=battles[3].trophies,
        )

        with self.assertRaisesRegex(ValueError, "minus one or plus one"):
            build_sequence_example(battles)


if __name__ == "__main__":
    unittest.main()
