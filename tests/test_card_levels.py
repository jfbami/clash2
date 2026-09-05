from __future__ import annotations

import unittest

import numpy as np

from crdata.card_levels import CardLevelConverter, fit_level_standardization


class CardLevelConverterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.converter = CardLevelConverter({1: "common", 2: "rare", 3: "epic", 4: "legendary", 5: "champion"})

    def test_rarity_offsets_produce_one_displayed_level_scale(self) -> None:
        card_ids = np.asarray([[1, 2, 3, 4, 5]])
        api_levels = np.asarray([[11, 9, 6, 3, 1]])

        displayed_levels = self.converter.convert(card_ids, api_levels)

        np.testing.assert_array_equal(displayed_levels, [[11, 11, 11, 11, 11]])

    def test_conversion_preserves_shape_and_level_differences(self) -> None:
        card_ids = np.asarray([[3, 3], [2, 2]])
        api_levels = np.asarray([[6, 7], [9, 11]])

        displayed_levels = self.converter.convert(card_ids, api_levels)

        np.testing.assert_array_equal(displayed_levels, [[11, 12], [11, 13]])

    def test_unknown_card_rarity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "has no known rarity"):
            self.converter.convert(np.asarray([999]), np.asarray([6]))


class LevelStandardizationTests(unittest.TestCase):
    def test_fit_uses_training_levels_and_transform_reuses_the_statistics(self) -> None:
        statistics = fit_level_standardization(np.asarray([10.0, 12.0, 14.0]))

        transformed = statistics.transform(np.asarray([statistics.mean]))

        self.assertEqual(float(transformed[0]), 0.0)
        self.assertAlmostEqual(statistics.mean, 12.0)
        self.assertAlmostEqual(statistics.standard_deviation, np.sqrt(8.0 / 3.0))

    def test_constant_training_levels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonzero variance"):
            fit_level_standardization(np.asarray([11.0, 11.0]))


if __name__ == "__main__":
    unittest.main()
