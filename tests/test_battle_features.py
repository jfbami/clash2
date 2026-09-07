from __future__ import annotations

import unittest

import numpy as np
import torch

from crdata.battle_features import (
    MODEL_FEATURE_NAMES,
    fit_battle_feature_standardization,
)
from crdata.models.battle_encoder import BattleTokenAssembler


class BattleFeatureStandardizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training = np.asarray([
            [-1.0, -2.0, 0.0, 0.0, 0.0, 5000.0],
            [1.0, 2.0, 1.0, 0.5, 1.0, 6000.0],
            [-1.0, 1.0, 0.0, 1.0, 2.0, np.nan],
        ], dtype=np.float32)

    def test_continuous_features_use_training_statistics(self) -> None:
        standardization = fit_battle_feature_standardization(self.training)

        transformed = standardization.transform(self.training)

        complete_continuous = transformed[:, [1, 3, 4]]
        np.testing.assert_allclose(
            complete_continuous.mean(axis=0),
            np.zeros(3),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            complete_continuous.std(axis=0),
            np.ones(3),
            atol=1e-6,
        )
        observed_trophies = transformed[:2, 5]
        self.assertAlmostEqual(float(observed_trophies.mean()), 0.0)
        self.assertAlmostEqual(float(observed_trophies.std()), 1.0)

    def test_binary_features_are_unchanged(self) -> None:
        transformed = fit_battle_feature_standardization(self.training).transform(
            self.training
        )

        np.testing.assert_array_equal(transformed[:, 0], self.training[:, 0])
        np.testing.assert_array_equal(transformed[:, 2], self.training[:, 2])

    def test_missing_trophies_are_mean_imputed_and_flagged(self) -> None:
        transformed = fit_battle_feature_standardization(self.training).transform(
            self.training
        )

        self.assertEqual(MODEL_FEATURE_NAMES[-1], "trophies_missing")
        self.assertEqual(float(transformed[2, 5]), 0.0)
        np.testing.assert_array_equal(transformed[:, 6], np.asarray([0.0, 0.0, 1.0]))

    def test_validation_data_reuses_training_statistics(self) -> None:
        standardization = fit_battle_feature_standardization(self.training)
        validation = np.asarray([[1.0, 4.0, 0.0, 0.25, 3.0, 6500.0]], dtype=np.float32)

        transformed = standardization.transform(validation)

        expected_trophies = (6500.0 - 5500.0) / 500.0
        self.assertEqual(float(transformed[0, 5]), expected_trophies)

    def test_fit_requires_observed_trophies(self) -> None:
        training = self.training.copy()
        training[:, 5] = np.nan

        with self.assertRaisesRegex(ValueError, "observed training value"):
            fit_battle_feature_standardization(training)


class BattleTokenAssemblerTests(unittest.TestCase):
    def test_concatenates_deck_vector_and_features(self) -> None:
        deck_vectors = torch.randn(2, 10, 48)
        battle_features = torch.randn(2, 10, 7)

        tokens = BattleTokenAssembler()(deck_vectors, battle_features)

        self.assertEqual(tokens.shape, (2, 10, 55))
        torch.testing.assert_close(tokens[..., :48], deck_vectors)
        torch.testing.assert_close(tokens[..., 48:], battle_features)

    def test_rejects_mismatched_leading_axes(self) -> None:
        with self.assertRaisesRegex(ValueError, "share leading axes"):
            BattleTokenAssembler()(torch.randn(2, 10, 48), torch.randn(2, 9, 7))

    def test_rejects_raw_feature_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 7 battle features"):
            BattleTokenAssembler()(torch.randn(2, 10, 48), torch.randn(2, 10, 6))


if __name__ == "__main__":
    unittest.main()
