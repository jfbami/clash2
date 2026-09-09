from __future__ import annotations

import unittest

import numpy as np

from crdata.summary_features import fit_summary_feature_standardization


class SummaryFeatureStandardizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training = np.asarray([
            [0.1, np.nan, 0.2, np.nan, 1.0, 2.0, 0.0, 2.0],
            [0.3, 0.4, 0.1, 0.2, 2.0, 3.0, 1.0, 3.0],
            [0.5, 0.2, 0.6, 0.8, 4.0, 5.0, 2.0, 4.0],
        ], dtype=np.float32)

    def test_observed_training_values_are_standardized(self) -> None:
        standardization = fit_summary_feature_standardization(self.training)

        transformed = standardization.transform(self.training)

        for column in range(self.training.shape[1]):
            observed = ~np.isnan(self.training[:, column])
            self.assertAlmostEqual(
                float(transformed[observed, column].mean()), 0.0, places=6
            )
            self.assertAlmostEqual(
                float(transformed[observed, column].std()), 1.0, places=6
            )

    def test_missing_rates_are_imputed_to_training_means(self) -> None:
        transformed = fit_summary_feature_standardization(self.training).transform(
            self.training
        )

        self.assertEqual(float(transformed[0, 1]), 0.0)
        self.assertEqual(float(transformed[0, 3]), 0.0)

    def test_validation_data_reuses_training_statistics(self) -> None:
        standardization = fit_summary_feature_standardization(self.training)
        validation = np.asarray([
            [0.7, 0.3, 0.4, 0.5, 3.0, 4.0, 1.5, 3.5]
        ], dtype=np.float32)

        transformed = standardization.transform(validation)

        expected = (0.7 - self.training[:, 0].mean()) / self.training[:, 0].std()
        self.assertAlmostEqual(float(transformed[0, 0]), float(expected), places=5)

    def test_fit_rejects_an_unobserved_feature(self) -> None:
        training = self.training.copy()
        training[:, 3] = np.nan

        with self.assertRaisesRegex(ValueError, "observed training value"):
            fit_summary_feature_standardization(training)


if __name__ == "__main__":
    unittest.main()
