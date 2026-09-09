"""Fit and apply leakage-safe preprocessing for player summary features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crdata.sequences import SUMMARY_FEATURE_NAMES


_FEATURE_COUNT = len(SUMMARY_FEATURE_NAMES)


@dataclass(frozen=True)
class SummaryFeatureStandardization:
    """Training-set statistics for the expanding-prefix summary vector."""

    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]

    def transform(self, summary_features: np.ndarray) -> np.ndarray:
        """Mean-impute undefined rates and standardize every summary column."""
        features = _validated_summary_array(summary_features).copy()
        means = np.asarray(self.means, dtype=np.float32)
        standard_deviations = np.asarray(self.standard_deviations, dtype=np.float32)
        if means.shape != (_FEATURE_COUNT,) or standard_deviations.shape != means.shape:
            raise ValueError("standardization statistics have the wrong shape")
        if not np.isfinite(means).all() or not np.isfinite(standard_deviations).all():
            raise ValueError("standardization statistics must be finite")
        if np.any(standard_deviations <= 0.0):
            raise ValueError("standard deviations must be positive")

        features = np.where(np.isnan(features), means, features)
        return (features - means) / standard_deviations


def fit_summary_feature_standardization(
    training_features: np.ndarray,
) -> SummaryFeatureStandardization:
    """Fit population statistics using only training-player summaries."""
    features = _validated_summary_array(training_features).astype(np.float64)
    reduction_axes = tuple(range(features.ndim - 1))
    observed_counts = np.sum(~np.isnan(features), axis=reduction_axes)
    if np.any(observed_counts == 0):
        raise ValueError("every summary feature needs an observed training value")
    means = np.nanmean(features, axis=reduction_axes)
    standard_deviations = np.nanstd(features, axis=reduction_axes)
    if not np.isfinite(standard_deviations).all() or np.any(
        standard_deviations == 0.0
    ):
        raise ValueError("summary training features must have nonzero variance")
    return SummaryFeatureStandardization(
        tuple(float(value) for value in means),
        tuple(float(value) for value in standard_deviations),
    )


def _validated_summary_array(summary_features: np.ndarray) -> np.ndarray:
    features = np.asarray(summary_features, dtype=np.float32)
    if features.ndim < 1 or features.shape[-1] != _FEATURE_COUNT:
        raise ValueError(f"summary_features must end with {_FEATURE_COUNT} columns")
    if features.size == 0:
        raise ValueError("summary_features cannot be empty")
    if np.isinf(features).any():
        raise ValueError("summary features cannot be infinite")
    return features
