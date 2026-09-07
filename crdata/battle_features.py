"""Fit and apply leakage-safe preprocessing for sequential battle features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crdata.sequences import FEATURE_NAMES


CONTINUOUS_FEATURE_NAMES = (
    "crown_difference",
    "switch_magnitude",
    "log1p_time_gap_hours",
    "trophies",
)
MODEL_FEATURE_NAMES = FEATURE_NAMES + ("trophies_missing",)
_CONTINUOUS_INDICES = tuple(FEATURE_NAMES.index(name) for name in CONTINUOUS_FEATURE_NAMES)
_TROPHIES_INDEX = FEATURE_NAMES.index("trophies")


@dataclass(frozen=True)
class BattleFeatureStandardization:
    """Training-set statistics for the four continuous battle features."""

    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]

    def transform(self, battle_features: np.ndarray) -> np.ndarray:
        """Standardize continuous columns and append trophy missingness."""
        features = _validated_feature_array(battle_features).copy()
        trophy_missing = np.isnan(features[..., _TROPHIES_INDEX])
        means = np.asarray(self.means, dtype=np.float32)
        standard_deviations = np.asarray(
            self.standard_deviations,
            dtype=np.float32,
        )
        if (
            means.shape != (len(_CONTINUOUS_INDICES),)
            or standard_deviations.shape != means.shape
        ):
            raise ValueError("standardization statistics have the wrong shape")
        if not np.isfinite(means).all() or not np.isfinite(standard_deviations).all():
            raise ValueError("standardization statistics must be finite")
        if np.any(standard_deviations <= 0.0):
            raise ValueError("standard deviations must be positive")

        features[..., _TROPHIES_INDEX] = np.where(
            trophy_missing,
            means[-1],
            features[..., _TROPHIES_INDEX],
        )
        features[..., _CONTINUOUS_INDICES] = (
            features[..., _CONTINUOUS_INDICES] - means
        ) / standard_deviations
        return np.concatenate(
            (features, trophy_missing.astype(np.float32)[..., None]),
            axis=-1,
        )


def fit_battle_feature_standardization(
    training_features: np.ndarray,
) -> BattleFeatureStandardization:
    """Fit population statistics using only training battle features."""
    features = _validated_feature_array(training_features)
    continuous = features[..., _CONTINUOUS_INDICES].astype(np.float64)
    reduction_axes = tuple(range(continuous.ndim - 1))
    observed_counts = np.sum(~np.isnan(continuous), axis=reduction_axes)
    if np.any(observed_counts == 0):
        raise ValueError("every continuous feature needs an observed training value")
    means = np.nanmean(continuous, axis=reduction_axes)
    standard_deviations = np.nanstd(
        continuous,
        axis=reduction_axes,
    )
    if not np.isfinite(standard_deviations).all() or np.any(
        standard_deviations == 0.0
    ):
        raise ValueError("continuous training features must have nonzero variance")
    return BattleFeatureStandardization(
        tuple(float(value) for value in means),
        tuple(float(value) for value in standard_deviations),
    )


def _validated_feature_array(battle_features: np.ndarray) -> np.ndarray:
    features = np.asarray(battle_features, dtype=np.float32)
    if features.ndim < 1 or features.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(f"battle_features must end with {len(FEATURE_NAMES)} columns")
    if features.size == 0:
        raise ValueError("battle_features cannot be empty")
    non_trophy = np.delete(features, _TROPHIES_INDEX, axis=-1)
    if not np.isfinite(non_trophy).all():
        raise ValueError("only trophies may be missing")
    trophies = features[..., _TROPHIES_INDEX]
    if np.isinf(trophies).any():
        raise ValueError("trophies cannot be infinite")
    return features
