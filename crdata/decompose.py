"""Split match outcome into deck design, player skill, and account investment.

`fit_additive_model` fits a logit that is linear and separable in three parts, so
each part's contribution to a battle can be read off independently:

    logit P(side A wins) = skill + investment + deck

`variance_shares` then reports how much each part varies across battles, which
answers what a win is actually made of.

The shares are conditional on matchmaking. Matchmaking equalises trophies before
a battle starts, so the skill share measures only the skill gap that survives
matchmaking, not the importance of skill in general.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from crdata.season18 import BattleMatrix


@dataclass(frozen=True)
class AdditiveModel:
    card_strength: np.ndarray
    card_ids: np.ndarray
    level_weight: float
    trophy_weight: float


@dataclass(frozen=True)
class VarianceShares:
    skill: float
    investment: float
    deck: float

    def as_percentages(self) -> dict[str, float]:
        total = self.skill + self.investment + self.deck
        return {"skill": 100 * self.skill / total,
                "investment": 100 * self.investment / total,
                "deck": 100 * self.deck / total}


def _design_matrix(battles: BattleMatrix) -> sparse.csr_matrix:
    scalars = np.c_[battles.trophy_difference, battles.level_difference]
    return sparse.hstack([sparse.csr_matrix(scalars), battles.card_difference]).tocsr()


def fit_additive_model(battles: BattleMatrix, regularisation: float = 1.0) -> AdditiveModel:
    """Fit the separable logit. No intercept, because the model must be antisymmetric."""
    model = LogisticRegression(
        max_iter=3000, C=regularisation, fit_intercept=False, solver="lbfgs")
    model.fit(_design_matrix(battles), battles.side_a_won)

    weights = model.coef_.ravel()
    return AdditiveModel(
        trophy_weight=float(weights[0]),
        level_weight=float(weights[1]),
        card_strength=weights[2:],
        card_ids=battles.card_ids)


def skill_component(model: AdditiveModel, battles: BattleMatrix) -> np.ndarray:
    return model.trophy_weight * battles.trophy_difference


def investment_component(model: AdditiveModel, battles: BattleMatrix) -> np.ndarray:
    return model.level_weight * battles.level_difference


def deck_component(model: AdditiveModel, battles: BattleMatrix) -> np.ndarray:
    return battles.card_difference @ model.card_strength


def variance_shares(model: AdditiveModel, battles: BattleMatrix) -> VarianceShares:
    """Variance of each component's contribution, on the log-odds scale."""
    return VarianceShares(
        skill=float(np.var(skill_component(model, battles))),
        investment=float(np.var(investment_component(model, battles))),
        deck=float(np.var(deck_component(model, battles))))


def deck_strengths(model: AdditiveModel, decks: np.ndarray) -> np.ndarray:
    """Summed card strength for each deck, where `decks` holds card ids per row."""
    strength_of = dict(zip(model.card_ids, model.card_strength))
    return np.array([sum(strength_of.get(card, 0.0) for card in deck) for deck in decks])


def win_probability(strength_gap: float) -> float:
    """Probability the stronger deck wins, given a log-odds gap, all else equal."""
    return float(1.0 / (1.0 + np.exp(-strength_gap)))
