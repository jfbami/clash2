"""Answer what a Clash Royale win is made of, on one Season 18 ladder file.

Usage:  python scripts/decompose.py <path-to-csv> [--subsample N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from crdata.decompose import (deck_strengths, fit_additive_model, variance_shares,
                              win_probability)
from crdata.season18 import load_season18


def report_shares(model, battles) -> None:
    shares = variance_shares(model, battles)
    percentages = shares.as_percentages()
    print("\nWHAT A WIN IS MADE OF (variance of each component, log-odds scale)")
    print(f"  player skill, via trophies : {percentages['skill']:5.1f}%   (var {shares.skill:.4f})")
    print(f"  account investment, levels : {percentages['investment']:5.1f}%   (var {shares.investment:.4f})")
    print(f"  deck design, card identity : {percentages['deck']:5.1f}%   (var {shares.deck:.4f})")
    print("  Shares are conditional on matchmaking, which equalises trophies before play.")


def report_deck_spread(model, decks) -> None:
    strengths = deck_strengths(model, decks)
    low, high = np.percentile(strengths, [25, 75])
    bottom, top = np.percentile(strengths, [5, 95])
    print("\nHOW MUCH BETTER ARE THE BEST DECKS")
    print(f"  deck strength spread, 25th to 75th percentile : {high - low:.4f} log-odds")
    print(f"  a 75th percentile deck beats a 25th           : {win_probability(high - low)*100:.2f}%")
    print(f"  a 95th percentile deck beats a 5th            : {win_probability(top - bottom)*100:.2f}%")
    print(f"  strongest observed deck beats weakest         : "
          f"{win_probability(strengths.max() - strengths.min())*100:.2f}%")


def report_extreme_cards(model, count: int = 8) -> None:
    order = np.argsort(model.card_strength)
    print(f"\nSTRONGEST AND WEAKEST CARDS (net of card level and trophies)")
    for label, indices in (("strongest", order[::-1][:count]), ("weakest", order[:count])):
        cards = ", ".join(f"{model.card_ids[i]}:{model.card_strength[i]:+.3f}" for i in indices)
        print(f"  {label:9s} {cards}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--subsample", type=int, default=None)
    arguments = parser.parse_args()

    battles = load_season18(arguments.csv, subsample=arguments.subsample)
    print(f"loaded {len(battles):,} battles, {len(battles.card_ids)} distinct cards")

    model = fit_additive_model(battles)
    print(f"\nweights: trophy={model.trophy_weight:+.6f}  level={model.level_weight:+.6f}")

    report_shares(model, battles)

    deck_columns = battles.card_difference
    a_decks = np.array([deck_columns[i].indices[deck_columns[i].data > 0]
                        for i in range(min(len(battles), 200_000))], dtype=object)
    observed = np.array([battles.card_ids[list(d)] for d in a_decks], dtype=object)
    report_deck_spread(model, observed)
    report_extreme_cards(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
