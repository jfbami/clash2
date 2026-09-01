"""Count and plot the win rate against card level advantage.

Usage:  python scripts/plot_card_level_effect.py

Writes `figures/card_level_effect.png` and prints the numbers behind it.
Nothing is fitted. Every point is a counted win rate.

Matchmaking pairs players within 50 trophies in 99.86 percent of battles, so a
counted win rate is already a comparison between players at the same trophy
count. `trophy_matched_win_rate` is printed as a check and deliberately not
plotted: standardising to the season-wide trophy mix pushes a plus 20 advantage
into trophy bands holding a few hundred battles, where the estimate is driven
by who ends up there rather than by card levels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crdata.level_effect import (CARDS_PER_DECK, Curve, LevelCounts, count_season,
                                 overall_win_rate, season_files, trophy_matched_win_rate)

PARQUET = Path(r"C:\Users\jfbaa\AppData\Local\Temp\claude"
               r"\C--Users-jfbaa-OneDrive-Documents-clash2"
               r"\d24c6794-c5fc-463a-925a-588dd12c92e6\scratchpad\season18_parquet")
EXCLUDED = ("01042021",)  # D9: the final day of the season is not representative.
PLOT_RANGE = 20  # in summed levels; the axis is drawn per card
CURVE_COLOUR = "#c1440e"


def _visible(curve: Curve) -> Curve:
    keep = np.abs(curve.advantage) <= PLOT_RANGE
    return Curve(curve.advantage[keep], curve.win_rate[keep], curve.battles[keep])


def _annotate(axes, curve: Curve, advantage: int, offset) -> None:
    rate = 100 * curve.at(advantage)
    axes.annotate(f"{rate:.0f}%", xy=(advantage / CARDS_PER_DECK, rate), xytext=offset,
                  textcoords="offset points", color=CURVE_COLOUR, fontsize=12,
                  fontweight="bold",
                  arrowprops=dict(arrowstyle="-", color=CURVE_COLOUR, linewidth=1))


def draw(curve: Curve, counts: LevelCounts, battles: int, destination: Path) -> None:
    figure, axes = plt.subplots(figsize=(10.5, 6.5))
    frequency = axes.twinx()

    per_card = curve.advantage / CARDS_PER_DECK
    share = counts.battles.sum(axis=0)[np.isin(counts.advantage_axis, curve.advantage)]
    frequency.bar(per_card, 100 * share / battles, width=0.85 / CARDS_PER_DECK,
                  color="#ddd8d1", zorder=0)
    frequency.set_ylim(0, 90)
    frequency.set_yticks([])
    for edge in ("top", "right"):
        frequency.spines[edge].set_visible(False)

    axes.set_zorder(frequency.get_zorder() + 1)
    axes.patch.set_visible(False)
    axes.axhline(50, color="#999999", linewidth=1, linestyle=":", zorder=1)
    axes.axvline(0, color="#999999", linewidth=1, linestyle=":", zorder=1)
    axes.plot(per_card, 100 * curve.win_rate, color=CURVE_COLOUR,
              linewidth=3, marker="o", markersize=5, zorder=3)

    axes.set_title("How much do card levels matter?", fontsize=18, pad=18, loc="left")
    axes.text(0, 1.025, f"{battles:,} Clash Royale ladder battles, Season 18",
              transform=axes.transAxes, fontsize=11, color="#555555")

    axes.set_xlabel("How many levels higher your cards are than your opponent's, on average",
                    fontsize=12.5, labelpad=10)
    axes.set_ylabel("Your win rate (%)", fontsize=12, labelpad=10)

    limit = PLOT_RANGE / CARDS_PER_DECK
    axes.set_xticks(np.arange(-limit, limit + 0.01, 0.5))
    axes.set_xlim(-limit - 0.12, limit + 0.12)
    axes.set_ylim(14, 88)
    axes.grid(axis="y", alpha=0.18, zorder=0)
    for edge in ("top", "right"):
        axes.spines[edge].set_visible(False)

    for advantage in (4, 8, 12):
        _annotate(axes, curve, advantage, (-46, 10))
    axes.annotate("Even match", xy=(0, 50), xytext=(-104, 26),
                  textcoords="offset points", fontsize=11, color="#666666",
                  arrowprops=dict(arrowstyle="->", color="#999999", linewidth=1))
    axes.text(0.5, -0.185, "Grey bars show how often each gap actually happens.",
              transform=axes.transAxes, fontsize=10, color="#888888", ha="center")

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    print(f"\nwritten {destination}")


def report(overall: Curve, matched: Curve) -> None:
    print(f"\n{'avg levels':>11} {'summed':>8} {'win rate':>10} {'battles':>13} "
          f"{'share':>8} {'trophy-standardised':>21}")
    total = overall.battles.sum()
    for advantage in range(-PLOT_RANGE, PLOT_RANGE + 1, 2):
        index = np.searchsorted(overall.advantage, advantage)
        print(f"{advantage / CARDS_PER_DECK:>+11.3f} {advantage:>+8} "
              f"{100 * overall.win_rate[index]:>9.2f}% "
              f"{overall.battles[index]:>13,} "
              f"{100 * overall.battles[index] / total:>7.2f}% "
              f"{100 * matched.at(advantage):>20.2f}%")


def load_counts(cache: Path) -> LevelCounts:
    """Count the season, or reuse a previous count so the plot can be redrawn."""
    if cache.exists():
        print(f"reusing counts from {cache}")
        stored = np.load(cache)
        return LevelCounts(battles=stored["battles"], wins=stored["wins"])

    files = season_files(PARQUET, EXCLUDED)
    print(f"{len(files)} day files, excluding {EXCLUDED}\n", flush=True)
    counts = count_season(files)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, battles=counts.battles, wins=counts.wins)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recount", action="store_true",
                        help="ignore the cached counts and stream the season again")
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache = root / "data" / "card_level_counts.npz"
    if arguments.recount and cache.exists():
        cache.unlink()

    counts = load_counts(cache)
    overall = _visible(overall_win_rate(counts))
    report(overall, _visible(trophy_matched_win_rate(counts)))

    draw(overall, counts, counts.total(), root / "figures" / "card_level_effect.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
