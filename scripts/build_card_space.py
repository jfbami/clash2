"""Build outcome-blind card embeddings from Season 18 deck co-occurrence.

Usage:  python scripts/build_card_space.py [--dimensions N]

Prints the singular value spectrum so the dimension count can be chosen from the
data rather than guessed, then runs pre-registered checks stated before the space
was built.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from crdata.embedding import LADDER_MODES, count_cooccurrence, factorise, pmi_matrix

SCRATCH = Path(r"C:\Users\jfbaa\AppData\Local\Temp\claude"
               r"\C--Users-jfbaa-OneDrive-Documents-clash2"
               r"\d24c6794-c5fc-463a-925a-588dd12c92e6\scratchpad")
PARQUET = SCRATCH / "season18_parquet"
REFERENCE = SCRATCH / "season18"
EXCLUDED_DAY = "01042021"

PREREGISTERED = [
    ("Fireball", "Poison", "negative", "compete for the same spell slot"),
    ("Hog Rider", "Ice Spirit", "positive", "cycle synergy"),
    ("Golem", "Hog Rider", "negative", "opposite archetypes"),
]


def card_names() -> dict[int, str]:
    frame = pd.read_csv(REFERENCE / "CardMasterListSeason18_12082020.csv")
    return dict(zip(frame["team.card1.id"], frame["team.card1.name"]))


def ladder_files() -> list[Path]:
    return sorted(f for f in PARQUET.glob("*.parquet") if EXCLUDED_DAY not in f.name)


def report_spectrum(singular: np.ndarray) -> None:
    share = singular / singular.sum()
    print("\nSINGULAR VALUE SPECTRUM")
    print(f"  {'k':>3s} {'value':>9s} {'share':>7s} {'cumulative':>11s}")
    for k in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
        if k <= len(singular):
            print(f"  {k:3d} {singular[k-1]:9.2f} {100*share[k-1]:6.2f}% "
                  f"{100*share[:k].sum():10.1f}%")


def check_predictions(pmi: np.ndarray, vectors: np.ndarray,
                      index: dict[str, int]) -> None:
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    print("\nPRE-REGISTERED CHECKS (stated before the space was built)")
    for first, second, expected, reason in PREREGISTERED:
        if first not in index or second not in index:
            print(f"  {first} / {second}: card not found")
            continue
        i, j = index[first], index[second]
        value = pmi[i, j]
        similarity = float(unit[i] @ unit[j])
        sign = "negative" if value < 0 else "positive"
        verdict = "PASS" if sign == expected else "FAIL"
        print(f"  [{verdict}] {first:14s} / {second:14s} "
              f"PMI={value:+6.3f} (expected {expected}, {reason})"
              f"  cosine={similarity:+.3f}")


def nearest(vectors: np.ndarray, index: dict[str, int], names: list[str],
            card: str, k: int = 6) -> None:
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = unit @ unit[index[card]]
    order = np.argsort(-similarity)[1:k + 1]
    neighbours = ", ".join(f"{names[i]} ({similarity[i]:.2f})" for i in order)
    print(f"  {card:18s} -> {neighbours}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=16)
    arguments = parser.parse_args()

    files = ladder_files()
    print(f"{len(files)} day files, Jan 4 excluded")
    print(f"modes kept: {LADDER_MODES}\n")

    started = time.time()
    counts, card_ids, decks_seen = count_cooccurrence(files)
    print(f"counted {decks_seen:,} decks in {time.time()-started:.0f}s")
    print(f"co-occurrence matrix: {counts.shape}, {counts.sum():,} pair observations")

    pmi = pmi_matrix(counts)
    vectors, singular = factorise(pmi, arguments.dimensions)
    report_spectrum(singular)

    lookup = card_names()
    names = [lookup.get(int(c), str(c)) for c in card_ids]
    index = {name: position for position, name in enumerate(names)}

    print(f"\n{len(card_ids)} cards embedded in {arguments.dimensions} dimensions")
    print(f"PMI range: {pmi.min():+.3f} to {pmi.max():+.3f}, "
          f"{100*(pmi < 0).mean():.1f} percent of pairs negative")

    check_predictions(pmi, vectors, index)

    print("\nNEAREST NEIGHBOURS (cosine in embedding space)")
    for card in ("Fireball", "Hog Rider", "Golem", "Skeletons", "Mega Knight"):
        if card in index:
            nearest(vectors, index, names, card)

    output = SCRATCH / "card_space.pkl"
    with open(output, "wb") as handle:
        pickle.dump({"card_ids": card_ids, "vectors": vectors, "pmi": pmi,
                     "counts": counts, "singular": singular, "names": names}, handle)
    print(f"\nsaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
