"""Verify the converted Season 18 Parquet before any modelling depends on it.

Usage:  python scripts/verify_season18.py

Answers three questions:
  1. Are battles duplicated across day files?
  2. How many battles per player across the season? This decides whether a
     per-player skill parameter is estimable at all.
  3. Does the final day differ from mid-season? This is assumption A1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from crdata.etl import season_files

PARQUET = Path(r"C:\Users\jfbaa\AppData\Local\Temp\claude"
               r"\C--Users-jfbaa-OneDrive-Documents-clash2"
               r"\d24c6794-c5fc-463a-925a-588dd12c92e6\scratchpad\season18_parquet")
FINAL_DAY = "01042021"


def summarise_days() -> pd.DataFrame:
    rows = []
    for path in season_files(PARQUET):
        frame = pd.read_parquet(path, columns=["battle_time", "a_won"])
        rows.append({"file": path.stem.replace("_WL_tagged", ""),
                     "battles": len(frame),
                     "from": frame.battle_time.min(),
                     "to": frame.battle_time.max()})
    return pd.DataFrame(rows)


def duplicate_count() -> tuple[int, int]:
    keys = []
    for path in season_files(PARQUET):
        frame = pd.read_parquet(path, columns=["battle_time", "a_tag", "b_tag"])
        keys.append(frame.battle_time.astype("int64").astype(str)
                    + "|" + frame.a_tag.astype(str) + "|" + frame.b_tag.astype(str))
    all_keys = pd.concat(keys, ignore_index=True)
    return len(all_keys), int(all_keys.duplicated().sum())


def battles_per_player() -> pd.Series:
    tags = []
    for path in season_files(PARQUET):
        frame = pd.read_parquet(path, columns=["a_tag", "b_tag"])
        tags.append(pd.concat([frame.a_tag.astype(str), frame.b_tag.astype(str)],
                              ignore_index=True))
    return pd.concat(tags, ignore_index=True).value_counts()


def compare_final_day_to_midseason() -> None:
    level_columns = [f"a_level{i}" for i in range(1, 9)]
    wanted = level_columns + ["a_trophies", "a_card1"]

    def profile(paths: list[Path]) -> dict:
        frame = pd.concat((pd.read_parquet(p, columns=wanted) for p in paths),
                          ignore_index=True)
        levels = frame[level_columns].to_numpy()
        return {"battles": len(frame),
                "mean card level": float(levels.mean()),
                "mean trophies": float(frame.a_trophies.mean()),
                "distinct cards in slot 1": int(frame.a_card1.nunique())}

    final = [p for p in season_files(PARQUET) if FINAL_DAY in p.name]
    mid = [p for p in season_files(PARQUET) if FINAL_DAY not in p.name]
    print(f"\n{'metric':28s} {'final day':>14s} {'mid-season':>14s} {'difference':>12s}")
    print("-" * 72)
    final_profile, mid_profile = profile(final), profile(mid)
    for key in final_profile:
        a, b = final_profile[key], mid_profile[key]
        gap = f"{a - b:+.3f}" if isinstance(a, float) else f"{a - b:+,}"
        a_text = f"{a:,.3f}" if isinstance(a, float) else f"{a:,}"
        b_text = f"{b:,.3f}" if isinstance(b, float) else f"{b:,}"
        print(f"{key:28s} {a_text:>14s} {b_text:>14s} {gap:>12s}")


def main() -> int:
    print("=" * 72)
    print("SEASON 18 VERIFICATION")
    print("=" * 72)

    days = summarise_days()
    print(days.to_string(index=False))
    print(f"\ntotal battles: {days.battles.sum():,}")

    total, duplicates = duplicate_count()
    print(f"\nduplicate battles across day files: {duplicates:,} of {total:,} "
          f"({100 * duplicates / total:.3f} percent)")

    counts = battles_per_player()
    print(f"\nBATTLES PER PLAYER ACROSS THE SEASON")
    print(f"  distinct players     : {len(counts):,}")
    print(f"  mean battles         : {counts.mean():.1f}")
    print(f"  median battles       : {counts.median():.0f}")
    for threshold in (5, 10, 25, 50, 100):
        qualifying = int((counts >= threshold).sum())
        print(f"  players with >= {threshold:3d}  : {qualifying:9,}  "
              f"covering {int(counts[counts >= threshold].sum()):,} appearances")

    print("\n" + "=" * 72)
    print("ASSUMPTION A1: is the final day representative?")
    print("=" * 72)
    compare_final_day_to_midseason()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
