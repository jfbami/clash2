"""Seed a player cohort balanced across trophy bands.

Usage:  python scripts/seed_spread_cohort.py --target 50000 --survey-only

Runs in three steps. Clan discovery searches by name to reach clans the ranking
endpoints never return. Member fetching turns clans into players with trophies.
Stratification takes an equal count from each trophy band.

`--survey-only` stops after reporting what each band could supply, so the
cohort size can be chosen against real availability rather than an estimate.

Supersedes `scripts/seed_cohort.py`, which seeds from the clan-war leaderboard
and returns members almost entirely at the 14,000 trophy ceiling.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from crdata.api import CRClient
from crdata.cohort import (DEFAULT_BUCKETS, discover_clans, fetch_members,
                           stratify)

REQUESTS_PER_SECOND = 2.3  # measured, see API_FINDINGS.md


def survey(players: pd.DataFrame, buckets: int) -> None:
    print(f"\n  {len(players):,} distinct players discovered")
    print(f"  trophies: min {players.trophies.min():,} "
          f"median {int(players.trophies.median()):,} max {players.trophies.max():,}")
    _, report = stratify(players, target=len(players), buckets=buckets)
    print("\n  what each trophy band can supply:")
    print(report.as_frame().to_string(index=False))
    smallest = report.available.min()
    print(f"\n  balanced cohort ceiling: {smallest * buckets:,} players "
          f"({smallest:,} per band)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=50_000,
                        help="total players wanted across all bands")
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS)
    parser.add_argument("--per-clan", type=int, default=40)
    parser.add_argument("--max-clans", type=int, default=1500,
                        help="cap on member requests, the expensive step")
    parser.add_argument("--survey-only", action="store_true",
                        help="report band availability and write nothing")
    parser.add_argument("--rediscover", action="store_true",
                        help="re-run clan discovery instead of reusing the cached pool")
    arguments = parser.parse_args()

    client = CRClient()
    root = Path(__file__).resolve().parents[1]
    pool = root / "data" / "state" / "clan_pool.parquet"

    print("step 1: discovering clans by name search", flush=True)
    if pool.exists() and not arguments.rediscover:
        clans = pd.read_parquet(pool)
        print(f"  reusing {len(clans):,} clans from {pool}")
    else:
        started = time.time()
        clans = discover_clans(client)
        pool.parent.mkdir(parents=True, exist_ok=True)
        clans.to_parquet(pool, index=False)
        print(f"  {len(clans):,} distinct clans in {time.time() - started:.0f}s")
    print(f"  clan score: min {clans.clan_score.min():,} "
          f"median {int(clans.clan_score.median()):,} max {clans.clan_score.max():,}")

    clans = clans.sample(n=min(len(clans), arguments.max_clans), random_state=0)
    estimate = len(clans) / REQUESTS_PER_SECOND / 60
    print(f"\nstep 2: fetching members from {len(clans):,} clans "
          f"(~{estimate:.0f} min)", flush=True)
    players = fetch_members(client, clans, arguments.per_clan)

    survey(players, arguments.buckets)
    if arguments.survey_only:
        print("\n  survey only, nothing written")
        return 0

    print(f"\nstep 3: stratifying to {arguments.target:,} players")
    cohort, report = stratify(players, arguments.target, arguments.buckets)
    print(report.as_frame().to_string(index=False))

    destination = root / "data" / "state" / "cohort_spread.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cohort.assign(added_at=pd.Timestamp.now(tz="UTC"), last_polled=pd.NaT,
                  n_polls=0, n_battles=0, n_fail=0).to_parquet(destination, index=False)

    sweep = len(cohort) / REQUESTS_PER_SECOND / 3600
    print(f"\n  {len(cohort):,} players written to {destination}")
    print(f"  one full battlelog sweep takes ~{sweep:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
