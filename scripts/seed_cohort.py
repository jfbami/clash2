"""Build a FIXED cohort of players to poll repeatedly.

A fixed cohort is the whole point: polling the same players over days is what
produces within-player deck variation, which is what lets deck effects be
estimated free of player skill and account investment. A one-shot BFS crawl
gives a wide shallow snapshot instead - the shape that makes leakage unavoidable.

Usage:  python scripts/seed_cohort.py --clans 60 --per-clan 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
import pandas as pd

from crdata.api import CRClient
from crdata.store import load_cohort, save_cohort

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clans", type=int, default=60, help="war clans to pull")
    ap.add_argument("--per-clan", type=int, default=30, help="members per clan")
    ap.add_argument("--locations", nargs="*", default=["global"],
                    help="e.g. global 57000000 57000001")
    args = ap.parse_args()

    c = CRClient()
    now = datetime.now(timezone.utc)
    rows, seen_clans = [], set()

    for loc in args.locations:
        try:
            clans = c.top_war_clans(limit=args.clans, location=loc)
        except Exception as e:
            print(f"  [{loc}] clan ranking failed: {e}")
            continue
        print(f"  [{loc}] {len(clans)} war clans")
        for cl in clans:
            if cl["tag"] in seen_clans:
                continue
            seen_clans.add(cl["tag"])
            try:
                members = c.clan_members(cl["tag"])
            except Exception:
                continue
            for m in members[: args.per_clan]:
                rows.append({
                    "tag": m["tag"], "name": m.get("name"),
                    "clan_tag": cl["tag"], "added_at": now,
                    "last_polled": pd.NaT, "n_polls": 0,
                    "n_battles": 0, "n_fail": 0,
                })

    new = pd.DataFrame(rows).drop_duplicates("tag")
    old = load_cohort()
    if len(old):
        new = new[~new.tag.isin(old.tag)]
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new
    save_cohort(merged)

    print(f"\n  clans visited : {len(seen_clans)}")
    print(f"  new players   : {len(new)}")
    print(f"  cohort total  : {len(merged)}")
    print(f"\n  At ~2.3 req/s one full pass takes ~{len(merged)/2.3/60:.0f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
