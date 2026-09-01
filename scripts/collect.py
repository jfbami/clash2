"""One polling pass over the cohort. Safe to run repeatedly - dedup is persistent.

Usage:  python scripts/collect.py [--limit N] [--sleep 0.15] [--types riverRacePvP pathOfLegend]
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
import pandas as pd

from crdata.api import CRClient, CRAuthError
from crdata.parse import parse_battle
from crdata.store import COHORT, SeenKeys, append_battles, load_cohort, save_cohort

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# The battlelog endpoint returns at most this many battles. A player who comes
# back with a full page may have played more since the last poll, and those
# battles are gone: the API keeps no history beyond the page. A short page
# proves nothing was missed for that player.
BATTLELOG_PAGE = 30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max players this pass (0=all)")
    ap.add_argument("--sleep", type=float, default=0.15, help="pause between requests")
    ap.add_argument("--types", nargs="*", default=None, help="keep only these battle types")
    ap.add_argument("--cohort", type=Path, default=None,
                    help="cohort file to poll (default: the trophy-balanced cohort)")
    args = ap.parse_args()

    cohort = load_cohort(args.cohort) if args.cohort else load_cohort()
    if cohort.empty:
        print("Cohort is empty. Run scripts/seed_spread_cohort.py first.")
        return 1

    # poll least-recently-polled first so coverage stays even across the cohort
    cohort = cohort.sort_values("last_polled", na_position="first")
    todo = cohort if args.limit == 0 else cohort.head(args.limit)

    client = CRClient()
    seen = SeenKeys()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    print(f"run {run_id} | cohort {len(cohort)} | polling {len(todo)} | already seen {len(seen):,}")

    rows: list[dict] = []
    stats = Counter()
    idx = cohort.set_index("tag")
    # accumulate updates separately: assigning tz-aware stamps into a
    # NaT-typed column row-by-row triggers a pandas dtype coercion warning
    upd: dict[str, dict] = {}
    t0 = time.time()

    for i, tag in enumerate(todo.tag.tolist(), 1):
        try:
            log = client.battlelog(tag)
        except CRAuthError as e:
            print(f"\nAUTH FAILURE - stopping: {e}")
            break
        except Exception:
            stats["player_fail"] += 1
            upd.setdefault(tag, {}).update(n_fail=int(idx.at[tag, "n_fail"]) + 1)
            continue

        # Only a repeat poll can reveal a gap. On a first poll a full page just
        # means the player has ever played 30 battles, not that any were missed.
        if int(idx.at[tag, "n_polls"]) > 0:
            stats["log_full" if len(log) >= BATTLELOG_PAGE else "log_partial"] += 1
        else:
            stats["first_poll"] += 1

        now = datetime.now(timezone.utc)
        kept = 0
        for b in log:
            stats["battles_seen"] += 1
            if args.types and b.get("type") not in args.types:
                stats["filtered_type"] += 1
                continue
            row = parse_battle(b, now, run_id)
            if row is None:
                stats["unparseable"] += 1
                continue
            if row["battle_key"] in seen:
                stats["duplicate"] += 1
                continue
            seen.add(row["battle_key"])
            rows.append(row)
            stats["new"] += 1
            stats[f"type:{row['type']}"] += 1
            if row["is_clean_1v1"]:
                stats["clean_1v1"] += 1
            kept += 1

        upd[tag] = {"last_polled": now,
                    "n_polls": int(idx.at[tag, "n_polls"]) + 1,
                    "n_battles": int(idx.at[tag, "n_battles"]) + kept}

        if i % 25 == 0 or i == len(todo):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(todo)} players | {stats['new']:,} new | "
                  f"{stats['duplicate']:,} dup | {rate:.1f} req/s", end="\r")
        time.sleep(args.sleep)

    if upd:
        u = pd.DataFrame.from_dict(upd, orient="index")
        u.index.name = "tag"
        for col in u.columns:
            idx[col] = u[col].combine_first(idx[col]) if col in idx else u[col]
        idx["last_polled"] = pd.to_datetime(idx["last_polled"], utc=True)

    path = append_battles(rows, run_id)
    seen.save()
    save_cohort(idx.reset_index(), args.cohort or COHORT)

    el = time.time() - t0
    print(f"\n\n{'='*58}\nPASS COMPLETE in {el/60:.1f} min")
    print(f"  battles seen      : {stats['battles_seen']:,}")
    print(f"  new (deduped)     : {stats['new']:,}")
    print(f"  duplicates dropped: {stats['duplicate']:,}"
          f"  ({100*stats['duplicate']/max(stats['battles_seen'],1):.1f}%)")
    print(f"  clean 1v1 usable  : {stats['clean_1v1']:,}")
    print(f"  player fetch fails: {stats['player_fail']:,}")

    polled = stats["log_full"] + stats["log_partial"]
    if stats["first_poll"]:
        print(f"\n  first-ever poll   : {stats['first_poll']:,} players "
              f"(backfill, coverage undefined)")
    if polled:
        print("\n  coverage, repeat polls only:")
        print(f"    caught everything : {stats['log_partial']:,} players "
              f"({100*stats['log_partial']/polled:.1f}%)")
        print(f"    log was full      : {stats['log_full']:,} players "
              f"({100*stats['log_full']/polled:.1f}%)  <- may have missed battles")
        if stats["log_full"] / polled > 0.10:
            print(f"    POLL MORE OFTEN: over 10% of players filled the "
                  f"{BATTLELOG_PAGE}-battle page.")
    if path:
        print(f"  written           : {path}")
    print(f"\n  by battle type:")
    for k, v in sorted(((k, v) for k, v in stats.items() if k.startswith("type:")),
                       key=lambda kv: -kv[1]):
        print(f"    {k[5:]:22s} {v:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
