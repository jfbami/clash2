"""Build the one-row-per-player table from the Season 18 battle Parquet.

Usage:  python scripts/build_player_table.py

Runs in two passes. The first melts every battle into its two sides and
hash-partitions them by player tag into shards. The second reduces each shard
to per-player behavioural features and concatenates the results.

Writes `data/players.parquet`. Shards land in the scratchpad because they are
an intermediate worth 1.6 GB, not a deliverable.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crdata.level_effect import season_files
from crdata.player_panel import SHARDS, shard_features, write_player_battles

SCRATCH = Path(r"C:\Users\jfbaa\AppData\Local\Temp\claude"
               r"\C--Users-jfbaa-OneDrive-Documents-clash2")
PARQUET = SCRATCH / "d24c6794-c5fc-463a-925a-588dd12c92e6" / "scratchpad" / "season18_parquet"
SHARD_DIR = SCRATCH / "faf9482d-f87c-4635-9340-80f0de0b2114" / "scratchpad" / "player_shards"
EXCLUDED = ("01042021",)  # D9: the final day of the season is not representative.


def melt(card_ids, force: bool) -> None:
    if SHARD_DIR.exists() and any(SHARD_DIR.glob("*.parquet")) and not force:
        print(f"reusing shards in {SHARD_DIR}")
        return

    files = season_files(PARQUET, EXCLUDED)
    print(f"pass 1: melting {len(files)} day files into {SHARDS} shards\n", flush=True)
    started = time.time()
    report = write_player_battles(files, SHARD_DIR, card_ids, SHARDS)
    print(f"\n  {report.rows_written:,} player-battle rows across "
          f"{report.shards} shards in {time.time() - started:.0f}s\n")


def reduce_shards() -> pd.DataFrame:
    shards = sorted(SHARD_DIR.glob("*.parquet"))
    print(f"pass 2: reducing {len(shards)} shards to per-player features", flush=True)
    frames = []
    for position, shard in enumerate(shards, start=1):
        frames.append(shard_features(shard))
        if position % 8 == 0 or position == len(shards):
            print(f"  {position}/{len(shards)} shards, "
                  f"{sum(len(f) for f in frames):,} players", flush=True)
    return pd.concat(frames, ignore_index=True)


def summarise(players: pd.DataFrame) -> None:
    print(f"\nplayers                     {len(players):,}")
    for threshold in (10, 20, 50):
        kept = int((players.battles >= threshold).sum())
        print(f"players with {threshold:>3}+ battles     {kept:,}")

    active = players[players.battles >= 20]
    print(f"\nbehaviour among the {len(active):,} players with 20+ battles:")
    columns = ["battles", "win_rate", "distinct_decks", "top_deck_share",
               "switch_rate", "switch_after_loss", "switch_after_win",
               "loss_reactivity", "switch_drift", "level_mean", "level_gain",
               "trophy_gain"]
    print(active[columns].describe().T[["mean", "std", "25%", "50%", "75%"]].to_string())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remelt", action="store_true", help="rebuild the shards")
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    card_ids = pd.read_parquet(root / "data" / "reference" / "cards.parquet")["id"].to_numpy()

    melt(card_ids, arguments.remelt)
    players = reduce_shards()

    destination = root / "data" / "players.parquet"
    players.to_parquet(destination, compression="zstd", index=False)
    summarise(players)
    print(f"\nwritten {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
