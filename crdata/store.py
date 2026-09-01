"""Parquet storage + persistent dedup + cohort state.

The dedup set is stored as 8-byte hashes rather than full keys so it stays
small: 10M battles ~ 80MB in memory.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BATTLES = DATA / "battles"
STATE = DATA / "state"
REF = DATA / "reference"
SEEN = STATE / "seen_keys.parquet"
COHORT = STATE / "cohort_spread.parquet"  # trophy-balanced; the war-clan seed is retired

for d in (BATTLES, STATE, REF):
    d.mkdir(parents=True, exist_ok=True)


def key_hash(k: str) -> int:
    """Stable 63-bit hash (Python's hash() is salted per process - unusable)."""
    return int.from_bytes(hashlib.blake2b(k.encode(), digest_size=8).digest(), "big") >> 1


class SeenKeys:
    def __init__(self) -> None:
        self.hashes: set[int] = set()
        if SEEN.exists():
            self.hashes = set(pq.read_table(SEEN)["h"].to_pylist())

    def __contains__(self, k: str) -> bool:
        return key_hash(k) in self.hashes

    def add(self, k: str) -> None:
        self.hashes.add(key_hash(k))

    def save(self) -> None:
        pq.write_table(pa.table({"h": pa.array(sorted(self.hashes), pa.uint64())}), SEEN)

    def __len__(self) -> int:
        return len(self.hashes)


def append_battles(rows: list[dict], run_id: str) -> Path | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = BATTLES / f"date={day}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_id}.parquet"
    df.to_parquet(path, index=False, compression="zstd")
    return path


def load_battles(clean_only: bool = False) -> pd.DataFrame:
    files = sorted(BATTLES.glob("date=*/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    # defensive: dedup again on read in case a run was interrupted mid-save
    df = df.drop_duplicates("battle_key")
    return df[df.is_clean_1v1] if clean_only else df


def load_cohort(path: Path = COHORT) -> pd.DataFrame:
    """Read the cohort to poll. Defaults to the trophy-balanced cohort."""
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(
        columns=["tag", "name", "clan_tag", "added_at", "last_polled",
                 "n_polls", "n_battles", "n_fail"]
    )


def save_cohort(df: pd.DataFrame, path: Path = COHORT) -> None:
    df.to_parquet(path, index=False)
