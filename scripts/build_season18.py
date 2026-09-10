"""Convert every extracted Season 18 CSV into Parquet.

Usage:  python scripts/build_season18.py --source-dir PATH

Reads extracted CSVs from the requested source directory and writes one Parquet
file per source into `data/season18_parquet/` by default. Safe to re-run: files
already converted are skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crdata.etl import convert_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def battle_files(root: Path) -> list[Path]:
    """Battle CSVs only. Excludes the card list and win condition reference files."""
    return sorted(path for path in root.rglob("*.csv")
                  if "attles" in path.name.lower())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="directory containing extracted Season 18 CSV files")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "season18_parquet")
    parser.add_argument("--force", action="store_true", help="reconvert existing files")
    arguments = parser.parse_args()

    sources = battle_files(arguments.source_dir)
    if not sources:
        print(f"No battle CSVs found under {arguments.source_dir}")
        return 1

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(sources)} source files\n")
    total_written = 0

    for source in sources:
        destination = arguments.output_dir / (source.stem + ".parquet")
        if destination.exists() and not arguments.force:
            print(f"  skip     {source.name}  (already converted)")
            continue

        started = time.time()
        report = convert_file(source, destination)
        total_written += report.rows_written
        print(f"  {report.rows_written:>10,} rows  "
              f"a_won={report.side_a_win_rate:.4f}  "
              f"{time.time() - started:5.0f}s  {source.name}")

    print(f"\ntotal rows written this run: {total_written:,}")
    print(f"output: {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
