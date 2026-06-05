"""Fetch supplementary nflverse data files used by the modeling pipeline.

The core data (player_stats, rosters, schedules) was previously fetched by
Notebook 01. This script extends that pattern to the supplementary feeds the
weekly fantasy model wants to consume:

    * snap counts        -> data/raw/snap_counts_<years>.csv
    * injury reports     -> data/raw/injuries_<years>.csv
    * depth charts       -> data/raw/depth_charts_<years>.csv

The pipeline modules treat these files as **optional**. If they are missing,
the corresponding features are skipped and the model still runs (with the
documented loss of accuracy). Once you run this script the next pipeline
execution picks them up automatically.

Usage:
    pip install nfl_data_py
    python scripts/fetch_nflverse_data.py --years 2016-2025

If nfl_data_py is not installed the script prints an actionable message and
exits non-zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_years(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch supplementary nflverse data (snap counts, injuries, depth charts)."
    )
    parser.add_argument(
        "--years",
        default="2016-2025",
        help='Year range or comma list. Default "2016-2025".',
    )
    parser.add_argument(
        "--skip",
        default="",
        help=(
            "Comma-separated feeds to skip. Options: snap_counts, injuries, "
            "depth_charts. Default: fetch all."
        ),
    )
    return parser.parse_args()


def _find_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    args = parse_args()
    years = _parse_years(args.years)
    skip = {part.strip() for part in args.skip.split(",") if part.strip()}

    try:
        import nfl_data_py as nfl
    except ImportError:
        sys.stderr.write(
            "nfl_data_py is not installed. Install it with:\n"
            "    pip install nfl_data_py\n"
        )
        return 2

    raw_dir = _find_project_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    year_span = f"{years[0]}_{years[-1]}"

    fetchers = {
        "snap_counts": (
            lambda: nfl.import_snap_counts(years),
            f"snap_counts_{year_span}.csv",
        ),
        "injuries": (
            lambda: nfl.import_injuries(years),
            f"injuries_{year_span}.csv",
        ),
        "depth_charts": (
            lambda: nfl.import_depth_charts(years),
            f"depth_charts_{year_span}.csv",
        ),
    }

    for name, (fetch, filename) in fetchers.items():
        if name in skip:
            print(f"Skipping {name} (--skip)")
            continue
        out_path = raw_dir / filename
        print(f"Fetching {name} -> {out_path.relative_to(_find_project_root())}")
        try:
            df = fetch()
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}", file=sys.stderr)
            continue
        df.to_csv(out_path, index=False)
        print(f"  wrote {len(df):,} rows")

    print("Done. Re-run `python scripts/run_pipeline.py --steps weekly_fantasy`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
