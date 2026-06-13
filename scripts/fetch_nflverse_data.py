"""Fetch supplementary nflverse data files used by the modeling pipeline.

The core data (player_stats, rosters, schedules) was previously fetched by
Notebook 01. This script extends that pattern to the supplementary feeds the
weekly fantasy + rookie models consume:

Original tier (small, fast):
    * snap counts        -> data/raw/snap_counts_<years>.csv
    * injury reports     -> data/raw/injuries_<years>.csv
    * depth charts       -> data/raw/depth_charts_<years>.csv

Rich tier (added later; larger, slower; more modeling lift):
    * NFL Combine        -> data/raw/combine_<years>.csv
    * Next Gen Stats     -> data/raw/ngs_<stat_type>_<years>.csv
    * PFR weekly stats   -> data/raw/pfr_weekly_<stat_type>_<years>.csv
    * Play-by-play       -> data/raw/pbp_<years>.parquet (parquet for size)
    * Draft picks        -> data/raw/draft_picks.csv

The pipeline modules treat these files as **optional**. If they are missing,
the corresponding features are skipped and the model still runs (with the
documented loss of accuracy). Once you run this script the next pipeline
execution picks them up automatically.

Usage:
    pip install nfl_data_py pyarrow
    python scripts/fetch_nflverse_data.py --years 2016-2025
    # Skip the heavy PBP fetch when you don't need it:
    python scripts/fetch_nflverse_data.py --skip pbp
    # Only fetch a specific feed:
    python scripts/fetch_nflverse_data.py --only ngs,combine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable


def _parse_years(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch supplementary nflverse data feeds."
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
            "depth_charts, combine, ngs, pfr_weekly, pbp, draft_picks. "
            "Default: fetch all."
        ),
    )
    parser.add_argument(
        "--only",
        default="",
        help="If set, fetch only the named feeds (overrides --skip).",
    )
    return parser.parse_args()


def _find_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fetch_and_save(
    name: str,
    fetch_fn: Callable,
    out_path: Path,
    project_root: Path,
    write_parquet: bool = False,
) -> bool:
    """Run a fetcher and save the result. Returns True on success."""
    print(f"Fetching {name} -> {out_path.relative_to(project_root)}")
    try:
        df = fetch_fn()
    except Exception as exc:  # noqa: BLE001
        print(f"  failed: {exc}", file=sys.stderr)
        return False
    if df is None or len(df) == 0:
        print(f"  empty result for {name}; skipping write", file=sys.stderr)
        return False
    if write_parquet:
        try:
            df.to_parquet(out_path, index=False)
        except Exception as exc:  # noqa: BLE001
            # Fall back to CSV if parquet engine is missing.
            csv_path = out_path.with_suffix(".csv")
            print(
                f"  parquet write failed ({exc}); falling back to "
                f"{csv_path.name}",
                file=sys.stderr,
            )
            df.to_csv(csv_path, index=False)
            out_path = csv_path
    else:
        df.to_csv(out_path, index=False)
    print(f"  wrote {len(df):,} rows")
    return True


def main() -> int:
    args = parse_args()
    years = _parse_years(args.years)
    skip = {part.strip() for part in args.skip.split(",") if part.strip()}
    only = {part.strip() for part in args.only.split(",") if part.strip()}

    try:
        import nfl_data_py as nfl
    except ImportError:
        sys.stderr.write(
            "nfl_data_py is not installed. Install it with:\n"
            "    pip install nfl_data_py pyarrow\n"
        )
        return 2

    project_root = _find_project_root()
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    year_span = f"{years[0]}_{years[-1]}"

    # --------------------------------------------------------------
    # Original feeds — small and fast
    # --------------------------------------------------------------
    feeds: list[tuple[str, Callable, Path, bool]] = [
        (
            "snap_counts",
            lambda: nfl.import_snap_counts(years),
            raw_dir / f"snap_counts_{year_span}.csv",
            False,
        ),
        (
            "injuries",
            lambda: nfl.import_injuries(years),
            raw_dir / f"injuries_{year_span}.csv",
            False,
        ),
        (
            "depth_charts",
            lambda: nfl.import_depth_charts(years),
            raw_dir / f"depth_charts_{year_span}.csv",
            False,
        ),
    ]

    # --------------------------------------------------------------
    # Rich feeds — bigger, slower, more modeling lift
    # --------------------------------------------------------------

    # Combine data is one-shot for a wide year window (not per-season).
    feeds.append(
        (
            "combine",
            lambda: nfl.import_combine_data(years=years),
            raw_dir / f"combine_{year_span}.csv",
            False,
        )
    )

    # Next Gen Stats — three stat types, one file each.
    for stat_type in ("passing", "rushing", "receiving"):
        feeds.append(
            (
                f"ngs_{stat_type}",
                # Capture stat_type by default-argument trick.
                lambda st=stat_type: nfl.import_ngs_data(stat_type=st, years=years),
                raw_dir / f"ngs_{stat_type}_{year_span}.csv",
                False,
            )
        )

    # PFR weekly stats — three stat types.
    for stat_type in ("pass", "rush", "rec"):
        feeds.append(
            (
                f"pfr_weekly_{stat_type}",
                lambda st=stat_type: nfl.import_weekly_pfr(s_type=st, years=years),
                raw_dir / f"pfr_weekly_{stat_type}_{year_span}.csv",
                False,
            )
        )

    # Draft picks — historical record with grades.
    feeds.append(
        (
            "draft_picks",
            lambda: nfl.import_draft_picks(years=years),
            raw_dir / f"draft_picks_{year_span}.csv",
            False,
        )
    )

    # Play-by-play — biggest single feed. Written as parquet for size.
    feeds.append(
        (
            "pbp",
            lambda: nfl.import_pbp_data(years=years, downcast=True),
            raw_dir / f"pbp_{year_span}.parquet",
            True,
        )
    )

    for name, fetch_fn, out_path, parquet in feeds:
        # Resolve which name groups apply to this feed for --skip/--only matching.
        # `ngs_passing` should match both itself and the group key `ngs`.
        applies = {name}
        for prefix in ("ngs_", "pfr_weekly_"):
            if name.startswith(prefix):
                applies.add(prefix.rstrip("_"))
        if only and not (applies & only):
            continue
        if not only and (applies & skip):
            print(f"Skipping {name} (--skip)")
            continue
        _fetch_and_save(name, fetch_fn, out_path, project_root, write_parquet=parquet)

    print(
        "Done. Re-run the pipeline (or restart Streamlit) to pick up the new "
        "files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
