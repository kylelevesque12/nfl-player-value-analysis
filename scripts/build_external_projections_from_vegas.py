"""Build Vegas-team-environment-implied PPR projections.

The Vegas market sets game lines (spread, total) through professional
traders, encoding a real market-derived projection of expected team
scoring. This script derives a per-(player, season, week) PPR projection
implied by those lines by fitting a per-(season, position) regression of
actual PPR on the three Vegas team-environment features the project already
attaches to each player-week:

    fantasy_points_ppr ~ implied_team_total
                        + spread_line_team_perspective
                        + is_home

The fitted value is the Vegas-implied PPR projection. It is a *weaker*
benchmark than the DK closing-line implied projection (DK encodes
player-level expectations via player-specific salaries; Vegas is only
team-level) but it has full coverage 2018-2025 and lets us validate
whether the +1.7% beat against DK in 2020-2021 extrapolates to a similar
beat against a different market signal across the full validation window.

Output: ``data/raw/external_projections_vegas.csv`` matching the schema
expected by ``src/external_benchmark.py`` (the multi-source version loaded
via glob).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python scripts/build_external_projections_from_vegas.py` from the
# project root without an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.weekly_fantasy_projection import build_modeling_frame  # noqa: E402


SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
MIN_REGRESSION_SAMPLE = 50  # per (season, position) — fewer than this falls back


def _find_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Vegas-team-environment-implied weekly PPR projections from "
            "existing schedule features."
        )
    )
    parser.add_argument(
        "--output",
        default="data/raw/external_projections_vegas.csv",
        help="Path (relative to project root) for the Vegas-implied CSV.",
    )
    return parser.parse_args()


def fit_vegas_implied(modeling_df: pd.DataFrame) -> pd.DataFrame:
    """Per (season, position), OLS regress PPR on Vegas team-environment.

    Falls back to the team-environment mean when a position-season has too
    few rows for a stable fit.
    """
    feature_cols = [
        "implied_team_total",
        "spread_line_team_perspective",
        "is_home",
    ]
    target_col = "fantasy_points_ppr"
    if any(c not in modeling_df.columns for c in feature_cols + [target_col]):
        missing = [c for c in feature_cols + [target_col] if c not in modeling_df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    rows: list[pd.DataFrame] = []
    for (season, position), group in modeling_df.groupby(["season", "position"]):
        usable = group.dropna(subset=feature_cols + [target_col]).copy()
        if len(usable) < MIN_REGRESSION_SAMPLE:
            mean_ppr = float(usable[target_col].mean()) if not usable.empty else 0.0
            group = group.assign(external_projection_ppr=mean_ppr)
            rows.append(group)
            continue

        X = usable[feature_cols].to_numpy(dtype="float64")
        y = usable[target_col].to_numpy(dtype="float64")
        # Solve OLS with an intercept column.
        X_design = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        all_rows = group.copy()
        full_X = all_rows[feature_cols].to_numpy(dtype="float64")
        # Fill rare NaNs in features with the position-season mean so we can
        # produce a projection for every row (the benchmark step will drop
        # rows where the actual PPR is NaN).
        col_means = np.nanmean(full_X, axis=0)
        nan_mask = np.isnan(full_X)
        full_X = np.where(nan_mask, col_means, full_X)
        full_X_design = np.column_stack([np.ones(len(full_X)), full_X])
        all_rows["external_projection_ppr"] = (full_X_design @ coef).clip(min=0.0)
        rows.append(all_rows)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    args = parse_args()
    root = _find_project_root()

    print("Building modeling frame...")
    player_stats = pd.read_csv(
        root / "data" / "raw" / "player_stats_2016_2025.csv", low_memory=False
    )
    schedules = pd.read_csv(
        root / "data" / "raw" / "schedules_2016_2025.csv", low_memory=False
    )
    rosters = pd.read_csv(
        root / "data" / "raw" / "rosters_2016_2025.csv", low_memory=False
    )
    modeling_df = build_modeling_frame(
        player_stats, schedules, rosters, project_root=root
    )
    # Restrict to skill positions and rows with a valid PPR observation.
    modeling_df = modeling_df[modeling_df["position"].isin(SKILL_POSITIONS)].copy()

    print("Fitting per-(season, position) Vegas implied projection...")
    projected = fit_vegas_implied(modeling_df)

    output_cols = [
        "season",
        "week",
        "player_id",
        "player_display_name",
        "position",
        "team",
        "external_projection_ppr",
    ]
    out = projected[output_cols].copy()
    out["source"] = "vegas_team_environment_implied"

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(
        f"Wrote {output_path.relative_to(root)} "
        f"({len(out):,} player-weeks, "
        f"seasons {int(out['season'].min())}-{int(out['season'].max())})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
