"""Build ``data/raw/external_projections.csv`` from RotoGuru DK salaries.

Pipeline:
  1. Load the concatenated DK salary file written by ``fetch_rotoguru_salaries.py``.
  2. Filter to skill positions and parse "Last, First" names to "First Last".
  3. Fuzzy-match to the nflverse roster to attach a stable ``gsis_id``.
  4. Join to weekly ``player_stats`` to pull the actual PPR scored.
  5. Per (season, position), fit a linear ``ppr_points ~ dk_salary`` regression.
     The fitted value is the **market-implied PPR projection** for that
     player-week. This is the strongest fantasy benchmark we can get for free:
     DK sets salaries pregame based on its own projection algorithm; the
     in-season points-vs-salary slope captures the implicit conversion the
     market is using. We allow an intercept so the fit is as accurate as
     possible (we want the benchmark to be as tough as possible).
  6. Write ``data/raw/external_projections.csv`` matching the schema expected
     by ``src/external_benchmark.py``.

Coverage caveat: RotoGuru's free DK archive currently ends at 2021. Years
2022+ produce no data and are documented in the report and roadmap.

Usage::

    python scripts/fetch_rotoguru_salaries.py --years 2014-2021
    python scripts/build_external_projections_from_dk.py
    python scripts/run_pipeline.py --steps external_benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
MATCH_SCORE_THRESHOLD = 84  # rapidfuzz token_set_ratio above this counts as a match
MIN_REGRESSION_SAMPLE = 50  # per (season, position) — below this, drop


def _find_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RotoGuru DK salary archive into external_projections.csv."
    )
    parser.add_argument(
        "--input",
        default="data/raw/dk_salaries_2014_2021.csv",
        help="Path (relative to project root) to the concatenated DK salaries CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/external_projections.csv",
        help="Path to write the converted external projections.",
    )
    return parser.parse_args()


def _last_first_to_first_last(name: str) -> str:
    """Convert "Mahomes II, Patrick" to "Patrick Mahomes II"."""
    if not isinstance(name, str) or "," not in name:
        return str(name).strip()
    last_part, first_part = name.split(",", 1)
    return f"{first_part.strip()} {last_part.strip()}".strip()


def load_dk_salaries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["season"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["dk_salary"] = pd.to_numeric(df["dk_salary"], errors="coerce")
    df["dk_points_scored"] = pd.to_numeric(df["dk_points_scored"], errors="coerce")
    df["position"] = df["position"].str.upper().str.strip()
    df = df[df["position"].isin(SKILL_POSITIONS)].copy()
    df = df.dropna(subset=["season", "week", "dk_salary"])
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["player_display_name"] = df["name_lastfirst"].apply(_last_first_to_first_last)
    return df


def load_roster_lookup(root: Path) -> pd.DataFrame:
    """One row per (season, gsis_id) with the canonical full name and position."""
    rosters = pd.read_csv(root / "data" / "raw" / "rosters_2016_2025.csv", dtype=str)
    rosters = rosters.rename(columns={"full_name": "player_display_name"})
    keep_cols = [
        c
        for c in ["season", "gsis_id", "player_display_name", "position", "team"]
        if c in rosters.columns
    ]
    rosters = rosters[keep_cols].dropna(subset=["season", "gsis_id", "player_display_name"])
    rosters["season"] = pd.to_numeric(rosters["season"], errors="coerce").astype("Int64")
    rosters = rosters.dropna(subset=["season"])
    rosters["season"] = rosters["season"].astype(int)
    if "position" in rosters.columns:
        rosters["position"] = rosters["position"].astype(str).str.upper()
    # If a player appears multiple times in a season (mid-year trade) keep
    # the first row; we only need a gsis_id.
    rosters = rosters.drop_duplicates(subset=["season", "gsis_id"]).reset_index(drop=True)
    return rosters


def attach_gsis_ids(
    dk: pd.DataFrame, rosters: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fuzzy-match each DK player-season to the roster by name.

    Matching is done **per season** (a player's name is unique within a season
    almost always; cross-season name collisions are rare and don't matter for
    the projection). We pre-build a per-season name list and use
    ``rapidfuzz.process.extractOne`` for each unique DK name in that season.
    """
    diagnostics = {"matched": 0, "unmatched": 0, "low_score": 0}
    matched_rows: list[pd.DataFrame] = []

    # Restrict roster to skill positions if available.
    if "position" in rosters.columns:
        rosters_skill = rosters[rosters["position"].isin(SKILL_POSITIONS)].copy()
    else:
        rosters_skill = rosters.copy()

    for season, dk_season in dk.groupby("season"):
        season_rosters = rosters_skill[rosters_skill["season"].eq(season)]
        if season_rosters.empty:
            diagnostics["unmatched"] += len(dk_season)
            continue
        candidate_names = season_rosters["player_display_name"].astype(str).tolist()
        name_to_gsis = dict(
            zip(
                season_rosters["player_display_name"].astype(str),
                season_rosters["gsis_id"].astype(str),
            )
        )

        unique_names = dk_season["player_display_name"].dropna().unique().tolist()
        lookup: dict[str, str] = {}
        for name in unique_names:
            best = process.extractOne(
                name,
                candidate_names,
                scorer=fuzz.token_set_ratio,
            )
            if best is None:
                continue
            cand_name, score, _ = best
            if score < MATCH_SCORE_THRESHOLD:
                continue
            lookup[name] = name_to_gsis[cand_name]

        dk_season = dk_season.copy()
        dk_season["player_id"] = dk_season["player_display_name"].map(lookup)
        diagnostics["matched"] += int(dk_season["player_id"].notna().sum())
        diagnostics["unmatched"] += int(dk_season["player_id"].isna().sum())
        matched_rows.append(dk_season)

    if not matched_rows:
        return pd.DataFrame(), diagnostics
    return pd.concat(matched_rows, ignore_index=True), diagnostics


def attach_actual_ppr(matched: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Join to ``player_stats`` to attach actual PPR for the regression target."""
    stats = pd.read_csv(
        root / "data" / "raw" / "player_stats_2016_2025.csv",
        usecols=["player_id", "season", "week", "fantasy_points_ppr"],
        low_memory=False,
    )
    stats["season"] = pd.to_numeric(stats["season"], errors="coerce").astype("Int64")
    stats["week"] = pd.to_numeric(stats["week"], errors="coerce").astype("Int64")
    stats = stats.dropna(subset=["season", "week", "player_id", "fantasy_points_ppr"])
    stats["season"] = stats["season"].astype(int)
    stats["week"] = stats["week"].astype(int)
    return matched.merge(
        stats,
        on=["player_id", "season", "week"],
        how="left",
    )


def fit_implied_projections(matched: pd.DataFrame) -> pd.DataFrame:
    """Per (season, position) regress PPR on DK salary; fitted value is the
    market-implied projection.
    """
    rows: list[pd.DataFrame] = []
    for (season, position), group in matched.groupby(["season", "position"]):
        train = group.dropna(subset=["fantasy_points_ppr", "dk_salary"])
        if len(train) < MIN_REGRESSION_SAMPLE:
            # Too few rows for a stable fit; fall back to a simple
            # mean(ppr) / mean(salary) ratio with no intercept.
            ratio = (
                train["fantasy_points_ppr"].mean() / train["dk_salary"].mean()
                if not train.empty and train["dk_salary"].mean() > 0
                else 0.001  # tiny default
            )
            implied = group["dk_salary"] * ratio
        else:
            x = train["dk_salary"].to_numpy()
            y = train["fantasy_points_ppr"].to_numpy()
            x_mean = x.mean()
            y_mean = y.mean()
            x_centered = x - x_mean
            denom = float(np.sum(x_centered**2))
            slope = (
                float(np.sum(x_centered * (y - y_mean)) / denom)
                if denom > 0
                else 0.0
            )
            intercept = y_mean - slope * x_mean
            implied = intercept + slope * group["dk_salary"]
        # Clip to non-negative (DK salaries are always positive; PPR can be
        # slightly negative due to turnovers but the projection should not be).
        implied = implied.clip(lower=0)
        out = group.copy()
        out["external_projection_ppr"] = implied
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    args = parse_args()
    root = _find_project_root()
    input_path = root / args.input
    if not input_path.exists():
        sys.stderr.write(
            f"Input DK salaries CSV not found at {input_path}.\n"
            "Run scripts/fetch_rotoguru_salaries.py first.\n"
        )
        return 2

    print("Loading DK salaries...")
    dk = load_dk_salaries(input_path)
    print(f"  {len(dk):,} skill-position player-weeks")

    print("Loading rosters and fuzzy-matching player names to gsis_id...")
    rosters = load_roster_lookup(root)
    matched, diagnostics = attach_gsis_ids(dk, rosters)
    matched_rate = (
        diagnostics["matched"] / max(diagnostics["matched"] + diagnostics["unmatched"], 1)
    )
    print(
        f"  matched={diagnostics['matched']:,} "
        f"unmatched={diagnostics['unmatched']:,} "
        f"({matched_rate:.1%} match rate)"
    )

    matched = matched.dropna(subset=["player_id"]).copy()
    if matched.empty:
        sys.stderr.write("No DK rows matched the roster. Aborting.\n")
        return 1

    print("Joining to player_stats for actual PPR...")
    enriched = attach_actual_ppr(matched, root)

    print(
        "Fitting per (season, position) salary-to-PPR conversion (the market's "
        "implied projection)..."
    )
    projected = fit_implied_projections(enriched)

    # Output schema for src/external_benchmark.py
    output_cols = [
        "season",
        "week",
        "player_id",
        "player_display_name",
        "position",
        "team_rotoguru",
        "external_projection_ppr",
    ]
    out = (
        projected[output_cols]
        .rename(columns={"team_rotoguru": "team"})
        .copy()
    )
    out["source"] = "draftkings_implied_via_rotoguru"

    output_path = root / args.output
    out.to_csv(output_path, index=False)
    print(
        f"Wrote {output_path.relative_to(root)} with {len(out):,} player-weeks "
        f"spanning {int(out['season'].min())}-{int(out['season'].max())}."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
