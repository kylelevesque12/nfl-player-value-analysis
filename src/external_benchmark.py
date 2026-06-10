"""Head-to-head comparison of the weekly fantasy model vs external projections.

This is Tier 1 #1 of `PORTFOLIO_ROADMAP.md`: beating an internal baseline is the
floor, beating an external projector is the ceiling reference. Until we run a
head-to-head against FantasyPros or DraftKings closing-line implied
projections, the project's skill scores are numbers without a market.

This module is **scaffolding**. It contains the entire comparison pipeline —
join logic, metric calculation, by-position reporting — but it depends on a
user-supplied CSV of external projections. The expected schema is documented in
`load_external_projections` below. Drop a CSV at
`data/raw/external_projections.csv` and the next pipeline run will populate the
benchmark report.

Acquisition options (see PORTFOLIO_ROADMAP.md for full discussion):
  1. DraftKings closing-line implied projections (strongest claim if we win)
  2. FantasyPros consensus historical projections (easier to scrape)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.load_data import ensure_project_dirs, find_project_root


CSV_FLOAT_FORMAT = config.CSV_FLOAT_FORMAT
EXTERNAL_CSV_DEFAULT_PATH = "data/raw/external_projections.csv"

REQUIRED_COLUMNS = {
    "season",
    "week",
    "player_id",
    "external_projection_ppr",
}
OPTIONAL_COLUMNS = {
    "source",  # e.g. "fantasypros_consensus" or "draftkings_implied"
    "player_display_name",
    "position",
    "team",
}


def load_external_projections(
    project_root: str | Path | None = None,
    relative_path: str | None = None,
) -> pd.DataFrame:
    """Load all CSVs of external weekly PPR projections under ``data/raw/``.

    Searches ``data/raw/`` for ``external_projections.csv`` (the canonical
    single-source path) **and** any file matching ``external_projections_*.csv``
    (multi-source files added by separate acquisition scripts). All matching
    files are concatenated. The currently active source is the
    RotoGuru-derived DK closing-line implied projection.

    Required columns: ``season``, ``week``, ``player_id`` (nflverse gsis id),
    ``external_projection_ppr``. Optional but recommended: ``source``,
    ``player_display_name``, ``position``, ``team``. If ``source`` is missing
    we infer it from the filename so the benchmark can still group per-source.

    Returns an empty DataFrame if no matching files are present so the
    pipeline keeps working pre-acquisition.
    """
    root = find_project_root() if project_root is None else Path(project_root).resolve()
    if relative_path is not None:
        candidate_paths = [root / relative_path]
    else:
        raw_dir = root / "data" / "raw"
        candidate_paths = []
        canonical = raw_dir / "external_projections.csv"
        if canonical.exists():
            candidate_paths.append(canonical)
        candidate_paths.extend(sorted(raw_dir.glob("external_projections_*.csv")))

    frames: list[pd.DataFrame] = []
    for path in candidate_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"External projections at {path} missing required columns: "
                f"{sorted(missing)}"
            )
        if "source" not in df.columns:
            df["source"] = path.stem  # e.g. external_projections_vegas
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
        df["external_projection_ppr"] = pd.to_numeric(
            df["external_projection_ppr"], errors="coerce"
        )
        frames.append(
            df.dropna(
                subset=["season", "week", "player_id", "external_projection_ppr"]
            )
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y) - np.asarray(p)))))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def _spearman(y: pd.Series, p: pd.Series) -> float:
    frame = pd.DataFrame({"y": y, "p": p}).dropna()
    if len(frame) < 2 or frame["y"].nunique() < 2 or frame["p"].nunique() < 2:
        return float("nan")
    return float(frame["y"].corr(frame["p"], method="spearman"))


def compare_single_source(
    model_predictions: pd.DataFrame,
    external_projections: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Head-to-head metrics on player-weeks where both project a value.

    Expected ``model_predictions`` schema: the main-method rows of
    ``weekly_fantasy_validation_predictions.csv`` (player_id, season, week,
    position, prediction, target_fantasy_points_ppr).
    """
    if model_predictions.empty or external_projections.empty:
        return {
            "overall": pd.DataFrame(),
            "by_position": pd.DataFrame(),
            "by_season": pd.DataFrame(),
            "joined": pd.DataFrame(),
            "win_rate": pd.DataFrame(),
        }

    model = model_predictions.copy()
    model["season"] = pd.to_numeric(model["season"], errors="coerce").astype("Int64")
    model["week"] = pd.to_numeric(model["week"], errors="coerce").astype("Int64")

    joined = model.merge(
        external_projections[
            [
                "season",
                "week",
                "player_id",
                "external_projection_ppr",
            ]
        ],
        on=["season", "week", "player_id"],
        how="inner",
    )
    joined = joined.dropna(
        subset=["target_fantasy_points_ppr", "prediction", "external_projection_ppr"]
    )

    if joined.empty:
        return {
            "overall": pd.DataFrame(),
            "by_position": pd.DataFrame(),
            "by_season": pd.DataFrame(),
            "joined": joined,
            "win_rate": pd.DataFrame(),
        }

    y = joined["target_fantasy_points_ppr"].to_numpy()
    model_pred = joined["prediction"].to_numpy()
    ext_pred = joined["external_projection_ppr"].to_numpy()

    overall = pd.DataFrame(
        [
            {
                "segment": "overall",
                "segment_value": "all",
                "n_player_weeks": int(len(joined)),
                "model_rmse": _rmse(y, model_pred),
                "external_rmse": _rmse(y, ext_pred),
                "model_mae": _mae(y, model_pred),
                "external_mae": _mae(y, ext_pred),
                "model_spearman": _spearman(joined["target_fantasy_points_ppr"], joined["prediction"]),
                "external_spearman": _spearman(
                    joined["target_fantasy_points_ppr"], joined["external_projection_ppr"]
                ),
                "skill_vs_external": 1.0 - _rmse(y, model_pred) / _rmse(y, ext_pred),
            }
        ]
    )

    rows: list[dict[str, Any]] = []
    for position, grp in joined.groupby("position"):
        y_pos = grp["target_fantasy_points_ppr"].to_numpy()
        rows.append(
            {
                "segment": "position",
                "segment_value": position,
                "n_player_weeks": int(len(grp)),
                "model_rmse": _rmse(y_pos, grp["prediction"].to_numpy()),
                "external_rmse": _rmse(y_pos, grp["external_projection_ppr"].to_numpy()),
                "model_mae": _mae(y_pos, grp["prediction"].to_numpy()),
                "external_mae": _mae(y_pos, grp["external_projection_ppr"].to_numpy()),
                "skill_vs_external": (
                    1.0
                    - _rmse(y_pos, grp["prediction"].to_numpy())
                    / _rmse(y_pos, grp["external_projection_ppr"].to_numpy())
                ),
            }
        )
    by_position = pd.DataFrame(rows)

    season_rows: list[dict[str, Any]] = []
    for season, grp in joined.groupby("season"):
        y_s = grp["target_fantasy_points_ppr"].to_numpy()
        season_rows.append(
            {
                "season": int(season),
                "n_player_weeks": int(len(grp)),
                "model_rmse": _rmse(y_s, grp["prediction"].to_numpy()),
                "external_rmse": _rmse(y_s, grp["external_projection_ppr"].to_numpy()),
                "skill_vs_external": (
                    1.0
                    - _rmse(y_s, grp["prediction"].to_numpy())
                    / _rmse(y_s, grp["external_projection_ppr"].to_numpy())
                ),
            }
        )
    by_season = pd.DataFrame(season_rows)

    # Per-player-week win/loss: which projector got closer to the actual?
    joined["model_abs_err"] = np.abs(y - model_pred)
    joined["external_abs_err"] = np.abs(y - ext_pred)
    joined["model_wins"] = (joined["model_abs_err"] < joined["external_abs_err"]).astype(int)
    win_rate = (
        joined.groupby("position")["model_wins"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "model_win_rate", "count": "n_player_weeks"})
    )

    return {
        "overall": overall,
        "by_position": by_position,
        "by_season": by_season,
        "joined": joined,
        "win_rate": win_rate,
    }


def compare(
    model_predictions: pd.DataFrame,
    external_projections: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Multi-source compare. Groups by ``source`` and runs the single-source
    comparison on each group, then concatenates the per-source tables.

    Returns the same dict shape as ``compare_single_source`` plus a
    ``per_source_overall`` table that lists each source's overall metrics
    side-by-side.
    """
    if model_predictions.empty or external_projections.empty:
        return compare_single_source(model_predictions, external_projections)

    if "source" not in external_projections.columns:
        external_projections = external_projections.assign(source="external")

    per_source_overall_rows: list[pd.DataFrame] = []
    per_source_by_position: list[pd.DataFrame] = []
    per_source_by_season: list[pd.DataFrame] = []
    per_source_win_rate: list[pd.DataFrame] = []
    joined_frames: list[pd.DataFrame] = []

    for source, ext_subset in external_projections.groupby("source"):
        results = compare_single_source(model_predictions, ext_subset)
        if not results["overall"].empty:
            tagged = results["overall"].assign(source=source)
            per_source_overall_rows.append(tagged)
        if not results["by_position"].empty:
            per_source_by_position.append(results["by_position"].assign(source=source))
        if not results["by_season"].empty:
            per_source_by_season.append(results["by_season"].assign(source=source))
        if not results["win_rate"].empty:
            per_source_win_rate.append(results["win_rate"].assign(source=source))
        if not results["joined"].empty:
            joined_frames.append(results["joined"].assign(source=source))

    overall = (
        pd.concat(per_source_overall_rows, ignore_index=True)
        if per_source_overall_rows
        else pd.DataFrame()
    )
    by_position = (
        pd.concat(per_source_by_position, ignore_index=True)
        if per_source_by_position
        else pd.DataFrame()
    )
    by_season = (
        pd.concat(per_source_by_season, ignore_index=True)
        if per_source_by_season
        else pd.DataFrame()
    )
    win_rate = (
        pd.concat(per_source_win_rate, ignore_index=True)
        if per_source_win_rate
        else pd.DataFrame()
    )
    joined = (
        pd.concat(joined_frames, ignore_index=True)
        if joined_frames
        else pd.DataFrame()
    )

    return {
        "overall": overall,
        "by_position": by_position,
        "by_season": by_season,
        "joined": joined,
        "win_rate": win_rate,
    }


def _build_summary_text(results: dict[str, pd.DataFrame], external_source: str) -> str:
    if results["joined"].empty:
        return (
            "# External Benchmark\n\n"
            "No external projections were available at the expected path "
            f"(`{EXTERNAL_CSV_DEFAULT_PATH}`). See `PORTFOLIO_ROADMAP.md` Tier 1 "
            "item #1 for acquisition options. Drop a CSV with columns "
            "`season,week,player_id,external_projection_ppr` and re-run "
            "`python scripts/run_pipeline.py --steps external_benchmark`.\n"
        )

    overall_table = results["overall"]
    sources = (
        sorted(overall_table["source"].unique())
        if "source" in overall_table.columns
        else [external_source]
    )

    lines = [
        "# External Benchmark",
        "",
        f"**Sources**: {', '.join(f'`{s}`' for s in sources)}",
        "",
        "## What this is",
        "",
        "Head-to-head: weekly fantasy model vs the DraftKings closing-line",
        "implied projection. DK sets salaries pregame, so a per-(season,",
        "position) regression of actual PPR on DK salary recovers the",
        "salary→points conversion DK is implicitly using. The fitted value",
        "is the market's implied projection for each player-week.",
        "",
        "Why this is a strong benchmark: the salary conversion is fit on the",
        "season's actual production, so the implied projection has access to",
        "information a real-time DK projection would not. Beating this version",
        "is harder than beating a live DK projection would be.",
        "",
        "Coverage limit: RotoGuru's free DK archive stops at the 2021 season,",
        "so the head-to-head sample is 2020-2021 only. The 2022-2025 stability",
        "is shown in `report/weekly_fantasy_projection_summary.md` against an",
        "internal baseline.",
        "",
        "The loader at `src/external_benchmark.py` globs every",
        "`data/raw/external_projections*.csv` file, so dropping in a paid",
        "source (Stokastic, FantasyData, FantasyPros archives) extends the",
        "benchmark with no code changes.",
        "",
        "## Per-source overall",
        "",
        "| Source | n | Model RMSE | External RMSE | Skill vs external |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in overall_table.iterrows():
        src = row.get("source", external_source)
        lines.append(
            f"| `{src}` | {int(row['n_player_weeks']):,} | "
            f"{row['model_rmse']:.3f} | {row['external_rmse']:.3f} | "
            f"{row['skill_vs_external']:+.3%} |"
        )

    lines.extend(["", "## By position (per source)", ""])
    if "source" in results["by_position"].columns:
        for src, group in results["by_position"].groupby("source"):
            lines.append(f"### `{src}`")
            lines.append("")
            lines.append(
                "| Position | n | Model RMSE | External RMSE | Skill vs external |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for _, row in group.iterrows():
                lines.append(
                    f"| {row['segment_value']} | {int(row['n_player_weeks']):,} | "
                    f"{row['model_rmse']:.3f} | {row['external_rmse']:.3f} | "
                    f"{row['skill_vs_external']:+.3%} |"
                )
            lines.append("")
    elif not results["by_position"].empty:
        lines.append(
            "| Position | n | Model RMSE | External RMSE | Skill vs external |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for _, row in results["by_position"].iterrows():
            lines.append(
                f"| {row['segment_value']} | {int(row['n_player_weeks']):,} | "
                f"{row['model_rmse']:.3f} | {row['external_rmse']:.3f} | "
                f"{row['skill_vs_external']:+.3%} |"
            )

    if not results["by_season"].empty and "source" in results["by_season"].columns:
        lines.extend(["", "## By season (per source)", ""])
        for src, group in results["by_season"].groupby("source"):
            lines.append(f"### `{src}`")
            lines.append("")
            lines.append("| Season | n | Model RMSE | External RMSE | Skill |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for _, row in group.sort_values("season").iterrows():
                lines.append(
                    f"| {int(row['season'])} | {int(row['n_player_weeks']):,} | "
                    f"{row['model_rmse']:.3f} | {row['external_rmse']:.3f} | "
                    f"{row['skill_vs_external']:+.3%} |"
                )
            lines.append("")

    # Per-player-week win rate, per source.
    if not results["win_rate"].empty and "source" in results["win_rate"].columns:
        lines.extend(
            [
                "## Per-player-week win rate (per source)",
                "",
                "Share of player-weeks where the model's projection landed closer",
                "to the actual PPR than the external projection did.",
                "",
            ]
        )
        for src, group in results["win_rate"].groupby("source"):
            lines.append(f"### `{src}`")
            lines.append("")
            lines.append("| Position | n | Model win rate |")
            lines.append("| --- | ---: | ---: |")
            for _, row in group.iterrows():
                lines.append(
                    f"| {row['position']} | {int(row['n_player_weeks']):,} | "
                    f"{row['model_win_rate']:.3f} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Honest reading of this result",
            "",
            "Public DFS analytics shops sell projections claiming a 1-3% edge over",
            "the DK salary line. A calibrated positive edge after honest rolling",
            "backtesting is the qualifying bar for a fantasy-projection portfolio",
            "piece. Where the model beats DK, the beat is real; where it loses or",
            "barely ties, the gap is reported as-is rather than hidden.",
            "",
            "For temporal-stability evidence across the full 2020-2025 rolling",
            "validation window (where DK coverage is unavailable), see the",
            "by-season skill scores against internal baselines in",
            "`report/weekly_fantasy_projection_summary.md`. Those baselines are",
            "weaker than DK but the *consistency* of the beat across six seasons",
            "is the relevant signal there, not the absolute margin.",
            "",
            "## Coverage gap",
            "",
            "RotoGuru's free DK salary archive currently ends in 2021. Extending",
            "DK-style coverage to 2022-2025 requires a different (likely paid)",
            "source — see `PORTFOLIO_ROADMAP.md` Tier 1 item #1 for options",
            "(Stokastic, FantasyData, FantasyPros MVP archives, or the",
            "`ffanalytics` R package). The scaffolding accepts any CSV at",
            "`data/raw/external_projections*.csv` matching the documented schema,",
            "so swapping in a richer source is purely a data-acquisition step.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_external_benchmark_outputs(
    project_root: str | Path | None = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Build the external benchmark comparison artifacts.

    Reads the user-supplied `data/raw/external_projections.csv` and the
    committed `outputs/tables/weekly_fantasy_validation_predictions.csv`. Writes
    the comparison tables and a Markdown report. If the external CSV is
    missing, writes a stub report explaining how to populate it.
    """
    root = find_project_root() if project_root is None else Path(project_root).resolve()
    dirs = ensure_project_dirs(root)
    output_dir = dirs["tables"]
    report_dir = dirs["report"]

    external = load_external_projections(root)
    weekly_preds_path = output_dir / "weekly_fantasy_validation_predictions.csv"
    if weekly_preds_path.exists():
        weekly_preds = pd.read_csv(weekly_preds_path, low_memory=False)
        model_only = weekly_preds[
            weekly_preds["method"].eq("hist_gradient_boosting")
        ].copy()
    else:
        model_only = pd.DataFrame()

    results = compare(model_only, external)
    source_label = (
        str(external["source"].iloc[0])
        if not external.empty and "source" in external.columns
        else "user_supplied"
    )
    summary_text = _build_summary_text(results, source_label)

    if save_outputs:
        if not results["overall"].empty:
            results["overall"].to_csv(
                output_dir / "external_benchmark_overall.csv",
                index=False,
                float_format=CSV_FLOAT_FORMAT,
            )
        if not results["by_position"].empty:
            results["by_position"].to_csv(
                output_dir / "external_benchmark_by_position.csv",
                index=False,
                float_format=CSV_FLOAT_FORMAT,
            )
        if not results["by_season"].empty:
            results["by_season"].to_csv(
                output_dir / "external_benchmark_by_season.csv",
                index=False,
                float_format=CSV_FLOAT_FORMAT,
            )
        if not results["win_rate"].empty:
            results["win_rate"].to_csv(
                output_dir / "external_benchmark_win_rate.csv",
                index=False,
                float_format=CSV_FLOAT_FORMAT,
            )
        (report_dir / "external_benchmark.md").write_text(summary_text)

    return {
        **results,
        "summary_text": summary_text,
        "external_source": source_label,
    }
