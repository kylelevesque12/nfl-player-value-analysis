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
    """Load a user-provided CSV of external weekly PPR projections.

    Required columns: ``season``, ``week``, ``player_id`` (nflverse gsis id),
    ``external_projection_ppr``. Optional but recommended: ``source``,
    ``player_display_name``, ``position``, ``team``.

    Returns an empty DataFrame if the file is absent so the pipeline keeps
    working pre-acquisition.
    """
    root = find_project_root() if project_root is None else Path(project_root).resolve()
    rel = relative_path or EXTERNAL_CSV_DEFAULT_PATH
    path = root / rel
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"External projections at {path} missing required columns: "
            f"{sorted(missing)}"
        )
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["external_projection_ppr"] = pd.to_numeric(
        df["external_projection_ppr"], errors="coerce"
    )
    return df.dropna(subset=["season", "week", "player_id", "external_projection_ppr"])


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y) - np.asarray(p)))))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def _spearman(y: pd.Series, p: pd.Series) -> float:
    frame = pd.DataFrame({"y": y, "p": p}).dropna()
    if len(frame) < 2 or frame["y"].nunique() < 2 or frame["p"].nunique() < 2:
        return float("nan")
    return float(frame["y"].corr(frame["p"], method="spearman"))


def compare(
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

    overall = results["overall"].iloc[0]
    lines = [
        "# External Benchmark",
        "",
        f"Source: `{external_source}`",
        f"Player-weeks matched: {int(overall['n_player_weeks']):,}",
        "",
        "## Overall",
        "",
        "| Projector | RMSE | MAE |",
        "| --- | ---: | ---: |",
        f"| Weekly fantasy model | {overall['model_rmse']:.3f} | {overall['model_mae']:.3f} |",
        f"| External ({external_source}) | {overall['external_rmse']:.3f} | {overall['external_mae']:.3f} |",
        "",
        f"**Skill vs external**: {overall['skill_vs_external']:+.3%}",
        "",
        "## By position",
        "",
        "| Position | n | Model RMSE | External RMSE | Skill vs external |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in results["by_position"].iterrows():
        lines.append(
            f"| {row['segment_value']} | {int(row['n_player_weeks']):,} | "
            f"{row['model_rmse']:.3f} | {row['external_rmse']:.3f} | "
            f"{row['skill_vs_external']:+.3%} |"
        )

    if not results["win_rate"].empty:
        lines.extend(
            [
                "",
                "## Per-player-week win rate",
                "",
                "Share of player-weeks where the weekly model's projection landed "
                "closer to the actual PPR than the external projection did.",
                "",
                "| Position | n | Model win rate |",
                "| --- | ---: | ---: |",
            ]
        )
        for _, row in results["win_rate"].iterrows():
            lines.append(
                f"| {row['position']} | {int(row['n_player_weeks']):,} | "
                f"{row['model_win_rate']:.3f} |"
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
