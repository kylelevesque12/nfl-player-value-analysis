"""Methodology and data-quality checks for the NFL player value project."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.load_data import ensure_project_dirs, find_project_root, load_csv


CSV_FLOAT_FORMAT = "%.12g"


def _status(passed: bool, warn: bool = False) -> str:
    if passed:
        return "PASS"
    if warn:
        return "WARN"
    return "FAIL"


def _record(
    check_name: str,
    status: str,
    detail: str,
    value: Any = "",
    threshold: Any = "",
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "status": status,
        "value": value,
        "threshold": threshold,
        "detail": detail,
    }


def _git_tracked_files(project_root: Path, pathspec: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", pathspec],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _load_optional_csv(project_root: Path, relative_path: str) -> pd.DataFrame | None:
    path = project_root / relative_path
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def _safe_max_abs(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.abs().max())


def build_methodology_checks(project_root: str | Path | None = None) -> pd.DataFrame:
    """Build a table of project methodology checks."""
    root = find_project_root() if project_root is None else Path(project_root)
    records: list[dict[str, Any]] = []

    gitignore = (root / ".gitignore").read_text() if (root / ".gitignore").exists() else ""
    for pattern in ["data/raw/", "data/processed/", ".ipynb_checkpoints/", ".venv"]:
        present = pattern in gitignore
        records.append(
            _record(
                f"gitignore_contains_{pattern.strip('/')}",
                _status(present),
                f"`{pattern}` is listed in .gitignore.",
                present,
                True,
            )
        )

    for pathspec in ["data/raw", "data/processed"]:
        tracked = _git_tracked_files(root, pathspec)
        records.append(
            _record(
                f"no_tracked_{pathspec.replace('/', '_')}",
                _status(len(tracked) == 0),
                "Raw and processed data should stay local and out of Git.",
                len(tracked),
                0,
            )
        )

    required_local_files = [
        "data/raw/player_stats_2016_2025.csv",
        "data/raw/rosters_2016_2025.csv",
        "data/raw/schedules_2016_2025.csv",
        "data/processed/skill_player_seasons_2016_2025.csv",
        "data/processed/player_value_scores_2016_2025.csv",
    ]
    for relative_path in required_local_files:
        exists = (root / relative_path).exists()
        records.append(
            _record(
                f"local_file_exists_{Path(relative_path).name}",
                _status(exists),
                "Required local data file is available for reproducing outputs.",
                exists,
                True,
            )
        )

    skill = _load_optional_csv(root, "data/processed/skill_player_seasons_2016_2025.csv")
    if skill is not None:
        duplicate_team_rows = (
            skill.groupby(["season", "player_id", "team"])
            .size()
            .gt(1)
            .sum()
        )
        records.append(
            _record(
                "no_duplicate_player_season_team_rows",
                _status(duplicate_team_rows == 0),
                "Cleaned data should have at most one row per player-season-team.",
                int(duplicate_team_rows),
                0,
            )
        )

        offensive_positions = set(skill["position"].dropna().unique())
        expected_positions = {"QB", "RB", "WR", "TE"}
        records.append(
            _record(
                "only_skill_positions_in_cleaned_data",
                _status(offensive_positions.issubset(expected_positions)),
                "Cleaned data is restricted to QB, RB, WR, and TE.",
                ", ".join(sorted(offensive_positions)),
                ", ".join(sorted(expected_positions)),
            )
        )

    values = _load_optional_csv(root, "data/processed/player_value_scores_2016_2025.csv")
    if values is not None:
        duplicate_player_seasons = (
            values.groupby(["season", "player_id"])
            .size()
            .gt(1)
            .sum()
        )
        records.append(
            _record(
                "no_duplicate_value_player_seasons",
                _status(duplicate_player_seasons == 0),
                "Value scoring should collapse multi-team stints to one player-season.",
                int(duplicate_player_seasons),
                0,
            )
        )

        min_games = int(values["games_played"].min())
        records.append(
            _record(
                "minimum_games_filter_applied",
                _status(min_games >= 4),
                "Value-score rows should meet the minimum-games threshold.",
                min_games,
                ">= 4",
            )
        )

        group_summary = (
            values.groupby(["season", "position"])["value_score"]
            .agg(["mean", "std"])
            .reset_index()
        )
        max_abs_mean = _safe_max_abs(group_summary["mean"])
        max_abs_std_gap = _safe_max_abs(group_summary["std"] - 1)
        records.append(
            _record(
                "value_score_group_means_near_zero",
                _status(max_abs_mean <= 1e-9),
                "Standardized value scores should average near zero within season-position groups.",
                max_abs_mean,
                "<= 1e-9",
            )
        )
        records.append(
            _record(
                "value_score_group_stds_near_one",
                _status(max_abs_std_gap <= 1e-9),
                "Standardized value scores should have sample standard deviation near one within season-position groups.",
                max_abs_std_gap,
                "<= 1e-9",
            )
        )

        if {"value_epa_total", "qb_epa", "scrimmage_epa", "position"}.issubset(values.columns):
            expected_value_epa = np.where(
                values["position"].eq("QB"),
                values["qb_epa"],
                values["scrimmage_epa"],
            )
            max_epa_gap = float(
                np.nanmax(np.abs(values["value_epa_total"].to_numpy() - expected_value_epa))
            )
            records.append(
                _record(
                    "value_epa_total_matches_position_definition",
                    _status(max_epa_gap <= 1e-8),
                    "QBs use QB EPA; RB/WR/TE use scrimmage EPA.",
                    max_epa_gap,
                    "<= 1e-8",
                )
            )

        percentile_in_range = values["position_season_percentile"].between(0, 1).all()
        records.append(
            _record(
                "position_percentiles_in_range",
                _status(bool(percentile_in_range)),
                "Position-season percentiles should stay between 0 and 1.",
                bool(percentile_in_range),
                True,
            )
        )

    predictions = _load_optional_csv(root, "outputs/tables/2026_player_value_predictions.csv")
    if predictions is not None:
        prediction_count = len(predictions)
        records.append(
            _record(
                "prediction_report_has_rows",
                _status(prediction_count > 0),
                "2026 prediction report should contain player projections.",
                prediction_count,
                "> 0",
            )
        )

        intervals_valid = (
            predictions["prediction_interval_low"]
            .le(predictions["prediction_interval_high"])
            .all()
        )
        records.append(
            _record(
                "prediction_intervals_ordered",
                _status(bool(intervals_valid)),
                "Prediction interval low values should not exceed high values.",
                bool(intervals_valid),
                True,
            )
        )

        finite_predictions = np.isfinite(
            pd.to_numeric(predictions["predicted_2026_value_score"], errors="coerce")
        ).all()
        records.append(
            _record(
                "predictions_are_finite",
                _status(bool(finite_predictions)),
                "Predicted value scores should be finite numeric values.",
                bool(finite_predictions),
                True,
            )
        )

    notes_path = root / "outputs/tables/2026_prediction_model_notes.json"
    if notes_path.exists():
        import json

        notes = json.loads(notes_path.read_text())
        features = notes.get("features", [])
        leakage_like_features = [feature for feature in features if feature.startswith("next_")]
        records.append(
            _record(
                "prediction_features_do_not_use_next_season_columns",
                _status(len(leakage_like_features) == 0),
                "The production prediction model should not use future target columns.",
                ", ".join(leakage_like_features) if leakage_like_features else "none",
                "none",
            )
        )

        context_features_used = [
            feature
            for feature in features
            if any(token in feature for token in ["share_team", "avg_wopr", "avg_target_share"])
        ]
        records.append(
            _record(
                "context_features_not_blindly_added_to_main_model",
                _status(len(context_features_used) == 0),
                "Context features were tested separately and not automatically added to the production model.",
                ", ".join(context_features_used) if context_features_used else "none",
                "none",
            )
        )

    salary_diagnostics = _load_optional_csv(root, "outputs/tables/salary_efficiency_merge_diagnostics.csv")
    if salary_diagnostics is not None and "match_rate" in salary_diagnostics.columns:
        match_rate = float(salary_diagnostics["match_rate"].iloc[0])
        records.append(
            _record(
                "salary_merge_match_rate_above_90_percent",
                _status(match_rate >= 0.90),
                "Salary-efficiency analysis should report a high match rate.",
                match_rate,
                ">= 0.90",
            )
        )

    markdown_files = sorted((root / "notebooks_markdown").glob("*.md"))
    records.append(
        _record(
            "markdown_notebook_mirrors_exist",
            _status(len(markdown_files) >= 8),
            "Markdown notebook mirrors provide a GitHub-friendly fallback when notebook preview fails.",
            len(markdown_files),
            ">= 8",
        )
    )

    return pd.DataFrame(records)


def _markdown_table(df: pd.DataFrame) -> str:
    cols = ["check_name", "status", "value", "threshold", "detail"]
    output = df[cols].copy()
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in output.iterrows():
        rows.append(
            "| "
            + " | ".join(str(row[col]).replace("\n", " ") for col in cols)
            + " |"
        )
    return "\n".join([header, separator, *rows])


def write_methodology_report(checks: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a concise methodology checks report."""
    output_path = Path(output_path)
    status_counts = checks["status"].value_counts().to_dict()
    failed = int(status_counts.get("FAIL", 0))
    warned = int(status_counts.get("WARN", 0))
    passed = int(status_counts.get("PASS", 0))

    lines = [
        "# Methodology Checks",
        "",
        "This report records reproducibility and methodology checks for the NFL player value project. The goal is to make the project easier to audit: data files should be handled correctly, value scores should follow the stated definition, and prediction outputs should avoid obvious leakage problems.",
        "",
        "## Summary",
        "",
        f"- Passed checks: {passed}",
        f"- Warning checks: {warned}",
        f"- Failed checks: {failed}",
        "",
    ]

    if failed == 0:
        lines.append("No failing methodology checks were found in the current local project state.")
    else:
        lines.append("At least one methodology check failed and should be reviewed before presenting the project.")

    lines.extend(
        [
            "",
            "## Checks",
            "",
            _markdown_table(checks),
            "",
            "## Notes",
            "",
            "- These checks do not prove the model is correct; they catch common project-quality problems.",
            "- Raw and processed data are still intentionally local because they can be regenerated.",
            "- The prediction model is still best interpreted as a screening and tiering tool, not an exact ranking system.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def build_methodology_check_outputs(
    project_root: str | Path | None = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Build methodology check tables and report."""
    root = find_project_root() if project_root is None else Path(project_root)
    dirs = ensure_project_dirs(root)
    checks = build_methodology_checks(root)
    report_path = dirs["report"] / "methodology_checks.md"

    if save_outputs:
        checks.to_csv(
            dirs["tables"] / "methodology_checks.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        write_methodology_report(checks, report_path)

    return {
        "checks": checks,
        "report_path": report_path,
    }
