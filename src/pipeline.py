"""Reproducible pipeline entry points for the NFL player value project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.clean_data import build_skill_player_seasons
from src.load_data import ensure_project_dirs, find_project_root, load_csv
from src.prediction_report import (
    MIN_VALUE_GAMES,
    build_2026_prediction_tables,
    create_player_season_value_scores,
)
from src.salary_efficiency import build_salary_efficiency_tables


PIPELINE_STEPS = ["clean", "value", "predictions", "salary"]


def _resolve_project_root(project_root: str | Path | None = None) -> Path:
    return find_project_root() if project_root is None else Path(project_root).resolve()


def build_cleaned_data(project_root: str | Path | None = None) -> pd.DataFrame:
    """Rebuild the cleaned player-season file from local raw CSVs."""
    root = _resolve_project_root(project_root)
    dirs = ensure_project_dirs(root)

    player_stats = load_csv("data/raw/player_stats_2016_2025.csv", root, low_memory=False)
    rosters = load_csv("data/raw/rosters_2016_2025.csv", root, low_memory=False)

    skill_seasons = build_skill_player_seasons(player_stats, rosters)
    output_path = dirs["processed"] / "skill_player_seasons_2016_2025.csv"
    skill_seasons.to_csv(output_path, index=False)
    return skill_seasons


def build_value_scores(project_root: str | Path | None = None) -> pd.DataFrame:
    """Rebuild player value scores from the cleaned player-season file."""
    root = _resolve_project_root(project_root)
    dirs = ensure_project_dirs(root)

    skill_seasons = load_csv(
        "data/processed/skill_player_seasons_2016_2025.csv",
        root,
    )
    value_scores = create_player_season_value_scores(
        skill_seasons,
        min_games=MIN_VALUE_GAMES,
    )
    output_path = dirs["processed"] / "player_value_scores_2016_2025.csv"
    value_scores.to_csv(output_path, index=False)
    return value_scores


def build_prediction_outputs(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild 2026 prediction tables and Excel workbook."""
    root = _resolve_project_root(project_root)
    return build_2026_prediction_tables(project_root=root, save_outputs=True)


def build_salary_outputs(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild salary-efficiency tables from value scores and local contracts."""
    root = _resolve_project_root(project_root)
    return build_salary_efficiency_tables(project_root=root, save_outputs=True)


def run_pipeline(
    steps: list[str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run selected reproducible pipeline steps in dependency order."""
    root = _resolve_project_root(project_root)
    if steps is None:
        steps = PIPELINE_STEPS.copy()

    unknown_steps = sorted(set(steps) - set(PIPELINE_STEPS))
    if unknown_steps:
        raise ValueError("Unknown pipeline steps: " + ", ".join(unknown_steps))

    results: dict[str, Any] = {"project_root": root}

    for step in PIPELINE_STEPS:
        if step not in steps:
            continue
        if step == "clean":
            results[step] = build_cleaned_data(root)
        elif step == "value":
            results[step] = build_value_scores(root)
        elif step == "predictions":
            results[step] = build_prediction_outputs(root)
        elif step == "salary":
            results[step] = build_salary_outputs(root)

    return results
