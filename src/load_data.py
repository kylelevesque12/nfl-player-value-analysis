"""Data-loading helpers for the NFL player value project.

The notebooks still tell the main story, but these helpers keep common path
and CSV-loading logic in one place. Raw and processed data remain local and
are intentionally ignored by Git.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT_MARKERS = [
    "README.md",
    "requirements.txt",
    "notebooks",
    "src",
]


def find_project_root(
    start: str | Path | None = None,
    markers: list[str] | None = None,
) -> Path:
    """Find the repository root from a notebook or terminal working directory."""
    if markers is None:
        markers = DEFAULT_PROJECT_MARKERS

    current = Path.cwd() if start is None else Path(start).resolve()
    candidates = [current, *current.parents]

    for candidate in candidates:
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    raise FileNotFoundError(
        "Could not find project root from " + str(current)
    )


def project_path(*parts: str, project_root: str | Path | None = None) -> Path:
    """Build an absolute path inside the project."""
    root = find_project_root() if project_root is None else Path(project_root)
    return root.joinpath(*parts)


def ensure_project_dirs(project_root: str | Path | None = None) -> dict[str, Path]:
    """Create and return the standard local data/output directories."""
    root = find_project_root() if project_root is None else Path(project_root)
    dirs = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "figures": root / "outputs" / "figures",
        "tables": root / "outputs" / "tables",
        "report": root / "report",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_csv(
    relative_path: str | Path,
    project_root: str | Path | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load a project CSV using a path relative to the repository root."""
    root = find_project_root() if project_root is None else Path(project_root)
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError("Missing data file: " + str(path))
    return pd.read_csv(path, **read_csv_kwargs)


def load_raw_player_stats(project_root: str | Path | None = None) -> pd.DataFrame:
    """Load locally saved weekly nflverse player stats."""
    return load_csv("data/raw/player_stats_2016_2025.csv", project_root)


def load_raw_rosters(project_root: str | Path | None = None) -> pd.DataFrame:
    """Load locally saved nflverse roster data."""
    return load_csv("data/raw/rosters_2016_2025.csv", project_root)


def load_raw_schedules(project_root: str | Path | None = None) -> pd.DataFrame:
    """Load locally saved nflverse schedule data."""
    return load_csv("data/raw/schedules_2016_2025.csv", project_root)


def load_skill_player_seasons(project_root: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned player-season dataset."""
    return load_csv("data/processed/skill_player_seasons_2016_2025.csv", project_root)


def load_player_value_scores(project_root: str | Path | None = None) -> pd.DataFrame:
    """Load the engineered value-score dataset."""
    return load_csv("data/processed/player_value_scores_2016_2025.csv", project_root)


def load_salary_efficiency_results(project_root: str | Path | None = None) -> pd.DataFrame:
    """Load the first-pass salary-efficiency results."""
    return load_csv("outputs/tables/salary_efficiency_2016_2025.csv", project_root)
