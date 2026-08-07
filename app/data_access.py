"""Where the app's numbers come from: project paths and the cached table loads.

Every table is read from outputs/tables and cached on its modification time, so
a rebuilt pipeline is picked up without restarting the app. The app never
recomputes a model — it only reads what the pipeline already wrote.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"

# The research write-ups live in the repository, not in the app. A handful of
# product surfaces link out to one specific report for readers who want the
# depth behind a number they are looking at.
GITHUB_BLOB_BASE = "https://github.com/kylelevesque12/nfl-player-value-analysis/blob/main"


@st.cache_data
def load_csv(filename: str, modified_at: float) -> pd.DataFrame:
    path = TABLE_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0




def show_missing_data_warning(missing: list[str]) -> None:
    if missing:
        st.warning(
            "Some expected output tables are missing: "
            + ", ".join(missing)
            + ". Rebuild outputs with `python scripts/run_pipeline.py`."
        )


def load_all_data() -> dict[str, pd.DataFrame]:
    files = {
        "salary": "salary_efficiency_2016_2025.csv",
        "methodology": "methodology_checks.csv",
        "fantasy": "2026_fantasy_football_projections.csv",
        "weekly_fantasy": "weekly_fantasy_validation_predictions.csv",
        "weekly_fantasy_live": "weekly_fantasy_live_projection.csv",
        # Season value decomposition: powers the stable/shaky role badge and
        # the regression watch (efficiency_variance_share per player).
        "two_stage_projection": "two_stage_2026_projection.csv",
        # Overall draft board: VORP, auction values, and ADP edge.
        "draft_board": "draft_board_2026.csv",
        # Replacement-level surplus: headline for the research card + the
        # surplus history in Player Detail.
        "replacement_top_surplus": "salary_findings_replacement_top_surplus.csv",
        # External benchmark vs DraftKings (Draft Board accuracy tab)
        "external_benchmark_overall": "external_benchmark_overall.csv",
        "external_benchmark_by_position": "external_benchmark_by_position.csv",
        "external_benchmark_by_season": "external_benchmark_by_season.csv",
        "external_benchmark_win_rate": "external_benchmark_win_rate.csv",
        # Bayesian rookie projections (player index / Player Detail)
        "rookie_modeling_frame": "rookie_modeling_frame.csv",
        "rookie_bayes_validation_predictions": "rookie_bayes_validation_predictions.csv",
        # Causal QB-injury study: headline ATT for the research-studies card,
        # events for the player index / Player Detail view. The full analysis
        # lives in the repo reports, not as an app page.
        "causal_s3_att": "causal_s3_att.csv",
        "causal_s3_events": "causal_s3_first_report_events.csv",
    }
    return {
        name: load_csv(filename, file_mtime(TABLE_DIR / filename))
        for name, filename in files.items()
    }

