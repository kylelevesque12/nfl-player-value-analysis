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
from src.salary_findings import build_salary_finding_tables
from src.context_features import build_contextual_player_features
from src.feature_impact import build_context_feature_impact_outputs
from src.fantasy_projection import build_fantasy_projection_outputs
from src.methodology_checks import build_methodology_check_outputs
from src.model_interpretation import build_model_interpretation_outputs
from src.causal.session1_driver import build_causal_session1_outputs
from src.causal.session2_driver import build_causal_session2_outputs
from src.external_benchmark import build_external_benchmark_outputs
from src.rookie_bayes import build_rookie_modeling_frame
from src.two_stage_weekly import build_two_stage_weekly_outputs
from src.weekly_fantasy_projection import build_weekly_fantasy_outputs
from src.weekly_win_projection import build_weekly_win_projection_outputs
from src.advanced_modeling import build_advanced_modeling_outputs
from src.model_benchmark import build_model_benchmark_outputs
from src.value_decomposition import build_value_decomposition_outputs
from src.two_stage_value import build_two_stage_value_outputs


PIPELINE_STEPS = [
    "clean",
    "value",
    "decompose",
    "predictions",
    "salary",
    "findings",
    "fantasy",
    "weekly_fantasy",
    "external_benchmark",
    "rookie_bayes",
    "two_stage_weekly",
    "causal_session1",
    "causal_session2",
    "weekly_wins",
    "context",
    "feature_impact",
    "checks",
    "interpretation",
    "benchmark",
    "two_stage",
    "advanced_modeling",
]
DEFAULT_PIPELINE_STEPS = [
    "clean",
    "value",
    "decompose",
    "predictions",
    "salary",
    "findings",
    "fantasy",
    "weekly_fantasy",
    "external_benchmark",
    "rookie_bayes",
    "two_stage_weekly",
    "causal_session1",
    "causal_session2",
    "weekly_wins",
    "checks",
    "interpretation",
    "benchmark",
    "two_stage",
]


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


def build_salary_findings(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild salary-efficiency finding tables and narrative report."""
    root = _resolve_project_root(project_root)
    return build_salary_finding_tables(project_root=root, save_outputs=True)


def build_fantasy_outputs(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild 2026 fantasy-football projection tables and report."""
    root = _resolve_project_root(project_root)
    return build_fantasy_projection_outputs(project_root=root, save_outputs=True)


def build_weekly_win_outputs(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild weekly game winner projection tables and report."""
    root = _resolve_project_root(project_root)
    return build_weekly_win_projection_outputs(project_root=root, save_outputs=True)


def build_weekly_fantasy_outputs_step(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild weekly fantasy point projection tables and report."""
    root = _resolve_project_root(project_root)
    return build_weekly_fantasy_outputs(project_root=root, save_outputs=True)


def build_external_benchmark_step(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild head-to-head external projector comparison.

    Acts as a no-op writer when no external projections CSV is present. See
    `PORTFOLIO_ROADMAP.md` Tier 1 item #1 for acquisition guidance.
    """
    root = _resolve_project_root(project_root)
    return build_external_benchmark_outputs(project_root=root, save_outputs=True)


def build_two_stage_weekly_step(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the structurally-constrained two-stage weekly outputs."""
    root = _resolve_project_root(project_root)
    return build_two_stage_weekly_outputs(project_root=root, save_outputs=True)


def build_causal_session1_step(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the causal session-1 treatment / parallel-trends artifacts."""
    root = _resolve_project_root(project_root)
    return build_causal_session1_outputs(project_root=root, save_outputs=True)


def build_causal_session2_step(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the causal session-2 mitigation + estimation artifacts."""
    root = _resolve_project_root(project_root)
    return build_causal_session2_outputs(project_root=root, save_outputs=True)


def build_rookie_bayes_step(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the rookie modeling frame.

    The Bayesian sampling step requires a dedicated PyMC venv (see
    requirements-bayes.txt). This pipeline hook builds the modeling frame
    (always available, no PyMC dependency) and writes a report stub.
    Validation tables are only produced when PyMC is available; until then
    the stub explains the venv setup.
    """
    import pandas as pd

    root = _resolve_project_root(project_root)
    rosters = pd.read_csv(
        root / "data" / "raw" / "rosters_2016_2025.csv", low_memory=False
    )
    player_stats = pd.read_csv(
        root / "data" / "raw" / "player_stats_2016_2025.csv", low_memory=False
    )
    modeling_df = build_rookie_modeling_frame(
        rosters, player_stats, project_root=root
    )
    output_dir = root / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    modeling_df.to_csv(
        output_dir / "rookie_modeling_frame.csv",
        index=False,
        float_format="%.12g",
    )

    from src.rookie_bayes import _build_summary_text

    summary_text = _build_summary_text(modeling_df, pd.DataFrame(), pd.DataFrame())
    report_path = root / "report" / "rookie_bayes_projection.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary_text)
    return {"modeling_frame": modeling_df, "summary_text": summary_text}


def build_context_features(project_root: str | Path | None = None) -> pd.DataFrame:
    """Rebuild player-season contextual football features from raw local data."""
    root = _resolve_project_root(project_root)
    dirs = ensure_project_dirs(root)

    player_stats = load_csv("data/raw/player_stats_2016_2025.csv", root, low_memory=False)
    schedules = load_csv("data/raw/schedules_2016_2025.csv", root, low_memory=False)
    context_features = build_contextual_player_features(player_stats, schedules)
    output_path = dirs["processed"] / "player_context_features_2016_2025.csv"
    context_features.to_csv(output_path, index=False)
    return context_features


def build_feature_impact_outputs(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild contextual feature-impact comparison tables and report."""
    root = _resolve_project_root(project_root)
    return build_context_feature_impact_outputs(project_root=root, save_outputs=True)


def build_check_outputs(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild methodology checks and report."""
    root = _resolve_project_root(project_root)
    return build_methodology_check_outputs(project_root=root, save_outputs=True)


def build_interpretation_outputs(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild model interpretation tables and report."""
    root = _resolve_project_root(project_root)
    return build_model_interpretation_outputs(project_root=root, save_outputs=True)


def build_advanced_outputs(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild optional advanced modeling outputs."""
    root = _resolve_project_root(project_root)
    return build_advanced_modeling_outputs(project_root=root, save_outputs=True)


def build_decomposition_outputs(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild efficiency/opportunity value decomposition and stability report."""
    root = _resolve_project_root(project_root)
    return build_value_decomposition_outputs(project_root=root, save_outputs=True)


def build_benchmark_outputs(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild baseline/skill-score benchmark tables and conformal intervals."""
    root = _resolve_project_root(project_root)
    return build_model_benchmark_outputs(project_root=root, save_outputs=True)


def build_two_stage_outputs(project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild two-stage value model outputs (Stage 1: opportunity)."""
    root = _resolve_project_root(project_root)
    return build_two_stage_value_outputs(project_root=root, save_outputs=True)


def run_pipeline(
    steps: list[str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run selected reproducible pipeline steps in dependency order."""
    root = _resolve_project_root(project_root)
    if steps is None:
        steps = DEFAULT_PIPELINE_STEPS.copy()

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
        elif step == "decompose":
            results[step] = build_decomposition_outputs(root)
        elif step == "predictions":
            results[step] = build_prediction_outputs(root)
        elif step == "salary":
            results[step] = build_salary_outputs(root)
        elif step == "findings":
            results[step] = build_salary_findings(root)
        elif step == "fantasy":
            results[step] = build_fantasy_outputs(root)
        elif step == "weekly_fantasy":
            results[step] = build_weekly_fantasy_outputs_step(root)
        elif step == "external_benchmark":
            results[step] = build_external_benchmark_step(root)
        elif step == "rookie_bayes":
            results[step] = build_rookie_bayes_step(root)
        elif step == "two_stage_weekly":
            results[step] = build_two_stage_weekly_step(root)
        elif step == "causal_session1":
            results[step] = build_causal_session1_step(root)
        elif step == "causal_session2":
            results[step] = build_causal_session2_step(root)
        elif step == "weekly_wins":
            results[step] = build_weekly_win_outputs(root)
        elif step == "context":
            results[step] = build_context_features(root)
        elif step == "feature_impact":
            results[step] = build_feature_impact_outputs(root)
        elif step == "checks":
            results[step] = build_check_outputs(root)
        elif step == "interpretation":
            results[step] = build_interpretation_outputs(root)
        elif step == "benchmark":
            results[step] = build_benchmark_outputs(root)
        elif step == "two_stage":
            results[step] = build_two_stage_outputs(root)
        elif step == "advanced_modeling":
            results[step] = build_advanced_outputs(root)

    return results
