"""Command-line runner for the NFL player value pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    return script_path.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild reproducible project outputs from local data files.",
    )
    parser.add_argument(
        "--steps",
        default="clean,value,decompose,predictions,salary,findings,fantasy,weekly_fantasy,external_benchmark,rookie_bayes,two_stage_weekly,causal_session1,weekly_wins,checks,interpretation,benchmark,two_stage",
        help=(
            "Comma-separated pipeline steps to run. "
            "Options: clean,value,decompose,predictions,salary,findings,context,"
            "fantasy,weekly_fantasy,external_benchmark,rookie_bayes,weekly_wins,"
            "feature_impact,checks,interpretation,benchmark,two_stage,advanced_modeling"
        ),
    )
    return parser.parse_args()


def main() -> int:
    project_root = _find_project_root()
    sys.path.insert(0, str(project_root))

    from src.pipeline import PIPELINE_STEPS, run_pipeline

    args = parse_args()
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    unknown_steps = sorted(set(steps) - set(PIPELINE_STEPS))
    if unknown_steps:
        print("Unknown pipeline steps: " + ", ".join(unknown_steps), file=sys.stderr)
        return 2

    print("Project root:", project_root)
    print("Running steps:", ", ".join(steps))
    results = run_pipeline(steps=steps, project_root=project_root)

    if "clean" in results:
        print("Cleaned player seasons:", results["clean"].shape)
    if "value" in results:
        print("Value scores:", results["value"].shape)
    if "predictions" in results:
        player_predictions = results["predictions"]["player_predictions"]
        print("2026 predictions:", player_predictions.shape)
    if "salary" in results:
        salary_efficiency = results["salary"]["salary_efficiency"]
        print("Salary-efficiency rows:", salary_efficiency.shape)
    if "findings" in results:
        finding_tables = results["findings"]["tables"]
        print("Salary finding sample:", finding_tables["finding_base"].shape)
    if "fantasy" in results:
        fantasy_predictions = results["fantasy"]["fantasy_predictions"]
        print("Fantasy projections:", fantasy_predictions.shape)
    if "weekly_fantasy" in results:
        weekly_fantasy = results["weekly_fantasy"]["predictions"]
        print("Weekly fantasy projection rows:", weekly_fantasy.shape)
    if "weekly_wins" in results:
        weekly_games = results["weekly_wins"]["weekly_win_games"]
        print("Weekly win backtest games:", weekly_games.shape)
    if "context" in results:
        print("Context feature rows:", results["context"].shape)
    if "feature_impact" in results:
        summary = results["feature_impact"]["summary"]
        print("Context feature-impact summary:", summary.shape)
    if "checks" in results:
        checks = results["checks"]["checks"]
        print("Methodology checks:", checks.shape)
    if "interpretation" in results:
        feature_importance = results["interpretation"]["feature_importance"]
        position_summary = results["interpretation"]["position_model_summary"]
        print("Model interpretation feature rows:", feature_importance.shape)
        print("Position model summary:", position_summary.shape)
    if "decompose" in results:
        decomposed = results["decompose"]["decomposed"]
        print("Value decomposition rows:", decomposed.shape)
    if "benchmark" in results:
        method_summary = results["benchmark"]["method_summary"]
        print("Model benchmark methods compared:", method_summary.shape)
    if "two_stage" in results:
        opp_summary = results["two_stage"]["opportunity_summary"]
        print("Two-stage opportunity methods compared:", opp_summary.shape)
    if "advanced_modeling" in results:
        advanced_summary = results["advanced_modeling"]["comparison_summary"]
        shap_importance = results["advanced_modeling"]["shap_importance"]
        print("Advanced modeling summary:", advanced_summary.shape)
        print("Advanced SHAP rows:", shap_importance.shape)

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
