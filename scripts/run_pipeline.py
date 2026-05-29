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
        default="clean,value,predictions,salary,findings,checks,interpretation",
        help=(
            "Comma-separated pipeline steps to run. "
            "Options: clean,value,predictions,salary,findings,context,"
            "feature_impact,checks,interpretation"
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

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
