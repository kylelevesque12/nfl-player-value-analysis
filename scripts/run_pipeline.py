"""Command-line entry point for the project's reproducible pipeline.

The pipeline has two perspectives (fantasy projection and front-office cap
allocation) sharing the same upstream cleaning / value-score / audit steps.
Each perspective has its own modeling and reporting steps below those.

Default is to run everything that doesn't need the dedicated PyMC venv. The
PyMC rookie sampling pass is its own command — see ``requirements-bayes.txt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    return script_path.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild project outputs from the local raw data files.",
    )
    parser.add_argument(
        "--steps",
        default=(
            # Shared upstream
            "clean,value,decompose,checks,interpretation,benchmark,"
            # Front office
            "predictions,salary,findings,two_stage,"
            # Fantasy
            "fantasy,rookie_rankings,draft_board,weekly_fantasy,"
            "external_benchmark,rookie_bayes,"
            "two_stage_weekly,causal_session1,causal_session2"
        ),
        help=(
            "Comma-separated pipeline steps. Two perspectives share the "
            "upstream cleaning/value/audit stages. Front-office steps: "
            "predictions, salary, findings, two_stage. Fantasy steps: "
            "fantasy, rookie_rankings, draft_board, weekly_fantasy, "
            "external_benchmark, rookie_bayes, two_stage_weekly, "
            "causal_session1, causal_session2. Shared: clean, value, "
            "decompose, checks, interpretation, benchmark. draft_board "
            "needs an ADP snapshot (`python scripts/fetch_adp.py --year "
            "2026`) for its market columns, but builds a full VORP-only "
            "board without one. rookie_rankings needs a separate "
            ".venv-bayes run to score the live rookie class first (see "
            "`src.rookie_bayes.build_2026_rookie_projection_outputs`); "
            "until then it's a no-op and the season table stays veterans-only."
        ),
    )
    return parser.parse_args()


def main() -> int:
    project_root = _find_project_root()
    sys.path.insert(0, str(project_root))

    from src.pipeline import PIPELINE_STEPS, run_pipeline

    args = parse_args()
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    unknown = sorted(set(steps) - set(PIPELINE_STEPS))
    if unknown:
        print("Unknown pipeline steps: " + ", ".join(unknown), file=sys.stderr)
        return 2

    print("Project root:", project_root)
    print("Running steps:", ", ".join(steps))
    results = run_pipeline(steps=steps, project_root=project_root)

    # Brief summary of what each step produced. Quiet on steps that didn't run.
    if "clean" in results:
        print("Cleaned player seasons:", results["clean"].shape)
    if "value" in results:
        print("Value scores:", results["value"].shape)
    if "decompose" in results:
        print("Value decomposition rows:", results["decompose"]["decomposed"].shape)
    if "predictions" in results:
        print("2026 predictions:", results["predictions"]["player_predictions"].shape)
    if "salary" in results:
        print("Salary-efficiency rows:", results["salary"]["salary_efficiency"].shape)
    if "findings" in results:
        print("Salary finding sample:", results["findings"]["tables"]["finding_base"].shape)
    if "fantasy" in results:
        print("Fantasy projections:", results["fantasy"]["fantasy_predictions"].shape)
    if "rookie_rankings" in results:
        n = results["rookie_rankings"]["n_rookies_added"]
        note = f"{n} rookies merged in" if n else "no-op (rookie projections not built yet)"
        print(f"Rookie rankings merge: {note}")
    if "draft_board" in results:
        board_diag = results["draft_board"]["diagnostics"]
        match_note = (
            f", ADP matched {board_diag['adp_match_rate']:.1%}"
            if "adp_match_rate" in board_diag
            else ", no ADP snapshot found (VORP-only board)"
        )
        print(f"Draft board: {results['draft_board']['board'].shape}{match_note}")
    if "weekly_fantasy" in results:
        print("Weekly fantasy projection rows:", results["weekly_fantasy"]["predictions"].shape)
    if "external_benchmark" in results:
        print("External benchmark complete.")
    if "rookie_bayes" in results:
        print("Rookie modeling frame:", results["rookie_bayes"]["modeling_frame"].shape)
    if "two_stage_weekly" in results:
        print("Two-stage weekly experiment complete.")
    if "causal_session1" in results:
        print("Causal QB-injury events:", results["causal_session1"]["events"].shape)
    if "causal_session2" in results:
        print("Causal session 2 complete.")
    if "checks" in results:
        print("Methodology checks:", results["checks"]["checks"].shape)
    if "interpretation" in results:
        print(
            "Model interpretation feature rows:",
            results["interpretation"]["feature_importance"].shape,
        )
    if "benchmark" in results:
        print(
            "Model benchmark methods compared:",
            results["benchmark"]["method_summary"].shape,
        )
    if "two_stage" in results:
        print(
            "Two-stage value opportunity summary:",
            results["two_stage"]["opportunity_summary"].shape,
        )

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
