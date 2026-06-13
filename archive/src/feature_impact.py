"""Evaluate whether contextual football features improve value prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from src.context_features import (
    CONTEXT_FEATURE_GROUPS,
    create_context_feature_dictionary,
    build_contextual_player_features,
)
from src.load_data import ensure_project_dirs, find_project_root, load_csv
from src.prediction_report import (
    ENHANCED_FEATURES,
    TUNED_RANDOM_FOREST_PARAMS,
    add_player_history_features,
    create_next_season_targets,
    create_player_season_value_scores,
    evaluate_predictions,
    make_model_pipeline,
)


TARGET_COL = "next_value_score"
DEFAULT_VALIDATION_YEARS = [2020, 2021, 2022, 2023, 2024]
CSV_FLOAT_FORMAT = "%.12g"


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def _feature_group_lookup(feature: str) -> str:
    for group_name, features in CONTEXT_FEATURE_GROUPS.items():
        if feature in features:
            return group_name
    if feature == "position":
        return "baseline_categorical"
    if feature.endswith("_prev") or "last2" in feature or "last3" in feature or "trend" in feature:
        return "baseline_history"
    return "baseline_current_season"


def make_context_feature_sets(modeling_df: pd.DataFrame) -> dict[str, list[str]]:
    """Create baseline and context-expanded feature sets that exist in the data."""
    baseline = _available(modeling_df, ENHANCED_FEATURES)
    usage = _available(modeling_df, CONTEXT_FEATURE_GROUPS["usage_context"])
    team = _available(modeling_df, CONTEXT_FEATURE_GROUPS["team_context"])
    schedule = _available(modeling_df, CONTEXT_FEATURE_GROUPS["schedule_context"])
    all_context = _dedupe(usage + team + schedule)

    return {
        "baseline": baseline,
        "baseline_plus_usage_context": _dedupe(baseline + usage),
        "baseline_plus_team_context": _dedupe(baseline + team),
        "baseline_plus_schedule_context": _dedupe(baseline + schedule),
        "baseline_plus_all_context": _dedupe(baseline + all_context),
    }


def _top_quintile_hit_rate(
    validation_df: pd.DataFrame,
    predictions: np.ndarray,
    target_col: str,
) -> float:
    scored = validation_df[["season", "position", target_col]].copy()
    scored["prediction"] = predictions

    actual_cutoff = scored.groupby(["season", "position"])[target_col].transform(
        lambda s: s.quantile(0.80)
    )
    predicted_cutoff = scored.groupby(["season", "position"])["prediction"].transform(
        lambda s: s.quantile(0.80)
    )
    actual_top = scored[target_col].ge(actual_cutoff)
    predicted_top = scored["prediction"].ge(predicted_cutoff)

    if predicted_top.sum() == 0:
        return np.nan
    return float((actual_top & predicted_top).sum() / predicted_top.sum())


def _spearman_corr(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(pd.Series(y_true.to_numpy()).corr(pd.Series(y_pred), method="spearman"))


def compare_context_feature_groups(
    modeling_df: pd.DataFrame,
    feature_sets: dict[str, list[str]] | None = None,
    target_col: str = TARGET_COL,
    validation_years: list[int] | None = None,
    model_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run rolling validation for each feature group and summarize impact."""
    if feature_sets is None:
        feature_sets = make_context_feature_sets(modeling_df)
    if validation_years is None:
        validation_years = DEFAULT_VALIDATION_YEARS
    if model_params is None:
        model_params = TUNED_RANDOM_FOREST_PARAMS

    records: list[dict[str, Any]] = []

    for feature_set_name, feature_cols in feature_sets.items():
        feature_cols = _available(modeling_df, feature_cols)
        context_feature_count = sum(
            1
            for feature in feature_cols
            if _feature_group_lookup(feature).endswith("_context")
        )

        for valid_year in validation_years:
            train_df = (
                modeling_df[modeling_df["season"].lt(valid_year)]
                .dropna(subset=[target_col])
                .copy()
            )
            valid_df = (
                modeling_df[modeling_df["season"].eq(valid_year)]
                .dropna(subset=[target_col])
                .copy()
            )

            if train_df.empty or valid_df.empty:
                continue

            model = RandomForestRegressor(**model_params)
            pipeline = make_model_pipeline(feature_cols, model)
            pipeline.fit(train_df[feature_cols], train_df[target_col])
            predictions = pipeline.predict(valid_df[feature_cols])

            metrics = evaluate_predictions(valid_df[target_col], predictions)
            records.append(
                {
                    "feature_set": feature_set_name,
                    "validation_season": valid_year,
                    "feature_count": len(feature_cols),
                    "context_feature_count": context_feature_count,
                    "train_rows": len(train_df),
                    "validation_rows": len(valid_df),
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "r2": float(metrics["r2"]),
                    "spearman_rank_corr": _spearman_corr(valid_df[target_col], predictions),
                    "top_quintile_hit_rate": _top_quintile_hit_rate(
                        valid_df,
                        predictions,
                        target_col,
                    ),
                    "features": ", ".join(feature_cols),
                }
            )

    comparison = pd.DataFrame(records)
    if comparison.empty:
        return comparison, pd.DataFrame()

    summary = (
        comparison.groupby("feature_set", as_index=False)
        .agg(
            validation_folds=("validation_season", "nunique"),
            feature_count=("feature_count", "first"),
            context_feature_count=("context_feature_count", "first"),
            avg_mae=("mae", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_r2=("r2", "mean"),
            avg_spearman_rank_corr=("spearman_rank_corr", "mean"),
            avg_top_quintile_hit_rate=("top_quintile_hit_rate", "mean"),
        )
    )

    baseline = summary[summary["feature_set"].eq("baseline")]
    if not baseline.empty:
        baseline_row = baseline.iloc[0]
        summary["mae_delta_vs_baseline"] = summary["avg_mae"] - baseline_row["avg_mae"]
        summary["rmse_delta_vs_baseline"] = summary["avg_rmse"] - baseline_row["avg_rmse"]
        summary["r2_delta_vs_baseline"] = summary["avg_r2"] - baseline_row["avg_r2"]
        summary["spearman_delta_vs_baseline"] = (
            summary["avg_spearman_rank_corr"] - baseline_row["avg_spearman_rank_corr"]
        )
        summary["rmse_pct_change_vs_baseline"] = (
            summary["rmse_delta_vs_baseline"] / baseline_row["avg_rmse"] * 100
        )
    else:
        summary["mae_delta_vs_baseline"] = np.nan
        summary["rmse_delta_vs_baseline"] = np.nan
        summary["r2_delta_vs_baseline"] = np.nan
        summary["spearman_delta_vs_baseline"] = np.nan
        summary["rmse_pct_change_vs_baseline"] = np.nan

    summary["impact_label"] = np.select(
        [
            summary["rmse_delta_vs_baseline"].lt(-0.01)
            & summary["spearman_delta_vs_baseline"].ge(0),
            summary["rmse_delta_vs_baseline"].gt(0.01),
        ],
        [
            "helped validation performance",
            "hurt validation performance",
        ],
        default="roughly neutral",
    )

    summary = summary.sort_values(["avg_rmse", "avg_mae"]).reset_index(drop=True)
    return comparison, summary


def build_context_modeling_data(
    project_root: str | Path | None = None,
    save_context_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the modeling table used for context feature impact tests."""
    root = find_project_root() if project_root is None else Path(project_root)
    dirs = ensure_project_dirs(root)

    player_stats = load_csv("data/raw/player_stats_2016_2025.csv", root, low_memory=False)
    schedules = load_csv("data/raw/schedules_2016_2025.csv", root, low_memory=False)
    skill_seasons = load_csv(
        "data/processed/skill_player_seasons_2016_2025.csv",
        root,
        low_memory=False,
    )

    context_features = build_contextual_player_features(player_stats, schedules)
    if save_context_features:
        context_features.to_csv(
            dirs["processed"] / "player_context_features_2016_2025.csv",
            index=False,
        )

    player_season = create_player_season_value_scores(skill_seasons)
    player_season = add_player_history_features(player_season)
    player_season = create_next_season_targets(player_season)
    modeling_df = player_season.merge(
        context_features,
        on=["season", "player_id"],
        how="left",
    )
    return modeling_df, context_features, create_context_feature_dictionary()


def calculate_permutation_importance_2024(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COL,
    model_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Estimate feature impact on the 2024 validation fold by permutation."""
    if model_params is None:
        model_params = TUNED_RANDOM_FOREST_PARAMS

    feature_cols = _available(modeling_df, feature_cols)
    train_df = modeling_df[modeling_df["season"].lt(2024)].dropna(subset=[target_col]).copy()
    valid_df = modeling_df[modeling_df["season"].eq(2024)].dropna(subset=[target_col]).copy()

    if train_df.empty or valid_df.empty:
        return pd.DataFrame()

    pipeline = make_model_pipeline(
        feature_cols,
        RandomForestRegressor(**model_params),
    )
    pipeline.fit(train_df[feature_cols], train_df[target_col])
    result = permutation_importance(
        pipeline,
        valid_df[feature_cols],
        valid_df[target_col],
        scoring="neg_root_mean_squared_error",
        n_repeats=8,
        random_state=42,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "feature_group": [_feature_group_lookup(feature) for feature in feature_cols],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    output = df[cols].copy()
    if max_rows is not None:
        output = output.head(max_rows)

    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in output.iterrows():
        rows.append(
            "| "
            + " | ".join(_format_markdown_value(row[col]) for col in cols)
            + " |"
        )
    return "\n".join([header, separator, *rows])


def write_context_feature_report(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    permutation: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write a short methodological report for the context feature tests."""
    output_path = Path(output_path)

    if summary.empty:
        text = (
            "# Context Feature Impact Review\n\n"
            "No validation results were produced. Check that the processed value-score "
            "data and raw player/schedule files exist locally.\n"
        )
        output_path.write_text(text)
        return output_path

    baseline = summary[summary["feature_set"].eq("baseline")]
    best = summary.iloc[0]
    baseline_rmse = baseline.iloc[0]["avg_rmse"] if not baseline.empty else np.nan

    summary_cols = [
        "feature_set",
        "feature_count",
        "context_feature_count",
        "avg_mae",
        "avg_rmse",
        "rmse_delta_vs_baseline",
        "avg_spearman_rank_corr",
        "impact_label",
    ]
    importance_cols = [
        "feature",
        "feature_group",
        "importance_mean",
        "importance_std",
    ]

    lines = [
        "# Context Feature Impact Review",
        "",
        "This review tests whether contextual football features improve the next-season value model. "
        "The comparison uses the same rolling-season setup as the main modeling workflow: train on earlier seasons, validate on a future season, and repeat across 2020-2024.",
        "",
        "The goal is not to keep every possible football feature. The goal is to add context only when it improves out-of-sample performance or makes the model more explainable.",
        "",
        "## Feature Sets Tested",
        "",
        "- `baseline`: current production, age/draft inputs, and multi-year history features.",
        "- `baseline_plus_usage_context`: baseline plus target share, air-yards share, WOPR, PACR/RACR, and CPOE.",
        "- `baseline_plus_team_context`: baseline plus team volume, team efficiency, and player role-share features.",
        "- `baseline_plus_schedule_context`: baseline plus rest, spread, total, home/road, division, roof, surface, temperature, and wind context.",
        "- `baseline_plus_all_context`: baseline plus every context group above.",
        "",
        "## Rolling-Validation Summary",
        "",
        _markdown_table(summary, summary_cols),
        "",
        "Lower RMSE and MAE are better. Higher Spearman correlation is better because it means the model is doing a better job sorting players into relative order.",
        "",
        "## Current Read",
        "",
        f"The baseline average RMSE is `{baseline_rmse:.3f}`. The best tested feature set is `{best['feature_set']}` with average RMSE `{best['avg_rmse']:.3f}`.",
    ]

    if best["feature_set"] == "baseline":
        lines.append(
            "At this stage, the added context features do not clearly beat the simpler baseline. "
            "That is still useful: it says the current model is not obviously missing easy signal from these context groups."
        )
    elif best["rmse_delta_vs_baseline"] < -0.01:
        lines.append(
            "The best context feature set improves rolling-validation RMSE by more than 0.01, "
            "so it is a reasonable candidate for the next production model after checking for interpretability and leakage."
        )
    else:
        lines.append(
            "The best context feature set is very close to baseline, so the safer interpretation is that the new features are roughly neutral until more evidence is added."
        )

    if not permutation.empty:
        lines.extend(
            [
                "",
                "## 2024 Permutation Importance",
                "",
                "Permutation importance estimates how much the 2024 validation score worsens when a feature is randomly shuffled. Positive values suggest the feature carried useful signal in that fold.",
                "",
                _markdown_table(permutation, importance_cols, max_rows=20),
            ]
        )

    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Context features are built from current-season information only, then tested against next-season value.",
            "- The schedule features describe the games the player actually played in that season. They should not be used as future-season features unless a separate preseason schedule forecast is built.",
            "- Team EPA context is treated as an environment signal, not a new value metric.",
            "- A feature group should be adopted only if it helps validation or gives a clear interpretability benefit without creating leakage.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def build_context_feature_impact_outputs(
    project_root: str | Path | None = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Build context features, compare feature groups, and save review outputs."""
    root = find_project_root() if project_root is None else Path(project_root)
    dirs = ensure_project_dirs(root)

    modeling_df, context_features, feature_dictionary = build_context_modeling_data(
        root,
        save_context_features=save_outputs,
    )
    feature_sets = make_context_feature_sets(modeling_df)
    comparison, summary = compare_context_feature_groups(modeling_df, feature_sets)
    permutation = calculate_permutation_importance_2024(
        modeling_df,
        feature_sets["baseline_plus_all_context"],
    )

    report_path = dirs["report"] / "context_feature_impact.md"
    if save_outputs:
        comparison.to_csv(
            dirs["tables"] / "context_feature_group_comparison.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        summary.to_csv(
            dirs["tables"] / "context_feature_group_summary.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        permutation.to_csv(
            dirs["tables"] / "context_feature_permutation_importance_2024.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        feature_dictionary.to_csv(
            dirs["tables"] / "context_feature_dictionary.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        write_context_feature_report(summary, comparison, permutation, report_path)

    return {
        "modeling_df": modeling_df,
        "context_features": context_features,
        "feature_dictionary": feature_dictionary,
        "comparison": comparison,
        "summary": summary,
        "permutation_importance": permutation,
        "report_path": report_path,
    }
