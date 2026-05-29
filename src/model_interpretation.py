"""Model interpretation and position-specific diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

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
VALIDATION_YEARS = [2020, 2021, 2022, 2023, 2024]
CSV_FLOAT_FORMAT = "%.12g"


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def _feature_group(feature: str) -> str:
    if feature == "position":
        return "position_indicator"
    if feature in {"age", "years_exp", "draft_number"}:
        return "player_profile"
    if feature in {"games_played", "value_epa_total", "value_epa_per_game", "yards_per_game", "tds_per_game"}:
        return "current_season_production"
    if "last2" in feature or "last3" in feature or feature.endswith("_prev") or "trend" in feature:
        return "player_history"
    if feature == "prior_qualifying_seasons":
        return "player_history"
    return "other"


def build_prediction_modeling_data(project_root: str | Path | None = None) -> pd.DataFrame:
    """Recreate the modeling table used by the prediction report."""
    root = find_project_root() if project_root is None else Path(project_root)
    skill_seasons = load_csv(
        "data/processed/skill_player_seasons_2016_2025.csv",
        root,
        low_memory=False,
    )
    player_season = create_player_season_value_scores(skill_seasons)
    player_season = add_player_history_features(player_season)
    player_season = create_next_season_targets(player_season)
    return player_season


def calculate_production_feature_importance(
    modeling_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate permutation importance on the 2024 validation fold."""
    if feature_cols is None:
        feature_cols = _available(modeling_df, ENHANCED_FEATURES)

    train_df = modeling_df[modeling_df["season"].lt(2024)].dropna(subset=[target_col]).copy()
    valid_df = modeling_df[modeling_df["season"].eq(2024)].dropna(subset=[target_col]).copy()
    if train_df.empty or valid_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pipeline = make_model_pipeline(
        feature_cols,
        RandomForestRegressor(**TUNED_RANDOM_FOREST_PARAMS),
    )
    pipeline.fit(train_df[feature_cols], train_df[target_col])
    result = permutation_importance(
        pipeline,
        valid_df[feature_cols],
        valid_df[target_col],
        scoring="neg_root_mean_squared_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "feature_group": [_feature_group(feature) for feature in feature_cols],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    group_importance = (
        importance.groupby("feature_group", as_index=False)
        .agg(
            total_importance=("importance_mean", "sum"),
            mean_importance=("importance_mean", "mean"),
            feature_count=("feature", "count"),
        )
        .sort_values("total_importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance.reset_index(drop=True), group_importance


def _evaluate_naive_current_value(valid_df: pd.DataFrame) -> dict[str, float]:
    predictions = valid_df["value_score"].fillna(0).to_numpy()
    return evaluate_predictions(valid_df[TARGET_COL], predictions)


def compare_pooled_position_and_naive_models(
    modeling_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    validation_years: list[int] | None = None,
) -> pd.DataFrame:
    """Compare pooled, position-specific, and simple baseline models."""
    if feature_cols is None:
        feature_cols = _available(modeling_df, ENHANCED_FEATURES)
    if validation_years is None:
        validation_years = VALIDATION_YEARS

    records: list[dict[str, Any]] = []
    positions = sorted(modeling_df["position"].dropna().unique())

    for valid_year in validation_years:
        train_df = modeling_df[modeling_df["season"].lt(valid_year)].dropna(subset=[TARGET_COL]).copy()
        valid_df = modeling_df[modeling_df["season"].eq(valid_year)].dropna(subset=[TARGET_COL]).copy()
        if train_df.empty or valid_df.empty:
            continue

        pooled_pipeline = make_model_pipeline(
            feature_cols,
            RandomForestRegressor(**TUNED_RANDOM_FOREST_PARAMS),
        )
        pooled_pipeline.fit(train_df[feature_cols], train_df[TARGET_COL])
        pooled_predictions = pooled_pipeline.predict(valid_df[feature_cols])
        valid_with_pooled = valid_df[["season", "position", TARGET_COL, "value_score"]].copy()
        valid_with_pooled["prediction"] = pooled_predictions

        for position in positions:
            fold_valid_pos = valid_with_pooled[valid_with_pooled["position"].eq(position)].copy()
            if fold_valid_pos.empty:
                continue

            pooled_metrics = evaluate_predictions(
                fold_valid_pos[TARGET_COL],
                fold_valid_pos["prediction"].to_numpy(),
            )
            records.append(
                {
                    "validation_season": valid_year,
                    "position": position,
                    "model_type": "pooled_model",
                    "train_rows": len(train_df),
                    "validation_rows": len(fold_valid_pos),
                    **pooled_metrics,
                }
            )

            naive_metrics = _evaluate_naive_current_value(fold_valid_pos)
            records.append(
                {
                    "validation_season": valid_year,
                    "position": position,
                    "model_type": "current_value_baseline",
                    "train_rows": len(train_df),
                    "validation_rows": len(fold_valid_pos),
                    **naive_metrics,
                }
            )

            fold_train_pos = train_df[train_df["position"].eq(position)].copy()
            fold_valid_raw_pos = valid_df[valid_df["position"].eq(position)].copy()
            position_features = [feature for feature in feature_cols if feature != "position"]
            if len(fold_train_pos) < 50 or fold_valid_raw_pos.empty:
                continue

            position_pipeline = make_model_pipeline(
                position_features,
                RandomForestRegressor(**TUNED_RANDOM_FOREST_PARAMS),
            )
            position_pipeline.fit(
                fold_train_pos[position_features],
                fold_train_pos[TARGET_COL],
            )
            position_predictions = position_pipeline.predict(fold_valid_raw_pos[position_features])
            position_metrics = evaluate_predictions(
                fold_valid_raw_pos[TARGET_COL],
                position_predictions,
            )
            records.append(
                {
                    "validation_season": valid_year,
                    "position": position,
                    "model_type": "position_specific_model",
                    "train_rows": len(fold_train_pos),
                    "validation_rows": len(fold_valid_raw_pos),
                    **position_metrics,
                }
            )

    return pd.DataFrame(records)


def summarize_position_model_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize position/model comparison across validation folds."""
    if comparison.empty:
        return pd.DataFrame()

    summary = (
        comparison.groupby(["position", "model_type"], as_index=False)
        .agg(
            validation_folds=("validation_season", "nunique"),
            avg_train_rows=("train_rows", "mean"),
            avg_validation_rows=("validation_rows", "mean"),
            avg_mae=("mae", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_r2=("r2", "mean"),
        )
    )

    pooled = summary[summary["model_type"].eq("pooled_model")][
        ["position", "avg_rmse", "avg_mae"]
    ].rename(columns={"avg_rmse": "pooled_avg_rmse", "avg_mae": "pooled_avg_mae"})
    summary = summary.merge(pooled, on="position", how="left")
    summary["rmse_delta_vs_pooled"] = summary["avg_rmse"] - summary["pooled_avg_rmse"]
    summary["mae_delta_vs_pooled"] = summary["avg_mae"] - summary["pooled_avg_mae"]
    return summary.sort_values(["position", "avg_rmse"]).reset_index(drop=True)


def _format_value(value: Any) -> str:
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
        rows.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    return "\n".join([header, separator, *rows])


def write_model_interpretation_report(
    feature_importance: pd.DataFrame,
    group_importance: pd.DataFrame,
    position_summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write a concise model interpretation report."""
    output_path = Path(output_path)
    top_feature = (
        feature_importance.iloc[0]["feature"]
        if not feature_importance.empty
        else "unknown"
    )

    lines = [
        "# Model Interpretation",
        "",
        "This report explains what the prediction model appears to use and where it struggles. The model should be read as a screening tool for player tiers and uncertainty, not as an exact ranking engine.",
        "",
        "## Main Takeaways",
        "",
        f"- The strongest 2024 permutation signal is `{top_feature}`.",
        "- Recent production and multi-year player history carry most of the useful signal.",
        "- Position-specific models are tested as a diagnostic, but the pooled model remains easier to explain and usually has more stable training data.",
        "- Sports forecasting is noisy; the model is more useful for grouping players than for claiming exact future ranks.",
        "",
        "## Feature Importance By Group",
        "",
    ]

    if group_importance.empty:
        lines.append("No feature-importance rows were available.")
    else:
        lines.append(
            _markdown_table(
                group_importance,
                ["feature_group", "feature_count", "total_importance", "mean_importance"],
            )
        )

    lines.extend(["", "## Top Feature Importance Rows", ""])
    if feature_importance.empty:
        lines.append("No feature-importance rows were available.")
    else:
        lines.append(
            _markdown_table(
                feature_importance,
                ["feature", "feature_group", "importance_mean", "importance_std"],
                max_rows=15,
            )
        )

    lines.extend(["", "## Position-Specific Model Comparison", ""])
    if position_summary.empty:
        lines.append("No position-specific comparison rows were available.")
    else:
        lines.append(
            _markdown_table(
                position_summary,
                [
                    "position",
                    "model_type",
                    "avg_train_rows",
                    "avg_rmse",
                    "rmse_delta_vs_pooled",
                    "avg_mae",
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Permutation importance was measured on the 2024 validation fold, so it should be treated as directional rather than permanent truth.",
            "- A feature with low importance is not necessarily useless; it may overlap with stronger related features.",
            "- Position-specific models can be appealing, but smaller samples make them easier to overfit.",
            "- The production model intentionally remains conservative because added context features only produced small validation gains.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def build_model_interpretation_outputs(
    project_root: str | Path | None = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Build model interpretation tables and report."""
    root = find_project_root() if project_root is None else Path(project_root)
    dirs = ensure_project_dirs(root)
    modeling_df = build_prediction_modeling_data(root)
    feature_cols = _available(modeling_df, ENHANCED_FEATURES)

    feature_importance, group_importance = calculate_production_feature_importance(
        modeling_df,
        feature_cols,
    )
    position_comparison = compare_pooled_position_and_naive_models(
        modeling_df,
        feature_cols,
    )
    position_summary = summarize_position_model_comparison(position_comparison)
    report_path = dirs["report"] / "model_interpretation.md"

    if save_outputs:
        feature_importance.to_csv(
            dirs["tables"] / "model_interpretation_feature_importance.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        group_importance.to_csv(
            dirs["tables"] / "model_interpretation_feature_group_importance.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        position_comparison.to_csv(
            dirs["tables"] / "position_model_comparison_by_fold.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        position_summary.to_csv(
            dirs["tables"] / "position_model_comparison_summary.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        write_model_interpretation_report(
            feature_importance,
            group_importance,
            position_summary,
            report_path,
        )

    return {
        "modeling_df": modeling_df,
        "feature_importance": feature_importance,
        "feature_group_importance": group_importance,
        "position_model_comparison": position_comparison,
        "position_model_summary": position_summary,
        "report_path": report_path,
    }
