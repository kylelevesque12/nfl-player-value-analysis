"""Advanced modeling methodology experiments.

This module keeps heavier modeling tools optional. The normal project pipeline
can run with pandas and scikit-learn, while this step adds Optuna tuning, SHAP
explanations, a Polars data profile, and local MLflow tracking when those
packages are installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.load_data import ensure_project_dirs, find_project_root, load_csv
from src.prediction_report import (
    ENHANCED_FEATURES,
    TUNED_RANDOM_FOREST_PARAMS,
    add_player_history_features,
    create_next_season_targets,
    create_player_season_value_scores,
    make_model_pipeline,
)


TARGET_COL = "next_value_score"
VALIDATION_YEARS = [2020, 2021, 2022, 2023, 2024]
CSV_FLOAT_FORMAT = "%.12g"
DEFAULT_OPTUNA_TRIALS = 20


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def _optional_import(package: str):
    try:
        return __import__(package)
    except ImportError:
        return None


def _feature_group(feature: str) -> str:
    if feature == "position":
        return "position_indicator"
    if feature in {"age", "years_exp", "draft_number"}:
        return "player_profile"
    if feature in {
        "games_played",
        "value_epa_total",
        "value_epa_per_game",
        "yards_per_game",
        "tds_per_game",
    }:
        return "current_season_production"
    if (
        feature == "prior_qualifying_seasons"
        or feature.endswith("_prev")
        or "last2" in feature
        or "last3" in feature
        or "trend" in feature
    ):
        return "player_history"
    return "other"


def _clean_feature_name(feature_name: str) -> str:
    cleaned = feature_name
    for prefix in ("num__", "cat__"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned


def _raw_feature_name(transformed_feature: str, feature_cols: list[str]) -> str:
    cleaned = _clean_feature_name(transformed_feature)
    if cleaned.startswith("position_"):
        return "position"
    for feature in sorted(feature_cols, key=len, reverse=True):
        if cleaned == feature:
            return feature
    return cleaned


def _metric_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def _markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 10) -> str:
    if df.empty:
        return "_No rows available._"
    display = df[cols].head(max_rows).copy()
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in display.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def build_advanced_modeling_frame(project_root: str | Path | None = None) -> pd.DataFrame:
    """Recreate the player-season modeling table used by the value model."""
    root = find_project_root() if project_root is None else Path(project_root).resolve()
    skill_seasons = load_csv(
        "data/processed/skill_player_seasons_2016_2025.csv",
        root,
        low_memory=False,
    )
    player_season = create_player_season_value_scores(skill_seasons)
    player_season = add_player_history_features(player_season)
    player_season = create_next_season_targets(player_season)
    return player_season


def rolling_validate_random_forest(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    model_id: str,
    validation_years: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one Random Forest configuration with rolling-origin validation."""
    if validation_years is None:
        validation_years = VALIDATION_YEARS

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for valid_year in validation_years:
        train_df = modeling_df[modeling_df["season"].lt(valid_year)].dropna(subset=[TARGET_COL]).copy()
        valid_df = modeling_df[modeling_df["season"].eq(valid_year)].dropna(subset=[TARGET_COL]).copy()
        if train_df.empty or valid_df.empty:
            continue

        pipeline = make_model_pipeline(
            feature_cols,
            RandomForestRegressor(**params),
        )
        pipeline.fit(train_df[feature_cols], train_df[TARGET_COL])
        predictions = pipeline.predict(valid_df[feature_cols])
        metrics = _metric_dict(valid_df[TARGET_COL], predictions)
        fold_rows.append(
            {
                "model_id": model_id,
                "validation_season": valid_year,
                "train_rows": len(train_df),
                "validation_rows": len(valid_df),
                **metrics,
            }
        )

        fold_predictions = valid_df[
            ["season", "player_id", "player_display_name", "position", TARGET_COL]
        ].copy()
        fold_predictions["model_id"] = model_id
        fold_predictions["prediction"] = predictions
        fold_predictions["residual"] = fold_predictions[TARGET_COL] - fold_predictions["prediction"]
        fold_predictions["abs_residual"] = fold_predictions["residual"].abs()
        fold_predictions["validation_season"] = valid_year
        prediction_rows.append(fold_predictions)

    fold_metrics = pd.DataFrame(fold_rows)
    validation_predictions = (
        pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    )
    return fold_metrics, validation_predictions


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize rolling fold metrics by model id."""
    if fold_metrics.empty:
        return pd.DataFrame()
    summary = (
        fold_metrics.groupby("model_id", as_index=False)
        .agg(
            validation_folds=("validation_season", "nunique"),
            avg_train_rows=("train_rows", "mean"),
            avg_validation_rows=("validation_rows", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
        )
        .sort_values(["mean_rmse", "mean_mae"])
        .reset_index(drop=True)
    )
    current_rmse = summary.loc[
        summary["model_id"].eq("current_depth_limited_random_forest"),
        "mean_rmse",
    ]
    summary["rmse_delta_vs_current"] = (
        summary["mean_rmse"] - float(current_rmse.iloc[0])
        if not current_rmse.empty
        else np.nan
    )
    return summary


def _suggest_random_forest_params(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "max_features": trial.suggest_categorical("max_features", [0.35, 0.50, 0.65, 0.80, "sqrt"]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 60, step=5),
        "random_state": 42,
        "n_jobs": -1,
    }


def run_optuna_random_forest_search(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    n_trials: int = DEFAULT_OPTUNA_TRIALS,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Tune the value model with Optuna when available."""
    optuna = _optional_import("optuna")
    if optuna is None:
        return (
            pd.DataFrame(
                {
                    "status": ["missing_dependency"],
                    "message": ["Install optuna to run hyperparameter search."],
                }
            ),
            TUNED_RANDOM_FOREST_PARAMS.copy(),
            "optuna_missing_used_current_params",
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Any) -> float:
        params = _suggest_random_forest_params(trial)
        fold_metrics, _ = rolling_validate_random_forest(
            modeling_df,
            feature_cols,
            params,
            model_id=f"trial_{trial.number}",
        )
        if fold_metrics.empty:
            return float("inf")
        mean_rmse = float(fold_metrics["rmse"].mean())
        trial.set_user_attr("mean_mae", float(fold_metrics["mae"].mean()))
        trial.set_user_attr("mean_r2", float(fold_metrics["r2"].mean()))
        return mean_rmse

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trial_rows = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "state": str(trial.state),
            "mean_rmse": trial.value,
            "mean_mae": trial.user_attrs.get("mean_mae", np.nan),
            "mean_r2": trial.user_attrs.get("mean_r2", np.nan),
        }
        row.update(trial.params)
        trial_rows.append(row)

    best_params = TUNED_RANDOM_FOREST_PARAMS.copy()
    best_params.update(study.best_trial.params)
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    return pd.DataFrame(trial_rows), best_params, "optuna_lowest_rolling_rmse"


def calculate_shap_importance(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    max_rows: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Calculate SHAP importance on the 2024 validation fold when available."""
    shap = _optional_import("shap")
    if shap is None:
        message = "Install shap to calculate local model explanations."
        return (
            pd.DataFrame({"status": ["missing_dependency"], "message": [message]}),
            pd.DataFrame({"status": ["missing_dependency"], "message": [message]}),
            "shap_missing",
        )

    train_df = modeling_df[modeling_df["season"].lt(2024)].dropna(subset=[TARGET_COL]).copy()
    valid_df = modeling_df[modeling_df["season"].eq(2024)].dropna(subset=[TARGET_COL]).copy()
    if train_df.empty or valid_df.empty:
        return pd.DataFrame(), pd.DataFrame(), "missing_2024_validation_fold"

    pipeline = make_model_pipeline(feature_cols, RandomForestRegressor(**params))
    pipeline.fit(train_df[feature_cols], train_df[TARGET_COL])

    explanation_df = valid_df.sample(
        n=min(max_rows, len(valid_df)),
        random_state=42,
    )
    transformed = pipeline.named_steps["preprocessor"].transform(explanation_df[feature_cols])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = [
        _clean_feature_name(name)
        for name in pipeline.named_steps["preprocessor"].get_feature_names_out()
    ]

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(transformed)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    importance = pd.DataFrame(
        {
            "transformed_feature": feature_names,
            "raw_feature": [
                _raw_feature_name(feature, feature_cols) for feature in feature_names
            ],
            "feature_group": [
                _feature_group(_raw_feature_name(feature, feature_cols))
                for feature in feature_names
            ],
            "mean_abs_shap": np.abs(shap_array).mean(axis=0),
            "mean_shap": shap_array.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    group_importance = (
        importance.groupby("feature_group", as_index=False)
        .agg(
            total_mean_abs_shap=("mean_abs_shap", "sum"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            feature_count=("transformed_feature", "count"),
        )
        .sort_values("total_mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance.reset_index(drop=True), group_importance, "shap_tree_explainer_2024_fold"


def build_polars_data_profile(project_root: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    """Profile the cleaned player-season file with Polars when available."""
    polars = _optional_import("polars")
    root = find_project_root() if project_root is None else Path(project_root).resolve()
    data_path = root / "data" / "processed" / "skill_player_seasons_2016_2025.csv"
    if polars is None:
        return (
            pd.DataFrame(
                {
                    "status": ["missing_dependency"],
                    "message": ["Install polars to run the fast data profile."],
                }
            ),
            "polars_missing",
        )
    if not data_path.exists():
        return pd.DataFrame(), "missing_cleaned_player_season_file"

    pl_df = polars.read_csv(data_path, infer_schema_length=10000)
    row_count = pl_df.height
    profile_rows = []
    for column in pl_df.columns:
        null_count = pl_df[column].null_count()
        profile_rows.append(
            {
                "source_file": str(data_path.relative_to(root)),
                "rows": row_count,
                "columns": pl_df.width,
                "column": column,
                "dtype": str(pl_df[column].dtype),
                "null_count": int(null_count),
                "null_rate": float(null_count / row_count) if row_count else np.nan,
            }
        )
    return pd.DataFrame(profile_rows), "polars_profile_cleaned_player_seasons"


def write_advanced_modeling_report(
    output_path: str | Path,
    comparison_summary: pd.DataFrame,
    optuna_trials: pd.DataFrame,
    shap_importance: pd.DataFrame,
    shap_group_importance: pd.DataFrame,
    best_params: dict[str, Any],
    methodology_status: dict[str, str],
) -> Path:
    """Write a readable methodology report for the advanced modeling step."""
    output_path = Path(output_path)
    best_trial_text = ""
    if not optuna_trials.empty and "mean_rmse" in optuna_trials.columns:
        best_trial = optuna_trials.sort_values("mean_rmse").head(1)
        best_trial_text = _markdown_table(
            best_trial,
            [col for col in ["trial_number", "mean_rmse", "mean_mae", "mean_r2"] if col in best_trial.columns],
            max_rows=1,
        )
    else:
        best_trial_text = "_Optuna trials were not available._"

    report = (
        "# Advanced Modeling Methodology\n\n"
        "This report adds a more formal modeling layer on top of the existing "
        "NFL player value pipeline. The point is not to make the model look more "
        "complicated. The point is to make the tuning, explanations, and data "
        "checks easier to defend.\n\n"
        "## What This Adds\n\n"
        "- Optuna searches Random Forest hyperparameters across rolling validation folds.\n"
        "- SHAP explains which transformed features most influence the selected tree model.\n"
        "- Polars profiles the cleaned modeling data quickly, mostly as a data-quality aid.\n"
        "- MLflow stores a local experiment run when the package is installed. The committed "
        "CSV and Markdown outputs remain the reviewer-friendly version.\n\n"
        "## Current Model vs Optuna-Tuned Candidate\n\n"
        + _markdown_table(
            comparison_summary,
            [
                "model_id",
                "validation_folds",
                "mean_mae",
                "mean_rmse",
                "mean_r2",
                "rmse_delta_vs_current",
            ],
            max_rows=10,
        )
        + "\n\n"
        "## Best Optuna Trial\n\n"
        + best_trial_text
        + "\n\n"
        "Best parameter set:\n\n"
        "```json\n"
        + json.dumps(best_params, indent=2)
        + "\n```\n\n"
        "## SHAP Feature Importance\n\n"
        "SHAP values are calculated on the 2024 validation fold. This is an "
        "interpretation diagnostic, not a causal claim. Correlated football "
        "features can share importance with each other.\n\n"
        + _markdown_table(
            shap_importance,
            ["transformed_feature", "raw_feature", "feature_group", "mean_abs_shap"],
            max_rows=15,
        )
        + "\n\n"
        "## SHAP Feature Groups\n\n"
        + _markdown_table(
            shap_group_importance,
            ["feature_group", "total_mean_abs_shap", "feature_count"],
            max_rows=10,
        )
        + "\n\n"
        "## Status\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in methodology_status.items())
        + "\n\n"
        "## Interpretation\n\n"
        "If Optuna only improves RMSE by a tiny amount, I would not automatically "
        "replace the current model. In a sports forecasting project, small metric "
        "wins can be noise. The more important result is whether the tuned model "
        "is consistently better across seasons and still simple enough to explain.\n"
    )
    output_path.write_text(report)
    return output_path


def _log_mlflow_run(
    project_root: Path,
    best_params: dict[str, Any],
    comparison_summary: pd.DataFrame,
    artifact_paths: list[Path],
) -> str:
    mlflow = _optional_import("mlflow")
    if mlflow is None:
        return "mlflow_missing"

    tracking_dir = project_root / "mlruns"
    tracking_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("nfl-player-value-advanced-modeling")

    best_row = comparison_summary.sort_values("mean_rmse").head(1)
    with mlflow.start_run(run_name="optuna_random_forest_value_model"):
        mlflow.log_params(best_params)
        if not best_row.empty:
            mlflow.log_metric("mean_rmse", float(best_row["mean_rmse"].iloc[0]))
            mlflow.log_metric("mean_mae", float(best_row["mean_mae"].iloc[0]))
            mlflow.log_metric("mean_r2", float(best_row["mean_r2"].iloc[0]))
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
    return "logged_local_mlruns"


def build_advanced_modeling_outputs(
    project_root: str | Path | None = None,
    save_outputs: bool = True,
    n_trials: int = DEFAULT_OPTUNA_TRIALS,
) -> dict[str, Any]:
    """Build optional advanced modeling outputs."""
    root = find_project_root() if project_root is None else Path(project_root).resolve()
    dirs = ensure_project_dirs(root)
    output_dir = dirs["tables"]
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)

    modeling_df = build_advanced_modeling_frame(root)
    feature_cols = _available(modeling_df, ENHANCED_FEATURES)
    modeling_target_df = modeling_df.dropna(subset=[TARGET_COL]).copy()

    current_fold_metrics, current_predictions = rolling_validate_random_forest(
        modeling_target_df,
        feature_cols,
        TUNED_RANDOM_FOREST_PARAMS,
        model_id="current_depth_limited_random_forest",
    )

    optuna_trials, best_params, tuning_status = run_optuna_random_forest_search(
        modeling_target_df,
        feature_cols,
        n_trials=n_trials,
    )
    optuna_fold_metrics, optuna_predictions = rolling_validate_random_forest(
        modeling_target_df,
        feature_cols,
        best_params,
        model_id="optuna_tuned_random_forest",
    )

    fold_metrics = pd.concat(
        [current_fold_metrics, optuna_fold_metrics],
        ignore_index=True,
    )
    validation_predictions = pd.concat(
        [current_predictions, optuna_predictions],
        ignore_index=True,
    )
    comparison_summary = summarize_fold_metrics(fold_metrics)

    shap_importance, shap_group_importance, shap_status = calculate_shap_importance(
        modeling_target_df,
        feature_cols,
        best_params,
    )
    polars_profile, polars_status = build_polars_data_profile(root)

    best_params_path = output_dir / "advanced_modeling_best_params.json"
    report_path = report_dir / "advanced_modeling_methodology.md"
    artifact_paths = [
        output_dir / "advanced_modeling_optuna_trials.csv",
        output_dir / "advanced_modeling_validation_summary.csv",
        output_dir / "advanced_modeling_validation_folds.csv",
        output_dir / "advanced_modeling_shap_importance.csv",
        output_dir / "advanced_modeling_shap_group_importance.csv",
        output_dir / "advanced_modeling_polars_data_profile.csv",
        best_params_path,
        report_path,
    ]

    methodology_status = {
        "optuna": tuning_status,
        "shap": shap_status,
        "polars": polars_status,
    }

    if save_outputs:
        optuna_trials.to_csv(
            output_dir / "advanced_modeling_optuna_trials.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        comparison_summary.to_csv(
            output_dir / "advanced_modeling_validation_summary.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        fold_metrics.to_csv(
            output_dir / "advanced_modeling_validation_folds.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        validation_predictions.to_csv(
            output_dir / "advanced_modeling_validation_predictions.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        shap_importance.to_csv(
            output_dir / "advanced_modeling_shap_importance.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        shap_group_importance.to_csv(
            output_dir / "advanced_modeling_shap_group_importance.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        polars_profile.to_csv(
            output_dir / "advanced_modeling_polars_data_profile.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        best_params_path.write_text(json.dumps(best_params, indent=2))
        write_advanced_modeling_report(
            report_path,
            comparison_summary,
            optuna_trials,
            shap_importance,
            shap_group_importance,
            best_params,
            methodology_status,
        )
        methodology_status["mlflow"] = _log_mlflow_run(
            root,
            best_params,
            comparison_summary,
            artifact_paths,
        )
        write_advanced_modeling_report(
            report_path,
            comparison_summary,
            optuna_trials,
            shap_importance,
            shap_group_importance,
            best_params,
            methodology_status,
        )

    return {
        "modeling_df": modeling_df,
        "feature_cols": feature_cols,
        "optuna_trials": optuna_trials,
        "best_params": best_params,
        "comparison_summary": comparison_summary,
        "fold_metrics": fold_metrics,
        "validation_predictions": validation_predictions,
        "shap_importance": shap_importance,
        "shap_group_importance": shap_group_importance,
        "polars_profile": polars_profile,
        "methodology_status": methodology_status,
    }
