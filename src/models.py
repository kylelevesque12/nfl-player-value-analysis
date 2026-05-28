"""Modeling helpers for NFL player value analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split feature names into numeric and categorical groups."""
    if categorical_cols is None:
        categorical_cols = ["position"]

    categorical = [col for col in feature_cols if col in categorical_cols]
    numeric = [col for col in feature_cols if col not in categorical]
    return numeric, categorical


def make_preprocessor(
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Create preprocessing for mixed numeric/categorical model inputs."""
    numeric, categorical = split_feature_types(feature_cols, categorical_cols)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ],
        remainder="drop",
    )


def make_model_pipeline(
    feature_cols: list[str],
    model: Any,
    categorical_cols: list[str] | None = None,
) -> Pipeline:
    """Create a preprocessing plus estimator pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(feature_cols, categorical_cols)),
            ("model", model),
        ]
    )


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return common regression metrics for value-score models."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def temporal_train_validation_test_split(
    df: pd.DataFrame,
    train_end: int = 2022,
    validation_season: int = 2023,
    test_season: int = 2024,
    season_col: str = "season",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split player-season data in chronological order."""
    train_df = df[df[season_col].le(train_end)].copy()
    validation_df = df[df[season_col].eq(validation_season)].copy()
    test_df = df[df[season_col].eq(test_season)].copy()
    return train_df, validation_df, test_df


def rolling_regression_validation(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model: Any,
    validation_years: list[int] | None = None,
    season_col: str = "season",
) -> pd.DataFrame:
    """Run rolling-origin validation for a regression model."""
    if validation_years is None:
        validation_years = [2020, 2021, 2022, 2023, 2024]

    records: list[dict[str, float]] = []
    for valid_year in validation_years:
        train_df = df[df[season_col].lt(valid_year)].dropna(subset=[target_col]).copy()
        valid_df = df[df[season_col].eq(valid_year)].dropna(subset=[target_col]).copy()

        if train_df.empty or valid_df.empty:
            continue

        pipeline = make_model_pipeline(feature_cols, clone(model))
        pipeline.fit(train_df[feature_cols], train_df[target_col])
        predictions = pipeline.predict(valid_df[feature_cols])
        metrics = evaluate_regression(valid_df[target_col], predictions)
        metrics["validation_season"] = int(valid_year)
        metrics["train_rows"] = int(len(train_df))
        metrics["validation_rows"] = int(len(valid_df))
        records.append(metrics)

    return pd.DataFrame(records)


def compare_regression_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    models: dict[str, Any],
) -> pd.DataFrame:
    """Fit several regressors and return a compact metric comparison table."""
    records: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        pipeline = make_model_pipeline(feature_cols, clone(model))
        pipeline.fit(train_df[feature_cols], train_df[target_col])
        predictions = pipeline.predict(test_df[feature_cols])
        metrics = evaluate_regression(test_df[target_col], predictions)
        metrics["model"] = model_name
        records.append(metrics)

    return (
        pd.DataFrame(records)
        .sort_values(["rmse", "mae"], ascending=[True, True])
        .reset_index(drop=True)
    )
