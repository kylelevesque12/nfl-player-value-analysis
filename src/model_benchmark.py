"""Baseline, skill-score, model, and conformal-interval benchmarking.

This module answers a question the existing report does not: *is the tuned
Random Forest actually better than a simple, defensible baseline?* Because the
prediction target (``next_value_score``) is standardized within each
season-position group, its per-season standard deviation is close to 1.0, so a
model that predicts the season mean already gets an RMSE near 1.0. Reporting
RMSE alone therefore overstates how much the model is learning. The honest
benchmark is a *skill score*: the percentage RMSE reduction versus a strong
naive baseline.

The module is deliberately leakage-safe. It reuses the exact same
player-season construction, history features, and next-season targets used by
``prediction_report`` so the comparison is apples-to-apples, and every metric
comes from rolling-origin validation (train strictly on earlier seasons,
validate on the held-out season).

Baselines compared:
- ``season_mean``: predict 0 (the standardized group mean).
- ``persistence``: predict the player's current-season value score.
- ``shrunken_persistence``: predict ``r * current value score`` where the
  shrinkage ``r`` is fit by least squares on the training fold only.
- ``age_curve``: an OLS age curve (age, age^2) with position effects.

Models compared:
- ``random_forest``: the project's tuned next-season RF.
- ``gradient_boosting``: a HistGradientBoostingRegressor (usually the strongest
  off-the-shelf tabular model and already used in the fantasy track).

It also adds split-conformal prediction intervals, which give distribution-free,
calibrated coverage without assuming the residuals are Gaussian.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src.prediction_report import (
    ENHANCED_FEATURES,
    PREDICTION_INTERVAL_TARGET_COVERAGE,
    TUNED_RANDOM_FOREST_PARAMS,
    add_player_history_features,
    create_next_season_targets,
    create_player_season_value_scores,
    find_project_root,
    make_model_pipeline,
)
from sklearn.ensemble import RandomForestRegressor

TARGET = "next_value_score"
CURRENT_VALUE_COL = "value_score"
DEFAULT_VALIDATION_YEARS = [2020, 2021, 2022, 2023, 2024]

GRADIENT_BOOSTING_PARAMS = {
    "loss": "squared_error",
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 400,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Data preparation (reuses leakage-safe helpers from prediction_report)
# ---------------------------------------------------------------------------
def load_modeling_frame(project_root: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Build the modeling frame and feature list used for benchmarking."""
    if project_root is None:
        project_root = find_project_root()

    processed_dir = project_root / "data" / "processed"
    skill_seasons_path = processed_dir / "skill_player_seasons_2016_2025.csv"
    value_scores_path = processed_dir / "player_value_scores_2016_2025.csv"

    if skill_seasons_path.exists():
        player_rows = pd.read_csv(skill_seasons_path)
    else:
        player_rows = pd.read_csv(value_scores_path)

    player_season = create_player_season_value_scores(player_rows)
    player_season = add_player_history_features(player_season)
    player_season = create_next_season_targets(player_season)

    feature_cols = [col for col in ENHANCED_FEATURES if col in player_season.columns]
    modeling_df = player_season.dropna(subset=[TARGET]).copy()
    return modeling_df, feature_cols


# ---------------------------------------------------------------------------
# Baseline predictors. Each takes (train_df, valid_df) and returns predictions
# for valid_df. They may only use information available before the validation
# season, so they are fit on train_df alone.
# ---------------------------------------------------------------------------
def _predict_season_mean(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray:
    # The target is standardized within season-position groups, so its
    # historical mean is ~0. Predicting the train mean is the honest "no skill"
    # reference point.
    return np.full(len(valid_df), float(train_df[TARGET].mean()))


def _predict_persistence(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray:
    current = pd.to_numeric(valid_df[CURRENT_VALUE_COL], errors="coerce")
    return current.fillna(float(train_df[TARGET].mean())).to_numpy()


def _predict_shrunken_persistence(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray:
    # Fit y_next = a + r * y_current on the training fold only, then apply.
    train_x = pd.to_numeric(train_df[CURRENT_VALUE_COL], errors="coerce")
    train_y = pd.to_numeric(train_df[TARGET], errors="coerce")
    mask = train_x.notna() & train_y.notna()
    if mask.sum() < 5:
        return _predict_persistence(train_df, valid_df)

    reg = LinearRegression()
    reg.fit(train_x[mask].to_numpy().reshape(-1, 1), train_y[mask].to_numpy())
    valid_x = pd.to_numeric(valid_df[CURRENT_VALUE_COL], errors="coerce").fillna(
        float(train_x[mask].mean())
    )
    return reg.predict(valid_x.to_numpy().reshape(-1, 1))


def _age_curve_design(df: pd.DataFrame) -> pd.DataFrame:
    age = pd.to_numeric(df.get("age"), errors="coerce")
    design = pd.DataFrame(index=df.index)
    design["age"] = age
    design["age_sq"] = age ** 2
    positions = df["position"].astype(str)
    for pos in ["RB", "WR", "TE"]:  # QB is the reference category
        design[f"pos_{pos}"] = (positions == pos).astype(float)
    return design


def _predict_age_curve(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray:
    train_design = _age_curve_design(train_df)
    train_y = pd.to_numeric(train_df[TARGET], errors="coerce")
    median = train_design.median(numeric_only=True)
    train_design = train_design.fillna(median)
    mask = train_y.notna()
    if mask.sum() < 10:
        return _predict_season_mean(train_df, valid_df)

    reg = LinearRegression()
    reg.fit(train_design[mask.to_numpy()], train_y[mask])
    valid_design = _age_curve_design(valid_df).fillna(median)
    return reg.predict(valid_design)


BASELINES: dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = {
    "season_mean": _predict_season_mean,
    "persistence": _predict_persistence,
    "shrunken_persistence": _predict_shrunken_persistence,
    "age_curve": _predict_age_curve,
}


def _make_regressor(name: str):
    if name == "random_forest":
        return RandomForestRegressor(**TUNED_RANDOM_FOREST_PARAMS)
    if name == "gradient_boosting":
        return HistGradientBoostingRegressor(**GRADIENT_BOOSTING_PARAMS)
    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ["random_forest", "gradient_boosting"]


# ---------------------------------------------------------------------------
# Rolling-origin prediction collection
# ---------------------------------------------------------------------------
def collect_rolling_predictions(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    validation_years: list[int] | None = None,
) -> pd.DataFrame:
    """Return long-format out-of-sample predictions for every method and fold."""
    if validation_years is None:
        validation_years = DEFAULT_VALIDATION_YEARS

    records: list[pd.DataFrame] = []
    for year in validation_years:
        train_df = modeling_df[modeling_df["season"].lt(year)].copy()
        valid_df = modeling_df[modeling_df["season"].eq(year)].copy()
        if train_df.empty or valid_df.empty:
            continue

        base = valid_df[["player_id", "position", "season", TARGET]].copy()

        for name, fn in BASELINES.items():
            out = base.copy()
            out["method"] = name
            out["method_type"] = "baseline"
            out["prediction"] = np.asarray(fn(train_df, valid_df), dtype="float64")
            records.append(out)

        for name in MODEL_NAMES:
            pipeline = make_model_pipeline(feature_cols, _make_regressor(name))
            pipeline.fit(train_df[feature_cols], train_df[TARGET])
            out = base.copy()
            out["method"] = name
            out["method_type"] = "model"
            out["prediction"] = pipeline.predict(valid_df[feature_cols])
            records.append(out)

    if not records:
        return pd.DataFrame()

    preds = pd.concat(records, ignore_index=True)
    preds["residual"] = preds[TARGET] - preds["prediction"]
    preds["abs_residual"] = preds["residual"].abs()
    return preds


# ---------------------------------------------------------------------------
# Summaries and skill scores
# ---------------------------------------------------------------------------
def summarize_methods(predictions: pd.DataFrame) -> pd.DataFrame:
    """Overall pooled metrics per method, with skill scores vs key baselines."""
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (method, method_type), grp in predictions.groupby(["method", "method_type"]):
        rows.append(
            {
                "method": method,
                "method_type": method_type,
                "n": int(len(grp)),
                "rmse": _rmse(grp[TARGET], grp["prediction"]),
                "mae": _mae(grp[TARGET], grp["prediction"]),
                "r2": _r2(grp[TARGET], grp["prediction"]),
            }
        )
    summary = pd.DataFrame(rows)

    def _ref_rmse(name: str) -> float:
        match = summary.loc[summary["method"] == name, "rmse"]
        return float(match.iloc[0]) if len(match) else float("nan")

    rmse_persist = _ref_rmse("shrunken_persistence")
    rmse_age = _ref_rmse("age_curve")
    summary["skill_vs_shrunken_persistence"] = 1.0 - summary["rmse"] / rmse_persist
    summary["skill_vs_age_curve"] = 1.0 - summary["rmse"] / rmse_age
    return summary.sort_values("rmse").reset_index(drop=True)


def summarize_methods_by_position(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-position metrics and skill score vs shrunken persistence."""
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (position, method, method_type), grp in predictions.groupby(
        ["position", "method", "method_type"]
    ):
        rows.append(
            {
                "position": position,
                "method": method,
                "method_type": method_type,
                "n": int(len(grp)),
                "rmse": _rmse(grp[TARGET], grp["prediction"]),
                "mae": _mae(grp[TARGET], grp["prediction"]),
                "r2": _r2(grp[TARGET], grp["prediction"]),
            }
        )
    by_pos = pd.DataFrame(rows)

    ref = (
        by_pos[by_pos["method"] == "shrunken_persistence"][["position", "rmse"]]
        .rename(columns={"rmse": "ref_rmse"})
    )
    by_pos = by_pos.merge(ref, on="position", how="left")
    by_pos["skill_vs_shrunken_persistence"] = 1.0 - by_pos["rmse"] / by_pos["ref_rmse"]
    by_pos = by_pos.drop(columns=["ref_rmse"])
    return by_pos.sort_values(["position", "rmse"]).reset_index(drop=True)


def summarize_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-validation-season RMSE for every method (rolling-origin folds)."""
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (season, method), grp in predictions.groupby(["season", "method"]):
        rows.append(
            {
                "validation_season": int(season),
                "method": method,
                "n": int(len(grp)),
                "rmse": _rmse(grp[TARGET], grp["prediction"]),
                "mae": _mae(grp[TARGET], grp["prediction"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["validation_season", "rmse"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split-conformal prediction intervals
# ---------------------------------------------------------------------------
def conformal_interval_validation(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str = "gradient_boosting",
    validation_years: list[int] | None = None,
    target_coverage: float = PREDICTION_INTERVAL_TARGET_COVERAGE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split-conformal intervals validated with rolling origin.

    For each validation season, the training years are split into a proper
    training set and a calibration set (the most recent prior season). The
    absolute residuals on the calibration set define a distribution-free
    quantile that sets the interval half-width. This yields calibrated coverage
    without assuming Gaussian residuals.
    """
    if validation_years is None:
        validation_years = DEFAULT_VALIDATION_YEARS
    alpha = 1.0 - target_coverage

    fold_frames: list[pd.DataFrame] = []
    for year in validation_years:
        train_df = modeling_df[modeling_df["season"].lt(year)].copy()
        valid_df = modeling_df[modeling_df["season"].eq(year)].copy()
        if train_df.empty or valid_df.empty:
            continue

        calib_season = int(train_df["season"].max())
        proper_train = train_df[train_df["season"].lt(calib_season)]
        calib = train_df[train_df["season"].eq(calib_season)]
        # Fall back to a random calibration split if there is only one train year.
        if proper_train.empty or len(calib) < 20:
            calib = train_df.sample(frac=0.25, random_state=42)
            proper_train = train_df.drop(calib.index)
            if proper_train.empty:
                proper_train = train_df

        pipeline = make_model_pipeline(feature_cols, _make_regressor(model_name))
        pipeline.fit(proper_train[feature_cols], proper_train[TARGET])

        calib_resid = np.abs(calib[TARGET].to_numpy() - pipeline.predict(calib[feature_cols]))
        n_calib = len(calib_resid)
        # Finite-sample split-conformal quantile level.
        level = min(1.0, np.ceil((n_calib + 1) * (1.0 - alpha)) / n_calib)
        q = float(np.quantile(calib_resid, level, method="higher"))

        preds = pipeline.predict(valid_df[feature_cols])
        out = valid_df[["player_id", "position", "season", TARGET]].copy()
        out["prediction"] = preds
        out["interval_low"] = preds - q
        out["interval_high"] = preds + q
        out["interval_width"] = 2.0 * q
        out["covered"] = out[TARGET].between(out["interval_low"], out["interval_high"])
        fold_frames.append(out)

    if not fold_frames:
        return pd.DataFrame(), pd.DataFrame()

    predictions = pd.concat(fold_frames, ignore_index=True)

    def _coverage_table(grouped, segment_col_value):
        agg = grouped.agg(
            n=("covered", "size"),
            coverage=("covered", "mean"),
            mean_width=("interval_width", "mean"),
        ).reset_index()
        return agg

    overall = predictions.assign(_all="all").groupby("_all")
    overall_tbl = _coverage_table(overall, "all").rename(columns={"_all": "segment_value"})
    overall_tbl.insert(0, "segment", "overall")

    pos_tbl = _coverage_table(predictions.groupby("position"), "position").rename(
        columns={"position": "segment_value"}
    )
    pos_tbl.insert(0, "segment", "position")

    summary = pd.concat([overall_tbl, pos_tbl], ignore_index=True)
    summary["target_coverage"] = target_coverage
    summary["coverage_gap"] = summary["coverage"] - target_coverage
    summary["method"] = f"split_conformal_{model_name}"
    return predictions, summary


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:.1f}%"


def build_benchmark_report_markdown(
    method_summary: pd.DataFrame,
    by_position: pd.DataFrame,
    conformal_summary: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Model Benchmark: Baselines, Skill Score, and Calibrated Intervals")
    lines.append("")
    lines.append(
        "All numbers come from rolling-origin validation: each season is "
        "predicted using only earlier seasons. The target `next_value_score` is "
        "standardized within each season-position group, so its per-season "
        "standard deviation is approximately 1.0. That means predicting the "
        "group mean already yields an RMSE near 1.0, and **RMSE alone overstates "
        "model quality**. The honest measure is the *skill score*: the "
        "percentage RMSE reduction versus a strong naive baseline."
    )
    lines.append("")
    lines.append("## Overall results")
    lines.append("")
    lines.append(
        "| Method | Type | RMSE | MAE | R² | Skill vs shrunken persistence | Skill vs age curve |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in method_summary.iterrows():
        lines.append(
            f"| {r['method']} | {r['method_type']} | {r['rmse']:.3f} | {r['mae']:.3f} | "
            f"{r['r2']:.3f} | {_fmt_pct(r['skill_vs_shrunken_persistence'])} | "
            f"{_fmt_pct(r['skill_vs_age_curve'])} |"
        )
    lines.append("")

    best = method_summary.iloc[0]
    best_model = method_summary[method_summary["method_type"] == "model"].iloc[0]
    lines.append(
        f"The lowest-RMSE method overall is **{best['method']}** "
        f"(RMSE {best['rmse']:.3f}). The best learned model is "
        f"**{best_model['method']}**, which reduces RMSE by "
        f"{_fmt_pct(best_model['skill_vs_shrunken_persistence'])} versus shrunken "
        f"persistence. A small or negative skill score is itself an important, "
        f"honest finding: it means a one-line baseline is hard to beat for this "
        f"target, and the model is best used for tiering rather than precise "
        f"ranking."
    )
    lines.append("")
    lines.append("## Skill score by position (vs shrunken persistence)")
    lines.append("")
    lines.append("| Position | Method | RMSE | R² | Skill vs shrunken persistence |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    models_only = by_position[by_position["method_type"] == "model"]
    for _, r in models_only.iterrows():
        lines.append(
            f"| {r['position']} | {r['method']} | {r['rmse']:.3f} | {r['r2']:.3f} | "
            f"{_fmt_pct(r['skill_vs_shrunken_persistence'])} |"
        )
    lines.append("")
    if not conformal_summary.empty:
        overall_cov = conformal_summary[conformal_summary["segment"] == "overall"].iloc[0]
        lines.append("## Split-conformal prediction intervals")
        lines.append("")
        lines.append(
            f"Split-conformal intervals targeting {_fmt_pct(overall_cov['target_coverage'])} "
            f"coverage achieved {_fmt_pct(overall_cov['coverage'])} empirical coverage "
            f"overall (mean width {overall_cov['mean_width']:.2f}). Unlike the "
            f"Gaussian-style bands in the main report, conformal intervals are "
            f"distribution-free and calibrated by construction."
        )
        lines.append("")
        lines.append("| Segment | Coverage | Target | Gap | Mean width |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for _, r in conformal_summary.iterrows():
            lines.append(
                f"| {r['segment_value']} | {_fmt_pct(r['coverage'])} | "
                f"{_fmt_pct(r['target_coverage'])} | {_fmt_pct(r['coverage_gap'])} | "
                f"{r['mean_width']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_model_benchmark_outputs(
    project_root: Path | None = None,
    save_outputs: bool = True,
    validation_years: list[int] | None = None,
) -> dict[str, Any]:
    """Build all benchmark tables and the markdown report."""
    if project_root is None:
        project_root = find_project_root()

    modeling_df, feature_cols = load_modeling_frame(project_root)
    predictions = collect_rolling_predictions(modeling_df, feature_cols, validation_years)
    method_summary = summarize_methods(predictions)
    by_position = summarize_methods_by_position(predictions)
    by_fold = summarize_by_fold(predictions)
    conformal_predictions, conformal_summary = conformal_interval_validation(
        modeling_df, feature_cols, validation_years=validation_years
    )
    report_md = build_benchmark_report_markdown(method_summary, by_position, conformal_summary)

    outputs = {
        "predictions": predictions,
        "method_summary": method_summary,
        "by_position": by_position,
        "by_fold": by_fold,
        "conformal_predictions": conformal_predictions,
        "conformal_summary": conformal_summary,
        "report_markdown": report_md,
        "feature_cols": feature_cols,
    }

    if save_outputs:
        tables_dir = project_root / "outputs" / "tables"
        report_dir = project_root / "report"
        tables_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        method_summary.to_csv(tables_dir / "model_benchmark_summary.csv", index=False)
        by_position.to_csv(tables_dir / "model_benchmark_by_position.csv", index=False)
        by_fold.to_csv(tables_dir / "model_benchmark_by_fold.csv", index=False)
        conformal_summary.to_csv(
            tables_dir / "model_benchmark_conformal_coverage.csv", index=False
        )
        (report_dir / "model_benchmark.md").write_text(report_md)

    return outputs


if __name__ == "__main__":
    result = build_model_benchmark_outputs()
    print(result["method_summary"].to_string(index=False))
