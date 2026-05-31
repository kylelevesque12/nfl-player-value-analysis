"""Fantasy football projection outputs for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.load_data import ensure_project_dirs, find_project_root, load_csv
from src.models import make_model_pipeline
from src.prediction_report import create_player_season_value_scores


FANTASY_RANDOM_FOREST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 8,
    "max_features": 0.65,
    "min_samples_leaf": 10,
    "random_state": 42,
    "n_jobs": -1,
}

FANTASY_HGB_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.04,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 0.10,
    "random_state": 42,
}

FANTASY_FEATURES = [
    "position",
    "age",
    "years_exp",
    "draft_number",
    "games_played",
    "fantasy_points_ppr",
    "fantasy_points_ppr_per_game",
    "fantasy_points",
    "targets",
    "targets_per_game",
    "receptions",
    "receptions_per_game",
    "carries",
    "carries_per_game",
    "receiving_yards",
    "rushing_yards",
    "receiving_tds",
    "rushing_tds",
    "scrimmage_touches_per_game",
    "scrimmage_yards_per_game",
    "yards_per_scrimmage_touch",
    "qb_yards_per_game",
    "qb_yards_per_play",
    "yards_per_game",
    "tds_per_game",
    "value_score",
    "value_epa_total",
    "value_epa_per_game",
    "prior_qualifying_seasons",
    "fantasy_points_ppr_prev",
    "fantasy_points_ppr_last2_avg",
    "fantasy_points_ppr_last3_avg",
    "fantasy_points_ppr_per_game_prev",
    "fantasy_points_ppr_per_game_last2_avg",
    "games_played_prev",
    "games_played_last2_avg",
    "targets_prev",
    "targets_last2_avg",
    "receptions_prev",
    "receptions_last2_avg",
    "carries_prev",
    "carries_last2_avg",
    "value_score_prev",
    "value_score_last2_avg",
]

FANTASY_MODEL_ORDER = [
    "current_year_baseline",
    "ridge_total",
    "elastic_net_total",
    "random_forest_total",
    "hist_gradient_boosting_total",
    "two_stage_hist_gradient_boosting",
]

MODEL_LABELS = {
    "current_year_baseline": "2025 PPR Baseline",
    "ridge_total": "Ridge Total-PPR Model",
    "elastic_net_total": "Elastic Net Total-PPR Model",
    "random_forest_total": "Random Forest Total-PPR Model",
    "hist_gradient_boosting_total": "Histogram Gradient Boosting Total-PPR Model",
    "two_stage_hist_gradient_boosting": "Two-Stage HGB: Games x PPR/Game",
}

MODEL_TYPES = {
    "current_year_baseline": "baseline",
    "ridge_total": "direct_total_points",
    "elastic_net_total": "direct_total_points",
    "random_forest_total": "direct_total_points",
    "hist_gradient_boosting_total": "direct_total_points",
    "two_stage_hist_gradient_boosting": "two_stage_games_x_rate",
}

CSV_FLOAT_FORMAT = "%.12g"
PREDICTION_INTERVAL_MULTIPLIER = 1.28


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _series_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype="float64")


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def _make_direct_model(model_name: str, feature_cols: list[str]):
    models = {
        "ridge_total": Ridge(alpha=25.0),
        "elastic_net_total": ElasticNet(
            alpha=0.02,
            l1_ratio=0.20,
            max_iter=20000,
            random_state=42,
        ),
        "random_forest_total": RandomForestRegressor(**FANTASY_RANDOM_FOREST_PARAMS),
        "hist_gradient_boosting_total": HistGradientBoostingRegressor(**FANTASY_HGB_PARAMS),
    }
    if model_name not in models:
        raise ValueError("Unknown direct fantasy model: " + model_name)
    return make_model_pipeline(feature_cols, models[model_name])


def _make_two_stage_model(feature_cols: list[str]):
    return make_model_pipeline(
        feature_cols,
        HistGradientBoostingRegressor(**FANTASY_HGB_PARAMS),
    )


def _evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    metrics["r2"] = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    return metrics


def _spearman_corr(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
    frame = pd.DataFrame({"actual": y_true, "prediction": y_pred}).dropna()
    if len(frame) < 2 or frame["actual"].nunique() < 2 or frame["prediction"].nunique() < 2:
        return np.nan
    return float(frame["actual"].corr(frame["prediction"], method="spearman"))


def _top_threshold(position: str) -> int:
    if position in {"QB", "TE"}:
        return 12
    if position in {"RB", "WR"}:
        return 24
    return 12


def _top_rank_hit_rate(group: pd.DataFrame) -> float:
    rates = []
    for (_, position), position_group in group.groupby(["valid_year", "position"]):
        threshold = min(_top_threshold(str(position)), len(position_group))
        if threshold <= 0:
            continue
        actual_top = set(
            position_group.nlargest(threshold, "next_fantasy_points_ppr")["player_id"]
        )
        predicted_top = set(
            position_group.nlargest(threshold, "prediction")["player_id"]
        )
        rates.append(len(actual_top & predicted_top) / threshold)
    return float(np.mean(rates)) if rates else np.nan


def _assign_fantasy_tier(position_percentile: float) -> str:
    if position_percentile >= 0.90:
        return "Elite Fantasy Profile"
    if position_percentile >= 0.75:
        return "Strong Starter"
    if position_percentile >= 0.50:
        return "Starter/Flex"
    if position_percentile >= 0.25:
        return "Depth/Volatile"
    return "Low Projection"


def _assign_confidence_level(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def _projection_change_label(change: float) -> str:
    if pd.isna(change):
        return "No 2025 comparison"
    if change >= 30:
        return "Projected increase"
    if change <= -30:
        return "Projected regression"
    return "Similar to 2025"


def _usage_profile(row: pd.Series) -> str:
    position = row.get("position")
    targets = row.get("targets_2025", 0)
    receptions = row.get("receptions_2025", 0)
    carries = row.get("carries_2025", 0)

    if position == "QB":
        if carries >= 75:
            return "QB with meaningful rushing volume"
        return "QB production profile"

    if position == "RB":
        if carries >= 150 and receptions >= 40:
            return "Three-down RB usage"
        if carries >= 150:
            return "Rush-volume RB"
        if receptions >= 40:
            return "Receiving-heavy RB"
        return "Limited RB usage"

    if position == "WR":
        if targets >= 120:
            return "Alpha WR target volume"
        if targets >= 80:
            return "Regular WR target volume"
        if receptions >= 45:
            return "Moderate WR receiving role"
        return "Limited WR volume"

    if position == "TE":
        if targets >= 90:
            return "High-volume receiving TE"
        if targets >= 55:
            return "Regular receiving TE"
        return "Limited TE receiving volume"

    return "Usage profile unavailable"


def _format_fantasy_note(row: pd.Series) -> str:
    notes = []
    if row["games_played_2025"] < 8:
        notes.append("small 2025 sample")
    if row["confidence_level"] == "High":
        notes.append("strong sample and tighter model range")
    if row["fantasy_points_ppr_2025"] >= 200:
        notes.append("high 2025 fantasy production")
    elif row["fantasy_points_ppr_2025"] < 75:
        notes.append("low 2025 fantasy baseline")
    if row["prediction_uncertainty"] >= row["high_uncertainty_cutoff"]:
        notes.append("wide projection range")
    return "; ".join(notes) if notes else "balanced fantasy profile"


def _format_fantasy_explanation(row: pd.Series) -> str:
    change = row.get("projection_change_from_2025", np.nan)
    projected = row.get("predicted_2026_fantasy_points_ppr", np.nan)
    prior = row.get("fantasy_points_ppr_2025", np.nan)
    interval_low = row.get("prediction_interval_low", np.nan)
    interval_high = row.get("prediction_interval_high", np.nan)
    predicted_games = row.get("predicted_2026_games_played", np.nan)
    predicted_ppg = row.get("predicted_2026_ppr_per_game", np.nan)

    if pd.isna(projected) or pd.isna(prior):
        opening = "Projection is based on the player's available 2025 profile."
    else:
        opening = (
            f"Projects {projected:.1f} PPR points vs {prior:.1f} in 2025 "
            f"({change:+.1f})."
        )

    details = [
        opening,
        "Usage: " + str(row.get("usage_profile", "profile unavailable")) + ".",
        "Tier: " + str(row.get("fantasy_projection_tier", "unavailable")) + ".",
        "Model: " + str(row.get("selected_model_label", "unavailable")) + ".",
    ]
    if pd.notna(predicted_games) and pd.notna(predicted_ppg):
        details.append(
            f"Two-stage context: {predicted_games:.1f} projected games at "
            f"{predicted_ppg:.1f} PPR/game."
        )
    if pd.notna(interval_low) and pd.notna(interval_high):
        details.append(f"Reasonable model range: {interval_low:.0f}-{interval_high:.0f} PPR.")

    return " ".join(details)


def add_fantasy_history_features(player_season: pd.DataFrame) -> pd.DataFrame:
    """Add fantasy-specific rate, lag, and rolling features."""
    featured = player_season.sort_values(["player_id", "season"]).copy()
    games = pd.to_numeric(featured["games_played"], errors="coerce")
    featured["fantasy_points_ppr_per_game"] = _safe_divide(
        pd.to_numeric(featured["fantasy_points_ppr"], errors="coerce"),
        games,
    )
    featured["targets_per_game"] = _safe_divide(_series_or_zero(featured, "targets"), games)
    featured["receptions_per_game"] = _safe_divide(_series_or_zero(featured, "receptions"), games)
    featured["carries_per_game"] = _safe_divide(_series_or_zero(featured, "carries"), games)

    grouped = featured.groupby("player_id", group_keys=False)
    featured["prior_qualifying_seasons"] = grouped.cumcount()

    history_cols = [
        "fantasy_points_ppr",
        "fantasy_points_ppr_per_game",
        "fantasy_points",
        "games_played",
        "targets",
        "receptions",
        "carries",
        "value_score",
    ]
    for col in _available(featured, history_cols):
        featured[f"{col}_prev"] = grouped[col].shift(1)
        featured[f"{col}_last2_avg"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(2, min_periods=1).mean()
        )
        featured[f"{col}_last3_avg"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

    return featured


def create_fantasy_modeling_frame(skill_seasons: pd.DataFrame) -> pd.DataFrame:
    """Create the player-season frame used for fantasy projection modeling."""
    player_season = create_player_season_value_scores(skill_seasons)
    player_season = add_fantasy_history_features(player_season)
    player_season = player_season.sort_values(["player_id", "season"]).copy()
    grouped = player_season.groupby("player_id")

    player_season["next_season"] = grouped["season"].shift(-1)
    for col in [
        "fantasy_points_ppr",
        "fantasy_points_ppr_per_game",
        "games_played",
    ]:
        player_season["next_" + col] = grouped[col].shift(-1)

    has_next_season = player_season["next_season"].eq(player_season["season"] + 1)
    player_season["next_season_fantasy_qualifier"] = has_next_season.astype(int)

    for col in [
        "next_fantasy_points_ppr",
        "next_fantasy_points_ppr_per_game",
        "next_games_played",
    ]:
        player_season.loc[~has_next_season, col] = 0

    return player_season


def _predict_direct_model(
    model_name: str,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model = _make_direct_model(model_name, feature_cols)
    model.fit(train_df[feature_cols], train_df[target_col])
    predictions = model.predict(predict_df[feature_cols]).clip(min=0)
    return predictions, {}


def _predict_two_stage_model(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    games_model = _make_two_stage_model(feature_cols)
    games_model.fit(train_df[feature_cols], train_df["next_games_played"])
    predicted_games = games_model.predict(predict_df[feature_cols]).clip(min=0, max=17)

    ppg_train = train_df[
        train_df["next_games_played"].gt(0)
        & train_df["next_fantasy_points_ppr_per_game"].notna()
    ].copy()
    if ppg_train.empty:
        predicted_ppg = np.repeat(0.0, len(predict_df))
    else:
        ppg_model = _make_two_stage_model(feature_cols)
        ppg_model.fit(ppg_train[feature_cols], ppg_train["next_fantasy_points_ppr_per_game"])
        predicted_ppg = ppg_model.predict(predict_df[feature_cols]).clip(min=0, max=40)

    predictions = predicted_games * predicted_ppg
    return predictions, {
        "predicted_games_played": predicted_games,
        "predicted_ppr_per_game": predicted_ppg,
    }


def _predict_fantasy_model(
    model_name: str,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "next_fantasy_points_ppr",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if model_name == "current_year_baseline":
        return _series_or_zero(predict_df, "fantasy_points_ppr").to_numpy(), {}
    if model_name == "two_stage_hist_gradient_boosting":
        return _predict_two_stage_model(train_df, predict_df, feature_cols)
    return _predict_direct_model(model_name, train_df, predict_df, feature_cols, target_col)


def rolling_fantasy_validation(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "next_fantasy_points_ppr",
    validation_years: list[int] | None = None,
) -> pd.DataFrame:
    """Run rolling validation for all fantasy projection candidates."""
    if validation_years is None:
        validation_years = [2020, 2021, 2022, 2023, 2024]

    records: list[pd.DataFrame] = []
    for valid_year in validation_years:
        train_df = modeling_df[modeling_df["season"].lt(valid_year)].dropna(subset=[target_col]).copy()
        valid_df = modeling_df[modeling_df["season"].eq(valid_year)].dropna(subset=[target_col]).copy()
        if train_df.empty or valid_df.empty:
            continue

        for model_name in FANTASY_MODEL_ORDER:
            predictions, extras = _predict_fantasy_model(
                model_name,
                train_df,
                valid_df,
                feature_cols,
                target_col,
            )

            fold_records = valid_df[
                [
                    "season",
                    "player_id",
                    "player_display_name",
                    "position",
                    "primary_team",
                    "fantasy_points_ppr",
                    "fantasy_points_ppr_per_game",
                    "games_played",
                    "next_games_played",
                    "next_fantasy_points_ppr_per_game",
                    target_col,
                ]
            ].copy()
            fold_records["model_name"] = model_name
            fold_records["model_label"] = MODEL_LABELS[model_name]
            fold_records["model_type"] = MODEL_TYPES[model_name]
            fold_records["prediction"] = predictions
            fold_records["residual"] = fold_records[target_col] - fold_records["prediction"]
            fold_records["abs_residual"] = fold_records["residual"].abs()
            fold_records["valid_year"] = valid_year
            fold_records["target_year"] = valid_year + 1
            fold_records["predicted_games_played"] = extras.get(
                "predicted_games_played",
                np.repeat(np.nan, len(fold_records)),
            )
            fold_records["predicted_ppr_per_game"] = extras.get(
                "predicted_ppr_per_game",
                np.repeat(np.nan, len(fold_records)),
            )
            records.append(fold_records)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _summarize_prediction_group(group: pd.DataFrame) -> dict[str, float]:
    metrics = _evaluate_regression(
        group["next_fantasy_points_ppr"],
        group["prediction"].to_numpy(),
    )
    return {
        "validation_rows": int(len(group)),
        "mean_actual_next_fantasy_points_ppr": float(group["next_fantasy_points_ppr"].mean()),
        "mean_predicted_next_fantasy_points_ppr": float(group["prediction"].mean()),
        "bias": float(group["residual"].mean()),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "spearman_rank_corr": _spearman_corr(
            group["next_fantasy_points_ppr"],
            group["prediction"],
        ),
        "top_rank_hit_rate": _top_rank_hit_rate(group),
    }


def summarize_fantasy_model_comparison(validation_predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize each fantasy model overall and by position."""
    if validation_predictions.empty:
        return pd.DataFrame()

    rows = []
    for model_name, group in validation_predictions.groupby("model_name"):
        summary = _summarize_prediction_group(group)
        summary.update(
            {
                "segment": "overall",
                "segment_value": "all",
                "model_name": model_name,
                "model_label": MODEL_LABELS[model_name],
                "model_type": MODEL_TYPES[model_name],
            }
        )
        rows.append(summary)

    for (model_name, position), group in validation_predictions.groupby(["model_name", "position"]):
        summary = _summarize_prediction_group(group)
        summary.update(
            {
                "segment": "position",
                "segment_value": position,
                "model_name": model_name,
                "model_label": MODEL_LABELS[model_name],
                "model_type": MODEL_TYPES[model_name],
            }
        )
        rows.append(summary)

    summary_df = pd.DataFrame(rows)
    sort_cols = ["segment", "segment_value", "rmse", "mae"]
    return summary_df.sort_values(sort_cols).reset_index(drop=True)


def select_final_fantasy_model(model_comparison: pd.DataFrame) -> tuple[str, str]:
    """Select the final projection model from rolling validation results."""
    overall = model_comparison[model_comparison["segment"].eq("overall")].copy()
    overall = overall.sort_values(["rmse", "mae"]).reset_index(drop=True)
    if overall.empty:
        return "random_forest_total", "fallback_random_forest"

    best_model = str(overall.iloc[0]["model_name"])
    return best_model, "lowest_overall_rolling_rmse"


def summarize_fantasy_validation(
    validation_predictions: pd.DataFrame,
    selected_model: str,
) -> pd.DataFrame:
    """Summarize the selected fantasy model overall and by position."""
    selected = validation_predictions[
        validation_predictions["model_name"].eq(selected_model)
    ].copy()
    if selected.empty:
        return pd.DataFrame()

    rows = []
    overall = selected.assign(overall="all")
    for segment_value, group in overall.groupby("overall"):
        summary = _summarize_prediction_group(group)
        summary.update(
            {
                "segment": "overall",
                "segment_value": segment_value,
                "model_name": selected_model,
                "model_label": MODEL_LABELS[selected_model],
                "model_type": MODEL_TYPES[selected_model],
            }
        )
        rows.append(summary)

    for position, group in selected.groupby("position"):
        summary = _summarize_prediction_group(group)
        summary.update(
            {
                "segment": "position",
                "segment_value": position,
                "model_name": selected_model,
                "model_label": MODEL_LABELS[selected_model],
                "model_type": MODEL_TYPES[selected_model],
            }
        )
        rows.append(summary)

    return pd.DataFrame(rows).reset_index(drop=True)


def build_fantasy_projection_outputs(
    project_root: str | Path | None = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Build 2026 fantasy-football projection outputs for the app."""
    root = find_project_root() if project_root is None else Path(project_root).resolve()
    dirs = ensure_project_dirs(root)
    output_dir = dirs["tables"]

    skill_seasons = load_csv(
        "data/processed/skill_player_seasons_2016_2025.csv",
        root,
        low_memory=False,
    )
    player_season = create_fantasy_modeling_frame(skill_seasons)
    feature_cols = _available(player_season, FANTASY_FEATURES)
    target_col = "next_fantasy_points_ppr"

    modeling_df = player_season[
        player_season["season"].between(2016, 2024)
    ].dropna(subset=[target_col]).copy()
    prediction_input = player_season[player_season["season"].eq(2025)].copy()

    validation_predictions = rolling_fantasy_validation(
        player_season,
        feature_cols,
        target_col,
    )
    model_comparison = summarize_fantasy_model_comparison(validation_predictions)
    selected_model, selection_reason = select_final_fantasy_model(model_comparison)
    validation_summary = summarize_fantasy_validation(validation_predictions, selected_model)
    selected_validation = validation_predictions[
        validation_predictions["model_name"].eq(selected_model)
    ].copy()
    overall_validation = validation_summary[validation_summary["segment"].eq("overall")]
    residual_rmse = (
        float(overall_validation["rmse"].iloc[0])
        if not overall_validation.empty
        else float(
            np.sqrt(
                mean_squared_error(
                    modeling_df[target_col],
                    _predict_fantasy_model(
                        selected_model,
                        modeling_df,
                        modeling_df,
                        feature_cols,
                        target_col,
                    )[0],
                )
            )
        )
    )

    predicted = prediction_input.copy()
    selected_predictions, selected_extras = _predict_fantasy_model(
        selected_model,
        modeling_df,
        predicted,
        feature_cols,
        target_col,
    )
    _, two_stage_extras = _predict_fantasy_model(
        "two_stage_hist_gradient_boosting",
        modeling_df,
        predicted,
        feature_cols,
        target_col,
    )
    predicted["predicted_2026_fantasy_points_ppr"] = selected_predictions
    predicted["predicted_2026_games_played"] = selected_extras.get(
        "predicted_games_played",
        two_stage_extras.get("predicted_games_played", np.repeat(np.nan, len(predicted))),
    )
    predicted["predicted_2026_ppr_per_game"] = selected_extras.get(
        "predicted_ppr_per_game",
        two_stage_extras.get("predicted_ppr_per_game", np.repeat(np.nan, len(predicted))),
    )

    candidate_predictions = {}
    for model_name in FANTASY_MODEL_ORDER:
        if model_name == "current_year_baseline":
            continue
        candidate_predictions[model_name], _ = _predict_fantasy_model(
            model_name,
            modeling_df,
            predicted,
            feature_cols,
            target_col,
        )
    candidate_prediction_frame = pd.DataFrame(candidate_predictions)
    predicted["model_disagreement"] = (
        candidate_prediction_frame.std(axis=1).fillna(0).to_numpy()
    )
    predicted["prediction_uncertainty"] = np.sqrt(
        np.square(residual_rmse) + np.square(predicted["model_disagreement"])
    )
    predicted["prediction_interval_low"] = (
        predicted["predicted_2026_fantasy_points_ppr"]
        - PREDICTION_INTERVAL_MULTIPLIER * predicted["prediction_uncertainty"]
    ).clip(lower=0)
    predicted["prediction_interval_high"] = (
        predicted["predicted_2026_fantasy_points_ppr"]
        + PREDICTION_INTERVAL_MULTIPLIER * predicted["prediction_uncertainty"]
    )

    predicted["fantasy_overall_rank"] = predicted["predicted_2026_fantasy_points_ppr"].rank(
        ascending=False,
        method="min",
    )
    predicted["fantasy_position_rank"] = predicted.groupby("position")[
        "predicted_2026_fantasy_points_ppr"
    ].rank(ascending=False, method="min")
    predicted["predicted_2026_overall_percentile"] = predicted[
        "predicted_2026_fantasy_points_ppr"
    ].rank(pct=True)
    predicted["predicted_2026_position_percentile"] = predicted.groupby("position")[
        "predicted_2026_fantasy_points_ppr"
    ].rank(pct=True)
    predicted["fantasy_projection_tier"] = predicted[
        "predicted_2026_position_percentile"
    ].apply(_assign_fantasy_tier)

    uncertainty_pct = predicted["prediction_uncertainty"].rank(pct=True)
    sample_score = predicted["games_played"].clip(lower=0, upper=17) / 17
    history_score = predicted["prior_qualifying_seasons"].clip(lower=0, upper=4) / 4
    predicted["confidence_score"] = (
        (1 - uncertainty_pct) * 45
        + sample_score * 30
        + history_score * 25
    ).round(1)
    predicted["confidence_level"] = predicted["confidence_score"].apply(_assign_confidence_level)
    predicted["high_uncertainty_cutoff"] = predicted["prediction_uncertainty"].quantile(0.67)
    predicted["selected_model"] = selected_model
    predicted["selected_model_label"] = MODEL_LABELS[selected_model]
    predicted["model_selection_reason"] = selection_reason

    report_df = predicted.rename(
        columns={
            "primary_team": "primary_team_2025",
            "teams": "teams_2025",
            "age": "age_2025",
            "years_exp": "years_exp_2025",
            "games_played": "games_played_2025",
            "fantasy_points_ppr": "fantasy_points_ppr_2025",
            "fantasy_points_ppr_per_game": "fantasy_points_ppr_per_game_2025",
            "targets": "targets_2025",
            "receptions": "receptions_2025",
            "carries": "carries_2025",
        }
    ).copy()
    report_df["projection_change_from_2025"] = (
        report_df["predicted_2026_fantasy_points_ppr"]
        - report_df["fantasy_points_ppr_2025"]
    )
    report_df["projection_change_label"] = report_df[
        "projection_change_from_2025"
    ].apply(_projection_change_label)
    report_df["usage_profile"] = report_df.apply(_usage_profile, axis=1)
    report_df["fantasy_note"] = report_df.apply(_format_fantasy_note, axis=1)
    report_df["fantasy_explanation"] = report_df.apply(_format_fantasy_explanation, axis=1)

    report_cols = [
        "player_id",
        "player_display_name",
        "position",
        "primary_team_2025",
        "teams_2025",
        "games_played_2025",
        "age_2025",
        "years_exp_2025",
        "draft_number",
        "fantasy_points_ppr_2025",
        "fantasy_points_ppr_per_game_2025",
        "targets_2025",
        "receptions_2025",
        "carries_2025",
        "value_score",
        "predicted_2026_fantasy_points_ppr",
        "predicted_2026_games_played",
        "predicted_2026_ppr_per_game",
        "projection_change_from_2025",
        "projection_change_label",
        "prediction_interval_low",
        "prediction_interval_high",
        "prediction_uncertainty",
        "model_disagreement",
        "fantasy_overall_rank",
        "fantasy_position_rank",
        "predicted_2026_overall_percentile",
        "predicted_2026_position_percentile",
        "fantasy_projection_tier",
        "usage_profile",
        "confidence_score",
        "confidence_level",
        "selected_model",
        "selected_model_label",
        "model_selection_reason",
        "fantasy_note",
        "fantasy_explanation",
    ]
    report_df = (
        report_df[_available(report_df, report_cols)]
        .sort_values("predicted_2026_fantasy_points_ppr", ascending=False)
        .reset_index(drop=True)
    )

    summary_text = (
        "# Fantasy Football Projection Summary\n\n"
        "This report adds a fantasy-football view to the project by projecting "
        "2026 season-long PPR fantasy points from 2025 player-season production, "
        "recent history, usage, and the existing EPA-based value features.\n\n"
        "The upgraded version compares several model families: a 2025 baseline, "
        "Ridge, Elastic Net, Random Forest, Histogram Gradient Boosting, and a "
        "two-stage model that predicts games played and PPR per game separately.\n\n"
        "The final projection model is selected by the lowest rolling-validation "
        "RMSE. The two-stage model is still reported as an interpretation aid "
        "because it translates fantasy value into projected games and scoring "
        "rate.\n\n"
        "The model is still not a finished fantasy ranking system. It does not "
        "yet include rookies, depth-chart changes, injuries, coaching changes, "
        "betting markets, or manual playing-time projections.\n\n"
        "To make the output easier to use, each player row includes a projection "
        "change label, a usage profile, and a plain-English fantasy explanation.\n\n"
        f"Projected players: {len(report_df):,}\n\n"
        f"Rolling validation rows: {len(selected_validation):,}\n\n"
        f"Selected model: {MODEL_LABELS[selected_model]}\n\n"
        f"Selection reason: {selection_reason}\n\n"
    )
    if not overall_validation.empty:
        row = overall_validation.iloc[0]
        summary_text += (
            f"Overall rolling MAE: {row['mae']:.2f} PPR points\n\n"
            f"Overall rolling RMSE: {row['rmse']:.2f} PPR points\n\n"
            f"Overall Spearman rank correlation: {row['spearman_rank_corr']:.3f}\n\n"
            f"Top-rank hit rate: {row['top_rank_hit_rate']:.3f}\n"
        )

    outputs = {
        "fantasy_predictions": report_df,
        "fantasy_validation_predictions": selected_validation,
        "fantasy_validation_summary": validation_summary,
        "fantasy_model_comparison": model_comparison,
        "feature_cols": feature_cols,
        "selected_model": selected_model,
        "selection_reason": selection_reason,
        "model_params": {
            "random_forest": FANTASY_RANDOM_FOREST_PARAMS,
            "hist_gradient_boosting": FANTASY_HGB_PARAMS,
        },
    }

    if save_outputs:
        report_df.to_csv(
            output_dir / "2026_fantasy_football_projections.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        selected_validation.to_csv(
            output_dir / "fantasy_projection_validation_predictions.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        validation_summary.to_csv(
            output_dir / "fantasy_projection_validation_by_position.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        model_comparison.to_csv(
            output_dir / "fantasy_model_comparison.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        report_path = root / "report" / "fantasy_football_projection_summary.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(summary_text)

    return outputs
