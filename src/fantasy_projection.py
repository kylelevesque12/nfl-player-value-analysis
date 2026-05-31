"""Fantasy football projection outputs for the dashboard draft."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
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
    "receptions",
    "carries",
    "receiving_yards",
    "rushing_yards",
    "receiving_tds",
    "rushing_tds",
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

CSV_FLOAT_FORMAT = "%.12g"
PREDICTION_INTERVAL_MULTIPLIER = 1.28


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def _make_fantasy_model(feature_cols: list[str]):
    model = RandomForestRegressor(**FANTASY_RANDOM_FOREST_PARAMS)
    return make_model_pipeline(feature_cols, model)


def _evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    metrics["r2"] = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    return metrics


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


def _format_fantasy_explanation(row: pd.Series) -> str:
    change = row.get("projection_change_from_2025", np.nan)
    projected = row.get("predicted_2026_fantasy_points_ppr", np.nan)
    prior = row.get("fantasy_points_ppr_2025", np.nan)
    interval_low = row.get("prediction_interval_low", np.nan)
    interval_high = row.get("prediction_interval_high", np.nan)

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
        "Confidence: " + str(row.get("confidence_level", "unavailable")) + ".",
    ]
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
    featured["targets_per_game"] = _safe_divide(
        pd.to_numeric(featured.get("targets", 0), errors="coerce"),
        games,
    )
    featured["receptions_per_game"] = _safe_divide(
        pd.to_numeric(featured.get("receptions", 0), errors="coerce"),
        games,
    )
    featured["carries_per_game"] = _safe_divide(
        pd.to_numeric(featured.get("carries", 0), errors="coerce"),
        games,
    )

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


def rolling_fantasy_validation(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "next_fantasy_points_ppr",
    validation_years: list[int] | None = None,
) -> pd.DataFrame:
    """Run rolling validation for next-season PPR fantasy projections."""
    if validation_years is None:
        validation_years = [2020, 2021, 2022, 2023, 2024]

    records: list[pd.DataFrame] = []
    for valid_year in validation_years:
        train_df = modeling_df[modeling_df["season"].lt(valid_year)].dropna(subset=[target_col]).copy()
        valid_df = modeling_df[modeling_df["season"].eq(valid_year)].dropna(subset=[target_col]).copy()
        if train_df.empty or valid_df.empty:
            continue

        pipeline = _make_fantasy_model(feature_cols)
        pipeline.fit(train_df[feature_cols], train_df[target_col])
        predictions = pipeline.predict(valid_df[feature_cols]).clip(min=0)

        fold_records = valid_df[
            [
                "season",
                "player_id",
                "player_display_name",
                "position",
                "primary_team",
                "fantasy_points_ppr",
                target_col,
            ]
        ].copy()
        fold_records["prediction"] = predictions
        fold_records["residual"] = fold_records[target_col] - fold_records["prediction"]
        fold_records["abs_residual"] = fold_records["residual"].abs()
        fold_records["valid_year"] = valid_year
        fold_records["target_year"] = valid_year + 1
        fold_records["baseline_current_year_prediction"] = fold_records["fantasy_points_ppr"]
        fold_records["baseline_abs_residual"] = (
            fold_records[target_col] - fold_records["baseline_current_year_prediction"]
        ).abs()
        records.append(fold_records)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def summarize_fantasy_validation(validation_predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize fantasy projection validation overall and by position."""
    if validation_predictions.empty:
        return pd.DataFrame()

    def summarize(grouped: pd.core.groupby.DataFrameGroupBy, segment: str) -> pd.DataFrame:
        rows = []
        for segment_value, group in grouped:
            metrics = _evaluate_regression(
                group["next_fantasy_points_ppr"],
                group["prediction"].to_numpy(),
            )
            baseline_rmse = float(
                np.sqrt(
                    mean_squared_error(
                        group["next_fantasy_points_ppr"],
                        group["baseline_current_year_prediction"],
                    )
                )
            )
            rows.append(
                {
                    "segment": segment,
                    "segment_value": segment_value,
                    "validation_rows": int(len(group)),
                    "mean_actual_next_fantasy_points_ppr": float(group["next_fantasy_points_ppr"].mean()),
                    "mean_predicted_next_fantasy_points_ppr": float(group["prediction"].mean()),
                    "bias": float(group["residual"].mean()),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "baseline_current_year_rmse": baseline_rmse,
                    "baseline_current_year_mae": float(group["baseline_abs_residual"].mean()),
                }
            )
        return pd.DataFrame(rows)

    overall = validation_predictions.assign(overall="all")
    return pd.concat(
        [
            summarize(overall.groupby("overall"), "overall"),
            summarize(validation_predictions.groupby("position"), "position"),
        ],
        ignore_index=True,
    )


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

    modeling_df = player_season[player_season["season"].between(2016, 2024)].dropna(subset=[target_col]).copy()
    prediction_input = player_season[player_season["season"].eq(2025)].copy()

    final_model = _make_fantasy_model(feature_cols)
    final_model.fit(modeling_df[feature_cols], modeling_df[target_col])

    validation_predictions = rolling_fantasy_validation(player_season, feature_cols, target_col)
    validation_summary = summarize_fantasy_validation(validation_predictions)
    overall_validation = validation_summary[
        validation_summary["segment"].eq("overall")
    ]
    residual_rmse = (
        float(overall_validation["rmse"].iloc[0])
        if not overall_validation.empty
        else float(np.sqrt(mean_squared_error(modeling_df[target_col], final_model.predict(modeling_df[feature_cols]))))
    )

    predicted = prediction_input.copy()
    predicted["predicted_2026_fantasy_points_ppr"] = final_model.predict(
        predicted[feature_cols]
    ).clip(min=0)

    transformed_2025 = final_model.named_steps["preprocessor"].transform(predicted[feature_cols])
    forest = final_model.named_steps["model"]
    tree_predictions = np.column_stack(
        [
            estimator.predict(transformed_2025)
            for estimator in forest.estimators_
        ]
    )
    predicted["tree_prediction_std"] = tree_predictions.std(axis=1)
    predicted["prediction_uncertainty"] = np.sqrt(
        np.square(residual_rmse) + np.square(predicted["tree_prediction_std"])
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
        "projection_change_from_2025",
        "projection_change_label",
        "prediction_interval_low",
        "prediction_interval_high",
        "prediction_uncertainty",
        "fantasy_overall_rank",
        "fantasy_position_rank",
        "predicted_2026_overall_percentile",
        "predicted_2026_position_percentile",
        "fantasy_projection_tier",
        "usage_profile",
        "confidence_score",
        "confidence_level",
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
        "The model is a draft dashboard layer, not a finished fantasy ranking "
        "system. It does not yet include rookies, depth-chart changes, injuries, "
        "coaching changes, betting markets, or manual playing-time projections.\n\n"
        "To make the output easier to use, each player row includes a projection "
        "change label, a usage profile, and a plain-English fantasy explanation.\n\n"
        f"Projected players: {len(report_df):,}\n\n"
        f"Rolling validation rows: {len(validation_predictions):,}\n\n"
    )
    if not overall_validation.empty:
        row = overall_validation.iloc[0]
        summary_text += (
            f"Overall rolling MAE: {row['mae']:.2f} PPR points\n\n"
            f"Overall rolling RMSE: {row['rmse']:.2f} PPR points\n\n"
            f"Current-year baseline RMSE: {row['baseline_current_year_rmse']:.2f} PPR points\n"
        )

    outputs = {
        "fantasy_predictions": report_df,
        "fantasy_validation_predictions": validation_predictions,
        "fantasy_validation_summary": validation_summary,
        "feature_cols": feature_cols,
        "model_params": FANTASY_RANDOM_FOREST_PARAMS,
    }

    if save_outputs:
        report_df.to_csv(
            output_dir / "2026_fantasy_football_projections.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        validation_predictions.to_csv(
            output_dir / "fantasy_projection_validation_predictions.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        validation_summary.to_csv(
            output_dir / "fantasy_projection_validation_by_position.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        report_path = root / "report" / "fantasy_football_projection_summary.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(summary_text)

    return outputs
