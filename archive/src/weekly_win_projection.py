"""Weekly game winner projection outputs for the dashboard draft."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.load_data import ensure_project_dirs, find_project_root, load_csv
from src.models import make_model_pipeline


WEEKLY_WIN_FEATURES = [
    "week",
    "spread_line",
    "total_line",
    "home_rest",
    "away_rest",
    "rest_advantage",
    "div_game",
    "temp",
    "wind",
    "home_recent_win_rate",
    "away_recent_win_rate",
    "recent_win_rate_diff",
    "home_recent_point_diff",
    "away_recent_point_diff",
    "recent_point_diff_diff",
    "home_recent_points_for",
    "away_recent_points_for",
    "home_recent_points_against",
    "away_recent_points_against",
]

CSV_FLOAT_FORMAT = "%.12g"


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    converted = df.copy()
    for col in cols:
        if col in converted.columns:
            converted[col] = pd.to_numeric(converted[col], errors="coerce")
    return converted


def _make_weekly_model(feature_cols: list[str]):
    model = LogisticRegression(max_iter=1000)
    return make_model_pipeline(feature_cols, model, categorical_cols=[])


def _winner_confidence(probability: float) -> str:
    winner_probability = max(probability, 1 - probability)
    if winner_probability >= 0.65:
        return "High"
    if winner_probability >= 0.58:
        return "Medium"
    return "Low"


def _market_signal(row: pd.Series) -> str:
    spread = row.get("spread_line")
    if pd.isna(spread):
        return "No spread signal available"
    if spread > 0:
        return f"Market leaned {row['home_team']} by {spread:.1f}"
    if spread < 0:
        return f"Market leaned {row['away_team']} by {abs(spread):.1f}"
    return "Market viewed the game near even"


def _edge_text(value: float, home_team: str, away_team: str, label: str, threshold: float) -> str:
    if pd.isna(value) or abs(value) < threshold:
        return ""
    team = home_team if value > 0 else away_team
    return f"{label} favored {team}"


def _format_pick_explanation(row: pd.Series) -> str:
    winner_probability = row.get("winner_probability", np.nan)
    opening = (
        f"Picked {row['predicted_winner']}"
        if pd.isna(winner_probability)
        else f"Picked {row['predicted_winner']} at {winner_probability:.0%}"
    )

    reasons = [_market_signal(row)]
    form_reason = _edge_text(
        row.get("recent_point_diff_diff"),
        row["home_team"],
        row["away_team"],
        "recent point differential",
        3,
    )
    win_rate_reason = _edge_text(
        row.get("recent_win_rate_diff"),
        row["home_team"],
        row["away_team"],
        "recent win rate",
        0.20,
    )
    rest_reason = _edge_text(
        row.get("rest_advantage"),
        row["home_team"],
        row["away_team"],
        "rest",
        1,
    )
    for reason in [form_reason, win_rate_reason, rest_reason]:
        if reason:
            reasons.append(reason)

    if not form_reason and not win_rate_reason:
        reasons.append("recent-form signal was limited or close")

    return opening + ". " + "; ".join(reasons) + "."


def prepare_completed_regular_games(schedules: pd.DataFrame) -> pd.DataFrame:
    """Return completed regular-season games with target labels."""
    games = schedules.copy()
    games = games[games["game_type"].eq("REG")].copy()
    games = games.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    games = _to_numeric(
        games,
        [
            "season",
            "week",
            "home_score",
            "away_score",
            "home_rest",
            "away_rest",
            "spread_line",
            "total_line",
            "div_game",
            "temp",
            "wind",
        ],
    )
    games = games[games["home_score"].ne(games["away_score"])].copy()
    games["home_win"] = games["home_score"].gt(games["away_score"]).astype(int)
    games["actual_winner"] = np.where(
        games["home_win"].eq(1),
        games["home_team"],
        games["away_team"],
    )
    games["matchup"] = games["away_team"].astype(str) + " at " + games["home_team"].astype(str)
    return games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def build_team_recent_form(games: pd.DataFrame) -> pd.DataFrame:
    """Create pregame recent-form features without using the current game."""
    home_rows = games[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "points_for",
            "away_score": "points_against",
        }
    )
    home_rows["is_home"] = 1

    away_rows = games[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
        ]
    ].rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "points_for",
            "home_score": "points_against",
        }
    )
    away_rows["is_home"] = 0

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games = team_games.sort_values(["season", "week", "game_id", "team"]).copy()
    team_games["win"] = team_games["points_for"].gt(team_games["points_against"]).astype(int)
    team_games["point_diff"] = team_games["points_for"] - team_games["points_against"]
    grouped = team_games.groupby(["season", "team"], group_keys=False)

    team_games["team_game_number"] = grouped.cumcount() + 1
    rolling_map = {
        "win": "recent_win_rate",
        "point_diff": "recent_point_diff",
        "points_for": "recent_points_for",
        "points_against": "recent_points_against",
    }
    for source_col, feature_col in rolling_map.items():
        team_games[feature_col] = grouped[source_col].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean()
        )

    return team_games[
        [
            "game_id",
            "team",
            "team_game_number",
            "recent_win_rate",
            "recent_point_diff",
            "recent_points_for",
            "recent_points_against",
        ]
    ].copy()


def create_weekly_modeling_frame(schedules: pd.DataFrame) -> pd.DataFrame:
    """Create one row per completed game with pregame winner features."""
    games = prepare_completed_regular_games(schedules)
    recent_form = build_team_recent_form(games)

    home_form = recent_form.rename(
        columns={
            "team": "home_team",
            "team_game_number": "home_team_game_number",
            "recent_win_rate": "home_recent_win_rate",
            "recent_point_diff": "home_recent_point_diff",
            "recent_points_for": "home_recent_points_for",
            "recent_points_against": "home_recent_points_against",
        }
    )
    away_form = recent_form.rename(
        columns={
            "team": "away_team",
            "team_game_number": "away_team_game_number",
            "recent_win_rate": "away_recent_win_rate",
            "recent_point_diff": "away_recent_point_diff",
            "recent_points_for": "away_recent_points_for",
            "recent_points_against": "away_recent_points_against",
        }
    )

    modeled = games.merge(home_form, on=["game_id", "home_team"], how="left")
    modeled = modeled.merge(away_form, on=["game_id", "away_team"], how="left")
    modeled["rest_advantage"] = modeled["home_rest"] - modeled["away_rest"]
    modeled["recent_win_rate_diff"] = (
        modeled["home_recent_win_rate"] - modeled["away_recent_win_rate"]
    )
    modeled["recent_point_diff_diff"] = (
        modeled["home_recent_point_diff"] - modeled["away_recent_point_diff"]
    )
    return modeled


def _evaluate_classifier(
    y_true: pd.Series,
    probability: np.ndarray,
) -> dict[str, float]:
    predicted = (probability >= 0.50).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "brier_score": float(brier_score_loss(y_true, probability)),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probability))
        metrics["log_loss"] = float(log_loss(y_true, probability))
    else:
        metrics["roc_auc"] = np.nan
        metrics["log_loss"] = np.nan
    return metrics


def rolling_weekly_win_validation(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    validation_years: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run rolling-origin validation for weekly winner probabilities."""
    if validation_years is None:
        validation_years = [2020, 2021, 2022, 2023, 2024, 2025]

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []

    for valid_year in validation_years:
        train_df = modeling_df[modeling_df["season"].lt(valid_year)].copy()
        valid_df = modeling_df[modeling_df["season"].eq(valid_year)].copy()
        if train_df.empty or valid_df.empty:
            continue

        pipeline = _make_weekly_model(feature_cols)
        pipeline.fit(train_df[feature_cols], train_df["home_win"])
        probability = pipeline.predict_proba(valid_df[feature_cols])[:, 1]
        predicted_home_win = (probability >= 0.50).astype(int)

        fold = valid_df[
            [
                "game_id",
                "season",
                "week",
                "gameday",
                "away_team",
                "home_team",
                "away_score",
                "home_score",
                "actual_winner",
                "home_win",
                "matchup",
                "spread_line",
                "total_line",
                "home_rest",
                "away_rest",
                "rest_advantage",
                "div_game",
                "home_recent_win_rate",
                "away_recent_win_rate",
                "recent_win_rate_diff",
                "home_recent_point_diff",
                "away_recent_point_diff",
                "recent_point_diff_diff",
                "home_recent_points_for",
                "away_recent_points_for",
            ]
        ].copy()
        fold["predicted_home_win_probability"] = probability
        fold["predicted_winner"] = np.where(
            predicted_home_win == 1,
            fold["home_team"],
            fold["away_team"],
        )
        fold["correct_prediction"] = fold["predicted_winner"].eq(fold["actual_winner"])
        fold["winner_probability"] = np.where(
            predicted_home_win == 1,
            probability,
            1 - probability,
        )
        fold["confidence_level"] = fold["predicted_home_win_probability"].apply(_winner_confidence)
        fold["market_signal"] = fold.apply(_market_signal, axis=1)
        fold["pick_explanation"] = fold.apply(_format_pick_explanation, axis=1)
        fold["prediction_source"] = "rolling_backtest"
        prediction_rows.append(fold)

        metrics = _evaluate_classifier(valid_df["home_win"], probability)
        metrics["season"] = int(valid_year)
        metrics["games"] = int(len(valid_df))
        metric_rows.append(metrics)

    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    by_season = pd.DataFrame(metric_rows)

    if not predictions.empty:
        overall_metrics = _evaluate_classifier(
            predictions["home_win"],
            predictions["predicted_home_win_probability"].to_numpy(),
        )
        overall_metrics["season"] = "overall"
        overall_metrics["games"] = int(len(predictions))
        by_season = pd.concat(
            [pd.DataFrame([overall_metrics]), by_season],
            ignore_index=True,
        )

    return predictions, by_season


def build_weekly_win_projection_outputs(
    project_root: str | Path | None = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Build weekly winner projection tables for the dashboard."""
    root = find_project_root() if project_root is None else Path(project_root).resolve()
    dirs = ensure_project_dirs(root)
    output_dir = dirs["tables"]

    schedules = load_csv("data/raw/schedules_2016_2025.csv", root, low_memory=False)
    modeling_df = create_weekly_modeling_frame(schedules)
    feature_cols = _available(modeling_df, WEEKLY_WIN_FEATURES)

    game_predictions, validation_summary = rolling_weekly_win_validation(
        modeling_df,
        feature_cols,
    )
    game_predictions = game_predictions.sort_values(
        ["season", "week", "game_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    summary_text = (
        "# Weekly Win Projection Summary\n\n"
        "This dashboard layer predicts the probability that the home team wins a "
        "regular-season game. The current version is a historical rolling "
        "backtest, so each season is predicted using only earlier seasons.\n\n"
        "Features include market line context, rest, divisional-game status, "
        "weather when available, and each team's recent in-season form. Because "
        "sportsbook lines are included, this should be read as a market-informed "
        "projection rather than a pure team-strength model.\n\n"
        "When future schedule rows are added locally, this same feature pipeline "
        "can be extended to score upcoming weeks.\n\n"
        "To make the table easier to interpret, each game row includes a market "
        "signal and a short pick explanation based on spread, recent form, and rest.\n\n"
        f"Backtested games: {len(game_predictions):,}\n\n"
    )
    if not validation_summary.empty:
        overall = validation_summary[validation_summary["season"].astype(str).eq("overall")]
        if not overall.empty:
            row = overall.iloc[0]
            summary_text += (
                f"Overall rolling accuracy: {row['accuracy']:.3f}\n\n"
                f"Overall Brier score: {row['brier_score']:.3f}\n\n"
                f"Overall ROC AUC: {row['roc_auc']:.3f}\n"
            )

    outputs = {
        "weekly_win_games": game_predictions,
        "weekly_win_validation": validation_summary,
        "modeling_frame": modeling_df,
        "feature_cols": feature_cols,
    }

    if save_outputs:
        game_predictions.to_csv(
            output_dir / "weekly_win_projection_games.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        validation_summary.to_csv(
            output_dir / "weekly_win_projection_validation.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        report_path = root / "report" / "weekly_win_projection_summary.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(summary_text)

    return outputs
