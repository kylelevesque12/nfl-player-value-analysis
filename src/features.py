"""Feature engineering helpers for player value and prediction modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


MIN_VALUE_GAMES = 4
VALUE_GROUP_COLS = ["season", "position"]


def _series_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype="float64")


def add_group_zscore(
    df: pd.DataFrame,
    value_col: str,
    z_col: str | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Standardize a column within season-position groups."""
    if group_cols is None:
        group_cols = VALUE_GROUP_COLS
    if z_col is None:
        z_col = value_col + "_z"

    scored = df.copy()
    group_mean = scored.groupby(group_cols)[value_col].transform("mean")
    group_std = scored.groupby(group_cols)[value_col].transform("std")
    scored[z_col] = (scored[value_col] - group_mean) / group_std
    return scored


def add_value_epa_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create the EPA columns used for the main value score.

    QBs use passing plus rushing EPA. RBs, WRs, and TEs use rushing plus
    receiving EPA. The total EPA version is the current primary metric, while
    the per-game version is kept as a supporting diagnostic.
    """
    scored = df.copy()
    games = pd.to_numeric(scored["games_played"], errors="coerce").replace(0, np.nan)

    if "qb_epa" not in scored.columns:
        scored["qb_epa"] = _series_or_zero(scored, "passing_epa") + _series_or_zero(scored, "rushing_epa")
    if "scrimmage_epa" not in scored.columns:
        scored["scrimmage_epa"] = _series_or_zero(scored, "rushing_epa") + _series_or_zero(scored, "receiving_epa")

    scored["value_epa_total"] = np.where(
        scored["position"].eq("QB"),
        scored["qb_epa"],
        scored["scrimmage_epa"],
    )
    scored["value_epa_per_game"] = scored["value_epa_total"] / games
    return scored


def add_position_rankings(
    df: pd.DataFrame,
    score_col: str = "value_score",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add within-position season rank and percentile columns."""
    if group_cols is None:
        group_cols = VALUE_GROUP_COLS

    ranked = df.copy()
    ranked["position_season_rank"] = (
        ranked.groupby(group_cols)[score_col].rank(ascending=False, method="min")
    )
    ranked["position_season_percentile"] = (
        ranked.groupby(group_cols)[score_col].rank(pct=True)
    )
    return ranked


def create_value_scores(
    player_seasons: pd.DataFrame,
    min_games: int = MIN_VALUE_GAMES,
) -> pd.DataFrame:
    """Create the project value-score columns from player-season data."""
    scored = player_seasons.copy()
    scored = scored[scored["games_played"].ge(min_games)].copy()
    scored = add_value_epa_columns(scored)
    scored = add_group_zscore(scored, "value_epa_total", "value_score")
    scored = add_group_zscore(scored, "value_epa_per_game", "value_score_per_game")
    scored["value_score_total_epa"] = scored["value_score"]
    scored["value_score_gap"] = scored["value_score_per_game"] - scored["value_score"]
    scored["value_metric"] = "position_adjusted_total_epa"
    scored = add_position_rankings(scored)
    return scored


def add_player_history_features(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Add lagged and rolling player-history features without future leakage."""
    featured = player_seasons.sort_values(["player_id", "season"]).copy()
    grouped = featured.groupby("player_id", group_keys=False)
    featured["prior_qualifying_seasons"] = grouped.cumcount()

    history_cols = [
        "value_score",
        "value_epa_total",
        "value_epa_per_game",
        "games_played",
        "yards_per_game",
        "tds_per_game",
    ]
    for col in history_cols:
        if col not in featured.columns:
            continue
        featured[f"{col}_prev"] = grouped[col].shift(1)
        featured[f"{col}_last2_avg"] = grouped[col].apply(
            lambda s: s.shift(1).rolling(2, min_periods=1).mean()
        )
        featured[f"{col}_last3_avg"] = grouped[col].apply(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

    if {"value_score_prev", "value_score_last2_avg"}.issubset(featured.columns):
        featured["value_score_trend_2yr"] = (
            featured["value_score_prev"] - featured["value_score_last2_avg"]
        )

    if "games_played" in featured.columns:
        featured["games_played_last2_sum"] = grouped["games_played"].apply(
            lambda s: s.shift(1).rolling(2, min_periods=1).sum()
        )
        featured["games_played_last3_avg"] = grouped["games_played"].apply(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

    return featured


def create_next_season_targets(
    player_seasons: pd.DataFrame,
    require_consecutive_season: bool = True,
) -> pd.DataFrame:
    """Attach next-season value targets for supervised modeling."""
    targeted = player_seasons.sort_values(["player_id", "season"]).copy()
    grouped = targeted.groupby("player_id")

    next_cols = [
        "season",
        "value_epa_total",
        "value_epa_per_game",
        "value_score",
        "value_score_per_game",
    ]
    for col in next_cols:
        if col in targeted.columns:
            targeted["next_" + col] = grouped[col].shift(-1)

    if require_consecutive_season and "next_season" in targeted.columns:
        has_next = targeted["next_season"].eq(targeted["season"] + 1)
        targeted["next_season_qualifier"] = has_next.astype(int)
        for col in [
            "next_value_epa_total",
            "next_value_epa_per_game",
            "next_value_score",
            "next_value_score_per_game",
        ]:
            if col in targeted.columns:
                targeted.loc[~has_next, col] = np.nan

    return targeted
