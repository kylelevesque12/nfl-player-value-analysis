"""Cleaning helpers for building NFL player-season datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


SKILL_POSITIONS = list(config.SKILL_POSITIONS)

DEFAULT_GROUP_COLS = [
    "season",
    "player_id",
    "player_display_name",
    "position",
    "team",
]

STAT_SUM_COLUMNS = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks",
    "sacks_suffered",
    "sack_yards",
    "sack_yards_lost",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_first_downs",
    "passing_epa",
    "passing_2pt_conversions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",
    "rushing_epa",
    "rushing_2pt_conversions",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_first_downs",
    "receiving_epa",
    "receiving_2pt_conversions",
    "fantasy_points",
    "fantasy_points_ppr",
]

ROSTER_COLUMNS = [
    "season",
    "gsis_id",
    "birth_date",
    "height",
    "weight",
    "years_exp",
    "entry_year",
    "rookie_year",
    "draft_club",
    "draft_number",
    "college",
]


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def _series_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype="float64")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def filter_skill_player_weeks(
    player_stats: pd.DataFrame,
    positions: list[str] | None = None,
    regular_season_only: bool = True,
) -> pd.DataFrame:
    """Filter weekly player stats to offensive skill positions."""
    if positions is None:
        positions = SKILL_POSITIONS

    skill_weekly = player_stats.copy()
    if regular_season_only and "season_type" in skill_weekly.columns:
        skill_weekly = skill_weekly[skill_weekly["season_type"].eq("REG")]

    if "position" not in skill_weekly.columns:
        raise ValueError("player_stats must include a position column.")

    return skill_weekly[skill_weekly["position"].isin(positions)].copy()


def aggregate_weekly_to_player_season(
    skill_weekly: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate weekly player rows to player-season-team rows."""
    if group_cols is None:
        group_cols = DEFAULT_GROUP_COLS

    group_cols = _available(skill_weekly, group_cols)
    if not {"season", "player_id", "position"}.issubset(group_cols):
        raise ValueError("Aggregation requires season, player_id, and position columns.")

    sum_cols = _available(skill_weekly, STAT_SUM_COLUMNS)
    agg_spec = {col: "sum" for col in sum_cols}
    for first_col in _available(skill_weekly, ["player_name"]):
        if first_col not in group_cols:
            agg_spec[first_col] = "first"

    if "week" in skill_weekly.columns:
        games_played = (
            skill_weekly.groupby(group_cols)["week"]
            .nunique()
            .rename("games_played")
            .reset_index()
        )
    else:
        games_played = (
            skill_weekly.groupby(group_cols)
            .size()
            .rename("games_played")
            .reset_index()
        )

    season_stats = skill_weekly.groupby(group_cols, as_index=False).agg(agg_spec)
    season_stats = season_stats.merge(games_played, on=group_cols, how="left")
    return season_stats


def add_offensive_rate_features(skill_season: pd.DataFrame) -> pd.DataFrame:
    """Add total, scrimmage, and QB-specific rate features."""
    df = skill_season.copy()
    games = pd.to_numeric(df["games_played"], errors="coerce")

    df["total_yards"] = (
        _series_or_zero(df, "passing_yards")
        + _series_or_zero(df, "rushing_yards")
        + _series_or_zero(df, "receiving_yards")
    )
    df["total_tds"] = (
        _series_or_zero(df, "passing_tds")
        + _series_or_zero(df, "rushing_tds")
        + _series_or_zero(df, "receiving_tds")
    )
    df["total_epa"] = (
        _series_or_zero(df, "passing_epa")
        + _series_or_zero(df, "rushing_epa")
        + _series_or_zero(df, "receiving_epa")
    )
    df["yards_per_game"] = _safe_divide(df["total_yards"], games)
    df["tds_per_game"] = _safe_divide(df["total_tds"], games)
    df["epa_per_game"] = _safe_divide(df["total_epa"], games)

    df["scrimmage_touches"] = _series_or_zero(df, "carries") + _series_or_zero(df, "receptions")
    df["scrimmage_yards"] = _series_or_zero(df, "rushing_yards") + _series_or_zero(df, "receiving_yards")
    df["scrimmage_tds"] = _series_or_zero(df, "rushing_tds") + _series_or_zero(df, "receiving_tds")
    df["scrimmage_epa"] = _series_or_zero(df, "rushing_epa") + _series_or_zero(df, "receiving_epa")
    df["scrimmage_yards_per_game"] = _safe_divide(df["scrimmage_yards"], games)
    df["scrimmage_touches_per_game"] = _safe_divide(df["scrimmage_touches"], games)
    df["scrimmage_tds_per_game"] = _safe_divide(df["scrimmage_tds"], games)
    df["scrimmage_epa_per_game"] = _safe_divide(df["scrimmage_epa"], games)
    df["yards_per_scrimmage_touch"] = _safe_divide(
        df["scrimmage_yards"],
        df["scrimmage_touches"],
    )

    df["qb_plays"] = _series_or_zero(df, "attempts") + _series_or_zero(df, "carries")
    df["qb_total_yards"] = _series_or_zero(df, "passing_yards") + _series_or_zero(df, "rushing_yards")
    df["qb_total_tds"] = _series_or_zero(df, "passing_tds") + _series_or_zero(df, "rushing_tds")
    df["qb_epa"] = _series_or_zero(df, "passing_epa") + _series_or_zero(df, "rushing_epa")
    df["qb_yards_per_play"] = _safe_divide(df["qb_total_yards"], df["qb_plays"])
    df["qb_yards_per_game"] = _safe_divide(df["qb_total_yards"], games)
    df["qb_tds_per_game"] = _safe_divide(df["qb_total_tds"], games)
    df["qb_epa_per_game"] = _safe_divide(df["qb_epa"], games)
    df["interceptions_per_game"] = _safe_divide(
        _series_or_zero(df, "passing_interceptions"),
        games,
    )

    return df


def merge_roster_context(skill_season: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Merge age, draft, and biographical roster fields onto player seasons."""
    roster_cols = _available(rosters, ROSTER_COLUMNS)
    if not {"season", "gsis_id"}.issubset(roster_cols):
        raise ValueError("rosters must include season and gsis_id columns.")

    roster_context = rosters[roster_cols].drop_duplicates(["season", "gsis_id"])
    merged = skill_season.merge(
        roster_context,
        left_on=["season", "player_id"],
        right_on=["season", "gsis_id"],
        how="left",
    )

    if "birth_date" in merged.columns:
        birth_date = pd.to_datetime(merged["birth_date"], errors="coerce")
        merged["age"] = merged["season"] - birth_date.dt.year

    return merged


def build_skill_player_seasons(
    player_stats: pd.DataFrame,
    rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the cleaned player-season table used by downstream notebooks."""
    skill_weekly = filter_skill_player_weeks(player_stats)
    skill_season = aggregate_weekly_to_player_season(skill_weekly)
    skill_season = add_offensive_rate_features(skill_season)

    if rosters is not None:
        skill_season = merge_roster_context(skill_season, rosters)

    return skill_season
