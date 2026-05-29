"""Contextual football feature engineering for player value models.

These features describe the environment around a player's production. They are
kept separate from the core value score so the project can test whether added
football context improves next-season prediction rather than assuming that more
columns are automatically better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
TEAM_ABBREVIATION_MAP = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
}

USAGE_CONTEXT_FEATURES = [
    "avg_target_share",
    "avg_air_yards_share",
    "avg_wopr",
    "avg_pacr",
    "avg_racr",
    "avg_passing_cpoe",
]

TEAM_CONTEXT_FEATURES = [
    "team_pass_attempts",
    "team_carries",
    "team_targets",
    "team_offensive_plays",
    "team_pass_rate_proxy",
    "team_passing_epa_per_attempt",
    "team_rushing_epa_per_carry",
    "team_receiving_epa_per_target",
    "team_total_offense_epa_proxy",
    "player_target_share_team",
    "player_carry_share_team",
    "player_scrimmage_touch_share_team",
    "player_receiving_air_yards_share_team",
    "player_qb_play_share_team",
]

SCHEDULE_CONTEXT_FEATURES = [
    "avg_team_rest",
    "avg_rest_advantage",
    "avg_spread_for_team",
    "avg_total_line",
    "home_game_rate",
    "favorite_game_rate",
    "division_game_rate",
    "dome_or_closed_game_rate",
    "outdoor_game_rate",
    "grass_game_rate",
    "turf_game_rate",
    "avg_temp",
    "avg_wind",
]

CONTEXT_FEATURE_GROUPS = {
    "usage_context": USAGE_CONTEXT_FEATURES,
    "team_context": TEAM_CONTEXT_FEATURES,
    "schedule_context": SCHEDULE_CONTEXT_FEATURES,
}


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def _standardize_team_series(series: pd.Series) -> pd.Series:
    return series.astype(str).replace(TEAM_ABBREVIATION_MAP)


def _numeric_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(0.0, index=df.index)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denominator


def _filter_regular_skill_weeks(player_stats: pd.DataFrame) -> pd.DataFrame:
    weekly = player_stats.copy()
    if "season_type" in weekly.columns:
        weekly = weekly[weekly["season_type"].eq("REG")].copy()
    weekly = weekly[weekly["position"].isin(SKILL_POSITIONS)].copy()
    weekly = weekly.dropna(subset=["season", "player_id"])
    if "team" in weekly.columns:
        weekly["team"] = _standardize_team_series(weekly["team"])
    return weekly


def _weighted_group_average(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    weight_col: str,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []

    for key_values, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)

        record = dict(zip(group_cols, key_values))
        weights = pd.to_numeric(group[weight_col], errors="coerce").fillna(0)

        for col in value_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            valid = values.notna() & weights.gt(0)
            if valid.any() and weights[valid].sum() > 0:
                record[col] = float(np.average(values[valid], weights=weights[valid]))
            else:
                record[col] = float(values.mean()) if values.notna().any() else np.nan

        records.append(record)

    return pd.DataFrame(records)


def build_player_usage_context(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level usage and efficiency context to player seasons."""
    weekly = _filter_regular_skill_weeks(player_stats)
    group_cols = ["season", "player_id"]

    agg_spec: dict[str, tuple[str, str]] = {}
    if "week" in weekly.columns:
        agg_spec["usage_games"] = ("week", "nunique")
    else:
        agg_spec["usage_games"] = ("player_id", "size")

    weekly_share_cols = {
        "target_share": "avg_target_share",
        "air_yards_share": "avg_air_yards_share",
        "wopr": "avg_wopr",
        "pacr": "avg_pacr",
        "racr": "avg_racr",
    }
    for source_col, output_col in weekly_share_cols.items():
        if source_col in weekly.columns:
            agg_spec[output_col] = (source_col, "mean")

    usage = weekly.groupby(group_cols, as_index=False).agg(**agg_spec)

    if {"passing_cpoe", "attempts"}.issubset(weekly.columns):
        cpoe = weekly[group_cols + ["passing_cpoe", "attempts"]].copy()
        cpoe["passing_cpoe"] = pd.to_numeric(cpoe["passing_cpoe"], errors="coerce")
        cpoe["attempts"] = pd.to_numeric(cpoe["attempts"], errors="coerce").fillna(0)
        cpoe["weighted_passing_cpoe"] = cpoe["passing_cpoe"] * cpoe["attempts"]

        weighted = (
            cpoe.groupby(group_cols, as_index=False)
            .agg(
                weighted_passing_cpoe=("weighted_passing_cpoe", "sum"),
                passing_cpoe_attempts=("attempts", "sum"),
                simple_passing_cpoe=("passing_cpoe", "mean"),
            )
        )
        weighted["avg_passing_cpoe"] = np.where(
            weighted["passing_cpoe_attempts"].gt(0),
            weighted["weighted_passing_cpoe"] / weighted["passing_cpoe_attempts"],
            weighted["simple_passing_cpoe"],
        )
        usage = usage.merge(
            weighted[group_cols + ["avg_passing_cpoe"]],
            on=group_cols,
            how="left",
        )

    return usage


def build_team_environment_context(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Build player-season features describing team volume and role share."""
    raw = player_stats.copy()
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"].eq("REG")].copy()
    raw = raw.dropna(subset=["season", "team"])
    raw["team"] = _standardize_team_series(raw["team"])

    raw["team_offensive_plays_component"] = (
        _numeric_or_zero(raw, "attempts") + _numeric_or_zero(raw, "carries")
    )

    team_totals = (
        raw.groupby(["season", "team"], as_index=False)
        .agg(
            team_pass_attempts=("attempts", "sum"),
            team_carries=("carries", "sum"),
            team_targets=("targets", "sum"),
            team_receptions=("receptions", "sum"),
            team_receiving_air_yards=("receiving_air_yards", "sum"),
            team_passing_epa=("passing_epa", "sum"),
            team_rushing_epa=("rushing_epa", "sum"),
            team_receiving_epa=("receiving_epa", "sum"),
            team_offensive_plays=("team_offensive_plays_component", "sum"),
        )
    )

    team_totals["team_scrimmage_touches"] = (
        team_totals["team_carries"] + team_totals["team_receptions"]
    )
    team_totals["team_pass_rate_proxy"] = _safe_divide(
        team_totals["team_pass_attempts"],
        team_totals["team_offensive_plays"],
    )
    team_totals["team_passing_epa_per_attempt"] = _safe_divide(
        team_totals["team_passing_epa"],
        team_totals["team_pass_attempts"],
    )
    team_totals["team_rushing_epa_per_carry"] = _safe_divide(
        team_totals["team_rushing_epa"],
        team_totals["team_carries"],
    )
    team_totals["team_receiving_epa_per_target"] = _safe_divide(
        team_totals["team_receiving_epa"],
        team_totals["team_targets"],
    )
    team_totals["team_total_offense_epa_proxy"] = (
        team_totals["team_passing_epa"] + team_totals["team_rushing_epa"]
    )

    weekly = _filter_regular_skill_weeks(player_stats)
    weekly["player_qb_plays_component"] = (
        _numeric_or_zero(weekly, "attempts") + _numeric_or_zero(weekly, "carries")
    )
    if "week" in weekly.columns:
        games_agg = ("week", "nunique")
    else:
        games_agg = ("player_id", "size")

    player_team = (
        weekly.groupby(["season", "player_id", "team"], as_index=False)
        .agg(
            team_stint_games=games_agg,
            player_targets=("targets", "sum"),
            player_carries=("carries", "sum"),
            player_receptions=("receptions", "sum"),
            player_receiving_air_yards=("receiving_air_yards", "sum"),
            player_qb_plays=("player_qb_plays_component", "sum"),
        )
    )
    player_team["player_scrimmage_touches"] = (
        player_team["player_carries"] + player_team["player_receptions"]
    )

    stint_context = player_team.merge(team_totals, on=["season", "team"], how="left")
    stint_context["player_target_share_team"] = _safe_divide(
        stint_context["player_targets"],
        stint_context["team_targets"],
    )
    stint_context["player_carry_share_team"] = _safe_divide(
        stint_context["player_carries"],
        stint_context["team_carries"],
    )
    stint_context["player_scrimmage_touch_share_team"] = _safe_divide(
        stint_context["player_scrimmage_touches"],
        stint_context["team_scrimmage_touches"],
    )
    stint_context["player_receiving_air_yards_share_team"] = _safe_divide(
        stint_context["player_receiving_air_yards"],
        stint_context["team_receiving_air_yards"],
    )
    stint_context["player_qb_play_share_team"] = _safe_divide(
        stint_context["player_qb_plays"],
        stint_context["team_offensive_plays"],
    )

    return _weighted_group_average(
        stint_context,
        ["season", "player_id"],
        _available(stint_context, TEAM_CONTEXT_FEATURES),
        "team_stint_games",
    )


def _team_game_rows(schedules: pd.DataFrame) -> pd.DataFrame:
    schedule = schedules.copy()
    if "game_type" in schedule.columns:
        schedule = schedule[schedule["game_type"].eq("REG")].copy()

    numeric_cols = [
        "away_rest",
        "home_rest",
        "spread_line",
        "total_line",
        "temp",
        "wind",
        "div_game",
    ]
    for col in _available(schedule, numeric_cols):
        schedule[col] = pd.to_numeric(schedule[col], errors="coerce")

    common_cols = ["season", "week", "game_id", "roof", "surface", "temp", "wind"]
    common_cols = _available(schedule, common_cols)

    away = schedule[common_cols + ["away_team", "home_team", "away_rest", "home_rest", "spread_line", "total_line", "div_game"]].copy()
    away = away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent_team",
            "away_rest": "team_rest",
            "home_rest": "opponent_rest",
        }
    )
    away["is_home"] = 0
    away["spread_for_team"] = -away["spread_line"]

    home = schedule[common_cols + ["home_team", "away_team", "home_rest", "away_rest", "spread_line", "total_line", "div_game"]].copy()
    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent_team",
            "home_rest": "team_rest",
            "away_rest": "opponent_rest",
        }
    )
    home["is_home"] = 1
    home["spread_for_team"] = home["spread_line"]

    team_games = pd.concat([away, home], ignore_index=True)
    team_games["team"] = _standardize_team_series(team_games["team"])
    team_games["opponent_team"] = _standardize_team_series(team_games["opponent_team"])
    team_games["rest_advantage"] = (
        pd.to_numeric(team_games["team_rest"], errors="coerce")
        - pd.to_numeric(team_games["opponent_rest"], errors="coerce")
    )
    team_games["favorite_game"] = pd.to_numeric(
        team_games["spread_for_team"],
        errors="coerce",
    ).lt(0)

    roof = team_games.get("roof", pd.Series("", index=team_games.index)).astype(str).str.lower()
    surface = team_games.get("surface", pd.Series("", index=team_games.index)).astype(str).str.lower()
    team_games["dome_or_closed_game"] = roof.isin(["dome", "closed"]).astype(int)
    team_games["outdoor_game"] = roof.eq("outdoors").astype(int)
    team_games["grass_game"] = surface.str.contains("grass", na=False).astype(int)
    team_games["turf_game"] = (
        surface.str.contains("turf|astro|artificial", na=False).astype(int)
    )

    return team_games


def build_schedule_context(
    player_stats: pd.DataFrame,
    schedules: pd.DataFrame | None,
) -> pd.DataFrame:
    """Aggregate schedule and game-environment context to player seasons."""
    weekly = _filter_regular_skill_weeks(player_stats)
    keys = ["season", "player_id"]
    if schedules is None or schedules.empty or "game_id" not in weekly.columns:
        return weekly[keys].drop_duplicates().copy()

    player_games = weekly[
        _available(weekly, ["season", "player_id", "team", "game_id", "week"])
    ].drop_duplicates()
    team_games = _team_game_rows(schedules)
    if {"season", "week", "team"}.issubset(player_games.columns):
        merged = player_games.merge(
            team_games,
            on=["season", "week", "team"],
            how="left",
            suffixes=("", "_schedule"),
        )
        schedule_games_agg = ("week", "nunique")
    else:
        merged = player_games.merge(
            team_games,
            on=["season", "game_id", "team"],
            how="left",
            suffixes=("", "_schedule"),
        )
        schedule_games_agg = ("game_id", "nunique")

    schedule_context = (
        merged.groupby(keys, as_index=False)
        .agg(
            schedule_games=schedule_games_agg,
            avg_team_rest=("team_rest", "mean"),
            avg_rest_advantage=("rest_advantage", "mean"),
            avg_spread_for_team=("spread_for_team", "mean"),
            avg_total_line=("total_line", "mean"),
            home_game_rate=("is_home", "mean"),
            favorite_game_rate=("favorite_game", "mean"),
            division_game_rate=("div_game", "mean"),
            dome_or_closed_game_rate=("dome_or_closed_game", "mean"),
            outdoor_game_rate=("outdoor_game", "mean"),
            grass_game_rate=("grass_game", "mean"),
            turf_game_rate=("turf_game", "mean"),
            avg_temp=("temp", "mean"),
            avg_wind=("wind", "mean"),
        )
    )
    return schedule_context


def build_contextual_player_features(
    player_stats: pd.DataFrame,
    schedules: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one contextual-feature row per player-season."""
    weekly = _filter_regular_skill_weeks(player_stats)
    base = weekly[["season", "player_id"]].drop_duplicates().copy()

    usage = build_player_usage_context(player_stats)
    team = build_team_environment_context(player_stats)
    schedule = build_schedule_context(player_stats, schedules)

    context = base.merge(usage, on=["season", "player_id"], how="left")
    context = context.merge(team, on=["season", "player_id"], how="left")
    context = context.merge(schedule, on=["season", "player_id"], how="left")

    return context.sort_values(["season", "player_id"]).reset_index(drop=True)


def create_context_feature_dictionary() -> pd.DataFrame:
    """Return a compact data dictionary for contextual football features."""
    definitions = {
        "avg_target_share": "Average weekly share of team targets for the player.",
        "avg_air_yards_share": "Average weekly share of team air yards for the player.",
        "avg_wopr": "Average weekly weighted opportunity rating for receivers.",
        "avg_pacr": "Average passer air conversion ratio context from player stats.",
        "avg_racr": "Average receiver air conversion ratio context from player stats.",
        "avg_passing_cpoe": "Attempts-weighted average completion percentage over expected for QBs.",
        "team_pass_attempts": "Team regular-season pass attempts in the player's team context.",
        "team_carries": "Team regular-season carries in the player's team context.",
        "team_targets": "Team regular-season targets in the player's team context.",
        "team_offensive_plays": "Team pass attempts plus carries, used as a volume proxy.",
        "team_pass_rate_proxy": "Team pass attempts divided by pass attempts plus carries.",
        "team_passing_epa_per_attempt": "Team passing EPA divided by pass attempts.",
        "team_rushing_epa_per_carry": "Team rushing EPA divided by carries.",
        "team_receiving_epa_per_target": "Team receiving EPA divided by targets.",
        "team_total_offense_epa_proxy": "Team passing EPA plus rushing EPA; avoids adding receiving EPA to reduce double counting.",
        "player_target_share_team": "Player targets divided by team targets.",
        "player_carry_share_team": "Player carries divided by team carries.",
        "player_scrimmage_touch_share_team": "Player carries plus receptions divided by team carries plus receptions.",
        "player_receiving_air_yards_share_team": "Player receiving air yards divided by team receiving air yards.",
        "player_qb_play_share_team": "Player pass attempts plus carries divided by team pass attempts plus carries.",
        "avg_team_rest": "Average rest days in the player's games.",
        "avg_rest_advantage": "Average rest days minus opponent rest days.",
        "avg_spread_for_team": "Average closing spread from the player's team perspective; negative generally means favored.",
        "avg_total_line": "Average game total betting line.",
        "home_game_rate": "Share of the player's games played at home.",
        "favorite_game_rate": "Share of games where the player's team was favored by the spread.",
        "division_game_rate": "Share of games against division opponents.",
        "dome_or_closed_game_rate": "Share of games in domed or closed-roof conditions.",
        "outdoor_game_rate": "Share of games listed as outdoors.",
        "grass_game_rate": "Share of games on grass surfaces.",
        "turf_game_rate": "Share of games on turf or artificial surfaces.",
        "avg_temp": "Average listed game temperature where available.",
        "avg_wind": "Average listed game wind speed where available.",
    }

    records = []
    for group_name, features in CONTEXT_FEATURE_GROUPS.items():
        for feature in features:
            records.append(
                {
                    "feature_group": group_name,
                    "feature": feature,
                    "definition": definitions.get(feature, ""),
                }
            )
    return pd.DataFrame(records)
