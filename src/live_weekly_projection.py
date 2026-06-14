"""Live weekly fantasy projection: score the upcoming week before it is played.

The historical pipeline only scores weeks that already happened (each row carries
the realized PPR target). This module synthesizes the *next* week's rows — one per
active fantasy-relevant player — so the model can project games that have no
outcome yet.

How a future row is built (carry-forward design, per the Session 7 plan):

  - Player-history features (rolling PPR, usage, depth rank, availability) are
    carried forward from the player's latest completed week. They are computed
    from games strictly before that week, so they contain no information about
    the upcoming game. The one documented approximation is that they are "as of
    the latest completed week" rather than re-rolled to include it — a one-game
    lag on a multi-game rolling window.
  - Game-context features (opponent, home/away, rest, spread/total/implied total,
    roof/temp/wind) come from the upcoming game's SCHEDULE row, which is known
    before kickoff. Opponent PPR-allowed is looked up for the upcoming opponent.
  - The outcome column is left missing — it does not exist yet and is never used.

Leakage discipline: nothing from the projected game (its PBP, NGS/PFR, box score,
or final weather) enters the feature row. If a future feature is unknowable, the
latest prior value or a documented neutral default is used and flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import weekly_fantasy_projection as wk

TARGET = "target_fantasy_points_ppr"
DEFAULT_RECENT_WEEKS = 3
MIN_POSITION_CAL_ROWS = 200  # below this a position falls back to the global halfwidth

# Game-context columns that must be overwritten with the UPCOMING game's values
# (the carried-forward player row holds the previous game's context).
_GAME_CONTEXT_COLS = [
    "is_home", "rest_days", "rest_advantage", "spread_line_team_perspective",
    "total_line", "implied_team_total", "div_game", "is_indoor", "game_temp",
    "game_wind", "opp_ppr_allowed_last4_avg", "opponent_team",
]


def get_latest_completed_week(player_stats: pd.DataFrame) -> tuple[int, int]:
    """Latest (season, week) with realized regular-season outcomes."""
    df = player_stats[player_stats["season_type"].eq("REG")].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["season", "week", "fantasy_points_ppr"])
    season = int(df["season"].max())
    week = int(df[df["season"].eq(season)]["week"].max())
    return season, week


def get_target_projection_week(
    schedules: pd.DataFrame, latest: tuple[int, int]
) -> tuple[int, int]:
    """Next week that has scheduled regular-season games after ``latest``."""
    s = schedules.copy()
    s = s[s["game_type"].eq("REG")] if "game_type" in s.columns else s
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    season, week = latest
    later_same = s[(s["season"].eq(season)) & (s["week"].gt(week))]
    if not later_same.empty:
        return season, int(later_same["week"].min())
    nxt = s[s["season"].gt(season)]
    if not nxt.empty:
        nseason = int(nxt["season"].min())
        return nseason, int(nxt[nxt["season"].eq(nseason)]["week"].min())
    # No future games in the static schedule: project "the week after latest".
    return season, week + 1


def _latest_opp_allowed_lookup(hist_weekly: pd.DataFrame) -> pd.DataFrame:
    """Most recent rolling PPR-allowed value for each (defense, position)."""
    opp = wk.build_opponent_ppr_allowed(hist_weekly)
    opp = opp.sort_values(["season", "week"]).drop_duplicates(
        ["def_team", "position"], keep="last"
    )
    return opp[["def_team", "position", "opp_ppr_allowed_last4_avg"]].rename(
        columns={"def_team": "opponent_team"}
    )


def build_live_projection_frame(
    player_stats: pd.DataFrame,
    schedules: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    recent_weeks: int = DEFAULT_RECENT_WEEKS,
    as_of: tuple[int, int] | None = None,
    project_root=None,
) -> pd.DataFrame:
    """Synthesize one feature row per active player for the upcoming week.

    ``as_of`` overrides the latest-completed-week detection (e.g. to demo or
    backtest "as of" an earlier week). It defaults to the true latest completed
    regular-season week. Returns a frame carrying every production feature
    column plus display fields and a ``weather_is_default`` flag. The outcome
    column is intentionally absent.
    """
    modeling = wk.build_modeling_frame(player_stats, schedules, rosters, project_root=project_root)
    latest = as_of if as_of is not None else get_latest_completed_week(player_stats)
    t_season, t_week = get_target_projection_week(schedules, latest)
    l_season, l_week = latest

    # Active pool: players with a completed row in the last ``recent_weeks`` weeks
    # of the latest season, fantasy-relevant positions only.
    pool = modeling[
        modeling["season"].eq(l_season)
        & modeling["week"].gt(l_week - recent_weeks)
        & modeling["week"].le(l_week)
        & modeling["position"].isin(wk.SKILL_POSITIONS)
    ].copy()
    if pool.empty:
        return pd.DataFrame()
    # Carry forward each player's most recent completed-week feature row.
    carry = pool.sort_values(["player_id", "season", "week"]).drop_duplicates(
        "player_id", keep="last"
    ).copy()

    # Upcoming game context for the target week, per team, via the production
    # schedule-context helper.
    stub = carry[["player_id", "position", "team"]].copy()
    stub["season"] = t_season
    stub["week"] = t_week
    ctx = wk._attach_schedule_context(stub, schedules)
    ctx = ctx.rename(columns={"opponent_from_sched": "opponent_team"})
    # Players whose team has no scheduled game that week (bye) drop out.
    ctx = ctx.dropna(subset=["game_id"]).copy()

    # Opponent PPR allowed for the UPCOMING opponent.
    opp_lookup = _latest_opp_allowed_lookup(
        wk.prepare_weekly_player_games(player_stats, schedules, rosters)
    )
    ctx = ctx.merge(opp_lookup, on=["opponent_team", "position"], how="left")

    # Assemble: start from carried-forward player features, overwrite context.
    live = carry.drop(columns=[c for c in _GAME_CONTEXT_COLS if c in carry.columns], errors="ignore")
    ctx_cols = ["player_id"] + [c for c in _GAME_CONTEXT_COLS if c in ctx.columns] + ["game_temp", "game_wind", "is_indoor"]
    ctx_cols = list(dict.fromkeys(ctx_cols))
    live = live.merge(ctx[[c for c in ctx_cols if c in ctx.columns]], on="player_id", how="inner")

    live["season"] = t_season
    live["week"] = t_week
    if "season_week_number" in live.columns:
        live["season_week_number"] = t_week

    # Weather default flag + neutral fill if the schedule lacks a reading.
    live["weather_is_default"] = live["game_temp"].isna() | live["game_wind"].isna()
    live["game_temp"] = live["game_temp"].fillna(70.0)
    live["game_wind"] = live["game_wind"].fillna(0.0)
    live["is_indoor"] = live["is_indoor"].fillna(0.0)

    # Recompute market interactions on the upcoming context.
    live = wk.add_market_interactions(live)

    # Strip realized-outcome columns: the upcoming game has no box score, and
    # the carried row's target belongs to the PRIOR game. Scoring never uses
    # them; removing them guarantees no outcome leaks into a live projection.
    outcome_cols = [TARGET, "season_ppr_total", "games_played", "season_ppr_per_game",
                    "residual", "abs_residual"]
    live = live.drop(columns=[c for c in outcome_cols if c in live.columns], errors="ignore")

    # One row per player; drop any accidental duplicate keys.
    live = live.drop_duplicates(["player_id", "season", "week"]).reset_index(drop=True)
    return live


# ---------------------------------------------------------------------------
# Per-position conformal intervals
# ---------------------------------------------------------------------------
def compute_position_conformal_halfwidths(
    cal_df: pd.DataFrame,
    feature_cols: list[str],
    model,
    coverage_levels: dict[str, float] | None = None,
    min_rows: int = MIN_POSITION_CAL_ROWS,
) -> dict:
    """Per-position conformal halfwidths from a fitted model's calibration
    residuals, with a global fallback for thin positions.

    Returns {"global": {level: hw}, "by_position": {pos: {level: hw}}}.
    """
    coverage_levels = coverage_levels or wk.INTERVAL_QUANTILES
    resid = cal_df[TARGET].to_numpy() - model.predict(cal_df[feature_cols])
    out = {"global": {}, "by_position": {}}
    for label, cov in coverage_levels.items():
        out["global"][label] = wk._conformal_halfwidth(resid, cov)
    for pos in wk.SKILL_POSITIONS:
        mask = cal_df["position"].eq(pos).to_numpy()
        out["by_position"][pos] = {}
        for label, cov in coverage_levels.items():
            if mask.sum() >= min_rows:
                out["by_position"][pos][label] = wk._conformal_halfwidth(resid[mask], cov)
            else:
                out["by_position"][pos][label] = out["global"][label]  # documented fallback
    return out


def _apply_position_halfwidths(live: pd.DataFrame, halfwidths: dict, label: str) -> pd.DataFrame:
    df = live.copy()
    hw = df["position"].map(
        lambda p: halfwidths["by_position"].get(p, {}).get(label, halfwidths["global"][label])
    ).astype(float)
    df[f"interval_low_{label}"] = (df["projected_points"] - hw).clip(lower=0)
    df[f"interval_high_{label}"] = df["projected_points"] + hw
    return df


def score_live_projection_frame(
    live_frame: pd.DataFrame,
    player_stats: pd.DataFrame,
    schedules: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    project_root=None,
) -> tuple[pd.DataFrame, dict]:
    """Train the production weekly model on all completed weeks and score the
    live frame, with per-position conformal intervals. Returns (projections,
    halfwidths)."""
    if live_frame.empty:
        return pd.DataFrame(), {}
    modeling = wk.build_modeling_frame(player_stats, schedules, rosters, project_root=project_root)
    feature_cols = wk._available(modeling, wk.WEEKLY_FANTASY_FEATURES)

    train_all = modeling.dropna(subset=[TARGET]).sort_values(["season", "week"]).reset_index(drop=True)
    cal_size = max(int(round(0.2 * len(train_all))), 1)
    train_fit = train_all.iloc[: len(train_all) - cal_size]
    cal = train_all.iloc[len(train_all) - cal_size :]

    model = wk._make_main_model(feature_cols)
    model.fit(train_fit[feature_cols], train_fit[TARGET])

    halfwidths = compute_position_conformal_halfwidths(cal, feature_cols, model)

    out = live_frame.copy()
    out["projected_points"] = np.clip(model.predict(out[feature_cols]), 0, None)
    for label in wk.INTERVAL_QUANTILES:
        out = _apply_position_halfwidths(out, halfwidths, label)

    display = [
        "season", "week", "player_id", "player_display_name", "position", "team",
        "opponent_team", "is_home", "projected_points",
        "interval_low_50", "interval_high_50", "interval_low_80", "interval_high_80",
        "pbp_depth_chart_rank_last1", "ppr_last4_avg", "implied_team_total",
        "game_temp", "game_wind", "is_indoor", "weather_is_default",
    ]
    display = [c for c in display if c in out.columns]
    return out[display].sort_values("projected_points", ascending=False).reset_index(drop=True), halfwidths
