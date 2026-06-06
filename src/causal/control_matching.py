"""Build matched control panels for the QB-injury DiD design.

For each treatment event at ``(team T, season S, transition_week W)``, the
treated panel is the affected receivers' weekly PPR observations across
``[W - pre_period_length, W + post_period_length - 1]``. The control panel is
the analogous observations from receivers on **other teams** whose own
starting QB stayed the same throughout the same calendar window.

Same-calendar-week matching is the design's identification engine. It
automatically controls for league-wide trends (rising pass rates, schedule
structure, weather) without needing to model them explicitly. The only
remaining thing we need to defend is parallel trends in the pre-period.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.causal.treatment_identification import (
    DEFAULT_POST_PERIOD_LENGTH,
    DEFAULT_PRE_PERIOD_LENGTH,
    DEFAULT_MIN_PRE_TARGETS_PER_GAME,
    REGULAR_SEASON,
)


# ---------------------------------------------------------------------------
# Panel construction helpers
# ---------------------------------------------------------------------------
def _restrict_receivers_frame(
    player_stats: pd.DataFrame, positions: Iterable[str]
) -> pd.DataFrame:
    df = player_stats[
        player_stats["season_type"].eq(REGULAR_SEASON)
        & player_stats["position"].isin(list(positions))
    ].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["targets"] = pd.to_numeric(df["targets"], errors="coerce").fillna(0)
    df["fantasy_points_ppr"] = pd.to_numeric(
        df["fantasy_points_ppr"], errors="coerce"
    )
    df = df.dropna(subset=["season", "week", "team", "player_id"])
    return df


def _identify_stable_qb_teams(
    starting_qbs: pd.DataFrame,
    season: int,
    week_range: tuple[int, int],
) -> set[str]:
    """Teams whose starting QB stayed the same throughout ``week_range``.

    "Stayed the same" means a single ``starting_qb_id`` covers every team-week
    inside the range. Teams that had ANY transition (or any NA starter) in the
    window are excluded — they would themselves be partially treated.
    """
    lo, hi = week_range
    window = starting_qbs[
        starting_qbs["season"].eq(season)
        & starting_qbs["week"].between(lo, hi)
    ]
    # Drop teams whose starting QB is NA anywhere in the window — that's a
    # platoon week and would confound the comparison.
    bad_teams = set(
        window.loc[window["starting_qb_id"].isna(), "team"].astype(str).unique()
    )
    valid = window.dropna(subset=["starting_qb_id"])
    per_team_unique = valid.groupby("team")["starting_qb_id"].nunique()
    stable_teams = set(per_team_unique[per_team_unique.eq(1)].index)
    return {t for t in stable_teams if t not in bad_teams}


def _build_panel_rows(
    receivers_window: pd.DataFrame,
    event_id: str,
    transition_week: int,
    role: str,
) -> pd.DataFrame:
    rows = receivers_window.copy()
    rows["event_id"] = event_id
    rows["role"] = role
    rows["transition_week"] = transition_week
    rows["week_offset"] = rows["week"].astype(int) - int(transition_week)
    rows["period"] = np.where(rows["week_offset"].lt(0), "pre", "post")
    return rows[
        [
            "event_id",
            "role",
            "season",
            "week",
            "transition_week",
            "week_offset",
            "period",
            "team",
            "player_id",
            "player_display_name",
            "position",
            "targets",
            "fantasy_points_ppr",
        ]
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct_control_panel(
    events: pd.DataFrame,
    affected_receivers: pd.DataFrame,
    player_stats: pd.DataFrame,
    starting_qbs: pd.DataFrame,
    *,
    pre_period_length: int = DEFAULT_PRE_PERIOD_LENGTH,
    post_period_length: int = DEFAULT_POST_PERIOD_LENGTH,
    min_pre_targets_per_game: float = DEFAULT_MIN_PRE_TARGETS_PER_GAME,
    positions: Iterable[str] = ("WR",),
) -> pd.DataFrame:
    """Construct the unified panel of treated + control WR-week observations.

    The returned long panel has one row per (event_id, player_id, week) with
    columns ``role`` ('treated' / 'control'), ``period`` ('pre' / 'post'),
    ``week_offset`` (week relative to transition; negative = pre).
    """
    if events.empty:
        return pd.DataFrame()

    receivers = _restrict_receivers_frame(player_stats, positions)
    panels: list[pd.DataFrame] = []
    affected_set = set(affected_receivers["event_id"].unique())

    for _, ev in events.iterrows():
        event_id = ev["event_id"]
        if event_id not in affected_set:
            continue
        team = ev["team"]
        season = int(ev["season"])
        t_week = int(ev["transition_week"])
        lo = t_week - pre_period_length
        hi = t_week + post_period_length - 1

        affected_for_event = affected_receivers[
            affected_receivers["event_id"].eq(event_id)
        ]
        affected_ids = set(affected_for_event["player_id"].astype(str))

        # Treated panel: affected receivers' rows on the treated team across
        # the full window.
        treated = receivers[
            receivers["team"].eq(team)
            & receivers["season"].eq(season)
            & receivers["week"].between(lo, hi)
            & receivers["player_id"].astype(str).isin(affected_ids)
        ]
        if treated.empty:
            continue
        panels.append(
            _build_panel_rows(treated, event_id, t_week, role="treated")
        )

        # Control universe: teams whose starting QB was stable across the
        # full [lo, hi] window.
        stable_teams = _identify_stable_qb_teams(
            starting_qbs, season=season, week_range=(lo, hi)
        )
        stable_teams.discard(team)  # never match a treated team to itself
        if not stable_teams:
            continue

        control_window = receivers[
            receivers["season"].eq(season)
            & receivers["team"].astype(str).isin(stable_teams)
            & receivers["week"].between(lo, hi)
        ]

        # Apply the same pre-period volume filter as the treated panel so the
        # comparison is apples-to-apples on baseline target volume.
        pre_window = control_window[control_window["week"].lt(t_week)]
        per_player_pre = (
            pre_window.groupby("player_id")
            .agg(
                pre_games=("week", "count"),
                pre_total_targets=("targets", "sum"),
            )
            .reset_index()
        )
        per_player_pre["pre_avg_targets"] = (
            per_player_pre["pre_total_targets"] / per_player_pre["pre_games"]
        )
        eligible_controls = set(
            per_player_pre.loc[
                per_player_pre["pre_avg_targets"].ge(min_pre_targets_per_game),
                "player_id",
            ].astype(str)
        )
        if not eligible_controls:
            continue

        control_panel = control_window[
            control_window["player_id"].astype(str).isin(eligible_controls)
        ]
        # Also require the control receiver to appear in BOTH pre AND post —
        # we cannot estimate the DiD effect on a player who only appears in
        # one half of the window.
        appearances = control_panel.groupby(["player_id", "period_label"], dropna=False).size() if False else None
        # The "appears in both" check is enforced more cleanly by tagging
        # period below and requiring per-player counts > 0 in each.
        panels.append(
            _build_panel_rows(control_panel, event_id, t_week, role="control")
        )

    if not panels:
        return pd.DataFrame()
    panel = pd.concat(panels, ignore_index=True)

    # Drop control receivers who don't appear in both periods within the
    # event's window. Treated receivers were already filtered by the
    # pre-period activity threshold in attach_affected_receivers; we still
    # need to enforce the same balance for controls.
    coverage = (
        panel.groupby(["event_id", "role", "player_id", "period"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    coverage["has_pre"] = coverage.get("pre", 0).gt(0)
    coverage["has_post"] = coverage.get("post", 0).gt(0)
    balanced = coverage[coverage["has_pre"] & coverage["has_post"]][
        ["event_id", "role", "player_id"]
    ]
    panel = panel.merge(balanced, on=["event_id", "role", "player_id"], how="inner")

    return panel.reset_index(drop=True)
