"""QB-injury treatment identification for the WR1-PPR causal analysis.

This module implements the data-engineering foundation of session 1 from the
``causal/qb_injury_session1.md`` plan. Five composable functions, each
independently testable:

1. ``identify_starting_qb_per_team_week`` — primary QB per team-week using
   the >=50% pass-attempt-share rule.
2. ``identify_qb_transitions`` — week-to-week starter changes per team.
3. ``classify_transitions_by_cause`` — injury vs benching vs unknown.
4. ``construct_treatment_events`` — injury-driven transitions where the
   backup remains starter for at least N weeks (filters one-week emergencies).
5. ``attach_affected_receivers`` — WRs on the affected team with sufficient
   pre-period activity.

No causal estimation happens here. The estimator (session 2) consumes the
``treatment_events`` and ``affected_receivers`` tables this module produces.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


STARTING_QB_MIN_SHARE = 0.50
DEFAULT_MIN_POST_WEEKS = 2
DEFAULT_PRE_PERIOD_LENGTH = 4
DEFAULT_POST_PERIOD_LENGTH = 4
DEFAULT_MIN_PRE_TARGETS_PER_GAME = 3.0
# Statuses we consider as direct evidence of injury when the prior QB also
# stops starting. We include Questionable here on purpose: a starting QB
# listed Questionable on Friday who then doesn't start is overwhelmingly
# injury-driven, not a benching. Real benchings of healthy starters have no
# injury-report row at all.
INJURY_TREATMENT_STATUSES = ("Out", "IR", "Doubtful", "Questionable")
# DNP / limited practice classifications used by nflverse.
INJURY_PRACTICE_KEYWORDS = ("Did Not Participate", "Limited Participation")
REGULAR_SEASON = "REG"


# ---------------------------------------------------------------------------
# 1. Starting QB per team-week
# ---------------------------------------------------------------------------
def identify_starting_qb_per_team_week(
    player_stats: pd.DataFrame,
    min_share: float = STARTING_QB_MIN_SHARE,
) -> pd.DataFrame:
    """Per (team, season, week) regular-season game, identify the starting QB.

    The starter is defined as the QB with the highest pass-attempt share whose
    share clears ``min_share``. Rows where no QB clears the threshold (QB-
    platoon games, emergency-passer outliers) get ``starting_qb_id = NA`` and
    are returned as such — downstream filters drop them rather than guess.

    Returned columns: ``team, season, week, starting_qb_id,
    starting_qb_display_name, primary_share``.
    """
    df = player_stats.copy()
    df = df[df["season_type"].eq(REGULAR_SEASON)]
    df = df[df["position"].eq("QB")]
    df = df.dropna(subset=["season", "week", "team", "player_id"])
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(0)

    team_totals = (
        df.groupby(["team", "season", "week"], as_index=False)["attempts"]
        .sum()
        .rename(columns={"attempts": "team_pass_attempts"})
    )
    enriched = df.merge(team_totals, on=["team", "season", "week"], how="left")
    enriched["share"] = np.where(
        enriched["team_pass_attempts"] > 0,
        enriched["attempts"] / enriched["team_pass_attempts"],
        0.0,
    )

    # For each team-week, pick the QB with the largest share, then filter to
    # those clearing the min-share threshold.
    enriched = enriched.sort_values(
        ["team", "season", "week", "share"], ascending=[True, True, True, False]
    )
    top = enriched.drop_duplicates(
        subset=["team", "season", "week"], keep="first"
    ).copy()
    top["starting_qb_id"] = np.where(
        top["share"].ge(min_share), top["player_id"], pd.NA
    )
    top["starting_qb_display_name"] = np.where(
        top["share"].ge(min_share),
        top["player_display_name"],
        pd.NA,
    )

    return top[
        [
            "team",
            "season",
            "week",
            "starting_qb_id",
            "starting_qb_display_name",
            "share",
            "team_pass_attempts",
        ]
    ].rename(columns={"share": "primary_share"}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Identify QB transitions (week-over-week starter changes)
# ---------------------------------------------------------------------------
def identify_qb_transitions(starting_qbs: pd.DataFrame) -> pd.DataFrame:
    """Identify weeks where the starting QB changes within a season.

    Walks each (team, season) chronologically by ``week`` (skipping byes
    correctly — a team going QB1 → bye → QB2 is one transition, not two,
    because bye weeks simply don't appear in ``starting_qbs``). Emits one row
    per transition with the prior and new QB IDs.

    Returned columns: ``team, season, prior_week, transition_week,
    prior_qb_id, new_qb_id``.
    """
    df = starting_qbs.dropna(subset=["starting_qb_id"]).copy()
    df = df.sort_values(["team", "season", "week"])

    grp = df.groupby(["team", "season"], group_keys=False)
    df["prior_qb_id"] = grp["starting_qb_id"].shift(1)
    df["prior_week"] = grp["week"].shift(1)

    transitions = df[
        df["prior_qb_id"].notna() & df["starting_qb_id"].ne(df["prior_qb_id"])
    ].copy()

    return transitions[
        [
            "team",
            "season",
            "prior_week",
            "week",
            "prior_qb_id",
            "starting_qb_id",
        ]
    ].rename(
        columns={
            "week": "transition_week",
            "starting_qb_id": "new_qb_id",
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Classify transitions by cause
# ---------------------------------------------------------------------------
def _prior_qb_returns_in_season(
    transition: pd.Series, starting_qbs: pd.DataFrame | None
) -> bool:
    """True if the prior QB starts again for any team later in the same season."""
    if starting_qbs is None:
        return True  # be conservative — without data we don't reclassify
    qb_id = transition["prior_qb_id"]
    season = transition["season"]
    t_week = transition["transition_week"]
    after = starting_qbs[
        starting_qbs["starting_qb_id"].eq(qb_id)
        & starting_qbs["season"].eq(season)
        & starting_qbs["week"].gt(t_week)
    ]
    return not after.empty


def classify_transitions_by_cause(
    transitions: pd.DataFrame,
    injuries: pd.DataFrame,
    starting_qbs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify each transition as injury, injury_dnp, benching, or unknown.

    Injury reports run Wednesday/Thursday/Friday and emit one row per QB per
    practice day. We collapse to (gsis_id, season, week, max-severity status)
    so a single Out report anywhere that week qualifies.

    Adds a ``cause`` column with values:

    - ``injury`` — the prior QB has a report_status in INJURY_TREATMENT_STATUSES
      during the transition_week.
    - ``injury_dnp`` — the prior QB was Did Not Participate at practice
      without explicit Out status.
    - ``benching`` — the prior QB shows no injury report row in the
      transition week (likely benched).
    - ``unknown`` — injury data missing for that season/week.
    """
    inj = injuries.copy()
    if inj.empty:
        out = transitions.copy()
        out["cause"] = "unknown"
        out["prior_qb_report_status"] = pd.NA
        out["prior_qb_practice_status"] = pd.NA
        return out

    inj["season"] = pd.to_numeric(inj["season"], errors="coerce").astype("Int64")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce").astype("Int64")
    inj = inj.dropna(subset=["season", "week", "gsis_id"])

    severity_order = {"Out": 4, "IR": 4, "Doubtful": 3, "Questionable": 2}
    inj["severity"] = inj["report_status"].map(severity_order).fillna(0)
    # Reduce to one (gsis_id, season, week) row keeping the highest-severity status.
    inj = inj.sort_values(["gsis_id", "season", "week", "severity"], ascending=False)
    inj = inj.drop_duplicates(subset=["gsis_id", "season", "week"], keep="first")

    keep_cols = ["gsis_id", "season", "week", "report_status", "practice_status"]
    inj = inj[[c for c in keep_cols if c in inj.columns]]

    merged = transitions.merge(
        inj.rename(
            columns={
                "gsis_id": "prior_qb_id",
                "week": "transition_week",
                "report_status": "prior_qb_report_status",
                "practice_status": "prior_qb_practice_status",
            }
        ),
        on=["prior_qb_id", "season", "transition_week"],
        how="left",
    )

    def _classify(row: pd.Series) -> str:
        status = row.get("prior_qb_report_status")
        practice = row.get("prior_qb_practice_status")
        if pd.notna(status) and status in INJURY_TREATMENT_STATUSES:
            return "injury"
        if pd.notna(practice) and any(
            kw in str(practice) for kw in INJURY_PRACTICE_KEYWORDS
        ):
            return "injury_dnp"
        if pd.notna(status) or pd.notna(practice):
            return "benching"
        # No injury data for this transition. Check whether the prior QB
        # starts again later in the same season — if they never do, that is
        # very strong evidence of injury (a healthy starter benched for the
        # rest of the season is rare for QBs).
        if not _prior_qb_returns_in_season(row, starting_qbs):
            return "presumed_injury"
        return "unknown"

    merged["cause"] = merged.apply(_classify, axis=1)
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Construct treatment events
# ---------------------------------------------------------------------------
def construct_treatment_events(
    classified: pd.DataFrame,
    starting_qbs: pd.DataFrame,
    min_post_weeks: int = DEFAULT_MIN_POST_WEEKS,
    causes: Iterable[str] = ("injury", "injury_dnp", "presumed_injury"),
) -> pd.DataFrame:
    """Build the treatment-event table.

    A treatment event requires:

    - The transition is injury-driven (cause in ``causes``).
    - The new QB remains starter for at least ``min_post_weeks`` weeks (filters
      one-week emergencies where the original starter returns immediately).

    Each event gets a unique ``event_id`` of the form ``YYYY_TEAM_W{NN}``.
    """
    causes_set = set(causes)
    candidates = classified[classified["cause"].isin(causes_set)].copy()
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "team",
                "season",
                "transition_week",
                "prior_qb_id",
                "new_qb_id",
                "cause",
                "post_period_starter_weeks",
            ]
        )

    starters = starting_qbs.dropna(subset=["starting_qb_id"]).copy()
    starters = starters.sort_values(["team", "season", "week"])

    rows: list[dict] = []
    for _, ev in candidates.iterrows():
        team = ev["team"]
        season = int(ev["season"])
        t_week = int(ev["transition_week"])
        new_qb = ev["new_qb_id"]
        post = starters[
            starters["team"].eq(team)
            & starters["season"].eq(season)
            & starters["week"].ge(t_week)
        ]
        # Count consecutive weeks starting from transition_week where the new
        # QB remained starter (allowing for bye-week gaps, which simply don't
        # appear in the data).
        post_weeks = 0
        for _, p_row in post.iterrows():
            if p_row["starting_qb_id"] == new_qb:
                post_weeks += 1
            else:
                break
        if post_weeks < min_post_weeks:
            continue
        rows.append(
            {
                "event_id": f"{season}_{team}_W{t_week:02d}",
                "team": team,
                "season": season,
                "transition_week": t_week,
                "prior_qb_id": ev["prior_qb_id"],
                "new_qb_id": new_qb,
                "cause": ev["cause"],
                "post_period_starter_weeks": post_weeks,
            }
        )

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Attach affected receivers
# ---------------------------------------------------------------------------
def attach_affected_receivers(
    events: pd.DataFrame,
    player_stats: pd.DataFrame,
    positions: Iterable[str] = ("WR",),
    pre_period_length: int = DEFAULT_PRE_PERIOD_LENGTH,
    min_pre_targets_per_game: float = DEFAULT_MIN_PRE_TARGETS_PER_GAME,
) -> pd.DataFrame:
    """For each treatment event, identify affected receivers.

    A receiver is affected if they played on the treated team during the
    pre-period (``transition_week - pre_period_length`` through
    ``transition_week - 1``) and averaged at least
    ``min_pre_targets_per_game`` targets per game during that window.
    """
    if events.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "player_id",
                "player_display_name",
                "position",
                "pre_period_games",
                "pre_period_avg_targets",
                "pre_period_avg_ppr",
            ]
        )

    receivers = player_stats[
        player_stats["season_type"].eq(REGULAR_SEASON)
        & player_stats["position"].isin(list(positions))
    ].copy()
    receivers["season"] = pd.to_numeric(receivers["season"], errors="coerce").astype("Int64")
    receivers["week"] = pd.to_numeric(receivers["week"], errors="coerce").astype("Int64")
    receivers["targets"] = pd.to_numeric(
        receivers["targets"], errors="coerce"
    ).fillna(0)
    receivers["fantasy_points_ppr"] = pd.to_numeric(
        receivers["fantasy_points_ppr"], errors="coerce"
    )
    receivers = receivers.dropna(subset=["season", "week", "team", "player_id"])

    rows: list[pd.DataFrame] = []
    for _, ev in events.iterrows():
        team = ev["team"]
        season = int(ev["season"])
        t_week = int(ev["transition_week"])
        pre_start = t_week - pre_period_length
        pre_window = receivers[
            receivers["team"].eq(team)
            & receivers["season"].eq(season)
            & receivers["week"].ge(pre_start)
            & receivers["week"].lt(t_week)
        ]
        if pre_window.empty:
            continue

        per_player = (
            pre_window.groupby(
                ["player_id", "player_display_name", "position"], as_index=False
            )
            .agg(
                pre_period_games=("week", "count"),
                pre_period_total_targets=("targets", "sum"),
                pre_period_total_ppr=("fantasy_points_ppr", "sum"),
            )
        )
        per_player["pre_period_avg_targets"] = (
            per_player["pre_period_total_targets"] / per_player["pre_period_games"]
        )
        per_player["pre_period_avg_ppr"] = (
            per_player["pre_period_total_ppr"] / per_player["pre_period_games"]
        )
        per_player = per_player[
            per_player["pre_period_avg_targets"].ge(min_pre_targets_per_game)
        ]
        per_player["event_id"] = ev["event_id"]
        rows.append(per_player)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "event_id",
            "player_id",
            "player_display_name",
            "position",
            "pre_period_games",
            "pre_period_avg_targets",
            "pre_period_avg_ppr",
        ]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level driver (used by the pipeline and the session 1 writeup)
# ---------------------------------------------------------------------------
def build_treatment_artifacts(
    player_stats: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    min_post_weeks: int = DEFAULT_MIN_POST_WEEKS,
    pre_period_length: int = DEFAULT_PRE_PERIOD_LENGTH,
    min_pre_targets_per_game: float = DEFAULT_MIN_PRE_TARGETS_PER_GAME,
    positions: Iterable[str] = ("WR",),
) -> dict[str, pd.DataFrame]:
    """End-to-end: produce all session-1 treatment artifacts in one call."""
    starting_qbs = identify_starting_qb_per_team_week(player_stats)
    transitions = identify_qb_transitions(starting_qbs)
    classified = classify_transitions_by_cause(
        transitions, injuries, starting_qbs=starting_qbs
    )
    events = construct_treatment_events(
        classified, starting_qbs, min_post_weeks=min_post_weeks
    )
    affected = attach_affected_receivers(
        events,
        player_stats,
        positions=positions,
        pre_period_length=pre_period_length,
        min_pre_targets_per_game=min_pre_targets_per_game,
    )
    return {
        "starting_qbs": starting_qbs,
        "transitions": transitions,
        "classified": classified,
        "events": events,
        "affected_receivers": affected,
    }
