"""Session-3 (causal) treatment re-definition: first injury-report appearance.

Sessions 1-2 defined treatment as the week a starting QB was actually *replaced*
by a backup, with an Out/Doubtful/Questionable designation. The session-2 verdict
was a null with a clear mechanism: by the time a QB is formally Out, his receivers
have already been declining for weeks — the formal designation is a *lagging*
indicator of QB health. The proposed fix, implemented here, is to move treatment
earlier: the **first week the team's established starting QB appears on the injury
report at all** — any status, including a Questionable tag or merely a limited /
DNP practice, even if he still starts the game.

This module builds that treatment, the matching-clean control starters (control
teams must have an injury-report-FREE starting QB through the window), and the
Out-only comparison event set. It reuses the session-1 starting-QB logic and the
session-1/2 panel + estimator machinery unchanged.

Leakage discipline: event timing is fixed from the injury report (known before
the game is played that week) and pre-period eligibility never looks at
post-treatment outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal.treatment_identification import (
    identify_starting_qb_per_team_week,
    REGULAR_SEASON,
)

DEFAULT_MIN_PRE_WEEKS = 3
DEFAULT_MIN_POST_WEEKS = 2
OUT_STATUSES = ("Out", "IR")


def _injuries_reg(injuries: pd.DataFrame) -> pd.DataFrame:
    inj = injuries.copy()
    # In this injuries feed ``game_type`` is the fully-populated regular/post
    # flag; ``season_type`` is present but null on ~90% of rows, so prefer
    # game_type and only fall back to season_type when game_type is absent.
    if "game_type" in inj.columns:
        inj = inj[inj["game_type"].eq(REGULAR_SEASON)]
    elif "season_type" in inj.columns:
        inj = inj[inj["season_type"].eq(REGULAR_SEASON)]
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce").astype("Int64")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce").astype("Int64")
    return inj.dropna(subset=["season", "week", "gsis_id"])


def qb_injury_report_weeks(injuries: pd.DataFrame) -> pd.DataFrame:
    """One row per (gsis_id, season, week) a player appears on the injury report
    (ANY status, including practice-only). Carries the week's report/practice
    status (highest severity) for labeling the event."""
    inj = _injuries_reg(injuries)
    severity = {"Out": 4, "IR": 4, "Doubtful": 3, "Questionable": 2, "Note": 1}
    inj["_sev"] = inj.get("report_status").map(severity).fillna(0) if "report_status" in inj else 0
    inj = inj.sort_values(["gsis_id", "season", "week", "_sev"], ascending=False)
    inj = inj.drop_duplicates(["gsis_id", "season", "week"], keep="first")
    cols = ["gsis_id", "season", "week"]
    for c in ("report_status", "practice_status", "report_primary_injury"):
        if c in inj.columns:
            cols.append(c)
    return inj[cols].reset_index(drop=True)


def _primary_starter_per_team_season(starting_qbs: pd.DataFrame) -> pd.DataFrame:
    """The modal starting QB per (team, season) and the weeks he started."""
    s = starting_qbs.dropna(subset=["starting_qb_id"]).copy()
    counts = (
        s.groupby(["team", "season", "starting_qb_id"])["week"]
        .agg(starts="count", first_start="min")
        .reset_index()
        .sort_values(["team", "season", "starts"], ascending=[True, True, False])
        .drop_duplicates(["team", "season"], keep="first")
        .rename(columns={"starting_qb_id": "qb_id"})
    )
    return counts[["team", "season", "qb_id", "starts", "first_start"]]


def build_first_report_events(
    player_stats: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    min_pre_weeks: int = DEFAULT_MIN_PRE_WEEKS,
    min_post_weeks: int = DEFAULT_MIN_POST_WEEKS,
    starting_qbs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the first-injury-report treatment events and an eligibility report.

    One candidate event per (team, season): the first week the team's modal
    starting QB — while he is the starter — appears on the injury report. Then
    eligibility rules are applied and each drop is counted.

    Returns (events, eligibility_report). ``events`` carries the schema the
    session-1/2 panel builder expects (``event_id, team, season,
    transition_week``) plus descriptive columns.
    """
    if starting_qbs is None:
        starting_qbs = identify_starting_qb_per_team_week(player_stats)
    s = starting_qbs.dropna(subset=["starting_qb_id"]).copy()
    s["season"] = pd.to_numeric(s["season"], errors="coerce").astype(int)
    s["week"] = pd.to_numeric(s["week"], errors="coerce").astype(int)

    primary = _primary_starter_per_team_season(s)
    report = qb_injury_report_weeks(injuries)
    report_keys = report.rename(columns={"gsis_id": "qb_id"})
    last_week = s.groupby(["team", "season"])["week"].max().rename("last_team_week")

    drops = {"candidates": 0, "no_injury_report": 0, "event_week_le_min_pre": 0,
             "insufficient_pre_starts": 0, "insufficient_post_weeks": 0, "eligible": 0}
    rows = []
    for _, p in primary.iterrows():
        drops["candidates"] += 1
        team, season, qb = p["team"], int(p["season"]), p["qb_id"]
        # Weeks this QB actually started for this team-season, and his starting
        # tenure [first_start, last_start]. We treat a report anywhere inside the
        # tenure as the event — this captures both the typical play-through case
        # (he is Questionable but starts) and the onset week he first sits.
        started_weeks = set(
            s[(s.team == team) & (s.season == season) & (s.starting_qb_id == qb)]["week"]
        )
        if not started_weeks:
            drops["no_injury_report"] += 1
            continue
        ten_lo, ten_hi = min(started_weeks), max(started_weeks)
        rep = report_keys[(report_keys.qb_id == qb) & (report_keys.season == season)
                          & report_keys["week"].between(ten_lo, ten_hi)]
        if rep.empty:
            drops["no_injury_report"] += 1
            continue
        event_week = int(rep["week"].min())
        first = rep.sort_values("week").iloc[0]

        # Eligibility ----------------------------------------------------------
        if event_week <= min_pre_weeks:  # need >= min_pre_weeks pre weeks (>=wk 4 for 3)
            drops["event_week_le_min_pre"] += 1
            continue
        pre_starts = len([w for w in started_weeks if event_week - min_pre_weeks <= w < event_week])
        if pre_starts < min_pre_weeks:
            drops["insufficient_pre_starts"] += 1
            continue
        team_last = int(last_week.get((team, season), event_week))
        if team_last - event_week < min_post_weeks:
            drops["insufficient_post_weeks"] += 1
            continue

        drops["eligible"] += 1
        rows.append({
            "event_id": f"{season}_{team}_W{event_week:02d}",
            "team": team, "season": season,
            "transition_week": event_week,      # name kept for panel-builder compat
            "event_week": event_week,
            "qb_id": qb, "new_qb_id": qb, "prior_qb_id": qb,
            "first_injury_status": first.get("report_status"),
            "first_practice_status": first.get("practice_status"),
            "injury_body_part": first.get("report_primary_injury"),
            "games_started_before_event": int(sum(1 for w in started_weeks if w < event_week)),
        })

    events = pd.DataFrame(rows).reset_index(drop=True)
    elig = pd.DataFrame(
        [{"rule": k, "count": v} for k, v in drops.items()]
    )
    return events, elig


def build_out_only_events(
    player_stats: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    min_pre_weeks: int = DEFAULT_MIN_PRE_WEEKS,
    min_post_weeks: int = DEFAULT_MIN_POST_WEEKS,
    starting_qbs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Comparison set: first week the starting QB is formally Out/IR (the old,
    stricter trigger). Same eligibility, for an apples-to-apples event count."""
    inj = _injuries_reg(injuries)
    if "report_status" in inj.columns:
        inj = inj[inj["report_status"].isin(OUT_STATUSES)]
    out_report = inj[["gsis_id", "season", "week"]].drop_duplicates()
    # Reuse the same machinery by passing an injuries frame restricted to Out.
    return build_first_report_events(
        player_stats,
        out_report.rename(columns={}).assign(season_type=REGULAR_SEASON),
        min_pre_weeks=min_pre_weeks, min_post_weeks=min_post_weeks,
        starting_qbs=starting_qbs,
    )[0]


def build_clean_control_starters(
    starting_qbs: pd.DataFrame, injuries: pd.DataFrame
) -> pd.DataFrame:
    """Return a copy of ``starting_qbs`` with the starter nulled in any team-week
    where that team's starting QB was on the injury report.

    Passing this to ``construct_control_panel`` makes the control universe teams
    whose starting QB was both stable AND injury-report-free through the window —
    so a would-be-treated team can't leak into another event's control pool.
    """
    report = qb_injury_report_weeks(injuries)[["gsis_id", "season", "week"]]
    report = report.rename(columns={"gsis_id": "starting_qb_id"})
    report["_on_report"] = 1
    sq = starting_qbs.copy()
    sq["season"] = pd.to_numeric(sq["season"], errors="coerce").astype("Int64")
    sq["week"] = pd.to_numeric(sq["week"], errors="coerce").astype("Int64")
    merged = sq.merge(report, on=["starting_qb_id", "season", "week"], how="left")
    flagged = merged["_on_report"].eq(1)
    merged.loc[flagged, "starting_qb_id"] = pd.NA
    merged.loc[flagged, "starting_qb_display_name"] = pd.NA
    return merged.drop(columns=["_on_report"]).reset_index(drop=True)
