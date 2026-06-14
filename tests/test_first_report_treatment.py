"""Tests for Causal Session 3: first-injury-report treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.load_data import find_project_root
from src.causal.treatment_identification import (
    attach_affected_receivers,
    identify_starting_qb_per_team_week,
)
from src.causal.control_matching import construct_control_panel
from src.causal.did_estimator import fit_event_study, summarize_att
from src.causal.first_report_treatment import (
    build_clean_control_starters,
    build_first_report_events,
    build_out_only_events,
    qb_injury_report_weeks,
)

PRE, POST = 3, 3


@pytest.fixture(scope="module")
def artifacts():
    root = find_project_root()
    ps = pd.read_csv(root / "data/raw/player_stats_2016_2025.csv", low_memory=False)
    inj = pd.read_csv(root / "data/raw/injuries_2016_2025.csv", low_memory=False)
    sq = identify_starting_qb_per_team_week(ps)
    events, elig = build_first_report_events(ps, inj, min_pre_weeks=PRE, starting_qbs=sq)
    out_only = build_out_only_events(ps, inj, min_pre_weeks=PRE, starting_qbs=sq)
    affected = attach_affected_receivers(events, ps, pre_period_length=PRE, min_pre_targets_per_game=3.0)
    clean = build_clean_control_starters(sq, inj)
    panel = construct_control_panel(events, affected, ps, clean, pre_period_length=PRE, post_period_length=POST)
    return dict(sq=sq, events=events, elig=elig, out_only=out_only, panel=panel, inj=inj)


def test_one_treated_event_per_team_season(artifacts):
    ev = artifacts["events"]
    assert not ev.duplicated(["team", "season"]).any()
    assert not ev.duplicated(["qb_id", "team", "season"]).any()
    assert ev["event_id"].is_unique


def test_event_week_has_minimum_pre_period(artifacts):
    ev = artifacts["events"]
    # Need >= PRE pre weeks, so the event can't be in the first PRE weeks.
    assert (ev["event_week"] > PRE).all()
    # And the QB must actually have started enough pre-period games.
    assert (ev["games_started_before_event"] >= PRE).all()


def test_first_report_on_or_before_first_out(artifacts):
    ev = artifacts["events"][["team", "season", "event_week"]].rename(columns={"event_week": "first_week"})
    out = artifacts["out_only"][["team", "season", "event_week"]].rename(columns={"event_week": "out_week"})
    both = ev.merge(out, on=["team", "season"], how="inner")
    assert len(both) > 0
    # First injury-report week must be <= first Out week wherever both exist.
    assert (both["first_week"] <= both["out_week"]).all()
    # And the earlier definition yields strictly more events.
    assert len(artifacts["events"]) > len(artifacts["out_only"])


def test_clean_control_starters_null_out_reported_qbs():
    # Synthetic: QB 'q1' starts AAA weeks 1-3, on the injury report week 2.
    sq = pd.DataFrame({
        "team": ["AAA"] * 3, "season": [2022] * 3, "week": [1, 2, 3],
        "starting_qb_id": ["q1", "q1", "q1"],
        "starting_qb_display_name": ["Q One"] * 3,
    })
    inj = pd.DataFrame({
        "gsis_id": ["q1"], "season": [2022], "week": [2],
        "game_type": ["REG"], "report_status": ["Questionable"],
        "practice_status": ["Limited Participation in Practice"],
    })
    clean = build_clean_control_starters(sq, inj)
    by_week = clean.set_index("week")["starting_qb_id"]
    assert pd.isna(by_week.loc[2])          # week on the report -> nulled
    assert by_week.loc[1] == "q1"           # clean weeks retained
    assert by_week.loc[3] == "q1"


def test_event_time_offsets_are_correct(artifacts):
    panel = artifacts["panel"]
    # week_offset must equal week minus the event (transition) week, everywhere.
    recomputed = panel["week"].astype(int) - panel["transition_week"].astype(int)
    assert (panel["week_offset"].astype(int) == recomputed).all()
    # Pre rows are strictly negative offsets; offsets span the configured window.
    assert panel["week_offset"].min() >= -PRE
    assert panel["week_offset"].max() <= POST - 1
    assert (panel.loc[panel["period"].eq("pre"), "week_offset"] < 0).all()


def test_controls_are_injury_report_free(artifacts):
    """No control receiver's team should have had its starting QB on the injury
    report during that event's window (the clean-control guarantee)."""
    panel = artifacts["panel"]
    sq = artifacts["sq"].copy()
    report = qb_injury_report_weeks(artifacts["inj"])[["gsis_id", "season", "week"]]
    report = report.rename(columns={"gsis_id": "starting_qb_id"}).assign(_rep=1)
    sq["season"] = pd.to_numeric(sq["season"], errors="coerce")
    sq["week"] = pd.to_numeric(sq["week"], errors="coerce")
    flagged = sq.merge(report, on=["starting_qb_id", "season", "week"], how="left")
    bad_team_weeks = set(
        map(tuple, flagged.loc[flagged["_rep"].eq(1), ["team", "season", "week"]]
            .dropna().astype({"season": int, "week": int}).values)
    )
    ctrl = panel[panel["role"].eq("control")]
    ctrl_team_weeks = set(map(tuple, ctrl[["team", "season", "week"]].astype({"season": int, "week": int}).values))
    assert ctrl_team_weeks.isdisjoint(bad_team_weeks)


def test_session3_pipeline_runs_and_returns_finite_estimate(artifacts):
    panel = artifacts["panel"]
    att = summarize_att(fit_event_study(panel))
    assert not att.empty
    val = float(att.iloc[0]["att_pooled_post_period"])
    assert np.isfinite(val)
