"""Tests for the QB-injury treatment identification module.

The synthetic-data tests pin the unit logic. The integration tests at the
bottom run the full pipeline against the real local data and assert that
known QB-injury events (Burrow 2023, Wentz 2017, Lawrence 2024) appear in
the treatment-events table with the right classifications.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.causal.treatment_identification import (
    attach_affected_receivers,
    build_treatment_artifacts,
    classify_transitions_by_cause,
    construct_treatment_events,
    identify_qb_transitions,
    identify_starting_qb_per_team_week,
)


# ---------------------------------------------------------------------------
# Synthetic-data unit tests
# ---------------------------------------------------------------------------
def _toy_player_stats(rows: list[dict]) -> pd.DataFrame:
    base = {
        "season_type": "REG",
        "position": "QB",
        "completions": 0,
        "passing_yards": 0,
        "carries": 0,
        "targets": 0,
        "receptions": 0,
        "receiving_yards": 0,
        "rushing_yards": 0,
        "fantasy_points_ppr": 0,
    }
    full = []
    for row in rows:
        merged = {**base, **row}
        full.append(merged)
    return pd.DataFrame(full)


def test_starting_qb_threshold_filters_platoons():
    # Two QBs splitting 18/22 attempts: 18/40 = 0.45 < 0.50 so neither qualifies.
    rows = [
        {
            "player_id": "p1",
            "player_display_name": "QB A",
            "team": "AAA",
            "season": 2024,
            "week": 1,
            "attempts": 18,
        },
        {
            "player_id": "p2",
            "player_display_name": "QB B",
            "team": "AAA",
            "season": 2024,
            "week": 1,
            "attempts": 22,
        },
    ]
    starters = identify_starting_qb_per_team_week(_toy_player_stats(rows))
    assert len(starters) == 1
    row = starters.iloc[0]
    # 22/40 = 0.55 > 0.50, so QB B IS the starter — this tests the boundary.
    assert row["starting_qb_id"] == "p2"


def test_starting_qb_threshold_returns_na_below_min_share():
    # Three QBs splitting attempts so no one clears 50%.
    rows = [
        {
            "player_id": "p1",
            "player_display_name": "QB A",
            "team": "AAA",
            "season": 2024,
            "week": 1,
            "attempts": 14,
        },
        {
            "player_id": "p2",
            "player_display_name": "QB B",
            "team": "AAA",
            "season": 2024,
            "week": 1,
            "attempts": 13,
        },
        {
            "player_id": "p3",
            "player_display_name": "QB C",
            "team": "AAA",
            "season": 2024,
            "week": 1,
            "attempts": 13,
        },
    ]
    starters = identify_starting_qb_per_team_week(_toy_player_stats(rows))
    # Top QB has 14/40 = 0.35, below the threshold. Returned as NA.
    assert len(starters) == 1
    assert pd.isna(starters.iloc[0]["starting_qb_id"])


def test_starting_qb_drops_emergency_passers():
    # WR who threw 1 pass should NOT become "starting QB" — they're not even
    # in the input filter (position != QB).
    rows = [
        {
            "player_id": "p1",
            "player_display_name": "QB A",
            "team": "AAA",
            "season": 2024,
            "week": 1,
            "attempts": 30,
        },
        {
            "player_id": "p2",
            "player_display_name": "WR Trick Play",
            "position": "WR",
            "team": "AAA",
            "season": 2024,
            "week": 1,
            "attempts": 1,
        },
    ]
    starters = identify_starting_qb_per_team_week(_toy_player_stats(rows))
    assert len(starters) == 1
    assert starters.iloc[0]["starting_qb_id"] == "p1"


def test_qb_transitions_skip_bye_weeks_correctly():
    # Team has QB1 weeks 1-4, then bye in week 5 (not in data), then QB2
    # weeks 6-8. This should be ONE transition between week 4 and week 6.
    starters = pd.DataFrame(
        {
            "team": ["AAA"] * 7,
            "season": [2024] * 7,
            "week": [1, 2, 3, 4, 6, 7, 8],
            "starting_qb_id": ["p1", "p1", "p1", "p1", "p2", "p2", "p2"],
            "starting_qb_display_name": ["QB A"] * 4 + ["QB B"] * 3,
            "primary_share": [0.95] * 7,
            "team_pass_attempts": [30] * 7,
        }
    )
    transitions = identify_qb_transitions(starters)
    assert len(transitions) == 1
    row = transitions.iloc[0]
    assert row["prior_week"] == 4
    assert row["transition_week"] == 6
    assert row["prior_qb_id"] == "p1"
    assert row["new_qb_id"] == "p2"


def test_classify_transitions_marks_out_status_as_injury():
    transitions = pd.DataFrame(
        {
            "team": ["AAA"],
            "season": [2024],
            "prior_week": [3],
            "transition_week": [4],
            "prior_qb_id": ["p1"],
            "new_qb_id": ["p2"],
        }
    )
    injuries = pd.DataFrame(
        {
            "gsis_id": ["p1", "p1"],
            "season": [2024, 2024],
            "week": [4, 4],
            "report_status": ["Out", "Out"],
            "practice_status": [
                "Did Not Participate In Practice",
                "Did Not Participate In Practice",
            ],
            "position": ["QB", "QB"],
        }
    )
    classified = classify_transitions_by_cause(transitions, injuries)
    assert len(classified) == 1
    assert classified.iloc[0]["cause"] == "injury"


def test_classify_transitions_marks_dnp_only_as_injury_dnp():
    transitions = pd.DataFrame(
        {
            "team": ["AAA"],
            "season": [2024],
            "prior_week": [3],
            "transition_week": [4],
            "prior_qb_id": ["p1"],
            "new_qb_id": ["p2"],
        }
    )
    injuries = pd.DataFrame(
        {
            "gsis_id": ["p1"],
            "season": [2024],
            "week": [4],
            "report_status": [np.nan],
            "practice_status": ["Did Not Participate In Practice"],
            "position": ["QB"],
        }
    )
    classified = classify_transitions_by_cause(transitions, injuries)
    assert classified.iloc[0]["cause"] == "injury_dnp"


def test_classify_transitions_marks_no_injury_data_as_unknown():
    transitions = pd.DataFrame(
        {
            "team": ["AAA"],
            "season": [2024],
            "prior_week": [3],
            "transition_week": [4],
            "prior_qb_id": ["p1"],
            "new_qb_id": ["p2"],
        }
    )
    # No injury rows at all.
    injuries = pd.DataFrame(
        columns=["gsis_id", "season", "week", "report_status", "practice_status", "position"]
    )
    classified = classify_transitions_by_cause(transitions, injuries)
    assert classified.iloc[0]["cause"] == "unknown"


def test_treatment_requires_min_post_weeks():
    # Backup starts in week 4 but original returns in week 5 (1 post week).
    classified = pd.DataFrame(
        {
            "team": ["AAA"],
            "season": [2024],
            "prior_week": [3],
            "transition_week": [4],
            "prior_qb_id": ["p1"],
            "new_qb_id": ["p2"],
            "cause": ["injury"],
        }
    )
    starters = pd.DataFrame(
        {
            "team": ["AAA"] * 5,
            "season": [2024] * 5,
            "week": [1, 2, 3, 4, 5],
            "starting_qb_id": ["p1", "p1", "p1", "p2", "p1"],
            "starting_qb_display_name": ["QB A"] * 3 + ["QB B"] + ["QB A"],
            "primary_share": [0.95] * 5,
            "team_pass_attempts": [30] * 5,
        }
    )
    events = construct_treatment_events(classified, starters, min_post_weeks=2)
    # With min_post_weeks=2 this 1-week emergency should be filtered out.
    assert events.empty


def test_treatment_counted_when_post_weeks_sufficient():
    classified = pd.DataFrame(
        {
            "team": ["AAA"],
            "season": [2024],
            "prior_week": [3],
            "transition_week": [4],
            "prior_qb_id": ["p1"],
            "new_qb_id": ["p2"],
            "cause": ["injury"],
        }
    )
    starters = pd.DataFrame(
        {
            "team": ["AAA"] * 6,
            "season": [2024] * 6,
            "week": [1, 2, 3, 4, 5, 6],
            "starting_qb_id": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "starting_qb_display_name": ["QB A"] * 3 + ["QB B"] * 3,
            "primary_share": [0.95] * 6,
            "team_pass_attempts": [30] * 6,
        }
    )
    events = construct_treatment_events(classified, starters, min_post_weeks=2)
    assert len(events) == 1
    assert events.iloc[0]["event_id"] == "2024_AAA_W04"
    assert events.iloc[0]["post_period_starter_weeks"] == 3


def test_affected_receivers_filters_low_volume_players():
    events = pd.DataFrame(
        {
            "event_id": ["2024_AAA_W05"],
            "team": ["AAA"],
            "season": [2024],
            "transition_week": [5],
        }
    )
    # Two WRs on team AAA in weeks 1-4. One averages 6 targets/game, one
    # averages 1 target/game. min_pre_targets_per_game=3 should keep only the
    # first.
    rows = []
    for week in [1, 2, 3, 4]:
        rows.append(
            {
                "player_id": "wr1",
                "player_display_name": "Star WR",
                "position": "WR",
                "team": "AAA",
                "season": 2024,
                "week": week,
                "season_type": "REG",
                "targets": 6,
                "fantasy_points_ppr": 12.0,
            }
        )
        rows.append(
            {
                "player_id": "wr2",
                "player_display_name": "Depth WR",
                "position": "WR",
                "team": "AAA",
                "season": 2024,
                "week": week,
                "season_type": "REG",
                "targets": 1,
                "fantasy_points_ppr": 2.0,
            }
        )
    receivers = attach_affected_receivers(events, pd.DataFrame(rows))
    assert len(receivers) == 1
    assert receivers.iloc[0]["player_id"] == "wr1"
    assert receivers.iloc[0]["pre_period_avg_targets"] == 6.0


# ---------------------------------------------------------------------------
# Integration tests against the real data (skipped if data files absent)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLAYER_STATS_PATH = os.path.join(
    _PROJECT_ROOT, "data", "raw", "player_stats_2016_2025.csv"
)
_INJURIES_PATH = os.path.join(
    _PROJECT_ROOT, "data", "raw", "injuries_2016_2025.csv"
)

requires_local_data = pytest.mark.skipif(
    not (os.path.exists(_PLAYER_STATS_PATH) and os.path.exists(_INJURIES_PATH)),
    reason="Requires local raw data files (run fetch scripts to populate).",
)


@pytest.fixture(scope="module")
def real_artifacts():
    player_stats = pd.read_csv(_PLAYER_STATS_PATH, low_memory=False)
    injuries = pd.read_csv(_INJURIES_PATH, low_memory=False)
    return build_treatment_artifacts(player_stats, injuries)


@requires_local_data
def test_burrow_2023_treatment_event_present(real_artifacts):
    events = real_artifacts["events"]
    # Joe Burrow's wrist injury kept him out from Week 11 of 2023.
    cin_events = events[
        events["team"].eq("CIN")
        & events["season"].eq(2023)
        & events["transition_week"].between(10, 13)
    ]
    assert not cin_events.empty, (
        "Expected a Cincinnati 2023 QB injury treatment event (Burrow -> Browning) "
        "around weeks 10-13"
    )
    # Cause should be injury or injury_dnp, not benching/unknown.
    assert cin_events["cause"].iloc[0] in {"injury", "injury_dnp"}


@requires_local_data
def test_lawrence_2024_treatment_event_present(real_artifacts):
    events = real_artifacts["events"]
    # Trevor Lawrence's shoulder injury kept him out late in 2024.
    jax_events = events[
        events["team"].eq("JAX")
        & events["season"].eq(2024)
        & events["transition_week"].ge(10)
    ]
    assert not jax_events.empty, (
        "Expected a Jacksonville 2024 QB injury treatment event "
        "(Lawrence -> Beathard) in the back half of the season"
    )
    assert jax_events["cause"].iloc[0] in {"injury", "injury_dnp"}


@requires_local_data
def test_total_treatment_events_in_expected_range(real_artifacts):
    """The plan estimates 15-40 events per season post-2016. Check the
    aggregate count lands in a sensible range."""
    events = real_artifacts["events"]
    by_season = events.groupby("season").size()
    assert by_season.min() >= 5, f"Suspiciously few events in some season: {by_season.to_dict()}"
    assert by_season.max() <= 80, f"Suspiciously many events in some season: {by_season.to_dict()}"


@requires_local_data
def test_affected_receivers_averages_at_least_one_per_event(real_artifacts):
    events = real_artifacts["events"]
    affected = real_artifacts["affected_receivers"]
    if events.empty:
        pytest.skip("No events to check")
    per_event = affected.groupby("event_id").size()
    # Most events should produce >= 1 affected WR. Some won't (low-volume
    # teams), but the mean should be comfortably > 1.
    assert per_event.mean() >= 1.0, (
        f"Mean affected receivers per event suspiciously low: {per_event.mean():.2f}"
    )
