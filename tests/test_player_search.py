"""Tests for the Session 10 global player search + detail assembly (pure)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app import player_search as ps


@pytest.fixture
def frames():
    # Two players: p1 (WR, MIN) with weekly+live+salary; p2 (QB, GB) rookie+causal.
    weekly = pd.DataFrame({
        "player_id": ["p1", "p1", "p1", "p1", "p2", "p2"],
        "player_display_name": ["Player One"] * 4 + ["Player Two"] * 2,
        "position": ["WR"] * 4 + ["QB"] * 2,
        "team": ["MIN"] * 4 + ["GB"] * 2,
        "season": [2024, 2024, 2025, 2025, 2025, 2025],
        "week": [1, 1, 2, 2, 3, 3],
        # Two methods per (season, week) — the assembler must keep only the model.
        "method": ["hist_gradient_boosting", "recent_4_avg"] * 3,
        "prediction": [12.0, 10.0, 14.0, 11.0, 18.0, 16.0],
        "target_fantasy_points_ppr": [13.0, 13.0, 9.0, 9.0, 20.0, 20.0],
    })
    live = pd.DataFrame({
        "player_id": ["p1"], "player_display_name": ["Player One"],
        "position": ["WR"], "team": ["MIN"], "season": [2025], "week": [3],
        "projected_points": [15.2], "interval_low_80": [6.0], "interval_high_80": [24.0],
        "opponent_team": ["CHI"],
    })
    salary = pd.DataFrame({
        "player_id": ["p1", "p1", "p2"],
        "player_display_name": ["Player One", "Player One", "Player Two"],
        "position": ["WR", "WR", "QB"], "team": ["MIN", "MIN", "GB"],
        "season": [2023, 2024, 2024], "salary_millions": [1.2, 18.0, 44.0],
        "value_score": [2.1, 3.0, 1.5], "value_above_expected_salary": [5.0, 9.0, -3.0],
    })
    rookie = pd.DataFrame({
        "player_id": ["p2"], "player_display_name": ["Player Two"], "position": ["QB"],
        "rookie_year": [2020], "draft_number": [26], "played_meaningfully": [0],
    })
    causal = pd.DataFrame({
        "qb_id": ["p2"], "team": ["GB"], "season": [2024], "event_week": [6],
        "first_injury_status": ["Questionable"], "games_started_before_event": [5],
    })
    return dict(weekly=weekly, live=live, salary=salary, rookie=rookie, causal=causal)


def test_player_index_has_unique_keys_and_flags(frames):
    idx = ps.build_player_index(**frames)
    assert idx["player_id"].is_unique
    assert set(idx["player_id"]) == {"p1", "p2"}
    p1 = idx[idx["player_id"] == "p1"].iloc[0]
    p2 = idx[idx["player_id"] == "p2"].iloc[0]
    assert p1["has_weekly"] and p1["has_live"] and p1["has_surplus"]
    assert not p1["has_rookie"] and not p1["has_causal"]
    assert p2["has_rookie"] and p2["has_causal"] and not p2["has_live"]


def test_search_is_case_insensitive_and_substring(frames):
    idx = ps.build_player_index(**frames)
    lower = ps.search_players(idx, "player one")
    upper = ps.search_players(idx, "PLAYER ONE")
    sub = ps.search_players(idx, "two")
    assert len(lower) == 1 and len(upper) == 1
    assert lower["player_id"].iloc[0] == "p1"
    assert sub["player_id"].iloc[0] == "p2"


def test_search_returns_expected_display_labels(frames):
    idx = ps.build_player_index(**frames)
    hit = ps.search_players(idx, "one").iloc[0]
    assert hit["label"] == "Player One · WR · MIN"


def test_selecting_a_player_resolves_to_stable_id(frames):
    idx = ps.build_player_index(**frames)
    hit = ps.search_players(idx, "player two").iloc[0]
    assert hit["player_id"] == "p2"          # stable gsis id, not the name


def test_detail_handles_missing_sections(frames):
    d = ps.assemble_player_detail("ghost", **frames)
    for key in ("weekly_history", "live", "surplus_history", "rookie", "causal", "rookie_pred"):
        assert d[key] is None
    # A player with only some modules still assembles cleanly.
    p1 = ps.assemble_player_detail("p1", **frames)
    assert p1["weekly_history"] is not None and p1["live"] is not None
    assert p1["rookie"] is None and p1["causal"] is None


def test_weekly_summary_has_no_duplicate_player_weeks(frames):
    d = ps.assemble_player_detail("p1", **frames)
    wk = d["weekly_history"]
    # The two-method input must collapse to one model row per (player, season, week).
    assert not wk.duplicated(["player_id", "season", "week"]).any()
    assert (wk["method"] == "hist_gradient_boosting").all()


def test_surplus_summary_has_no_duplicate_player_seasons(frames):
    d = ps.assemble_player_detail("p1", **frames)
    sal = d["surplus_history"]
    assert not sal.duplicated(["player_id", "season"]).any()


def test_causal_section_keyed_on_qb_id(frames):
    d = ps.assemble_player_detail("p2", **frames)
    assert d["causal"] is not None and len(d["causal"]) == 1
    # p1 is not a treated QB.
    assert ps.assemble_player_detail("p1", **frames)["causal"] is None


def test_nav_labels_still_match_app_source():
    src = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text()
    assert 'NAV_PLAYER = "Player Detail"' in src
    assert "render_player_search(player_index)" in src
    assert "player_detail_page(data, player_index)" in src
    # Single-section navigation routes through nav_section.
    assert 'key="nav_section"' in src and "Draft Board" in src


def test_empty_index_search_does_not_crash():
    empty = ps.build_player_index()
    assert empty.empty
    assert ps.search_players(empty, "anyone").empty
