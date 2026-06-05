"""Tests for the structurally-constrained two-stage weekly module.

The key invariant to pin: after renormalization, raw target-share predictions
must sum to 1 within each (team, season, week). This is the structural
property the model is built around.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.two_stage_weekly import (
    _renormalize_target_share,
    build_two_stage_frame,
)


def test_renormalize_target_share_sums_to_one_per_team_week():
    valid = pd.DataFrame(
        {
            "team": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "season": [2024, 2024, 2024, 2024, 2024],
            "week": [1, 1, 1, 1, 1],
        }
    )
    raw = np.array([0.6, 0.4, 0.2, 0.7, 0.3])
    renormalized = _renormalize_target_share(valid, raw)
    # Team AAA: 0.6 / 1.2, 0.4 / 1.2, 0.2 / 1.2 → sum to 1.
    assert abs(renormalized[:3].sum() - 1.0) < 1e-9
    # Team BBB: 0.7 / 1.0, 0.3 / 1.0 → sum to 1.
    assert abs(renormalized[3:].sum() - 1.0) < 1e-9


def test_renormalize_zero_team_predictions_stays_zero():
    valid = pd.DataFrame(
        {"team": ["AAA", "AAA"], "season": [2024, 2024], "week": [1, 1]}
    )
    raw = np.array([0.0, 0.0])
    renormalized = _renormalize_target_share(valid, raw)
    # Sum was zero — we leave it zero rather than divide by zero.
    assert (renormalized == 0.0).all()


def test_renormalize_negative_raw_predictions_clamped_then_normalized():
    valid = pd.DataFrame(
        {"team": ["AAA", "AAA"], "season": [2024, 2024], "week": [1, 1]}
    )
    raw = np.array([-0.1, 0.5])  # negative gets clamped to 0
    renormalized = _renormalize_target_share(valid, raw)
    # After clamp: 0.0 + 0.5 = 0.5 sum. Renormalized: 0.0 + 1.0 = 1.0.
    assert renormalized[0] == 0.0
    assert renormalized[1] == 1.0
    assert abs(renormalized.sum() - 1.0) < 1e-9


def test_renormalize_handles_nan_raw_predictions():
    valid = pd.DataFrame(
        {"team": ["AAA", "AAA"], "season": [2024, 2024], "week": [1, 1]}
    )
    raw = np.array([np.nan, 0.5])  # nan is treated as 0 before normalization
    renormalized = _renormalize_target_share(valid, raw)
    assert renormalized[0] == 0.0
    assert renormalized[1] == 1.0


def test_build_two_stage_frame_includes_per_stage_targets():
    # Minimal synthetic player_stats / schedules / rosters wide enough to
    # exercise build_two_stage_frame.
    player_stats = pd.DataFrame(
        {
            "player_id": ["p1", "p2", "p3"],
            "player_display_name": ["WR One", "WR Two", "TE One"],
            "position": ["WR", "WR", "TE"],
            "season": [2024, 2024, 2024],
            "week": [1, 1, 1],
            "season_type": ["REG"] * 3,
            "team": ["AAA", "AAA", "AAA"],
            "opponent_team": ["BBB", "BBB", "BBB"],
            "fantasy_points_ppr": [15.0, 8.0, 7.0],
            "targets": [8, 5, 2],
            "receptions": [6, 4, 1],
            "carries": [0, 0, 0],
            "attempts": [0, 0, 0],
            "passing_yards": [0, 0, 0],
            "rushing_yards": [0, 0, 0],
            "receiving_yards": [85, 50, 20],
        }
    )
    schedules = pd.DataFrame(
        {
            "game_id": ["2024_01_AAA_BBB"],
            "season": [2024],
            "week": [1],
            "game_type": ["REG"],
            "gameday": ["2024-09-08"],
            "home_team": ["AAA"],
            "away_team": ["BBB"],
            "home_score": [27],
            "away_score": [14],
            "home_rest": [7],
            "away_rest": [7],
            "spread_line": [-3.0],
            "total_line": [45.0],
            "div_game": [0],
        }
    )
    rosters = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "gsis_id": ["p1", "p2", "p3"],
            "position": ["WR", "WR", "TE"],
            "birth_date": ["1998-01-01", "1996-01-01", "1995-01-01"],
        }
    )
    frame = build_two_stage_frame(player_stats, schedules, rosters)
    assert len(frame) == 3
    # team_targets = 8 + 5 + 2 = 15; share for player p1 = 8/15
    p1 = frame[frame["player_id"].eq("p1")].iloc[0]
    assert p1["team_targets"] == 15
    assert abs(p1["target_share"] - 8 / 15) < 1e-12
