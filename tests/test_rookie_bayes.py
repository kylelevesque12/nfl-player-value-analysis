"""Tests for the non-PyMC pieces of the Bayesian rookie module.

PyMC is intentionally not importable in the main project venv, so this file
covers the modeling-frame construction, height/age parsing, and the
training-fold standardization. The PyMC fit/predict are covered by manual
execution in the dedicated bayes venv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rookie_bayes import (
    UNDRAFTED_PICK_NUMBER,
    _build_rookie_player_season_targets,
    _height_to_inches,
    _safe_age_at_draft,
    build_rookie_modeling_frame,
    standardize_features,
)


def test_height_to_inches_parses_feet_dash_inches_and_pure_inches():
    height = pd.Series(["6-2", "5-11", "74", "", None, "6-0"])
    inches = _height_to_inches(height)
    assert inches.iloc[0] == 74.0
    assert inches.iloc[1] == 71.0
    assert inches.iloc[2] == 74.0
    assert pd.isna(inches.iloc[3])
    assert pd.isna(inches.iloc[4])
    assert inches.iloc[5] == 72.0


def test_age_at_draft_uses_mid_april():
    birth = pd.Series(["2000-04-15"])
    draft_year = pd.Series([2022])
    age = _safe_age_at_draft(birth, draft_year)
    # Exactly 22 years old on April 15, 2022.
    assert abs(age.iloc[0] - 22.0) < 0.01


def test_player_season_targets_collapses_weekly_rows():
    weekly = pd.DataFrame(
        {
            "player_id": ["p1"] * 3 + ["p2"] * 2,
            "season": [2022, 2022, 2022, 2022, 2022],
            "week": [1, 2, 3, 1, 2],
            "season_type": ["REG"] * 5,
            "fantasy_points_ppr": [10.0, 20.0, 30.0, 5.0, 15.0],
        }
    )
    agg = _build_rookie_player_season_targets(weekly)
    p1 = agg[agg["player_id"].eq("p1")].iloc[0]
    p2 = agg[agg["player_id"].eq("p2")].iloc[0]
    assert p1["season_ppr_total"] == 60.0
    assert p1["games_played"] == 3
    assert p1["season_ppr_per_game"] == 20.0
    assert p2["season_ppr_total"] == 20.0
    assert p2["games_played"] == 2
    assert p2["season_ppr_per_game"] == 10.0


def test_modeling_frame_includes_only_rookie_year_rows():
    rosters = pd.DataFrame(
        {
            "gsis_id": ["p1", "p1", "p2", "p2"],
            "full_name": ["Test One", "Test One", "Test Two", "Test Two"],
            "position": ["WR", "WR", "RB", "RB"],
            "season": [2022, 2023, 2021, 2022],
            "rookie_year": [2022, 2022, 2021, 2021],
            "draft_number": [50, 50, 175, 175],
            "birth_date": ["2000-01-01", "2000-01-01", "1999-06-15", "1999-06-15"],
            "height": ["6-1", "6-1", "5-9", "5-9"],
            "weight": [200, 200, 215, 215],
            "college": ["LSU", "LSU", "Alabama", "Alabama"],
            "draft_club": ["DAL", "DAL", "PIT", "PIT"],
            "entry_year": [2022, 2022, 2021, 2021],
        }
    )
    weekly = pd.DataFrame(
        {
            "player_id": ["p1"] * 5 + ["p2"] * 4,
            "season": [2022] * 5 + [2021] * 4,
            "week": [1, 2, 3, 4, 5, 1, 2, 3, 4],
            "season_type": ["REG"] * 9,
            "fantasy_points_ppr": [10.0, 12.0, 11.0, 13.0, 14.0, 5.0, 7.0, 6.0, 8.0],
        }
    )
    frame = build_rookie_modeling_frame(rosters, weekly)
    # Only one row per player (their rookie season).
    assert len(frame) == 2
    # Rookie-year-only filter applied: p1's 2023 row is dropped.
    assert frame["rookie_year"].tolist() == [2022, 2021]
    p1 = frame[frame["player_id"].eq("p1")].iloc[0]
    p2 = frame[frame["player_id"].eq("p2")].iloc[0]
    assert p1["season_ppr_per_game"] == 12.0
    assert p2["season_ppr_per_game"] == 6.5
    assert p1["height_inches"] == 73.0
    assert p2["height_inches"] == 69.0
    assert p1["draft_log"] == np.log(50)
    assert p2["draft_log"] == np.log(175)


def test_modeling_frame_imputes_undrafted_pick():
    rosters = pd.DataFrame(
        {
            "gsis_id": ["p1"],
            "full_name": ["UDFA Guy"],
            "position": ["WR"],
            "season": [2022],
            "rookie_year": [2022],
            "draft_number": [np.nan],
            "birth_date": ["2000-01-01"],
            "height": ["6-0"],
            "weight": [195],
            "college": ["Wyoming"],
            "draft_club": [None],
            "entry_year": [2022],
        }
    )
    weekly = pd.DataFrame(
        {
            "player_id": ["p1"],
            "season": [2022],
            "week": [1],
            "season_type": ["REG"],
            "fantasy_points_ppr": [3.0],
        }
    )
    frame = build_rookie_modeling_frame(rosters, weekly)
    assert frame.iloc[0]["draft_number"] == UNDRAFTED_PICK_NUMBER
    assert frame.iloc[0]["draft_log"] == np.log(UNDRAFTED_PICK_NUMBER)


def test_standardize_features_uses_training_stats_only():
    train = pd.DataFrame(
        {
            "player_id": ["a", "b", "c", "d"],
            "position": ["WR"] * 4,
            "draft_log": [1.0, 2.0, 3.0, 4.0],
            "age_at_draft": [22.0, 23.0, 22.0, 21.0],
            "height_inches": [72.0, 73.0, 74.0, 75.0],
            "weight": [200.0, 210.0, 220.0, 230.0],
            "season_ppr_per_game": [10.0, 12.0, 14.0, 16.0],
            "games_played": [16, 16, 16, 16],
        }
    )
    # Test has a clearly out-of-distribution player to verify scaling.
    test = pd.DataFrame(
        {
            "player_id": ["x"],
            "position": ["WR"],
            "draft_log": [5.0],  # outside train range
            "age_at_draft": [25.0],
            "height_inches": [80.0],
            "weight": [250.0],
            "season_ppr_per_game": [np.nan],
            "games_played": [0],
        }
    )
    train_z, test_z, stats = standardize_features(train, test)
    train_draft_mu, train_draft_sd = stats["draft_log"]
    assert train_draft_mu == 2.5
    # draft_log std of [1, 2, 3, 4] = ~1.291
    assert abs(train_draft_sd - np.std([1, 2, 3, 4], ddof=1)) < 1e-12
    # Test row's z-score uses train mean/std.
    expected = (5.0 - train_draft_mu) / train_draft_sd
    assert abs(test_z.iloc[0]["draft_log_z"] - expected) < 1e-12
