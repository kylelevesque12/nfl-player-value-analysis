"""Leakage and structural tests for the weekly fantasy projection module.

The pregame features for a player-week must NEVER include data from that same
game or any future game. These tests build small synthetic player-week histories
with known values and assert the rolling and lag columns are strictly pregame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.weekly_fantasy_projection import (
    add_availability_features,
    add_pregame_player_features,
    add_target,
    build_opponent_ppr_allowed,
)


def _toy_player_weeks() -> pd.DataFrame:
    # One WR, five regular-season weeks with known PPR values 10, 20, 30, 40, 50.
    return pd.DataFrame(
        {
            "player_id": ["p1"] * 5,
            "player_display_name": ["Test Player"] * 5,
            "position": ["WR"] * 5,
            "team": ["AAA"] * 5,
            "opponent_team": ["BBB", "CCC", "DDD", "EEE", "FFF"],
            "season": [2024] * 5,
            "week": [1, 2, 3, 4, 5],
            "fantasy_points_ppr": [10.0, 20.0, 30.0, 40.0, 50.0],
            "targets": [5, 6, 7, 8, 9],
            "receptions": [3, 4, 5, 6, 7],
            "carries": [0, 0, 0, 0, 0],
            "attempts": [0, 0, 0, 0, 0],
            "passing_yards": [0, 0, 0, 0, 0],
            "rushing_yards": [0, 0, 0, 0, 0],
            "receiving_yards": [40, 50, 60, 70, 80],
            "birth_date": ["1998-01-01"] * 5,
            "gameday": [
                "2024-09-08",
                "2024-09-15",
                "2024-09-22",
                "2024-09-29",
                "2024-10-06",
            ],
        }
    )


def test_ppr_last1_is_strictly_lagged():
    featured = add_pregame_player_features(_toy_player_weeks())
    featured = featured.sort_values("week").reset_index(drop=True)
    # Week 1 has no prior game -> NaN; later weeks equal the previous PPR.
    assert np.isnan(featured.loc[0, "ppr_last1"])
    assert featured.loc[1, "ppr_last1"] == 10.0
    assert featured.loc[2, "ppr_last1"] == 20.0
    assert featured.loc[3, "ppr_last1"] == 30.0
    assert featured.loc[4, "ppr_last1"] == 40.0


def test_ppr_last4_avg_excludes_current_game():
    featured = add_pregame_player_features(_toy_player_weeks())
    featured = featured.sort_values("week").reset_index(drop=True)
    # At week 5 the last4 window should be weeks 1-4: avg(10,20,30,40) = 25.
    # If the current game's 50 were leaking in it would be 30 or 35.
    assert featured.loc[4, "ppr_last4_avg"] == 25.0
    # At week 4 it should be weeks 1-3: avg(10,20,30) = 20.
    assert featured.loc[3, "ppr_last4_avg"] == 20.0
    # Week 1 has no prior games at all.
    assert np.isnan(featured.loc[0, "ppr_last4_avg"])


def test_season_to_date_avg_excludes_current_game():
    featured = add_pregame_player_features(_toy_player_weeks())
    featured = featured.sort_values("week").reset_index(drop=True)
    # Season-to-date at week 5: mean of weeks 1-4 = 25.
    assert featured.loc[4, "ppr_season_to_date_avg"] == 25.0
    # Season-to-date at week 1 is NaN (no prior games this season).
    assert np.isnan(featured.loc[0, "ppr_season_to_date_avg"])


def test_targets_last4_avg_excludes_current_game():
    featured = add_pregame_player_features(_toy_player_weeks())
    featured = featured.sort_values("week").reset_index(drop=True)
    # Week 5 last4 targets window covers weeks 1-4: mean(5,6,7,8) = 6.5.
    assert featured.loc[4, "targets_last4_avg"] == 6.5


def test_target_is_current_game_ppr():
    # The target for each player-week is the *current* game's PPR. This is the
    # ESPN/DFS-style framing: project this Sunday's PPR using everything we knew
    # before kickoff. Pregame features must already be shift(1)-lagged so the
    # current game's stats never enter its own features. The combination yields
    # a model that learns from prior-game features to predict the current game.
    featured = add_pregame_player_features(_toy_player_weeks())
    with_target = add_target(featured)
    with_target = with_target.sort_values("week").reset_index(drop=True)
    assert with_target.loc[0, "target_fantasy_points_ppr"] == 10.0
    assert with_target.loc[4, "target_fantasy_points_ppr"] == 50.0
    # And the pregame feature at the same row is strictly prior:
    assert with_target.loc[4, "ppr_last4_avg"] == 25.0  # mean of weeks 1-4


def _toy_schedules() -> pd.DataFrame:
    # Team AAA plays in every week 1-5; team OPP1/.../OPP5 are just opponents.
    rows = [
        {
            "season": 2024,
            "week": w,
            "game_type": "REG",
            "home_team": "AAA",
            "away_team": f"OPP{w}",
        }
        for w in range(1, 6)
    ]
    return pd.DataFrame(rows)


def test_availability_features_count_missed_games():
    # Player appeared in weeks 1, 2, 4 only. Team AAA played all 5 weeks.
    # At week 5 (the row we project), the team's last 4 games were weeks 1-4,
    # of which the player appeared in 3 (weeks 1, 2, 4) and missed 1 (week 3).
    player_stats = pd.DataFrame(
        {
            "player_id": ["p1", "p1", "p1"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 4],
            "team": ["AAA", "AAA", "AAA"],
            "season_type": ["REG"] * 3,
            "position": ["WR"] * 3,
            "fantasy_points_ppr": [10.0, 20.0, 30.0],
        }
    )
    # We're projecting week 5 — that row is in featured but the player has not
    # yet played it. Construct featured manually to match what the pipeline
    # would pass in.
    featured = pd.DataFrame(
        {
            "player_id": ["p1"],
            "team": ["AAA"],
            "season": [2024],
            "week": [5],
        }
    )
    result = add_availability_features(featured, player_stats, _toy_schedules())
    row = result.iloc[0]
    # Rolling window at week 5 covers prior team games (weeks 1-4) with
    # appeared = [1, 1, 0, 1]. So 3 active, 1 missed; last team game (week 4)
    # was played; streak ending before week 5 is just the week-4 appearance (1).
    assert row["active_games_last4"] == 3.0
    assert row["weeks_missed_last4"] == 1.0
    assert row["active_last_game"] == 1.0
    assert row["consecutive_games_active"] == 1.0


def test_availability_does_not_leak_current_game():
    # A player who is about to play week 4 should NOT have their week-4 appearance
    # counted in the availability features for week 4.
    player_stats = pd.DataFrame(
        {
            "player_id": ["p1", "p1", "p1", "p1"],
            "season": [2024] * 4,
            "week": [1, 2, 3, 4],
            "team": ["AAA"] * 4,
            "season_type": ["REG"] * 4,
            "position": ["WR"] * 4,
            "fantasy_points_ppr": [10.0, 20.0, 30.0, 40.0],
        }
    )
    featured = pd.DataFrame(
        {
            "player_id": ["p1"],
            "team": ["AAA"],
            "season": [2024],
            "week": [4],
        }
    )
    result = add_availability_features(featured, player_stats, _toy_schedules())
    row = result.iloc[0]
    # Window at week 4 is prior 4 weeks; only weeks 1-3 exist. All 3 appeared.
    assert row["active_games_last4"] == 3.0
    assert row["weeks_missed_last4"] == 0.0
    assert row["consecutive_games_active"] == 3.0  # weeks 1,2,3 in a row


def test_opponent_ppr_allowed_excludes_current_game():
    # Build a tiny history where defense BBB has been scored on for known PPR
    # totals in weeks 1, 2, 3, then we check the rolling pregame average at
    # week 4. Two WRs always face the same defenses in lockstep so we can hand-
    # compute the expected allowed totals.
    rows = []
    for week, totals in enumerate(
        [
            ("BBB", 10.0, 5.0),  # week 1, allowed = 15
            ("BBB", 20.0, 10.0),  # week 2, allowed = 30
            ("BBB", 30.0, 5.0),  # week 3, allowed = 35
            ("BBB", 0.0, 0.0),  # week 4, current game; should NOT enter feature
        ],
        start=1,
    ):
        def_team, p1_pts, p2_pts = totals
        rows.append(
            {
                "season": 2024,
                "week": week,
                "team": "AAA",
                "opponent_team": def_team,
                "position": "WR",
                "fantasy_points_ppr": p1_pts,
            }
        )
        rows.append(
            {
                "season": 2024,
                "week": week,
                "team": "AAA",
                "opponent_team": def_team,
                "position": "WR",
                "fantasy_points_ppr": p2_pts,
            }
        )
    weekly = pd.DataFrame(rows)

    opp = build_opponent_ppr_allowed(weekly)
    week4 = opp[(opp["def_team"] == "BBB") & (opp["week"] == 4)]
    # last4 average of weeks 1-3 allowed (15, 30, 35) = 80 / 3.
    assert not week4.empty
    assert week4.iloc[0]["opp_ppr_allowed_last4_avg"] == (15.0 + 30.0 + 35.0) / 3.0
