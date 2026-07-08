"""Tests for the injury-return feature block on the season fantasy model.

The point of these features is to separate "hurt" from "washed": a strong
per-game player who missed half a season to injury should not project like a
backup just because his season TOTAL is low. The interaction term is the piece
a linear model cannot build for itself, so it is checked directly. Every
feature is computed on the current-season row (the model input), so a leakage
test confirms it never reads the season being projected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fantasy_projection import (
    FANTASY_FEATURES,
    attach_injury_return_features,
    build_player_season_injury_summary,
    create_fantasy_modeling_frame,
)


def _injuries_fixture() -> pd.DataFrame:
    # Player A: on the report weeks 5, 6, 9 in 2023 (Out weeks 5, 6).
    # Player B: never on the report.
    rows = []
    for wk, status in [(5, "Out"), (6, "Out"), (9, "Questionable"), (9, "Questionable")]:
        rows.append({"gsis_id": "A", "season": 2023, "week": wk, "report_status": status})
    rows.append({"gsis_id": "A", "season": 2023, "week": 1, "report_status": np.nan})
    return pd.DataFrame(rows)


def test_injury_summary_counts_report_and_out_weeks():
    summary = build_player_season_injury_summary(_injuries_fixture())
    row = summary.set_index(["player_id", "season"]).loc[("A", 2023)]
    # Distinct reported weeks (5, 6, 9) = 3; week 1 had a NaN status.
    assert row["injury_report_weeks"] == 3
    # Out/Doubtful weeks (5, 6) = 2.
    assert row["injury_out_weeks"] == 2


def test_injury_summary_empty_or_missing_id_is_safe():
    assert build_player_season_injury_summary(pd.DataFrame()).empty
    no_id = pd.DataFrame({"season": [2023], "week": [1], "report_status": ["Out"]})
    assert build_player_season_injury_summary(no_id).empty


def test_games_missed_uses_correct_season_length():
    df = pd.DataFrame(
        {
            "player_id": ["A", "B", "C"],
            "season": [2019, 2023, 2023],
            "games_played": [12, 4, 17],
            "fantasy_points_ppr_per_game": [10.0, 15.0, 12.0],
        }
    )
    out = attach_injury_return_features(df, injury_summary=None)
    # 2019 season is 16 games: 16 - 12 = 4 missed. 2023 is 17: 17 - 4 = 13.
    assert out.set_index("player_id").loc["A", "games_missed"] == 4
    assert out.set_index("player_id").loc["B", "games_missed"] == 13
    assert out.set_index("player_id").loc["C", "games_missed"] == 0


def test_interaction_term_flags_the_healthy_but_injured_player():
    df = pd.DataFrame(
        {
            "player_id": ["Star", "Backup"],
            "season": [2023, 2023],
            "games_played": [4, 4],
            # Same low games total, wildly different per-game quality.
            "fantasy_points_ppr_per_game": [18.0, 4.0],
        }
    )
    out = attach_injury_return_features(df, injury_summary=None).set_index("player_id")
    # Both missed 13 games, but the interaction separates them: the star's
    # bounce-back signal is 18*13 vs the backup's 4*13.
    assert out.loc["Star", "ppr_per_game_x_games_missed"] == 18.0 * 13
    assert out.loc["Backup", "ppr_per_game_x_games_missed"] == 4.0 * 13
    assert (
        out.loc["Star", "ppr_per_game_x_games_missed"]
        > out.loc["Backup", "ppr_per_game_x_games_missed"]
    )


def test_injury_counts_zero_fill_without_a_feed():
    df = pd.DataFrame(
        {
            "player_id": ["A"],
            "season": [2023],
            "games_played": [4],
            "fantasy_points_ppr_per_game": [15.0],
        }
    )
    out = attach_injury_return_features(df, injury_summary=None)
    assert out["injury_report_weeks"].eq(0).all()
    assert out["injury_out_weeks"].eq(0).all()


def test_injury_counts_join_when_summary_present():
    df = pd.DataFrame(
        {
            "player_id": ["A", "B"],
            "season": [2023, 2023],
            "games_played": [4, 16],
            "fantasy_points_ppr_per_game": [15.0, 10.0],
        }
    )
    summary = build_player_season_injury_summary(_injuries_fixture())
    out = attach_injury_return_features(df, injury_summary=summary).set_index("player_id")
    assert out.loc["A", "injury_report_weeks"] == 3
    assert out.loc["A", "injury_out_weeks"] == 2
    # Player B had no injury rows -> zero-filled, not NaN.
    assert out.loc["B", "injury_report_weeks"] == 0


def test_injury_features_were_tested_and_dropped_from_production():
    # They were evaluated and did not clear the ablation threshold, so they
    # must NOT be in the production feature list (see the module comment and
    # report/fantasy/injury_return_features.md). The helpers stay available
    # for the eval and the UI flag, but the model does not consume them.
    for col in (
        "games_missed",
        "injury_report_weeks",
        "injury_out_weeks",
        "ppr_per_game_x_games_missed",
    ):
        assert col not in FANTASY_FEATURES


def test_injury_features_are_leakage_safe():
    """The interaction for season N must depend only on season N's own games
    and per-game rate — never on season N+1. Changing a player's FUTURE season
    must leave his current-season injury features untouched."""
    base = pd.DataFrame(
        {
            "player_id": ["A", "A"],
            "player_display_name": ["A", "A"],
            "position": ["WR", "WR"],
            "season": [2022, 2023],
            "games_played": [4, 17],
            "fantasy_points_ppr": [60.0, 200.0],
            "targets": [40, 150],
            "receptions": [30, 100],
            "carries": [0, 0],
            "value_score": [0.5, 1.0],
            "value_epa_total": [10.0, 30.0],
            "value_epa_per_game": [2.5, 1.8],
            "primary_team": ["NYG", "NYG"],
            "age": [23, 24],
            "years_exp": [1, 2],
            "draft_number": [6, 6],
        }
    )
    # attach after the per-game feature exists
    from src.fantasy_projection import add_fantasy_history_features

    featured = add_fantasy_history_features(base)
    out = attach_injury_return_features(featured).sort_values("season")
    injured_year = out[out["season"].eq(2022)].iloc[0]
    # 2022: 4 games in a 17-game season, 15 PPG -> interaction 15*13 = 195.
    assert injured_year["games_missed"] == 13
    assert injured_year["ppr_per_game_x_games_missed"] == 15.0 * 13
