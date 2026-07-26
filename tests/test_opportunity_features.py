"""Tests for share-normalized opportunity features.

The subtle correctness properties matter more than the arithmetic here: shares
must come from aggregated season totals (not averaged weekly ratios), traded
players must be weighted by where they played, and the prior-season lag must
never see the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.opportunity_features import (
    add_team_opportunity_shares,
    attach_opportunity_features,
    collapse_shares_to_player_season,
)


def _team_frame() -> pd.DataFrame:
    """One team-season: 3 receivers splitting 100 targets, plus a QB."""
    return pd.DataFrame(
        {
            "season": [2022] * 4,
            "team": ["NE"] * 4,
            "player_id": ["wr1", "wr2", "rb1", "qb1"],
            "position": ["WR", "WR", "RB", "QB"],
            "targets": [50, 30, 20, 0],
            "receiving_air_yards": [700, 200, 100, 0],
            "carries": [0, 0, 80, 20],
            "games_played": [17, 17, 17, 17],
        }
    )


def test_target_share_uses_season_totals():
    out = add_team_opportunity_shares(_team_frame())
    shares = out.set_index("player_id")["target_share"]
    assert shares["wr1"] == 0.5      # 50 of 100 team targets
    assert shares["wr2"] == 0.3
    assert shares["rb1"] == 0.2


def test_air_yards_share_separates_role_from_volume():
    # wr1 and wr2 differ far more in air yards than in targets — exactly the
    # distinction (field-stretcher vs possession) raw counts cannot express.
    out = add_team_opportunity_shares(_team_frame()).set_index("player_id")
    assert out.loc["wr1", "air_yards_share"] == 0.7
    assert out.loc["wr2", "air_yards_share"] == 0.2
    assert out.loc["wr1", "target_share"] - out.loc["wr2", "target_share"] == 0.2


def test_wopr_matches_standard_formula():
    out = add_team_opportunity_shares(_team_frame()).set_index("player_id")
    expected = 1.5 * 0.5 + 0.7 * 0.7
    assert out.loc["wr1", "wopr"] == expected


def test_quarterback_receiving_shares_are_nan_not_zero():
    # A 0 would read as "had a receiving role and lost it"; QBs simply have none.
    out = add_team_opportunity_shares(_team_frame()).set_index("player_id")
    assert np.isnan(out.loc["qb1", "target_share"])
    assert np.isnan(out.loc["qb1", "wopr"])
    # Carry share is meaningful for a QB, so it survives.
    assert out.loc["qb1", "carry_share"] == 0.2


def test_zero_team_total_gives_nan_not_inf():
    df = _team_frame()
    df["targets"] = 0
    out = add_team_opportunity_shares(df)
    assert out["target_share"].isna().all()
    assert not np.isinf(out["target_share"].fillna(0)).any()


def test_traded_player_share_is_games_weighted():
    # 12 games at a 40% share, then 5 games at a 10% share.
    df = pd.DataFrame(
        {
            "season": [2022] * 4,
            "team": ["NE", "NE", "NYJ", "NYJ"],
            "player_id": ["wr1", "other_ne", "wr1", "other_nyj"],
            "position": ["WR"] * 4,
            "targets": [40, 60, 10, 90],
            "receiving_air_yards": [0, 0, 0, 0],
            "carries": [0, 0, 0, 0],
            "games_played": [12, 12, 5, 5],
        }
    )
    collapsed = collapse_shares_to_player_season(add_team_opportunity_shares(df))
    got = collapsed.set_index("player_id").loc["wr1", "target_share"]
    expected = (0.4 * 12 + 0.1 * 5) / 17
    assert abs(got - expected) < 1e-9


def test_prev_share_lag_does_not_leak_future():
    season_team = pd.DataFrame(
        {
            "season": [2021, 2021, 2022, 2022],
            "team": ["NE"] * 4,
            "player_id": ["wr1", "wr2", "wr1", "wr2"],
            "position": ["WR"] * 4,
            "targets": [20, 80, 60, 40],
            "receiving_air_yards": [0, 0, 0, 0],
            "carries": [0, 0, 0, 0],
            "games_played": [17] * 4,
        }
    )
    player_season = season_team[["season", "player_id"]].copy()
    out = attach_opportunity_features(player_season, season_team).set_index(
        ["player_id", "season"]
    )
    # 2021 has no prior season; 2022 sees 2021's share and nothing later.
    assert np.isnan(out.loc[("wr1", 2021), "target_share_prev"])
    assert out.loc[("wr1", 2022), "target_share_prev"] == 0.2
    assert out.loc[("wr1", 2022), "target_share"] == 0.6
