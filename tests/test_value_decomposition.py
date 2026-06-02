"""Unit tests for value decomposition: rate math, z-scoring, and stability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import value_decomposition as vd


def _toy_skill_seasons() -> pd.DataFrame:
    # Two WRs with controlled rate inputs over two seasons each.
    return pd.DataFrame(
        {
            "season": [2022, 2023, 2022, 2023],
            "player_id": ["a", "a", "b", "b"],
            "player_display_name": ["A", "A", "B", "B"],
            "position": ["WR", "WR", "WR", "WR"],
            "team": ["X", "X", "Y", "Y"],
            "games_played": [16, 16, 16, 16],
            "targets": [100, 100, 40, 40],
            "receptions": [70, 70, 20, 20],
            "receiving_yards": [1000, 1000, 200, 200],
            "receiving_air_yards": [1200, 1200, 500, 500],
            "receiving_yards_after_catch": [350, 350, 100, 100],
            "carries": [0, 0, 0, 0],
            "rushing_yards": [0, 0, 0, 0],
            "attempts": [0, 0, 0, 0],
            "completions": [0, 0, 0, 0],
            "passing_yards": [0, 0, 0, 0],
            "passing_air_yards": [0, 0, 0, 0],
            "scrimmage_touches": [70, 70, 20, 20],
            "value_epa_total": [60.0, 55.0, 5.0, 8.0],
            "value_score": [1.2, 1.0, -0.8, -0.5],
        }
    )


def test_rate_features_match_manual():
    out = vd.add_talent_rate_features(_toy_skill_seasons())
    row = out.iloc[0]  # player A 2022
    assert abs(row["catch_rate"] - 0.70) < 1e-9          # 70/100
    assert abs(row["yards_per_target"] - 10.0) < 1e-9    # 1000/100
    assert abs(row["yards_per_reception"] - (1000 / 70)) < 1e-9
    assert abs(row["adot"] - 12.0) < 1e-9                # 1200/100
    assert abs(row["yac_per_reception"] - 5.0) < 1e-9    # 350/70
    assert abs(row["racr"] - (1000 / 1200)) < 1e-9


def test_rate_is_nan_when_denominator_zero():
    df = _toy_skill_seasons()
    df.loc[:, ["targets", "receptions"]] = 0
    out = vd.add_talent_rate_features(df)
    assert out["yards_per_target"].isna().all()
    assert out["catch_rate"].isna().all()


def test_opportunity_and_efficiency_zscore_properties():
    out = vd.add_opportunity_and_efficiency(_toy_skill_seasons())
    # opportunity_z standardized within season-position group -> mean ~0 per group
    grp_mean = out.groupby(["season", "position"])["opportunity_z"].mean().abs().max()
    assert grp_mean < 1e-9
    # Higher-volume player A should have higher opportunity_z than B in 2022.
    a22 = out[(out.player_id == "a") & (out.season == 2022)]["opportunity_z"].iloc[0]
    b22 = out[(out.player_id == "b") & (out.season == 2022)]["opportunity_z"].iloc[0]
    assert a22 > b22


def test_efficiency_qualification_gate():
    # Player B (40 targets) is below the WR efficiency floor (default uses
    # MIN_EFFICIENCY_OPPORTUNITY["WR"]=30, so 40 qualifies; drop to 10 to fail).
    df = _toy_skill_seasons()
    df.loc[df.player_id == "b", "scrimmage_touches"] = 10
    df.loc[df.player_id == "b", "targets"] = 10
    out = vd.add_opportunity_and_efficiency(df)
    b_rows = out[out.player_id == "b"]
    assert (~b_rows["efficiency_qualified"]).all()
    assert b_rows["efficiency_z"].isna().all()


def test_stability_analysis_runs_and_orders_axes():
    decomposed = vd.build_decomposed_player_seasons(_toy_skill_seasons())
    stab = vd.stability_analysis(decomposed)
    assert {"total_value", "efficiency", "opportunity"}.issubset(set(stab["axis"]))
    # Every reported correlation is within [-1, 1] or NaN.
    vals = stab["yoy_correlation"].dropna()
    assert ((vals >= -1.0) & (vals <= 1.0)).all()
