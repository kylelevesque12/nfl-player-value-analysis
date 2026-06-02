"""Unit tests for two-stage value model, Stage 1 (opportunity).

These cover the leakage-safety of the opportunity history features and target,
and the persistence baseline — all pandas/numpy only (no model training).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import two_stage_value as ts


def _toy() -> pd.DataFrame:
    # One player, four consecutive seasons with known opportunity per game.
    return pd.DataFrame(
        {
            "player_id": ["p1", "p1", "p1", "p1"],
            "season": [2018, 2019, 2020, 2021],
            "position": ["RB", "RB", "RB", "RB"],
            "opportunity_per_game": [10.0, 12.0, 14.0, 16.0],
            "games_played": [16, 16, 16, 16],
        }
    )


def test_history_prev_is_strictly_lagged():
    out = ts.add_opportunity_history_features(_toy())
    out = out.sort_values("season").reset_index(drop=True)
    assert np.isnan(out.loc[0, "opportunity_per_game_prev"])
    assert out.loc[1, "opportunity_per_game_prev"] == 10.0
    assert out.loc[3, "opportunity_per_game_prev"] == 14.0


def test_history_rolling_excludes_current():
    out = ts.add_opportunity_history_features(_toy())
    out = out.sort_values("season").reset_index(drop=True)
    # last2_avg at 2021 = mean(2019, 2020) = 13.0, not including 2021's 16.0
    assert out.loc[3, "opportunity_per_game_last2_avg"] == 13.0


def test_trend_feature_definition():
    out = ts.add_opportunity_history_features(_toy())
    out = out.sort_values("season").reset_index(drop=True)
    # trend = prev - last2_avg; at 2021: 14.0 - 13.0 = 1.0
    assert abs(out.loc[3, "opportunity_per_game_trend_2yr"] - 1.0) < 1e-9


def test_next_season_target_consecutive_only():
    out = ts.add_next_season_opportunity_target(_toy())
    out = out.sort_values("season").reset_index(drop=True)
    assert out.loc[0, ts.OPPORTUNITY_TARGET] == 12.0
    assert np.isnan(out.loc[3, ts.OPPORTUNITY_TARGET])  # no season after 2021


def test_target_dropped_on_season_gap():
    df = _toy()
    df = df[df["season"] != 2020].reset_index(drop=True)  # gap: 2019 -> 2021
    out = ts.add_next_season_opportunity_target(df)
    row_2019 = out[out["season"] == 2019].iloc[0]
    assert np.isnan(row_2019[ts.OPPORTUNITY_TARGET])
    assert row_2019["has_next_season"] == 0


def test_persistence_baseline_uses_current_opportunity():
    train = pd.DataFrame({ts.OPPORTUNITY_CURRENT: [5.0, 7.0]})
    valid = pd.DataFrame(
        {ts.OPPORTUNITY_CURRENT: [11.0, 13.0], ts.OPPORTUNITY_TARGET: [np.nan, np.nan]}
    )
    preds = ts.predict_opportunity_persistence(train, valid)
    assert np.allclose(preds, [11.0, 13.0])


def test_summary_skill_score_arithmetic():
    preds = pd.DataFrame(
        {
            "method": ["persistence"] * 3 + ["random_forest"] * 3,
            "method_type": ["baseline"] * 3 + ["model"] * 3,
            "position": ["RB"] * 6,
            "season": [2024] * 6,
            ts.OPPORTUNITY_TARGET: [0.0] * 6,
            "prediction": [2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
        }
    )
    preds["residual"] = preds[ts.OPPORTUNITY_TARGET] - preds["prediction"]
    preds["abs_residual"] = preds["residual"].abs()
    summary = ts.summarize_opportunity_methods(preds)
    rf = summary[summary["method"] == "random_forest"].iloc[0]
    # ref rmse 2.0, rf rmse 1.0 -> skill 0.5
    assert abs(rf["skill_vs_persistence"] - 0.5) < 1e-9


def _toy_efficiency() -> pd.DataFrame:
    # Two players, three consecutive seasons, all above the RB efficiency floor
    # (MIN_EFFICIENCY_OPPORTUNITY["RB"] = 50 -> opportunity_total >= 50).
    return pd.DataFrame(
        {
            "player_id": ["a", "a", "a", "b", "b", "b"],
            "season": [2019, 2020, 2021, 2019, 2020, 2021],
            "position": ["RB"] * 6,
            "games_played": [16] * 6,
            "efficiency_per_opportunity": [0.10, 0.12, 0.08, -0.02, 0.01, 0.03],
            "efficiency_qualified": [True, True, True, True, True, True],
            "opportunity_per_game": [12.0, 13.0, 11.0, 9.0, 10.0, 8.0],
        }
    )


def test_efficiency_history_is_lagged():
    out = ts.add_efficiency_history_features(_toy_efficiency())
    out = out[out.player_id == "a"].sort_values("season").reset_index(drop=True)
    assert np.isnan(out.loc[0, "efficiency_per_opportunity_prev"])
    assert abs(out.loc[1, "efficiency_per_opportunity_prev"] - 0.10) < 1e-9
    assert abs(out.loc[2, "efficiency_per_opportunity_prev"] - 0.12) < 1e-9


def test_efficiency_target_requires_both_ends_qualified():
    df = _toy_efficiency()
    # Disqualify player a's 2020 season -> the 2019->2020 pair must be dropped.
    df.loc[(df.player_id == "a") & (df.season == 2020), "efficiency_qualified"] = False
    out = ts.add_next_season_efficiency_target(df)
    a2019 = out[(out.player_id == "a") & (out.season == 2019)].iloc[0]
    assert np.isnan(a2019[ts.EFFICIENCY_TARGET])  # next end unqualified


def test_shrink_to_mean_baseline_predicts_positional_mean():
    train = pd.DataFrame(
        {"position": ["RB", "RB", "WR", "WR"], ts.EFFICIENCY_TARGET: [0.1, 0.3, 0.5, 0.7]}
    )
    valid = pd.DataFrame({"position": ["RB", "WR"], ts.EFFICIENCY_TARGET: [np.nan, np.nan]})
    preds = ts.predict_efficiency_shrink_to_mean(train, valid)
    assert abs(preds[0] - 0.2) < 1e-9  # RB mean
    assert abs(preds[1] - 0.6) < 1e-9  # WR mean


def test_reconstruct_value_predictions_is_product():
    opp = pd.Series([10.0, 5.0])
    eff = pd.Series([0.2, -0.1])
    out = ts.reconstruct_value_predictions(opp, eff)
    assert np.allclose(out.to_numpy(), [2.0, -0.5])


def test_standardize_to_value_score_uses_frozen_stats():
    stats = pd.DataFrame({"mean": {"RB": 2.0}, "std": {"RB": 0.5}})
    pred = pd.Series([3.0, 1.0])
    pos = pd.Series(["RB", "RB"])
    z = ts.standardize_to_value_score(pred, pos, stats)
    assert np.allclose(z, [2.0, -2.0])  # (3-2)/.5, (1-2)/.5


def test_propagate_product_interval_matches_closed_form():
    # E=2, O=10, se=0.5, so=1.0 -> var_eff=25, var_opp=4, var_int=0.25
    out = ts.propagate_product_interval(
        np.array([2.0]), np.array([10.0]), np.array([0.5]), np.array([1.0]), z=2.0
    )
    assert out["value_pred"][0] == 20.0
    assert out["var_from_efficiency"][0] == 25.0
    assert out["var_from_opportunity"][0] == 4.0
    assert abs(out["var_interaction"][0] - 0.25) < 1e-12
    assert abs(out["sigma"][0] - np.sqrt(29.25)) < 1e-12
    assert abs(out["interval_width"][0] - 2 * 2.0 * np.sqrt(29.25)) < 1e-12
    assert abs(out["efficiency_variance_share"][0] - 25.0 / 29.25) < 1e-12


def test_propagate_interval_is_symmetric_around_prediction():
    out = ts.propagate_product_interval(
        np.array([1.5]), np.array([8.0]), np.array([0.3]), np.array([0.7]), z=1.28
    )
    mid = 0.5 * (out["interval_low"][0] + out["interval_high"][0])
    assert abs(mid - out["value_pred"][0]) < 1e-12


def test_propagate_zero_variance_guard():
    out = ts.propagate_product_interval(
        np.array([1.0]), np.array([1.0]), np.array([0.0]), np.array([0.0])
    )
    assert out["interval_width"][0] == 0.0
    assert np.isnan(out["efficiency_variance_share"][0])


def test_per_position_residual_sigma_uses_overall_fallback():
    # A position with a single calibration row has undefined std -> fallback.
    df = pd.DataFrame(
        {
            "position": ["WR", "WR", "WR", "QB"],
            "resid": [1.0, -1.0, 0.5, 2.0],
        }
    )
    per_pos, overall = ts._per_position_residual_sigma(df, "resid")
    assert overall > 0
    # QB has one row -> std is NaN -> replaced by overall.
    assert abs(per_pos["QB"] - overall) < 1e-12
    assert per_pos["WR"] > 0
