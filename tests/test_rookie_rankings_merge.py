"""Tests for merging rookie projections into the season fantasy table
(src/rookie_rankings_merge.py) — pure, no PyMC dependency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rookie_rankings_merge import (
    build_rookie_rows,
    merge_rookies_into_season_table,
)

SEASON_COLUMNS = [
    "player_id", "player_display_name", "position", "primary_team_2025",
    "teams_2025", "games_played_2025", "age_2025", "years_exp_2025",
    "draft_number", "fantasy_points_ppr_2025", "fantasy_points_ppr_per_game_2025",
    "targets_2025", "receptions_2025", "carries_2025", "value_score",
    "predicted_2026_fantasy_points_ppr", "predicted_2026_games_played",
    "predicted_2026_ppr_per_game", "projection_change_from_2025",
    "projection_change_label", "prediction_interval_low",
    "prediction_interval_high", "prediction_uncertainty", "model_disagreement",
    "fantasy_overall_rank", "fantasy_position_rank",
    "predicted_2026_overall_percentile", "predicted_2026_position_percentile",
    "fantasy_projection_tier", "usage_profile", "breakout_potential",
    "slump_potential", "draft_board_bucket", "confidence_score",
    "confidence_level", "selected_model", "selected_model_label",
    "model_selection_reason", "fantasy_note", "fantasy_explanation",
]


def _veterans_fixture() -> pd.DataFrame:
    n = 20
    proj = np.linspace(300, 50, n)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "player_id": [f"00-{1000+i}" for i in range(n)],
            "player_display_name": [f"Veteran {i}" for i in range(n)],
            "position": (["RB"] * 6 + ["WR"] * 8 + ["QB"] * 3 + ["TE"] * 3),
            "primary_team_2025": ["KC"] * n,
            "teams_2025": ["KC"] * n,
            "games_played_2025": rng.integers(4, 17, n),
            "age_2025": rng.integers(23, 33, n),
            "years_exp_2025": rng.integers(1, 10, n),
            "draft_number": rng.integers(1, 200, n),
            "fantasy_points_ppr_2025": proj + rng.normal(0, 5, n),
            "fantasy_points_ppr_per_game_2025": proj / 15,
            "targets_2025": rng.integers(0, 150, n),
            "receptions_2025": rng.integers(0, 100, n),
            "carries_2025": rng.integers(0, 250, n),
            "value_score": rng.normal(0, 1, n),
            "predicted_2026_fantasy_points_ppr": proj,
            "predicted_2026_games_played": np.full(n, 15.0),
            "predicted_2026_ppr_per_game": proj / 15,
            "projection_change_from_2025": np.full(n, 5.0),
            "projection_change_label": ["Similar to 2025"] * n,
            "prediction_interval_low": proj - 40,
            "prediction_interval_high": proj + 40,
            "prediction_uncertainty": np.full(n, 30.0),
            "model_disagreement": np.full(n, 5.0),
            "fantasy_overall_rank": pd.Series(proj).rank(ascending=False, method="min"),
            "fantasy_position_rank": 1,
            "predicted_2026_overall_percentile": pd.Series(proj).rank(pct=True),
            "predicted_2026_position_percentile": pd.Series(proj).rank(pct=True),
            "fantasy_projection_tier": ["Strong Starter"] * n,
            "usage_profile": ["Regular WR target volume"] * n,
            "breakout_potential": ["Low"] * n,
            "slump_potential": ["Low"] * n,
            "draft_board_bucket": ["Stable Option"] * n,
            "confidence_score": np.full(n, 60.0),
            "confidence_level": ["Medium"] * n,
            "selected_model": ["elastic_net_total"] * n,
            "selected_model_label": ["Elastic Net Total-PPR Model"] * n,
            "model_selection_reason": ["lowest validation error"] * n,
            "fantasy_note": ["balanced fantasy profile"] * n,
            "fantasy_explanation": ["Projects some points."] * n,
        }
    )


def _rookie_projections_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["00-9001", "00-9002"],
            "player_display_name": ["Rookie Star", "Rookie Deep"],
            "position": ["RB", "WR"],
            "draft_number": [3.0, 120.0],
            "draft_club": ["ARI", "SF"],
            "predicted_p_plays_meaningfully": [0.95, 0.30],
            "predicted_ppr_per_game_if_plays_mean": [15.0, 6.0],
            "predicted_games_played": [11.0, 3.0],
            "predicted_season_ppr_mean": [200.0, 15.0],
            "predicted_season_ppr_p10": [120.0, -5.0],
            "predicted_season_ppr_p25": [160.0, 2.0],
            "predicted_season_ppr_p75": [240.0, 30.0],
            "predicted_season_ppr_p90": [280.0, 50.0],
        }
    )


# ---------------------------------------------------------------------------
# build_rookie_rows
# ---------------------------------------------------------------------------
RANK_DEPENDENT_COLUMNS = {
    "fantasy_overall_rank", "fantasy_position_rank",
    "predicted_2026_overall_percentile", "predicted_2026_position_percentile",
    "fantasy_projection_tier", "breakout_potential", "slump_potential",
    "draft_board_bucket", "confidence_score", "confidence_level",
}


def test_rookie_rows_have_every_non_rank_dependent_column():
    # Rank/percentile/tier/confidence are pool-relative, so build_rookie_rows
    # deliberately leaves them for merge_rookies_into_season_table to compute
    # over the combined set — checked separately below.
    rows = build_rookie_rows(_rookie_projections_fixture())
    expected = set(SEASON_COLUMNS) - RANK_DEPENDENT_COLUMNS
    missing = expected - set(rows.columns)
    assert not missing, f"rookie rows missing season-table columns: {missing}"


def test_merged_rows_have_the_full_season_table_schema():
    combined = merge_rookies_into_season_table(_veterans_fixture(), _rookie_projections_fixture())
    missing = set(SEASON_COLUMNS) - set(combined.columns)
    assert not missing, f"combined table missing season-table columns: {missing}"


def test_rookie_rows_have_no_fake_2025_baseline():
    rows = build_rookie_rows(_rookie_projections_fixture())
    assert rows["fantasy_points_ppr_2025"].isna().all()
    assert (rows["games_played_2025"] == 0).all()
    assert rows["projection_change_label"].eq("Rookie — no 2025 comparison").all()


def test_rookie_interval_low_never_negative():
    rows = build_rookie_rows(_rookie_projections_fixture())
    # Rookie Deep's p10 is -5 (a real possible posterior tail below zero
    # points) but the displayed floor must be clipped at 0.
    assert (rows["prediction_interval_low"] >= 0).all()


def test_rookie_explanation_mentions_draft_slot_and_no_history():
    rows = build_rookie_rows(_rookie_projections_fixture())
    star = rows.set_index("player_display_name").loc["Rookie Star", "fantasy_explanation"]
    assert "3 overall pick" in star
    assert "No 2025 NFL history" in star


# ---------------------------------------------------------------------------
# merge_rookies_into_season_table
# ---------------------------------------------------------------------------
def test_merge_is_noop_when_rookie_projections_missing():
    vets = _veterans_fixture()
    out = merge_rookies_into_season_table(vets, None)
    pd.testing.assert_frame_equal(
        out.sort_values("player_id").reset_index(drop=True),
        vets.sort_values("player_id").reset_index(drop=True),
    )


def test_merge_is_noop_when_rookie_projections_empty():
    vets = _veterans_fixture()
    out = merge_rookies_into_season_table(vets, pd.DataFrame())
    assert len(out) == len(vets)


def test_merge_adds_rookies_and_recomputes_ranks_over_combined_set():
    vets = _veterans_fixture()
    rookies = _rookie_projections_fixture()
    combined = merge_rookies_into_season_table(vets, rookies)

    assert len(combined) == len(vets) + len(rookies)
    # Rookie Star projects higher than every veteran (200 vs max 300... wait,
    # check: veteran max proj is 300, so Rookie Star at 200 should NOT be
    # rank 1 -- assert his rank reflects his actual value among all 22 rows.
    star = combined.set_index("player_display_name").loc["Rookie Star"]
    expected_rank = int((combined["predicted_2026_fantasy_points_ppr"] > 200.0).sum()) + 1
    assert star["fantasy_overall_rank"] == expected_rank
    assert star["is_rookie_projection"] == True  # noqa: E712


def test_merge_flags_rookie_and_veteran_rows_correctly():
    combined = merge_rookies_into_season_table(_veterans_fixture(), _rookie_projections_fixture())
    rookie_mask = combined["player_id"].isin(["00-9001", "00-9002"])
    assert combined.loc[rookie_mask, "is_rookie_projection"].all()
    assert not combined.loc[~rookie_mask, "is_rookie_projection"].any()


def test_merge_gives_low_confidence_bust_risk_rookie_a_low_score():
    combined = merge_rookies_into_season_table(_veterans_fixture(), _rookie_projections_fixture())
    deep = combined.set_index("player_display_name").loc["Rookie Deep"]
    # Zero sample, zero history, and (per the wide 15-297 range) his
    # uncertainty is near the top of the combined pool -> confidence must
    # land in the bottom half of the 0-100 scale, not a coin flip "Medium".
    assert deep["confidence_score"] < 50


def test_merge_recomputes_tier_uniformly_over_combined_pool():
    combined = merge_rookies_into_season_table(_veterans_fixture(), _rookie_projections_fixture())
    # Every row's tier must be internally consistent with its own recomputed
    # percentile (not left over from before the merge).
    from src.fantasy_projection import _assign_fantasy_tier

    for _, row in combined.iterrows():
        assert row["fantasy_projection_tier"] == _assign_fantasy_tier(
            row["predicted_2026_position_percentile"]
        )
