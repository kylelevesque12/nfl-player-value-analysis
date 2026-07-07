"""Tests for the PyMC-free parts of scoring the live rookie class:
expected-games-by-position and the season-total scaling math. The actual
hurdle fit (score_rookie_class) needs PyMC and is exercised manually via
.venv-bayes, same convention as the rest of src/rookie_bayes.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.rookie_bayes import (
    expected_games_if_plays_by_position,
    scale_hurdle_output_to_season_totals,
)


def _train_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": ["RB", "RB", "RB", "WR", "WR", "QB"],
            "played_meaningfully": [1, 1, 0, 1, 0, 1],
            "games_played": [16, 10, 1, 14, 2, 17],
        }
    )


def test_expected_games_averages_only_players_who_cleared_the_hurdle():
    result = expected_games_if_plays_by_position(_train_fixture())
    # RB: mean of [16, 10] (the 1-game player didn't clear the hurdle) = 13.
    assert result["RB"] == pytest.approx(13.0)
    assert result["WR"] == pytest.approx(14.0)
    assert result["QB"] == pytest.approx(17.0)


def test_expected_games_falls_back_to_overall_mean_for_position_with_no_hits():
    train = pd.concat(
        [
            _train_fixture(),
            pd.DataFrame({"position": ["TE"], "played_meaningfully": [0], "games_played": [0]}),
        ],
        ignore_index=True,
    )
    result = expected_games_if_plays_by_position(train)
    # TE has zero rookies who cleared the hurdle; falls back to the overall
    # mean across every position's hurdle-clearing rookies, not a crash/NaN.
    overall = pd.Series([16, 10, 14, 17]).mean()
    assert result["TE"] == pytest.approx(overall)


def test_season_scaling_is_percentile_equivariant():
    """A constant multiplier applied to every hurdle percentile column must
    preserve the same relative shape — this is the mathematical argument for
    reusing predict_hurdle's output unmodified rather than re-deriving the
    posterior at the season-total scale."""
    scored = pd.DataFrame(
        {
            "position": ["RB", "WR"],
            "predicted_p_plays_meaningfully": [0.8, 0.5],
            "predicted_rookie_year_ppr_per_game_mean": [10.0, 6.0],
            "predicted_rookie_year_ppr_per_game_p10": [2.0, 1.0],
            "predicted_rookie_year_ppr_per_game_p25": [5.0, 3.0],
            "predicted_rookie_year_ppr_per_game_p75": [15.0, 9.0],
            "predicted_rookie_year_ppr_per_game_p90": [20.0, 12.0],
        }
    )
    games_by_position = {"RB": 13.0, "WR": 14.0}
    out = scale_hurdle_output_to_season_totals(scored, games_by_position)

    assert out.loc[0, "predicted_games_played"] == pytest.approx(0.8 * 13.0)
    assert out.loc[0, "predicted_season_ppr_mean"] == pytest.approx(10.0 * 13.0)
    assert out.loc[0, "predicted_season_ppr_p10"] == pytest.approx(2.0 * 13.0)
    assert out.loc[0, "predicted_season_ppr_p90"] == pytest.approx(20.0 * 13.0)
    assert out.loc[1, "predicted_season_ppr_mean"] == pytest.approx(6.0 * 14.0)
    # Percentile ordering must survive the scaling.
    assert out.loc[0, "predicted_season_ppr_p10"] < out.loc[0, "predicted_season_ppr_mean"]
    assert out.loc[0, "predicted_season_ppr_mean"] < out.loc[0, "predicted_season_ppr_p90"]
