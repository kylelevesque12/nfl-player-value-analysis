"""Unit tests for leakage-safety in feature engineering.

The history and target helpers must only use information available before the
season being predicted. These tests build a tiny synthetic player history with
known values and assert the lag/rolling/target columns are exactly right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    add_player_history_features,
    create_next_season_targets,
)


def _toy_player_seasons() -> pd.DataFrame:
    # One player, four consecutive seasons with simple value scores.
    return pd.DataFrame(
        {
            "player_id": ["p1", "p1", "p1", "p1"],
            "season": [2018, 2019, 2020, 2021],
            "position": ["WR", "WR", "WR", "WR"],
            "value_score": [1.0, 2.0, 3.0, 4.0],
            "value_epa_total": [10.0, 20.0, 30.0, 40.0],
            "value_epa_per_game": [1.0, 2.0, 3.0, 4.0],
            "games_played": [16, 16, 16, 16],
            "yards_per_game": [50.0, 60.0, 70.0, 80.0],
            "tds_per_game": [0.5, 0.6, 0.7, 0.8],
        }
    )


def test_prev_value_is_strictly_lagged():
    featured = add_player_history_features(_toy_player_seasons())
    featured = featured.sort_values("season").reset_index(drop=True)
    # First season has no prior -> NaN; later seasons equal the previous value.
    assert np.isnan(featured.loc[0, "value_score_prev"])
    assert featured.loc[1, "value_score_prev"] == 1.0
    assert featured.loc[2, "value_score_prev"] == 2.0
    assert featured.loc[3, "value_score_prev"] == 3.0


def test_rolling_history_excludes_current_season():
    featured = add_player_history_features(_toy_player_seasons())
    featured = featured.sort_values("season").reset_index(drop=True)
    # last2_avg at season 2021 should average 2019+2020 values (2.0, 3.0) = 2.5,
    # and must NOT include the current 2021 value (4.0).
    assert featured.loc[3, "value_score_last2_avg"] == 2.5
    assert featured.loc[2, "value_score_last2_avg"] == 1.5


def test_next_season_target_is_forward_shift():
    targeted = create_next_season_targets(_toy_player_seasons())
    targeted = targeted.sort_values("season").reset_index(drop=True)
    # next_value_score should be the following season's value, last row NaN.
    assert targeted.loc[0, "next_value_score"] == 2.0
    assert targeted.loc[2, "next_value_score"] == 4.0
    assert np.isnan(targeted.loc[3, "next_value_score"])


def test_non_consecutive_season_target_is_dropped():
    df = _toy_player_seasons()
    # Remove 2020 so 2019 -> next is 2021 (a gap), which must be invalidated.
    df = df[df["season"] != 2020].reset_index(drop=True)
    targeted = create_next_season_targets(df).sort_values("season").reset_index(drop=True)
    row_2019 = targeted[targeted["season"] == 2019].iloc[0]
    assert np.isnan(row_2019["next_value_score"])
    assert row_2019["next_season_qualifier"] == 0
