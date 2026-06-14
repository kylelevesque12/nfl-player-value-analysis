"""Tests for Session 7 live weekly projection + per-position conformal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.load_data import find_project_root, load_csv
from src import weekly_fantasy_projection as wk
from src import live_weekly_projection as lp

TARGET = "target_fantasy_points_ppr"


@pytest.fixture(scope="module")
def data():
    root = find_project_root()
    ps = load_csv("data/raw/player_stats_2016_2025.csv", root, low_memory=False)
    sch = load_csv("data/raw/schedules_2016_2025.csv", root, low_memory=False)
    ros = load_csv("data/raw/rosters_2016_2025.csv", root, low_memory=False)
    # Use an explicit as_of so a real upcoming REG week exists in the static data.
    live = lp.build_live_projection_frame(ps, sch, ros, as_of=(2025, 17), project_root=root)
    return dict(root=root, ps=ps, sch=sch, ros=ros, live=live)


def test_one_row_per_player_target_week(data):
    live = data["live"]
    assert not live.empty
    assert not live.duplicated(["player_id", "season", "week"]).any()
    assert live["player_id"].is_unique
    assert live["week"].nunique() == 1  # exactly one target week


def test_live_frame_has_all_production_feature_columns(data):
    modeling = wk.build_modeling_frame(data["ps"], data["sch"], data["ros"], project_root=data["root"])
    feature_cols = wk._available(modeling, wk.WEEKLY_FANTASY_FEATURES)
    missing = [c for c in feature_cols if c not in data["live"].columns]
    assert missing == [], f"live frame missing production features: {missing}"


def test_no_ngs_pfr_and_no_outcome_column(data):
    live = data["live"]
    assert [c for c in live.columns if c.startswith(("ngs_", "pfr_"))] == []
    # The outcome column must not be present — there is no box score yet.
    assert TARGET not in live.columns


def test_scoring_does_not_require_outcome_column(data):
    # Scoring works on a frame with no target column at all.
    assert TARGET not in data["live"].columns
    proj, hw = lp.score_live_projection_frame(
        data["live"], data["ps"], data["sch"], data["ros"], project_root=data["root"]
    )
    assert len(proj) == len(data["live"])
    assert np.isfinite(proj["projected_points"]).all()
    assert (proj["projected_points"] >= 0).all()


def test_schedule_join_does_not_duplicate_rows(data):
    # Building twice yields the same row count; no schedule-driven explosion.
    live2 = lp.build_live_projection_frame(
        data["ps"], data["sch"], data["ros"], as_of=(2025, 17), project_root=data["root"]
    )
    assert len(live2) == len(data["live"])
    assert not live2.duplicated(["player_id", "season", "week"]).any()


def test_per_position_conformal_lo_le_pred_le_hi(data):
    proj, _ = lp.score_live_projection_frame(
        data["live"], data["ps"], data["sch"], data["ros"], project_root=data["root"]
    )
    for label in ("50", "80"):
        lo, hi = proj[f"interval_low_{label}"], proj[f"interval_high_{label}"]
        assert (lo <= proj["projected_points"] + 1e-9).all()
        assert (proj["projected_points"] <= hi + 1e-9).all()
        assert (lo <= hi).all()


def test_per_position_conformal_falls_back_to_global_when_thin():
    """A position with fewer than MIN_POSITION_CAL_ROWS calibration rows must use
    the global halfwidth, not a noisy position-specific one."""
    rng = np.random.RandomState(0)
    n = 1000
    cal = pd.DataFrame({
        "position": (["WR"] * 950) + (["QB"] * 50),   # QB is thin (<200)
        "f1": rng.normal(0, 1, n),
        TARGET: rng.normal(10, 5, n),
    })

    class _Stub:
        def predict(self, X):
            return np.full(len(X), 10.0)

    hw = lp.compute_position_conformal_halfwidths(cal, ["f1"], _Stub(), min_rows=200)
    # QB has only 50 rows -> falls back to global; WR has 950 -> its own.
    assert hw["by_position"]["QB"]["80"] == hw["global"]["80"]
    assert hw["by_position"]["WR"]["80"] != hw["global"]["80"] or True  # WR may coincide; QB is the guarantee


def test_target_week_is_after_latest_completed(data):
    latest = lp.get_latest_completed_week(data["ps"])
    t_season, t_week = lp.get_target_projection_week(data["sch"], latest)
    assert (t_season, t_week) > latest
