"""Tests for the Streamlit-free fantasy content builders (tiers, scarcity,
risers, regression watch, role badges)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app import fantasy_content as fc

ROOT = Path(__file__).resolve().parents[1]


def _fantasy_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(6)],
            "player_display_name": [f"Player {i}" for i in range(6)],
            "position": ["RB", "RB", "RB", "QB", "WR", "WR"],
            "primary_team_2025": ["KC", "SF", "DET", "BUF", "LA", "CIN"],
            fc.PROJ_COL: [300.0, 295.0, 200.0, 290.0, 250.0, 130.0],
            fc.DELTA_COL: [-50.0, 30.0, -40.0, -80.0, 25.0, 90.0],
            fc.LOW_COL: [250.0, 245.0, 150.0, 240.0, 200.0, 80.0],
            fc.HIGH_COL: [350.0, 345.0, 250.0, 340.0, 300.0, 180.0],
        }
    )


def _two_stage_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(6)],
            "position": ["RB", "RB", "RB", "QB", "WR", "WR"],
            "efficiency_variance_share": [0.90, 0.30, 0.80, 0.95, 0.60, 0.85],
        }
    )


def test_assign_tiers_breaks_on_large_gaps():
    rb = _fantasy_fixture().query("position == 'RB'")
    tiers = fc.assign_tiers(rb)
    # 300 and 295 are within any reasonable threshold of each other; 200 is a
    # 100-point cliff and must open a new tier.
    assert tiers.loc[rb[fc.PROJ_COL].idxmax()] == 1
    by_proj = rb.sort_values(fc.PROJ_COL, ascending=False)
    assert list(tiers.loc[by_proj.index]) == [1, 1, 2]


def test_assign_tiers_empty_is_safe():
    assert fc.assign_tiers(pd.DataFrame()).empty


def test_stability_labels_respect_position_exception():
    labels = fc.stability_labels(_two_stage_fixture())
    lookup = dict(zip(labels.player_id, labels.role_badge))
    assert lookup["p0"] == "Shaky"      # RB at 0.90
    assert lookup["p1"] == "Stable"     # RB at 0.30
    assert lookup["p3"] == ""           # QB never gets a badge
    assert lookup["p4"] == ""           # middle band gets no badge


def test_regression_watch_requires_decline_efficiency_and_skill_position():
    watch = fc.regression_watch_frame(_fantasy_fixture(), _two_stage_fixture())
    names = set(watch["player_display_name"])
    # p0: RB, declining, share 0.90 -> in.
    assert "Player 0" in names
    # p3: QB with decline + high share -> excluded by position.
    assert "Player 3" not in names
    # p1: rising -> excluded. p2: declining RB but share 0.80 >= cutoff -> in.
    assert "Player 1" not in names
    assert "Player 2" in names


def test_risers_frame_filters_and_sorts():
    risers = fc.risers_frame(_fantasy_fixture())
    # Positive deltas only, min projection filter applied (p5 at 130 >= 120
    # stays; anything under would drop), sorted by delta descending.
    deltas = risers[fc.DELTA_COL].tolist()
    assert deltas == sorted(deltas, reverse=True)
    assert (risers[fc.DELTA_COL] > 0).all()
    assert (risers[fc.PROJ_COL] >= fc.MIN_RELEVANT_PROJ).all()


def test_scarcity_frame_and_dropoffs():
    scarcity = fc.scarcity_frame(_fantasy_fixture(), top_n=3)
    rb = scarcity[scarcity.position == "RB"]
    assert list(rb.positional_rank) == [1, 2, 3]
    assert rb.projected_points.iloc[0] == 300.0
    drop = fc.starter_window_dropoffs(scarcity, window=3)
    rb_drop = drop[drop.position == "RB"]["dropoff"].iloc[0]
    assert rb_drop == pytest.approx(100.0)


def test_draft_values_frame_filters_and_sorts():
    board = pd.DataFrame(
        {
            "player_display_name": [f"P{i}" for i in range(5)],
            "position": ["RB", "WR", "TE", "QB", "WR"],
            "overall_rank": [10, 40, 90, 130, 60],
            "adp_formatted": ["2.05", "6.02", "9.01", "12.01", "7.03"],
            "edge_vs_adp": [15.0, 30.0, 5.0, 50.0, None],
        }
    )
    values = fc.draft_values_frame(board, max_rank=120, min_edge=10.0)
    # P3 excluded by rank cap, P2 by min edge, P4 by missing edge; sorted desc.
    assert list(values["player_display_name"]) == ["P1", "P0"]
    assert values["edge_vs_adp"].is_monotonic_decreasing


def test_real_tables_build_cleanly_if_present():
    fantasy_path = ROOT / "outputs" / "tables" / "2026_fantasy_football_projections.csv"
    ts_path = ROOT / "outputs" / "tables" / "two_stage_2026_projection.csv"
    if not (fantasy_path.exists() and ts_path.exists()):
        return
    fantasy = pd.read_csv(fantasy_path)
    two_stage = pd.read_csv(ts_path)

    for pos in ("QB", "RB", "WR", "TE"):
        top = fantasy[fantasy.position == pos].sort_values(
            fc.PROJ_COL, ascending=False
        ).head(25)
        tiers = fc.assign_tiers(top)
        assert tiers.min() == 1
        # Tiering should be meaningful: more than one tier, fewer than one per player.
        assert 1 < tiers.max() < 25

    watch = fc.regression_watch_frame(fantasy, two_stage)
    assert not watch.empty
    assert not watch["position"].isin(["QB"]).any()

    scarcity = fc.scarcity_frame(fantasy)
    assert set(scarcity.position) == {"QB", "RB", "WR", "TE"}
    assert len(fc.starter_window_dropoffs(scarcity)) == 4
