"""Tests for the Session 2 NGS/PFR external feature investigation.

The headline of Session 2 is a *negative* result: NGS/PFR weekly stats add no
leakage-safe signal to the weekly model, and a naive lagged-value join actually
leaks same-week availability. These tests pin both the leakage-safe mechanics
and the decision to keep those columns OUT of the production model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.load_data import find_project_root
from src import weekly_fantasy_projection as wk
from src import external_player_features as ext


# ---------------------------------------------------------------------------
# Synthetic: the lag/roll builder must be strictly shifted (no same-week value)
# ---------------------------------------------------------------------------
def test_add_lag_and_roll_is_shifted():
    df = pd.DataFrame(
        {
            "player_id": ["A"] * 4 + ["B"] * 2,
            "season": [2020] * 4 + [2020] * 2,
            "week": [1, 2, 3, 4, 1, 2],
            "metric": [10.0, 20.0, 30.0, 40.0, 5.0, 7.0],
        }
    )
    out = ext._add_lag_and_roll(df, ["metric"]).sort_values(["player_id", "week"])
    a = out[out["player_id"] == "A"].set_index("week")

    # First game has no prior -> NaN, so no current-week value can leak in.
    assert pd.isna(a.loc[1, "metric_lag1"])
    # lag1 at week t equals the same-week value at week t-1, never week t.
    assert a.loc[2, "metric_lag1"] == 10.0
    assert a.loc[3, "metric_lag1"] == 20.0
    # roll3 at week 4 = mean of weeks 1-3 (shifted), not weeks 2-4.
    assert a.loc[4, "metric_roll3"] == pytest.approx((10 + 20 + 30) / 3)
    # The lagged column must never equal the same-week metric where a prior
    # game exists (strict shift).
    merged = df.merge(out, on=["player_id", "season", "week"])
    has_prior = merged["metric_lag1"].notna()
    assert (merged.loc[has_prior, "metric_lag1"] != merged.loc[has_prior, "metric"]).all()


# ---------------------------------------------------------------------------
# Data-backed: key uniqueness and join safety (skipped if raw files absent)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def root():
    return find_project_root()


def _has(root, stem):
    return ext._find_file(root, stem) is not None


def test_value_feature_table_has_unique_player_week_keys(root):
    if not _has(root, "ngs_receiving"):
        pytest.skip("NGS data not present")
    table, cols = ext.build_external_weekly_features(root)
    assert table is not None and cols
    assert not table.duplicated(["player_id", "season", "week"]).any()


def test_pfr_bridge_does_not_explode_rows(root):
    if not _has(root, "pfr_weekly_rec"):
        pytest.skip("PFR data not present")
    pfr = ext.load_pfr_weekly_features(root)
    assert pfr is not None
    # Exactly one row per player-week after the pfr_id -> gsis_id bridge.
    assert not pfr.duplicated(["player_id", "season", "week"]).any()


def test_coverage_flag_attach_preserves_rows_and_is_binary(root):
    if not _has(root, "ngs_receiving"):
        pytest.skip("NGS data not present")
    # A small synthetic modeling frame; most ids won't be in NGS (flags 0).
    frame = pd.DataFrame(
        {
            "player_id": ["00-0000001", "00-0000001", "00-0000002"],
            "season": [2022, 2022, 2022],
            "week": [1, 2, 1],
            "junk": [1, 2, 3],
        }
    )
    out = ext.attach_ngs_coverage_flags(frame, project_root=root)
    assert len(out) == len(frame)  # join never adds rows
    for flag in ext.NGS_COVERAGE_FLAGS:
        assert flag in out.columns
        assert set(np.unique(out[flag])).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# The negative-result guard: NGS/PFR must NOT be in the production model.
# ---------------------------------------------------------------------------
def test_weekly_feature_list_excludes_ngs_and_pfr():
    leaky = [c for c in wk.WEEKLY_FANTASY_FEATURES if c.startswith(("ngs_", "pfr_"))]
    assert leaky == [], (
        "Session 2 concluded NGS/PFR add no leakage-safe signal; these columns "
        f"must not be registered in WEEKLY_FANTASY_FEATURES: {leaky}"
    )


def test_value_join_leaks_same_week_availability(root):
    """Documents WHY the value columns are excluded: a non-null lagged NGS value
    on the modeling frame coincides almost perfectly with being tracked in the
    SAME week (the join keys on the current week), i.e. it leaks availability.
    The leak-free coverage flag, by contrast, is not identical to same-week
    tracking."""
    if not _has(root, "ngs_receiving"):
        pytest.skip("NGS data not present")
    presence = ext._load_ngs_presence(root, "ngs_receiving").assign(_now=1.0)
    table, _ = ext.build_external_weekly_features(root, include_pfr=False)
    merged = table.merge(presence, on=["player_id", "season", "week"], how="left")
    merged["_now"] = merged["_now"].fillna(0.0)
    val_present = merged["ngs_avg_separation_lag1"].notna().astype(float)
    # Strong (near-deterministic) coupling with same-week tracking = the leak.
    assert val_present.corr(merged["_now"]) > 0.8
