"""Tests for the Session 6 ensemble + quantile-interval experiment harness.

These pin the leakage-safety of stacking, the position-model fallback, and the
interval mechanics on a small synthetic frame so they run fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import weekly_ensemble_experiment as ex
from src import weekly_fantasy_projection as wk


@pytest.fixture(scope="module")
def synth():
    rng = np.random.RandomState(0)
    rows = []
    for season in range(2018, 2024):
        for _ in range(240):
            pos = rng.choice(["QB", "RB", "WR", "TE"])
            f1, f2 = rng.normal(0, 1), rng.normal(0, 1)
            y = max(0.0, 8 + 3 * f1 + 2 * f2 + {"QB": 4, "RB": 1, "WR": 1, "TE": 0}[pos] + rng.normal(0, 3))
            rows.append({"season": season, "week": rng.randint(1, 17), "position": pos,
                         "f1": f1, "f2": f2, "target_fantasy_points_ppr": y})
    return pd.DataFrame(rows)


FEATS = ["f1", "f2", "position"]


def test_point_backtest_has_four_arms_aligned_to_validation(synth):
    preds = ex.run_point_backtest(synth, FEATS, folds=[2022, 2023], n_oof_seasons=2)
    assert set(preds["arm"]) == {"pooled_hgb", "per_position_hgb", "linear_ridge", "stacked_ensemble"}
    # Each arm has exactly one prediction per validation row.
    n_valid = (synth["season"].isin([2022, 2023])).sum()
    for arm, g in preds.groupby("arm"):
        assert len(g) == n_valid
        assert np.isfinite(g["prediction"]).all()
        assert (g["prediction"] >= 0).all()


def test_stacking_meta_uses_out_of_fold_predictions(monkeypatch, synth):
    """The meta-model must only ever be fit on base predictions for rows the
    base models did NOT train on. We assert that by recording, for every
    base-prediction call, that the prediction rows' seasons are strictly later
    than the max training season (i.e. genuinely held out)."""
    violations = []
    orig = ex._base_predictions

    def spy(train, predict_df, feature_cols):
        max_train = train["season"].max()
        # OOF calls predict a season strictly greater than every training season.
        if predict_df["season"].min() <= max_train:
            # The only legitimate equal-or-overlap call is the final validation
            # scoring, where predict season > all train seasons too. So any
            # overlap is a leak.
            if (predict_df["season"] <= max_train).any():
                violations.append((max_train, predict_df["season"].unique().tolist()))
        return orig(train, predict_df, feature_cols)

    monkeypatch.setattr(ex, "_base_predictions", spy)
    ex.run_point_backtest(synth, FEATS, folds=[2023], n_oof_seasons=2)
    assert violations == [], f"base model scored rows it trained on: {violations}"


def test_per_position_fallback_to_pooled(synth):
    train = synth[synth["season"] < 2023]
    valid = synth[synth["season"] == 2023].copy()
    pooled = ex._fit_pooled(train, FEATS)
    # Force a position to have no specific model -> must fall back to pooled.
    models = ex._fit_per_position(train, FEATS)
    models.pop("TE", None)
    pred = ex._predict_per_position(models, pooled, valid, FEATS)
    te_mask = valid["position"].eq("TE").to_numpy()
    pooled_pred = np.clip(pooled.predict(valid[FEATS]), 0, None)
    assert np.allclose(pred[te_mask], pooled_pred[te_mask])  # TE used pooled fallback


def test_quantile_intervals_lo_le_hi_and_coverage_valid(synth):
    iv = ex.run_interval_backtest(synth, FEATS, folds=[2023])
    assert (iv["lo"] <= iv["hi"] + 1e-9).all()       # lower bound never exceeds upper
    assert (iv["lo"] >= 0).all()                      # fantasy points are non-negative
    q = ex.interval_quality(iv)
    assert (q["empirical_coverage"].between(0, 1)).all()
    assert set(q["method"]) == {"conformal", "quantile"}


def test_no_ngs_or_pfr_in_session1_feature_cols():
    leaky = [c for c in wk.WEEKLY_FANTASY_FEATURES if c.startswith(("ngs_", "pfr_"))]
    assert leaky == [], f"NGS/PFR must not be in the weekly feature set: {leaky}"
