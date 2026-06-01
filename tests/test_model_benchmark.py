"""Unit tests for the benchmark metrics and baselines.

These avoid training heavy models; they check the metric math and the baseline
predictors on small synthetic frames, plus the skill-score arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import model_benchmark as mb


def test_rmse_mae_r2_basic():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    yhat = np.array([0.0, 1.0, 2.0, 3.0])
    assert mb._rmse(y, yhat) == 0.0
    assert mb._mae(y, yhat) == 0.0
    assert mb._r2(y, yhat) == 1.0


def test_rmse_matches_manual():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.5, 2.0, 2.0])
    expected = float(np.sqrt(np.mean([(0.5) ** 2, 0.0, (1.0) ** 2])))
    assert abs(mb._rmse(y, yhat) - expected) < 1e-12


def test_season_mean_baseline_predicts_train_mean():
    train = pd.DataFrame({mb.TARGET: [0.0, 2.0, 4.0]})  # mean 2.0
    valid = pd.DataFrame({mb.TARGET: [9.0, 9.0]})
    preds = mb._predict_season_mean(train, valid)
    assert np.allclose(preds, 2.0)
    assert len(preds) == 2


def test_persistence_baseline_uses_current_value():
    train = pd.DataFrame({mb.TARGET: [0.0, 0.0]})
    valid = pd.DataFrame(
        {mb.CURRENT_VALUE_COL: [1.5, -0.5], mb.TARGET: [np.nan, np.nan]}
    )
    preds = mb._predict_persistence(train, valid)
    assert np.allclose(preds, [1.5, -0.5])


def test_shrunken_persistence_recovers_linear_relationship():
    # Build train where next = 0.5 * current exactly; fit should recover slope.
    current = np.linspace(-2, 2, 50)
    nxt = 0.5 * current
    train = pd.DataFrame({mb.CURRENT_VALUE_COL: current, mb.TARGET: nxt})
    valid = pd.DataFrame({mb.CURRENT_VALUE_COL: [4.0], mb.TARGET: [np.nan]})
    pred = mb._predict_shrunken_persistence(train, valid)
    assert abs(pred[0] - 2.0) < 1e-6  # 0.5 * 4.0


def test_summarize_methods_skill_score_arithmetic():
    # Two methods; skill vs shrunken_persistence should follow 1 - rmse/ref.
    preds = pd.DataFrame(
        {
            "method": ["shrunken_persistence"] * 3 + ["random_forest"] * 3,
            "method_type": ["baseline"] * 3 + ["model"] * 3,
            "position": ["WR"] * 6,
            "season": [2024] * 6,
            mb.TARGET: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "prediction": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
        }
    )
    preds["residual"] = preds[mb.TARGET] - preds["prediction"]
    preds["abs_residual"] = preds["residual"].abs()
    summary = mb.summarize_methods(preds)
    rf = summary[summary["method"] == "random_forest"].iloc[0]
    # rmse ref = 1.0, rf rmse = 0.5 -> skill = 0.5
    assert abs(rf["skill_vs_shrunken_persistence"] - 0.5) < 1e-9
