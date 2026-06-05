"""Unit tests for the replacement-level surplus module.

These tests fix small hand-checked inputs and assert the replacement baselines
and dollar surplus math match what we'd compute by hand. The point is to pin
the framing — if someone changes the replacement quantile or the price-slope
formula, these break loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.replacement_level import (
    compute_replacement_baselines,
    compute_replacement_level_surplus,
)


def _toy_finding_base() -> pd.DataFrame:
    # One position-season (2024 WR) with 10 players. Salary percentiles
    # are hand-set so the bottom quartile (4 players) is well-defined and the
    # value/salary slope is unambiguous.
    rows = []
    salaries = [1.0, 1.1, 1.2, 1.3, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]
    values = [-0.8, -0.5, -0.3, 0.0, 0.3, 0.6, 0.8, 1.2, 1.5, 2.0]
    for i, (sal, val) in enumerate(zip(salaries, values)):
        rows.append(
            {
                "season": 2024,
                "player_id": f"p{i+1}",
                "player_display_name": f"Player {i+1}",
                "position": "WR",
                "team": "AAA",
                "games_played": 16,
                "years_exp": 4,
                "value_score": val,
                "salary_millions": sal,
                "salary_percentile": (i + 1) / len(salaries),  # 0.1..1.0
            }
        )
    return pd.DataFrame(rows)


def test_replacement_baseline_drops_thin_samples():
    base = _toy_finding_base()
    baselines = compute_replacement_baselines(base)
    # With only 2 players at salary_percentile <= 0.25, the baseline is below
    # the minimum sample size and should be excluded.
    assert baselines.empty


def test_replacement_baseline_when_sample_is_thick():
    # Build a base with 20 players so the bottom quartile (5 players) clears
    # the MIN_REPLACEMENT_SAMPLE=5 threshold.
    rows = []
    for i in range(20):
        rows.append(
            {
                "season": 2024,
                "player_id": f"p{i+1}",
                "player_display_name": f"Player {i+1}",
                "position": "WR",
                "team": "AAA",
                "games_played": 16,
                "years_exp": 4,
                "value_score": -1.0 + 0.1 * i,
                "salary_millions": 1.0 + 0.5 * i,
                "salary_percentile": (i + 1) / 20.0,
            }
        )
    base = pd.DataFrame(rows)
    baselines = compute_replacement_baselines(base)
    assert len(baselines) == 1
    row = baselines.iloc[0]
    # Bottom quartile = first 5 players (salary_percentile 0.05..0.25).
    # Median salary of [1.0, 1.5, 2.0, 2.5, 3.0] = 2.0
    # Median value of [-1.0, -0.9, -0.8, -0.7, -0.6] = -0.8
    assert row["replacement_salary_millions"] == 2.0
    assert np.isclose(row["replacement_value_score"], -0.8)
    assert row["replacement_sample_size"] == 5


def test_surplus_dollar_math_is_consistent():
    # Build a base where the slope is exactly known.
    rows = []
    # 12 players, value evenly spaced from -1.0 to +1.2, salary = 5*value + 2 + noise=0
    for i in range(12):
        val = -1.0 + 0.2 * i
        sal = 5.0 * val + 2.0
        rows.append(
            {
                "season": 2024,
                "player_id": f"p{i+1}",
                "player_display_name": f"Player {i+1}",
                "position": "WR",
                "team": "AAA",
                "games_played": 16,
                "years_exp": 4,
                "value_score": val,
                "salary_millions": sal,
                "salary_percentile": (i + 1) / 12.0,
            }
        )
    base = pd.DataFrame(rows)
    enriched, _baselines, prices = compute_replacement_level_surplus(base)
    # The slope of salary on value is exactly 5.0.
    assert np.isclose(prices.iloc[0]["price_per_value_unit_millions"], 5.0)
    # For each player, dollar_surplus = value_over_replacement*5 - cap_over_replacement.
    # Since salary = 5*value + 2, cap_over_replacement = 5 * value_over_replacement.
    # So the dollar surplus should be near zero for every player by construction.
    valid = enriched.dropna(subset=["dollar_surplus_millions"])
    assert (valid["dollar_surplus_millions"].abs() < 1e-9).all()
