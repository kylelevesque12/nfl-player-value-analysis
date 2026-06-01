"""Central configuration for the NFL player value project.

Thresholds and modeling constants were previously defined independently in
several modules (for example ``MIN_VALUE_GAMES`` in both ``features.py`` and
``prediction_report.py``, and a separate 8-game floor used in the salary
findings). Scattering these makes the project harder to reason about and easy
to make internally inconsistent.

This module is the single source of truth. Existing modules keep their own
module-level constants for backward compatibility, but they should read the
canonical value from here, and a consistency test in ``tests/`` asserts that
the duplicated definitions still agree with this file.
"""

from __future__ import annotations

# --- Player-season eligibility -------------------------------------------------
# Minimum games for a player-season to receive a value score. A season-level
# value metric built on only a handful of games is noisy, so this is the floor
# for the core value-score table and next-season modeling.
MIN_VALUE_GAMES: int = 4

# Stricter floor used for salary-efficiency findings, where we want stable
# season-level production before drawing contract-value conclusions.
SALARY_FINDINGS_MIN_GAMES: int = 8

# --- Positions -----------------------------------------------------------------
SKILL_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

# Grouping used when standardizing value within a peer set.
VALUE_GROUP_COLS: tuple[str, ...] = ("season", "position")

# --- Temporal validation -------------------------------------------------------
# Seasons used as held-out folds for rolling-origin validation. Each fold trains
# strictly on earlier seasons.
ROLLING_VALIDATION_YEARS: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)

# Most recent completed season of input data; predictions are made for the
# following season.
CURRENT_INPUT_SEASON: int = 2025
PREDICTION_TARGET_SEASON: int = 2026

# --- Prediction intervals ------------------------------------------------------
PREDICTION_INTERVAL_TARGET_COVERAGE: float = 0.80
# z multiplier for an approximate central 80% Gaussian interval (one-sided 0.90).
PREDICTION_INTERVAL_MULTIPLIER: float = 1.28

# --- Reproducibility -----------------------------------------------------------
RANDOM_STATE: int = 42

# --- IO ------------------------------------------------------------------------
CSV_FLOAT_FORMAT: str = "%.12g"
