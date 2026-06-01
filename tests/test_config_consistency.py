"""Guard against the scattered-threshold problem returning.

These tests assert that constants duplicated across modules still agree with
the single source of truth in ``src.config``. If someone changes a threshold in
one place but not the central config, these fail loudly.
"""

from __future__ import annotations

from src import config


def test_min_value_games_matches_features_module():
    from src import features

    assert features.MIN_VALUE_GAMES == config.MIN_VALUE_GAMES


def test_min_value_games_matches_prediction_report():
    from src import prediction_report

    assert prediction_report.MIN_VALUE_GAMES == config.MIN_VALUE_GAMES


def test_interval_constants_match_prediction_report():
    from src import prediction_report

    assert (
        prediction_report.PREDICTION_INTERVAL_TARGET_COVERAGE
        == config.PREDICTION_INTERVAL_TARGET_COVERAGE
    )
    assert (
        prediction_report.PREDICTION_INTERVAL_MULTIPLIER
        == config.PREDICTION_INTERVAL_MULTIPLIER
    )


def test_skill_positions_match_clean_data():
    from src import clean_data

    assert set(clean_data.SKILL_POSITIONS) == set(config.SKILL_POSITIONS)
