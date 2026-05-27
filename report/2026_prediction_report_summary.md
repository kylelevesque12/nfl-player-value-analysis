# 2026 NFL Player Value Prediction Report

This report turns the modeling pipeline into a recruiter-facing analytics artifact. It uses 2025 player-season data to project 2026 offensive player value for QBs, RBs, WRs, and TEs.

## Report Files

- [2026 Player Value Predictions Excel report](../outputs/tables/2026_player_value_predictions.xlsx)
- [2026 Player Value Predictions CSV](../outputs/tables/2026_player_value_predictions.csv)
- [Value-model validation by position CSV](../outputs/tables/2026_value_validation_by_position.csv)
- [Prediction interval calibration CSV](../outputs/tables/2026_prediction_interval_validation.csv)
- [Availability validation metrics CSV](../outputs/tables/2026_availability_validation_metrics.csv)

## What The Report Includes

The Excel workbook starts with a simplified decision layer, then keeps the technical details in later tabs for auditability.

- `predicted_2026_value_score`: projected next-season position-adjusted value score
- `predicted_2026_qualifying_probability`: probability the player has a qualifying 2026 season
- `availability_adjusted_2026_value`: projected value weighted by qualifying probability
- `prediction_interval_low` and `prediction_interval_high`: approximate central 80% range around the projection
- `availability_risk_level`: Low, Medium, or High availability risk
- `confidence_level`: practical confidence label based on uncertainty and sample size
- `prediction_driver`: short explanation of the main signals behind the projection
- value-model validation by position: rolling-validation error for QBs, RBs, WRs, and TEs

Front-facing workbook tabs:

- `Dashboard`
- `Player Predictions`
- `Team Summary`
- `Position Summary`
- `Validation Summary`

Audit/detail tabs:

- `Full Model Data`
- `Data Dictionary`
- `Model Notes`

## Model Design

The report uses two modeling layers:

1. An enhanced-history Random Forest regression model predicts next-season value score.
2. A Random Forest classification model estimates next-season qualifying availability.

The feature set includes 2025 production, age, experience, draft information, EPA-based value, and multi-year history features such as prior value, rolling value averages, trend, and recent games played. The model is tuned directly on the enhanced-history feature set using rolling, time-aware validation. The final value model uses a depth-limited Random Forest because the unconstrained-depth and depth-limited versions were effectively tied, and the simpler model is easier to defend.

Player value scores are rebuilt from the cleaned player-season-team data by collapsing multi-team stints before the minimum-games filter. That prevents traded players from being scored as separate partial-season samples.

## Validation Snapshot

- 505 player projections
- value model rolling-validation RMSE: about 0.92
- central 80% prediction interval rolling-validation coverage: about 83.9%
- availability model rolling-validation mean ROC AUC: about 0.79

## How To Interpret It

This is a screening and prioritization tool. It is useful for finding players worth deeper review, especially when a player has strong projected value, high qualifying probability, reasonable prediction interval, and interpretable driver notes.

It should not be interpreted as a guaranteed ranking of 2026 performance. Limited precision is expected in sports forecasting; the value is in calibrated ranges, tiers, and transparent uncertainty.

## Limitations

The model does not know future injuries, free agency moves, coaching changes, depth-chart changes, rookies, or 2026 team context. EPA-based production also reflects team environment and usage, not only individual player talent.
