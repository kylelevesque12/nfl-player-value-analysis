# 2026 NFL Player Value Prediction Report

This report turns the modeling pipeline into a recruiter-facing analytics artifact. It uses 2025 player-season data to project 2026 offensive player value for QBs, RBs, WRs, and TEs.

## Report Files

- [2026 Player Value Predictions Excel report](../outputs/tables/2026_player_value_predictions.xlsx)
- [2026 Player Value Predictions CSV](../outputs/tables/2026_player_value_predictions.csv)
- [Availability validation metrics CSV](../outputs/tables/2026_availability_validation_metrics.csv)

## What The Report Includes

- `predicted_2026_value_score`: projected next-season position-adjusted value score
- `predicted_2026_qualifying_probability`: probability the player has a qualifying 2026 season
- `availability_adjusted_2026_value`: projected value weighted by qualifying probability
- `availability_risk_level`: Low, Medium, or High availability risk
- `confidence_level`: practical confidence label based on uncertainty and sample size
- `prediction_driver`: short explanation of the main signals behind the projection

## Model Design

The report uses two modeling layers:

1. A Random Forest regression model predicts next-season value score.
2. A Random Forest classification model estimates next-season qualifying availability.

The feature set includes 2025 production, age, experience, draft information, EPA-based value, and multi-year history features such as prior value, rolling value averages, trend, and recent games played.

## Validation Snapshot

- 501 player projections
- value model rolling-validation RMSE: about 0.93
- availability model rolling-validation mean ROC AUC: about 0.78

## How To Interpret It

This is a screening and prioritization tool. It is useful for finding players worth deeper review, especially when a player has strong projected value, high qualifying probability, and interpretable driver notes.

It should not be interpreted as a guaranteed ranking of 2026 performance.

## Limitations

The model does not know future injuries, free agency moves, coaching changes, depth-chart changes, rookies, or 2026 team context. EPA-based production also reflects team environment and usage, not only individual player talent.
