# NFL Player Value Analysis

End-to-end NFL analytics project measuring offensive player value with nflverse data, feature engineering, exploratory analysis, predictive modeling, and 2026 projection reporting.

## Project Question

How can we measure and predict NFL offensive player value in a way that is transparent, position-aware, and useful for salary-efficiency analysis later?

The project currently focuses on QBs, RBs, WRs, and TEs. The main value metric is EPA-based production standardized within season-position groups, so players are compared against peers at the same position in the same season.

## Current Deliverable: 2026 Prediction Report

The project includes a recruiter-facing Excel report built from the predictive modeling pipeline:

- [Download the 2026 Player Value Predictions Excel report](outputs/tables/2026_player_value_predictions.xlsx)
- [View the player prediction CSV](outputs/tables/2026_player_value_predictions.csv)
- [View value-model validation by position](outputs/tables/2026_value_validation_by_position.csv)
- [View prediction interval calibration](outputs/tables/2026_prediction_interval_validation.csv)
- [Read the report summary](report/2026_prediction_report_summary.md)

The report includes:

- predicted 2026 value score
- probability of a qualifying 2026 season
- confidence level
- availability risk level
- approximate prediction interval
- plain-English prediction drivers
- value-model validation by position
- team and position summaries

The Excel workbook is organized with a clean front-facing layer first: `Dashboard`, `Player Predictions`, `Team Summary`, `Position Summary`, and `Validation Summary`. More technical columns are preserved in `Full Model Data`, `Data Dictionary`, and `Model Notes` for auditability.

## Method Summary

1. Load nflverse player stats, rosters, and schedules.
2. Clean weekly player data into player-season data.
3. Engineer EPA-based value scores by position and season.
4. Explore value by position, age, experience, and production profile.
5. Train models to predict next-season value.
6. Generate a 2026 Excel prediction report using 2025 player-season inputs.

## Modeling Notes

The 2026 report uses two modeling layers:

- an enhanced-history value model that predicts next-season position-adjusted value score
- an availability model that estimates whether a player will have a qualifying next-season row

The value model uses current-season production plus multi-year history features such as prior value, rolling value averages, trend, and recent games played. The report also includes rolling-validation error by position so pooled-model performance can be checked for QBs, RBs, WRs, and TEs.

Player value scores are rebuilt from the cleaned player-season-team data by collapsing multi-team stints before applying the minimum-games filter. This keeps traded-player seasons from being split into misleading partial samples.

The 2026 report includes approximate central 80% prediction intervals and checks their rolling-validation coverage. These intervals are meant to make uncertainty visible; the model is better suited for tiering and screening than exact player ranking.

This helps avoid hiding survivorship risk inside the value prediction. The report should be interpreted as a screening tool for deeper football analysis, not as a guarantee of future player performance.

## Limitations

EPA-based value captures production, not pure individual talent. It does not fully isolate scheme, offensive line quality, teammate effects, injuries, depth-chart changes, coaching changes, or future roster movement. Tight ends are especially affected because blocking value is not fully represented in the current data.

Raw and processed data files are intentionally excluded from GitHub.
