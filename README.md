# NFL Player Value Analysis

End-to-end NFL analytics project measuring offensive player value with nflverse data, feature engineering, exploratory analysis, predictive modeling, and 2026 projection reporting.

## Project Question

How can we measure and predict NFL offensive player value in a way that is transparent, position-aware, and useful for salary-efficiency analysis later?

The project currently focuses on QBs, RBs, WRs, and TEs. The main value metric is EPA-based production standardized within season-position groups, so players are compared against peers at the same position in the same season.

## Current Deliverable: 2026 Prediction Report

The project includes a recruiter-facing Excel report built from the predictive modeling pipeline:

- [Download the 2026 Player Value Predictions Excel report](outputs/tables/2026_player_value_predictions.xlsx)
- [View the player prediction CSV](outputs/tables/2026_player_value_predictions.csv)
- [Read the report summary](report/2026_prediction_report_summary.md)

The report includes:

- predicted 2026 value score
- probability of a qualifying 2026 season
- availability-adjusted value
- confidence level
- availability risk level
- plain-English prediction drivers
- team and position summaries

## Method Summary

1. Load nflverse player stats, rosters, and schedules.
2. Clean weekly player data into player-season data.
3. Engineer EPA-based value scores by position and season.
4. Explore value by position, age, experience, and production profile.
5. Train models to predict next-season value.
6. Generate a 2026 Excel prediction report using 2025 player-season inputs.

## Modeling Notes

The 2026 report uses two modeling layers:

- a value model that predicts next-season position-adjusted value score
- an availability model that estimates whether a player will have a qualifying next-season row

This helps avoid hiding survivorship risk inside the value prediction. The report should be interpreted as a screening tool for deeper football analysis, not as a guarantee of future player performance.

## Limitations

EPA-based value captures production, not pure individual talent. It does not fully isolate scheme, offensive line quality, teammate effects, injuries, depth-chart changes, coaching changes, or future roster movement. Tight ends are especially affected because blocking value is not fully represented in the current data.

Raw and processed data files are intentionally excluded from GitHub.
