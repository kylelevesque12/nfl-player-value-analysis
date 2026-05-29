# Measuring NFL Player Value and Salary Efficiency

## Summary

This project builds an end-to-end NFL analytics workflow for measuring offensive player value, forecasting next-season production, and evaluating salary efficiency. It uses nflverse player data, pandas cleaning, feature engineering, exploratory analysis, model validation, contract data, and a recruiter-facing Excel report.

The current project phase focuses on QBs, RBs, WRs, and TEs. The main value metric is position-adjusted total EPA, standardized within season-position groups. Quarterbacks use passing plus rushing EPA; running backs, wide receivers, and tight ends use rushing plus receiving EPA.

## Current Deliverable

The primary deliverables are the 2026 Player Value Prediction Report and the first-pass Salary Efficiency Analysis:

- Excel workbook: `outputs/tables/2026_player_value_predictions.xlsx`
- Player prediction CSV: `outputs/tables/2026_player_value_predictions.csv`
- Validation by position: `outputs/tables/2026_value_validation_by_position.csv`
- Prediction interval calibration: `outputs/tables/2026_prediction_interval_validation.csv`
- Salary-efficiency results: `outputs/tables/salary_efficiency_2016_2025.csv`
- Salary-efficiency diagnostics: `outputs/tables/salary_efficiency_merge_diagnostics.csv`
- Salary-efficiency summary: `report/salary_efficiency_summary.md`
- Salary-efficiency findings: `report/salary_efficiency_findings.md`

The workbook has a simplified front-facing layer for decision makers and separate audit tabs for technical review.

## Reproducibility

The main project outputs can be rebuilt with:

`python scripts/run_pipeline.py`

This command rebuilds the cleaned player-season data, value scores, 2026 prediction outputs, Excel workbook, salary-efficiency tables, and salary-efficiency findings from the local raw data files.

## Modeling Approach

The prediction report uses two models:

- An enhanced-history Random Forest regression model predicts next-season value score.
- A Random Forest classification model estimates the probability that a player has a qualifying next-season row.

The value model uses current-season production, age, experience, draft information, EPA-based value, prior value, rolling value averages, trend, and recent games played.

The salary-efficiency analysis merges value scores with nflverse historical contract data from OverTheCap. It uses inflated APY as an approximate annual contract-cost metric, then estimates value above expected salary using salary, position, age, experience, draft slot, and games played.

## Validation Snapshot

- 505 player projections in the 2026 report.
- Rolling-validation RMSE is about 0.92.
- Rolling-validation MAE is about 0.68.
- Approximate central 80% prediction intervals covered about 83.9% of historical rolling-validation outcomes.
- The availability model has mean rolling-validation ROC AUC of about 0.79.
- Salary-efficiency merge match rate is about 96.1%.
- Salary-efficiency findings use a filtered sample of 3,531 matched player-seasons with at least 8 games played.

## Interpretation

The prediction model should be treated as a screening and prioritization tool, not a precise ranking engine. Sports forecasting is noisy, so the strongest output is the combination of projected value, prediction interval, availability risk, and plain-English driver notes.

The salary-efficiency analysis should also be treated as a first-pass contract-efficiency view. It is useful for finding possible surplus-value and underperforming contracts, but exact cap-hit or cash-paid data would make it stronger.

## Limitations

EPA-based value captures production, not pure individual talent. It does not fully isolate injuries, scheme, offensive line quality, coaching changes, future depth charts, contract incentives, or teammate effects. Tight ends are especially affected because blocking value is not fully captured in the current dataset.

The salary-efficiency section uses inflated APY from historical contracts, not exact season-level cap hit. Contract restructures, incentives, void years, dead cap, and in-season movement are not fully modeled.

## Next Phase

The next major project improvement is replacing approximate contract APY with true season-level cap hit or cash paid, then extending the analysis to team-level cap allocation.
