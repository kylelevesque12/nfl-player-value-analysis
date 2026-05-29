# Measuring NFL Player Value and Salary Efficiency

## Executive Summary

This project builds an end-to-end NFL analytics workflow for measuring offensive player value, predicting next-season value, and evaluating salary efficiency. The analysis focuses on quarterbacks, running backs, wide receivers, and tight ends from 2016 through 2025.

The main value metric is position-season standardized total EPA. In the code, this column is named `value_score`, but the plain-English meaning is standardized EPA relative to players at the same position in the same season. For quarterbacks, value is based on passing plus rushing EPA. For running backs, wide receivers, and tight ends, value is based on rushing plus receiving EPA.

The project also builds a next-season prediction model and a first-pass salary-efficiency analysis using historical contract data. The final outputs include cleaned datasets, notebooks, reusable Python source code, a 2026 Excel prediction report, salary-efficiency findings, and a reproducible command-line pipeline.

## Project Question

The central question is:

How can NFL offensive player value be measured, predicted, and compared to salary in a way that is transparent, position-aware, and useful for football decision-making?

The project answers this in three stages:

1. Build a production-based player value score.
2. Predict next-season player value using current and historical features.
3. Compare value to contract cost to identify salary-efficient and salary-inefficient player-seasons.

## Data Sources

The project uses nflverse data loaded with `nflreadpy`:

- weekly player statistics
- roster data
- schedule data

The salary-efficiency section uses nflverse historical contract data sourced from OverTheCap. The current cost variable is `inflated_apy`, which adjusts contract average annual value across seasons. This is useful for first-pass contract-cost comparisons, but it is not the same as exact cap hit or cash paid.

Raw and processed data are intentionally excluded from GitHub. Final lightweight output tables and reports are included so the project can be reviewed without committing large raw files.

## Data Cleaning

The raw weekly player data is filtered to regular-season offensive skill players:

- QB
- RB
- WR
- TE

Weekly rows are aggregated to player-season data. The project collapses multi-team stints to one player-season row before value scoring so traded players are not split into misleading partial samples.

The cleaned dataset includes production totals, per-game rates, roster context, age, years of experience, and draft information. It also keeps position-specific features separate. For example, the project avoids using a universal yards-per-touch metric because passing yards and scrimmage touches are not compatible. Instead:

- QBs use QB-specific play and EPA features.
- RBs, WRs, and TEs use scrimmage production and scrimmage EPA features.

## Value Score Methodology

The primary value metric is `value_score`, which can be read as position-season standardized total EPA.

For quarterbacks:

`value_epa_total = passing_epa + rushing_epa`

For running backs, wide receivers, and tight ends:

`value_epa_total = rushing_epa + receiving_epa`

Then, within each season-position group:

`value_score = z-score(value_epa_total)`

This means:

- `0.0` is average for that position in that season.
- `1.0` is one standard deviation above that position-season average.
- `-1.0` is one standard deviation below that position-season average.
- A `2024 WR` and a `2024 RB` can be compared after standardization, but each was first judged against their own position group.

Calling this metric `standardized EPA` would be reasonable. I keep `value_score` as the code column because it is short and works cleanly across the prediction and salary-efficiency pipeline, but the report treats it as standardized EPA. The most precise label is position-season standardized total EPA because it states both the grouping and the EPA version.

The project also keeps `value_epa_total` and `value_epa_per_game` as supporting interpretation columns. These are useful because they show the raw EPA scale behind the standardized score.

## Why Not Only Model Raw EPA?

Directly modeling raw EPA is tempting because EPA is already a football value metric. The issue is that raw EPA is not always the best comparison target by itself.

Raw EPA has several interpretation problems:

- It is heavily affected by opportunity and playing time.
- QBs naturally accumulate much more EPA than most non-QBs because they touch the ball on nearly every offensive play.
- Raw totals can reward volume as much as efficiency.
- Per-game EPA can overrate smaller samples if a player was highly efficient in fewer games.
- Offensive environment matters: scheme, quarterback quality, offensive line, teammates, and play-calling all affect EPA.
- Salary comparisons become less fair if players are not first compared within position.

The standardized EPA value score solves a different problem. It asks:

How much better or worse was this player than other players at the same position in the same season?

That is more useful for ranking, modeling, and salary-efficiency analysis. Raw EPA is still important, but in this project it is treated as the underlying production signal, while `value_score` is the comparison metric.

A reasonable future extension would be to model both:

- raw `value_epa_total` to estimate actual production volume
- standardized `value_score` to estimate relative player value

For this project, the standardized score is the better primary target because it is more position-aware and easier to compare across player types.

## Exploratory Analysis

The exploratory notebooks examine value by:

- position
- season
- player age
- years of experience
- total EPA
- per-game EPA
- salary tier

One important finding is that high-end value is concentrated among a small number of elite player-seasons. This is expected in NFL offensive data because a few players drive a large share of passing and receiving efficiency.

Another important finding is that `value_epa_per_game` and total value do not always tell the same story. Per-game production is useful context, but total EPA better reflects a full-season contribution. The project keeps both views and uses the gap between them as a diagnostic.

## Predictive Modeling

The modeling target is next-season value score. This is more useful than predicting current-season value because it asks whether past production, age, usage, and experience can predict future performance.

The final value model is an enhanced-history Random Forest regression model. It uses:

- current-season production
- age
- years of experience
- draft information
- current EPA production
- prior value score
- rolling value averages
- value trend
- recent games played

The project also includes an availability model. This model estimates whether a player is likely to have a qualifying next-season row. This helps separate player value from the risk that a player may not have enough future playing time to qualify.

## Model Validation

The value model is evaluated with rolling validation rather than a random split. This is important because NFL forecasting is time-based: future seasons should be predicted from past seasons, not from randomly mixed data.

Current validation results:

| Metric | Result |
| --- | ---: |
| Rolling-validation RMSE | 0.92 |
| Rolling-validation MAE | 0.68 |
| Approximate 80% interval coverage | 83.9% |
| Availability model ROC AUC | 0.79 |

Position-level RMSE:

| Position | RMSE |
| --- | ---: |
| QB | 0.88 |
| RB | 1.02 |
| TE | 0.91 |
| WR | 0.89 |

The model is not precise enough to treat individual rankings as guarantees. That is normal for sports forecasting. The strongest use case is tiering and screening: identifying players who project well, players with wide uncertainty, and players whose future value is harder to trust.

## Model Interpretation

The model interpretation report adds a more direct explanation of what drives the prediction model. In the 2024 validation fold, the strongest permutation signal is `value_epa_total`, followed by recent multi-year value features such as `value_score_last3_avg` and `value_score_last2_avg`.

This is a useful result because it matches the football intuition behind the project: recent production matters, but multi-year history helps stabilize noisy one-season outcomes.

The project also compares pooled and position-specific models. Position-specific models slightly improve RMSE for RBs and WRs, but the differences are small. The pooled model remains the preferred production model because it is simpler, uses more training data, and performs similarly across positions.

## Context Feature Impact

The project now tests whether additional football context actually improves the model instead of simply adding more variables. The context-feature workflow creates usage, team-environment, and schedule-context features, then compares each group with rolling validation.

The best context feature set in the current test is `baseline_plus_team_context`. It slightly improves average rolling-validation RMSE from 0.925 to 0.922 and improves Spearman rank correlation from 0.417 to 0.429. This is directionally useful, but small enough to treat as roughly neutral rather than a major breakthrough.

This finding is still valuable. It shows that team context and role-share variables may add signal, while also keeping the project honest about the fact that more features do not automatically create a better model. The current production prediction report remains conservative, and the context results are documented separately in `report/context_feature_impact.md`.

## Methodology Checks

The project now includes a methodology-check report that audits common project-quality risks. The current local run has 26 passing checks and no failing checks.

The checks verify that:

- raw and processed data are not tracked in Git
- local raw and processed files exist for reproduction
- cleaned player-season-team rows are unique after aggregation
- value scores are standardized correctly within season-position groups
- the final value-score data has one row per player-season
- prediction intervals are ordered correctly
- model features do not include next-season target columns
- salary merge quality is reported
- Markdown notebook mirrors exist as a GitHub-friendly fallback

This does not prove the model is perfect, but it makes the pipeline easier to audit and helps catch avoidable methodology mistakes.

## 2026 Prediction Report

The project generates a recruiter-facing Excel workbook:

`outputs/tables/2026_player_value_predictions.xlsx`

The report contains 505 player projections. It includes:

- predicted 2026 value score
- approximate prediction interval
- confidence level
- availability risk level
- position percentile
- plain-English prediction drivers
- validation summaries

Top projected 2026 player values in the current report:

| Player | Pos | 2025 Team | Predicted 2026 Value | Approx. 80% Interval | Qualifying Probability |
| --- | --- | --- | ---: | ---: | ---: |
| Amon-Ra St. Brown | WR | DET | 2.35 | 1.02 to 3.67 | 94.9% |
| George Kittle | TE | SF | 2.30 | 0.95 to 3.65 | 67.9% |
| Josh Allen | QB | BUF | 2.07 | 0.67 to 3.48 | 92.4% |
| Ja'Marr Chase | WR | CIN | 1.92 | 0.47 to 3.38 | 93.6% |
| Puka Nacua | WR | LA | 1.63 | 0.20 to 3.06 | 95.7% |

## Salary Efficiency

The salary-efficiency analysis merges player value scores with historical contract data. The current salary variable is `inflated_apy`, which is an inflation-adjusted average annual contract value.

The merge matched 4,569 of 4,753 value-score rows, for a 96.1% match rate.

The main salary-efficiency metric is:

`value_above_expected_salary = actual value_score - expected value_score`

Expected value is estimated with a regression model using:

- salary
- position
- age
- years of experience
- draft slot
- games played

Positive residuals identify players who produced more value than expected given their cost and context. Negative residuals identify players who produced less value than expected.

## Salary Efficiency Findings

The salary findings use a cleaner sample of 3,531 matched player-seasons with at least 8 games played.

Key findings:

- Median salary in the findings sample is about $2.4 million in inflated APY.
- The strongest individual surplus season is 2025 Puka Nacua.
- The strongest rookie-contract proxy season is also 2025 Puka Nacua.
- The top team-season by total surplus is 2018 Kansas City.
- Across all seasons, Kansas City has the highest total surplus in the filtered skill-position sample.
- High-cost RB seasons show negative average salary-efficiency residuals, which suggests RB contract timing and decline risk are important.

Top surplus player-seasons:

| Season | Player | Pos | Team | Salary Millions | Value Score | Value Above Expected |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 2025 | Puka Nacua | WR | LA | 1.37 | 6.06 | 5.96 |
| 2017 | Rob Gronkowski | TE | NE | 22.48 | 5.72 | 5.18 |
| 2017 | Alvin Kamara | RB | NO | 1.74 | 4.95 | 4.75 |
| 2022 | Travis Kelce | TE | KC | 21.75 | 5.17 | 4.47 |
| 2023 | CeeDee Lamb | WR | DAL | 5.32 | 4.53 | 4.17 |

Team-season surplus leaders:

| Season | Team | Player-Seasons | Total Salary Millions | Total Surplus |
| --- | --- | ---: | ---: | ---: |
| 2018 | KC | 10 | 68.58 | 14.03 |
| 2020 | GB | 15 | 101.55 | 14.01 |
| 2022 | KC | 13 | 131.47 | 13.04 |
| 2024 | BAL | 10 | 127.27 | 11.92 |
| 2017 | NE | 12 | 117.57 | 11.08 |

## Reproducibility

The main outputs can be rebuilt with:

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

This command rebuilds:

- cleaned player-season data
- value scores
- 2026 prediction outputs
- the Excel workbook
- salary-efficiency tables
- salary-efficiency findings
- methodology checks
- model interpretation diagnostics

The notebooks remain the narrative layer, while `src/` contains reusable project logic.

## Limitations

The value score is production-based. It does not fully isolate:

- offensive scheme
- quarterback quality for receivers
- offensive line effects
- play-calling
- injuries
- coaching changes
- teammate effects
- defensive attention
- future depth-chart changes

Tight ends are especially difficult because blocking value is not fully captured in the current production data.

The salary-efficiency section also has limitations:

- `inflated_apy` is not exact cap hit.
- It does not capture restructures, incentives, void years, dead cap, or cash timing.
- Team-level salary findings include only QB/RB/WR/TE rows, not full roster spending.
- Residuals are descriptive, not causal.

## Future Improvements

The strongest next improvements would be:

1. Add exact season-level cap hit or cash-paid data.
2. Compare raw EPA and standardized value score as parallel modeling targets.
3. Decide whether to promote team-context features into the production prediction model after reviewing stability by position.
4. Add richer external context such as offensive line metrics, quarterback situation, injuries, depth-chart changes, and coaching changes.
5. Build a Streamlit dashboard for filtering player predictions and salary-efficiency findings.
6. Extend the salary analysis to team-level cap allocation once better salary data is available.

## Conclusion

This project creates a full NFL player value workflow: raw data collection, cleaning, feature engineering, value scoring, modeling, prediction reporting, and salary-efficiency analysis.

The main takeaway is that position-season standardized total EPA is a practical compromise. It preserves EPA as the underlying football production signal, but transforms it into a position-aware comparison metric. That makes it more useful for cross-position ranking, prediction, and salary-efficiency analysis than raw EPA alone.

The prediction model should be used as a screening tool rather than a ranking oracle. The salary-efficiency results should be read as first-pass contract-cost findings rather than final cap accounting. Within those limits, the project is now reproducible, interpretable, and ready to present as a portfolio-quality data science project.
