# Model Benchmark: Baselines, Skill Score, and Calibrated Intervals

All numbers come from rolling-origin validation: each season is predicted using only earlier seasons. The target `next_value_score` is standardized within each season-position group, so its per-season standard deviation is approximately 1.0. That means predicting the group mean already yields an RMSE near 1.0, and **RMSE alone overstates model quality**. The honest measure is the *skill score*: the percentage RMSE reduction versus a strong naive baseline.

## Overall results

| Method | Type | RMSE | MAE | R² | Skill vs shrunken persistence | Skill vs age curve |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random_forest | model | 0.925 | 0.675 | 0.231 | 4.3% | 12.1% |
| gradient_boosting | model | 0.958 | 0.694 | 0.175 | 0.9% | 9.0% |
| shrunken_persistence | baseline | 0.966 | 0.710 | 0.160 | 0.0% | 8.2% |
| age_curve | baseline | 1.052 | 0.789 | 0.004 | -8.9% | 0.0% |
| season_mean | baseline | 1.054 | 0.780 | -0.000 | -9.1% | -0.2% |
| persistence | baseline | 1.161 | 0.841 | -0.213 | -20.2% | -10.3% |

The lowest-RMSE method overall is **random_forest** (RMSE 0.925). The best learned model is **random_forest**, which reduces RMSE by 4.3% versus shrunken persistence. A small or negative skill score is itself an important, honest finding: it means a one-line baseline is hard to beat for this target, and the model is best used for tiering rather than precise ranking.

## Skill score by position (vs shrunken persistence)

| Position | Method | RMSE | R² | Skill vs shrunken persistence |
| --- | --- | ---: | ---: | ---: |
| QB | random_forest | 0.877 | 0.292 | 2.7% |
| QB | gradient_boosting | 0.981 | 0.114 | -8.8% |
| RB | random_forest | 1.017 | 0.007 | 3.3% |
| RB | gradient_boosting | 1.052 | -0.063 | -0.1% |
| TE | random_forest | 0.908 | 0.287 | 5.3% |
| TE | gradient_boosting | 0.909 | 0.286 | 5.2% |
| WR | random_forest | 0.887 | 0.301 | 5.0% |
| WR | gradient_boosting | 0.916 | 0.255 | 1.9% |

## Split-conformal prediction intervals

Split-conformal intervals targeting 80.0% coverage achieved 81.1% empirical coverage overall (mean width 2.30). Unlike the Gaussian-style bands in the main report, conformal intervals are distribution-free and calibrated by construction.

| Segment | Coverage | Target | Gap | Mean width |
| --- | ---: | ---: | ---: | ---: |
| all | 81.1% | 80.0% | 1.1% | 2.30 |
| QB | 75.5% | 80.0% | -4.5% | 2.30 |
| RB | 77.5% | 80.0% | -2.5% | 2.30 |
| TE | 82.9% | 80.0% | 2.9% | 2.30 |
| WR | 84.0% | 80.0% | 4.0% | 2.30 |
