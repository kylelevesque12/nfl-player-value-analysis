# Advanced Modeling Methodology

This report adds a more formal modeling layer on top of the existing NFL player value pipeline. The point is not to make the model look more complicated. The point is to make the tuning, explanations, and data checks easier to defend.

## What This Adds

- Optuna searches Random Forest hyperparameters across rolling validation folds.
- SHAP explains which transformed features most influence the selected tree model.
- Polars profiles the cleaned modeling data quickly, mostly as a data-quality aid.
- MLflow stores a local experiment run when the package is installed. The committed CSV and Markdown outputs remain the reviewer-friendly version.

## Current Model vs Optuna-Tuned Candidate

| model_id | validation_folds | mean_mae | mean_rmse | mean_r2 | rmse_delta_vs_current |
| --- | --- | --- | --- | --- | --- |
| optuna_tuned_random_forest | 5 | 0.6728 | 0.9207 | 0.2367 | -0.0040 |
| current_depth_limited_random_forest | 5 | 0.6760 | 0.9247 | 0.2302 | 0.0000 |

## Best Optuna Trial

| trial_number | mean_rmse | mean_mae | mean_r2 |
| --- | --- | --- | --- |
| 17.0000 | 0.9207 | 0.6728 | 0.2367 |

Best parameter set:

```json
{
  "n_estimators": 400,
  "max_depth": 12,
  "max_features": "sqrt",
  "min_samples_leaf": 14,
  "random_state": 42,
  "n_jobs": -1,
  "min_samples_split": 30
}
```

## SHAP Feature Importance

SHAP values are calculated on the 2024 validation fold. This is an interpretation diagnostic, not a causal claim. Correlated football features can share importance with each other.

| transformed_feature | raw_feature | feature_group | mean_abs_shap |
| --- | --- | --- | --- |
| value_epa_total | value_epa_total | current_season_production | 0.1108 |
| value_epa_per_game | value_epa_per_game | current_season_production | 0.0938 |
| value_score_last3_avg | value_score_last3_avg | player_history | 0.0539 |
| value_score_last2_avg | value_score_last2_avg | player_history | 0.0488 |
| yards_per_game | yards_per_game | current_season_production | 0.0392 |
| value_epa_total_prev | value_epa_total_prev | player_history | 0.0274 |
| games_played | games_played | current_season_production | 0.0271 |
| value_score_prev | value_score_prev | player_history | 0.0253 |
| tds_per_game | tds_per_game | current_season_production | 0.0197 |
| years_exp | years_exp | player_profile | 0.0188 |
| age | age | player_profile | 0.0183 |
| value_epa_per_game_prev | value_epa_per_game_prev | player_history | 0.0167 |
| position_RB | position | position_indicator | 0.0163 |
| yards_per_game_prev | yards_per_game_prev | player_history | 0.0130 |
| position_WR | position | position_indicator | 0.0126 |

## SHAP Feature Groups

| feature_group | total_mean_abs_shap | feature_count |
| --- | --- | --- |
| current_season_production | 0.2905 | 5 |
| player_history | 0.2197 | 12 |
| player_profile | 0.0484 | 3 |
| position_indicator | 0.0389 | 4 |

## Status

- optuna: optuna_lowest_rolling_rmse
- shap: shap_tree_explainer_2024_fold
- polars: polars_profile_cleaned_player_seasons
- mlflow: logged_local_mlruns

## Interpretation

If Optuna only improves RMSE by a tiny amount, I would not automatically replace the current model. In a sports forecasting project, small metric wins can be noise. The more important result is whether the tuned model is consistently better across seasons and still simple enough to explain.
