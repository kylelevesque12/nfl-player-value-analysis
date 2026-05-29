# Model Interpretation

This report explains what the prediction model appears to use and where it struggles. The model should be read as a screening tool for player tiers and uncertainty, not as an exact ranking engine.

## Main Takeaways

- The strongest 2024 permutation signal is `value_epa_total`.
- Recent production and multi-year player history carry most of the useful signal.
- Position-specific models are tested as a diagnostic, but the pooled model remains easier to explain and usually has more stable training data.
- Sports forecasting is noisy; the model is more useful for grouping players than for claiming exact future ranks.

## Feature Importance By Group

| feature_group | feature_count | total_importance | mean_importance |
| --- | --- | --- | --- |
| current_season_production | 5 | 0.048 | 0.010 |
| player_history | 12 | 0.030 | 0.003 |
| player_profile | 3 | 0.008 | 0.003 |
| position_indicator | 1 | 0.007 | 0.007 |

## Top Feature Importance Rows

| feature | feature_group | importance_mean | importance_std |
| --- | --- | --- | --- |
| value_epa_total | current_season_production | 0.039 | 0.012 |
| value_score_last3_avg | player_history | 0.013 | 0.004 |
| value_score_last2_avg | player_history | 0.013 | 0.004 |
| position | position_indicator | 0.007 | 0.001 |
| value_score_prev | player_history | 0.006 | 0.003 |
| value_epa_per_game | current_season_production | 0.006 | 0.007 |
| draft_number | player_profile | 0.004 | 0.001 |
| age | player_profile | 0.003 | 0.003 |
| yards_per_game | current_season_production | 0.003 | 0.002 |
| games_played | current_season_production | 0.002 | 0.001 |
| years_exp | player_profile | 0.001 | 0.002 |
| value_epa_total_prev | player_history | 0.001 | 0.001 |
| yards_per_game_prev | player_history | 0.000 | 0.001 |
| games_played_prev | player_history | 0.000 | 0.000 |
| prior_qualifying_seasons | player_history | 0.000 | 0.001 |

## Position-Specific Model Comparison

| position | model_type | avg_train_rows | avg_rmse | rmse_delta_vs_pooled | avg_mae |
| --- | --- | --- | --- | --- | --- |
| QB | pooled_model | 2017.400 | 0.876 | 0.000 | 0.669 |
| QB | position_specific_model | 219.600 | 0.892 | 0.016 | 0.680 |
| QB | current_value_baseline | 2017.400 | 1.013 | 0.137 | 0.781 |
| RB | position_specific_model | 521.600 | 1.007 | -0.008 | 0.727 |
| RB | pooled_model | 2017.400 | 1.016 | 0.000 | 0.734 |
| RB | current_value_baseline | 2017.400 | 1.354 | 0.339 | 0.967 |
| TE | pooled_model | 2017.400 | 0.909 | 0.000 | 0.655 |
| TE | position_specific_model | 450.600 | 0.917 | 0.008 | 0.667 |
| TE | current_value_baseline | 2017.400 | 1.122 | 0.213 | 0.824 |
| WR | position_specific_model | 825.600 | 0.882 | -0.004 | 0.643 |
| WR | pooled_model | 2017.400 | 0.886 | 0.000 | 0.654 |
| WR | current_value_baseline | 2017.400 | 1.087 | 0.201 | 0.793 |

## Interpretation Notes

- Permutation importance was measured on the 2024 validation fold, so it should be treated as directional rather than permanent truth.
- A feature with low importance is not necessarily useless; it may overlap with stronger related features.
- Position-specific models can be appealing, but smaller samples make them easier to overfit.
- The production model intentionally remains conservative because added context features only produced small validation gains.
