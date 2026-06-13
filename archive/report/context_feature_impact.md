# Context Feature Impact Review

This review tests whether contextual football features improve the next-season value model. The comparison uses the same rolling-season setup as the main modeling workflow: train on earlier seasons, validate on a future season, and repeat across 2020-2024.

The goal is not to keep every possible football feature. The goal is to add context only when it improves out-of-sample performance or makes the model more explainable.

## Feature Sets Tested

- `baseline`: current production, age/draft inputs, and multi-year history features.
- `baseline_plus_usage_context`: baseline plus target share, air-yards share, WOPR, PACR/RACR, and CPOE.
- `baseline_plus_team_context`: baseline plus team volume, team efficiency, and player role-share features.
- `baseline_plus_schedule_context`: baseline plus rest, spread, total, home/road, division, roof, surface, temperature, and wind context.
- `baseline_plus_all_context`: baseline plus every context group above.

## Rolling-Validation Summary

| feature_set | feature_count | context_feature_count | avg_mae | avg_rmse | rmse_delta_vs_baseline | avg_spearman_rank_corr | impact_label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_plus_team_context | 35 | 14 | 0.671 | 0.922 | -0.003 | 0.429 | roughly neutral |
| baseline_plus_all_context | 54 | 33 | 0.674 | 0.922 | -0.002 | 0.424 | roughly neutral |
| baseline | 21 | 0 | 0.676 | 0.925 | 0.000 | 0.417 | roughly neutral |
| baseline_plus_schedule_context | 34 | 13 | 0.678 | 0.925 | 0.001 | 0.415 | roughly neutral |
| baseline_plus_usage_context | 27 | 6 | 0.676 | 0.926 | 0.001 | 0.412 | roughly neutral |

Lower RMSE and MAE are better. Higher Spearman correlation is better because it means the model is doing a better job sorting players into relative order.

## Current Read

The baseline average RMSE is `0.925`. The best tested feature set is `baseline_plus_team_context` with average RMSE `0.922`.
The best context feature set is very close to baseline, so the safer interpretation is that the new features are roughly neutral until more evidence is added.

## 2024 Permutation Importance

Permutation importance estimates how much the 2024 validation score worsens when a feature is randomly shuffled. Positive values suggest the feature carried useful signal in that fold.

| feature | feature_group | importance_mean | importance_std |
| --- | --- | --- | --- |
| value_epa_total | baseline_current_season | 0.021 | 0.008 |
| value_score_last2_avg | baseline_history | 0.010 | 0.002 |
| value_score_last3_avg | baseline_history | 0.008 | 0.002 |
| position | baseline_categorical | 0.007 | 0.001 |
| player_target_share_team | team_context | 0.007 | 0.001 |
| value_score_prev | baseline_history | 0.004 | 0.001 |
| avg_target_share | usage_context | 0.003 | 0.001 |
| age | baseline_current_season | 0.003 | 0.001 |
| player_scrimmage_touch_share_team | team_context | 0.003 | 0.001 |
| avg_wopr | usage_context | 0.003 | 0.001 |
| player_receiving_air_yards_share_team | team_context | 0.003 | 0.001 |
| value_epa_per_game | baseline_current_season | 0.002 | 0.004 |
| avg_air_yards_share | usage_context | 0.001 | 0.001 |
| avg_wind | schedule_context | 0.001 | 0.000 |
| avg_racr | usage_context | 0.001 | 0.000 |
| years_exp | baseline_current_season | 0.001 | 0.001 |
| draft_number | baseline_current_season | 0.001 | 0.000 |
| yards_per_game_prev | baseline_history | 0.001 | 0.000 |
| games_played_last3_avg | baseline_history | 0.000 | 0.000 |
| value_epa_total_prev | baseline_history | 0.000 | 0.000 |

## Method Notes

- Context features are built from current-season information only, then tested against next-season value.
- The schedule features describe the games the player actually played in that season. They should not be used as future-season features unless a separate preseason schedule forecast is built.
- Team EPA context is treated as an environment signal, not a new value metric.
- A feature group should be adopted only if it helps validation or gives a clear interpretability benefit without creating leakage.
