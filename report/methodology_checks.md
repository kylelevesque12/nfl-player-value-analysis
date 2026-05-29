# Methodology Checks

This report records reproducibility and methodology checks for the NFL player value project. The goal is to make the project easier to audit: data files should be handled correctly, value scores should follow the stated definition, and prediction outputs should avoid obvious leakage problems.

## Summary

- Passed checks: 26
- Warning checks: 0
- Failed checks: 0

No failing methodology checks were found in the current local project state.

## Checks

| check_name | status | value | threshold | detail |
| --- | --- | --- | --- | --- |
| gitignore_contains_data/raw | PASS | True | True | `data/raw/` is listed in .gitignore. |
| gitignore_contains_data/processed | PASS | True | True | `data/processed/` is listed in .gitignore. |
| gitignore_contains_.ipynb_checkpoints | PASS | True | True | `.ipynb_checkpoints/` is listed in .gitignore. |
| gitignore_contains_.venv | PASS | True | True | `.venv` is listed in .gitignore. |
| no_tracked_data_raw | PASS | 0 | 0 | Raw and processed data should stay local and out of Git. |
| no_tracked_data_processed | PASS | 0 | 0 | Raw and processed data should stay local and out of Git. |
| local_file_exists_player_stats_2016_2025.csv | PASS | True | True | Required local data file is available for reproducing outputs. |
| local_file_exists_rosters_2016_2025.csv | PASS | True | True | Required local data file is available for reproducing outputs. |
| local_file_exists_schedules_2016_2025.csv | PASS | True | True | Required local data file is available for reproducing outputs. |
| local_file_exists_skill_player_seasons_2016_2025.csv | PASS | True | True | Required local data file is available for reproducing outputs. |
| local_file_exists_player_value_scores_2016_2025.csv | PASS | True | True | Required local data file is available for reproducing outputs. |
| no_duplicate_player_season_team_rows | PASS | 0 | 0 | Cleaned data should have at most one row per player-season-team. |
| only_skill_positions_in_cleaned_data | PASS | QB, RB, TE, WR | QB, RB, TE, WR | Cleaned data is restricted to QB, RB, WR, and TE. |
| no_duplicate_value_player_seasons | PASS | 0 | 0 | Value scoring should collapse multi-team stints to one player-season. |
| minimum_games_filter_applied | PASS | 4 | >= 4 | Value-score rows should meet the minimum-games threshold. |
| value_score_group_means_near_zero | PASS | 8.458842092382145e-17 | <= 1e-9 | Standardized value scores should average near zero within season-position groups. |
| value_score_group_stds_near_one | PASS | 6.661338147750939e-16 | <= 1e-9 | Standardized value scores should have sample standard deviation near one within season-position groups. |
| value_epa_total_matches_position_definition | PASS | 0.0 | <= 1e-8 | QBs use QB EPA; RB/WR/TE use scrimmage EPA. |
| position_percentiles_in_range | PASS | True | True | Position-season percentiles should stay between 0 and 1. |
| prediction_report_has_rows | PASS | 505 | > 0 | 2026 prediction report should contain player projections. |
| prediction_intervals_ordered | PASS | True | True | Prediction interval low values should not exceed high values. |
| predictions_are_finite | PASS | True | True | Predicted value scores should be finite numeric values. |
| prediction_features_do_not_use_next_season_columns | PASS | none | none | The production prediction model should not use future target columns. |
| context_features_not_blindly_added_to_main_model | PASS | none | none | Context features were tested separately and not automatically added to the production model. |
| salary_merge_match_rate_above_90_percent | PASS | 0.961287607827 | >= 0.90 | Salary-efficiency analysis should report a high match rate. |
| markdown_notebook_mirrors_exist | PASS | 8 | >= 8 | Markdown notebook mirrors provide a GitHub-friendly fallback when notebook preview fails. |

## Notes

- These checks do not prove the model is correct; they catch common project-quality problems.
- Raw and processed data are still intentionally local because they can be regenerated.
- The prediction model is still best interpreted as a screening and tiering tool, not an exact ranking system.
