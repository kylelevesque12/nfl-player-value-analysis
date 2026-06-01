# Fantasy Football Projection Summary

This report adds a fantasy-football view to the project by projecting 2026 season-long PPR fantasy points from 2025 player-season production, recent history, usage, and the existing EPA-based value features.

The upgraded version compares several model families: a 2025 baseline, Ridge, Elastic Net, Random Forest, Histogram Gradient Boosting, and a two-stage model that predicts games played and PPR per game separately.

The final projection model is selected by the lowest rolling-validation RMSE. The two-stage model is still reported as an interpretation aid because it translates fantasy value into projected games and scoring rate.

The model is still not a finished fantasy ranking system. It does not yet include rookies, depth-chart changes, injuries, coaching changes, betting markets, or manual playing-time projections.

To make the output easier to use, each player row includes a projection change label, a usage profile, breakout and slump potential labels, a draft-board bucket, and a plain-English fantasy explanation.

Projected players: 505

Rolling validation rows: 2,414

Selected model: Elastic Net Total-PPR Model

Selection reason: lowest_overall_rolling_rmse

Overall rolling MAE: 41.60 PPR points

Overall rolling RMSE: 59.09 PPR points

Overall Spearman rank correlation: 0.722

Top-rank hit rate: 0.615
