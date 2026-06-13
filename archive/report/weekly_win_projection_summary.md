# Weekly Win Projection Summary

This dashboard layer predicts the probability that the home team wins a regular-season game. The current version is a historical rolling backtest, so each season is predicted using only earlier seasons.

Features include market line context, rest, divisional-game status, weather when available, and each team's recent in-season form. Because sportsbook lines are included, this should be read as a market-informed projection rather than a pure team-strength model.

When future schedule rows are added locally, this same feature pipeline can be extended to score upcoming weeks.

To make the table easier to interpret, each game row includes a market signal and a short pick explanation based on spread, recent form, and rest.

Backtested games: 1,610

Overall rolling accuracy: 0.671

Overall Brier score: 0.211

Overall ROC AUC: 0.726
