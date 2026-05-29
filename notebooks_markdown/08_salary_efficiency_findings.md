# Salary Efficiency Findings

This notebook turns the salary-efficiency output into a smaller set of findings. The goal is not just to list players, but to understand which player-seasons, teams, and position groups created the most contract-cost surplus.

Important caveat: salary is measured with `inflated_apy`, so this is a contract-efficiency view rather than exact cap-hit accounting.

## Load Findings Pipeline

The findings are built from `src.salary_findings` so the same tables can be regenerated from the command-line pipeline. This keeps the notebook focused on interpretation instead of copying analysis logic across places.


```python
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def find_project_root(expected_file="outputs/tables/salary_efficiency_2016_2025.csv"):
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / expected_file).exists():
            return candidate
    raise FileNotFoundError(f"Could not find {expected_file} from {Path.cwd()}")


project_root = find_project_root()
sys.path.insert(0, str(project_root))

from src.salary_findings import build_salary_finding_tables

project_root
```

## Build Finding Tables

I filter to matched salary rows with at least 8 games played. That removes many tiny-sample seasons while still keeping enough players for position and team comparisons.


```python
outputs = build_salary_finding_tables(project_root=project_root, save_outputs=True)
tables = outputs["tables"]

summary_metrics = tables["summary_metrics"]
top_surplus = tables["top_surplus"]
high_cost_underperformers = tables["high_cost_underperformers"]
rookie_surplus = tables["rookie_surplus"]
veteran_values = tables["veteran_values"]
team_season = tables["team_season"]
team_summary = tables["team_summary"]
position_salary_tiers = tables["position_salary_tiers"]
season_trends = tables["season_trends"]
finding_base = tables["finding_base"]

print("Finding sample:", finding_base.shape)
print("Report saved to:", outputs["report_path"])
```

## Summary Metrics

This quick snapshot checks the size of the findings sample and the main headline result. The sample is smaller than the full value-score dataset because it requires a salary match and a minimum-games filter.


```python
display(summary_metrics)
```

## Top Surplus Player-Seasons

These are the player-seasons where actual value score was farthest above expected value after accounting for salary, position, age, experience, draft slot, and games played.


```python
display(
    top_surplus[[
        "season", "player_display_name", "position", "team",
        "games_played", "salary_millions", "value_score",
        "value_above_expected_salary"
    ]].head(15).round(3)
)
```

## High-Cost Underperformers

This table only considers players at or above the 75th salary percentile within their season-position group. I read this as contract-cost underperformance, not exact cap-hit underperformance.


```python
display(
    high_cost_underperformers[[
        "season", "player_display_name", "position", "team",
        "games_played", "salary_millions", "value_score",
        "value_above_expected_salary"
    ]].head(15).round(3)
)
```

## Rookie-Contract Proxy Surplus

This view uses `years_exp <= 3` as a rough rookie-contract proxy. It is not perfect because some players can sign extensions early, but it is useful for finding low-cost surplus production.


```python
display(
    rookie_surplus[[
        "season", "player_display_name", "position", "team",
        "games_played", "years_exp", "salary_millions",
        "value_above_expected_salary"
    ]].head(15).round(3)
)
```

## Team-Season Salary Efficiency

Team-season totals show where several efficient player-seasons stacked together. Because this uses skill-position `inflated_apy`, I would describe this as offensive contract-cost efficiency rather than total roster cap efficiency.


```python
display(
    team_season[[
        "season", "team", "player_seasons", "total_salary_millions",
        "total_value_above_expected_salary", "high_efficiency_players",
        "low_efficiency_players"
    ]].head(15).round(3)
)
```


```python
plt.figure(figsize=(10, 6))
plot_df = team_season.head(12).copy()
plot_df["team_season"] = plot_df["season"].astype(str) + " " + plot_df["team"]
sns.barplot(
    data=plot_df,
    y="team_season",
    x="total_value_above_expected_salary",
    color="#1f77b4",
)
plt.title("Top Team-Seasons by Total Salary Efficiency Surplus")
plt.xlabel("Total Value Above Expected Salary")
plt.ylabel("Team-Season")
plt.tight_layout()
plt.show()
```

## Position And Salary-Tier Pattern

This table helps check whether the residual model is still leaving obvious position patterns. Running backs stand out: high-cost RB seasons have a negative average residual, which supports the idea that RB salary efficiency is especially sensitive to contract timing and decline risk.


```python
display(position_salary_tiers.round(3))
```


```python
plt.figure(figsize=(10, 6))
sns.barplot(
    data=position_salary_tiers,
    x="position",
    y="mean_value_above_expected_salary",
    hue="salary_tier",
)
plt.axhline(0, color="black", linewidth=1)
plt.title("Mean Salary-Efficiency Residual by Position and Salary Tier")
plt.xlabel("Position")
plt.ylabel("Mean Value Above Expected Salary")
plt.legend(title="Salary Tier", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
```

## Main Interpretation

The strongest findings are about surplus concentration. A few elite player-seasons, especially on rookie or below-market contracts, create very large positive residuals. At the team level, the best seasons tend to combine one or two stars with several useful contributors.

The high-cost underperformer table is useful, but it needs more caution. A negative residual does not mean the contract was a bad decision by itself; injuries, role changes, team context, and non-receiving value can all matter.

## Exported Outputs

The findings tables and written report are saved so they can be reviewed without rerunning the notebook.


```python
for path in sorted((project_root / "outputs" / "tables").glob("salary_findings_*.csv")):
    print(path)

print(outputs["report_path"])
```
