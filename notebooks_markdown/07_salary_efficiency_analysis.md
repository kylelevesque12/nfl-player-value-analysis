# Salary Efficiency Analysis

This notebook starts the salary-efficiency phase of the project. Up to this point, the project measured and predicted player value. Here I start asking a different question: which players produced more value than expected relative to their contract cost?

I am using nflverse historical contract data, which comes from OverTheCap through the nflreadr/nflreadpy ecosystem. This is a useful first salary source because it includes `gsis_id`, which makes the merge much cleaner than a name-only match.

One important limitation: this is contract/APY data, not exact season-level cap-hit accounting. I use inflated APY as an approximate annual cost metric, then clearly label it as an approximation.

## Load Project Data and Salary-Efficiency Helpers

The helper functions live in `src/salary_efficiency.py`. They handle the contract cleaning, player-season expansion, merge diagnostics, simple efficiency metrics, and residual-based salary-efficiency model.

The raw contract file is saved locally under `data/raw/historical_contracts.csv` and is intentionally ignored by Git.


```python
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def find_project_root(expected_file="data/processed/player_value_scores_2016_2025.csv"):
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / expected_file).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find " + expected_file + " from working directory " + str(Path.cwd())
    )


project_root = find_project_root()
sys.path.insert(0, str(project_root / "src"))

from salary_efficiency import build_salary_efficiency_tables

project_root
```

## Build Salary-Efficiency Tables

This step merges value scores with inferred active contracts. The contract-season match is based on `gsis_id`, `year_signed`, and contract length.

Methodologically, I am treating this as a first-pass contract-efficiency analysis. A later version with exact cap hits or cash paid by season would be stronger, but this is still useful because it compares value to a consistent annual contract-cost estimate.


```python
outputs = build_salary_efficiency_tables(project_root=project_root, save_outputs=True)

salary_efficiency = outputs["salary_efficiency"]
diagnostics = outputs["diagnostics"]
top_efficient = outputs["top_efficient"]
lowest_efficiency = outputs["lowest_efficiency"]
output_dir = outputs["output_dir"]

print("Salary-efficiency rows:", salary_efficiency.shape)
print("Output directory:", output_dir)
salary_efficiency.head()
```

## Merge Diagnostics

Before interpreting salary efficiency, I need to know how much of the value-score dataset actually matched to contract data.

A high match rate is encouraging, but unmatched rows still matter. They may be low-salary players, short-term players, or players whose IDs/contracts are missing from the historical contract file.


```python
display(diagnostics["overall"])
display(diagnostics["by_position"].round(3))
display(diagnostics["by_season"].round(3))
```

## Main Salary-Efficiency Metrics

I use two types of metrics:

- Simple ratios: `value_per_million` and `epa_per_million`.
- Residual value: `value_above_expected_salary`.

The residual metric is the more important one. It compares actual value to expected value after accounting for salary, position, age, experience, draft slot, and games played. That helps avoid a naive conclusion where every cheap player looks efficient and every expensive player looks inefficient.


```python
display(
    salary_efficiency[[
        "season", "player_display_name", "position", "team", "games_played",
        "value_score", "salary_millions", "salary_source",
        "value_per_million", "salary_percentile", "value_cost_percentile_gap",
        "expected_value_given_salary", "value_above_expected_salary",
        "salary_efficiency_percentile", "salary_efficiency_tier"
    ]]
    .dropna(subset=["salary_millions"])
    .sort_values("value_above_expected_salary", ascending=False)
    .head(10)
    .round(3)
)
```

## Most Salary-Efficient Player-Seasons

These players produced the most value above what the salary model expected. This is the closest current version of “undervalued” in the project.

I still read this table carefully. A player can look efficient because he is on a rookie contract, because he had an unusually strong season, or because contract APY does not perfectly represent that season's cap cost.


```python
top_cols = [
    "season", "player_display_name", "position", "team", "games_played",
    "value_score", "salary_millions", "value_above_expected_salary",
    "salary_efficiency_percentile", "salary_efficiency_tier"
]

display(top_efficient[top_cols].head(20).round(3))
```

## Lowest Salary-Efficiency Player-Seasons

These are not automatically “bad players.” They are players whose production came in far below what the salary model expected for their contract/context.

This table is especially sensitive to injuries, role changes, missed games, and contract timing, so I would use it as a flag for deeper review rather than a final judgment.


```python
display(lowest_efficiency[top_cols].head(20).round(3))
```

## Salary Efficiency by Position

This view summarizes whether some positions look structurally more or less efficient in the current setup.

Because the model includes position as a feature, the residual metric should already account for broad positional salary differences. I still look at this summary because systematic position patterns can reveal model limitations.


```python
position_summary = (
    salary_efficiency[salary_efficiency["has_salary"]]
    .groupby("position", as_index=False)
    .agg(
        player_seasons=("player_id", "count"),
        median_salary_millions=("salary_millions", "median"),
        median_value_score=("value_score", "median"),
        median_value_above_expected_salary=("value_above_expected_salary", "median"),
        high_efficiency_rate=("salary_efficiency_tier", lambda s: (s == "High Efficiency").mean()),
        low_efficiency_rate=("salary_efficiency_tier", lambda s: (s == "Low Efficiency").mean()),
    )
)

display(position_summary.round(3))

plt.figure(figsize=(8, 4))
sns.boxplot(
    data=salary_efficiency[salary_efficiency["has_salary"]],
    x="position",
    y="value_above_expected_salary",
    order=["QB", "RB", "WR", "TE"],
)
plt.axhline(0, color="black", linewidth=1)
plt.title("Salary Efficiency Residuals by Position")
plt.xlabel("Position")
plt.ylabel("Value above expected salary")
plt.show()
```

## Rookie-Contract Style View

A major reason to study salary efficiency is that rookie contracts can create surplus value. I use `years_exp <= 3` as a simple rookie-contract style proxy.

This is imperfect because actual contract status depends on draft round, extensions, fifth-year options, and timing, but it gives a useful first look.


```python
rookie_view = salary_efficiency[salary_efficiency["has_salary"]].copy()
rookie_view["rookie_contract_proxy"] = rookie_view["years_exp"].le(3)

rookie_summary = (
    rookie_view
    .groupby(["position", "rookie_contract_proxy"], as_index=False)
    .agg(
        player_seasons=("player_id", "count"),
        median_salary_millions=("salary_millions", "median"),
        median_value_score=("value_score", "median"),
        median_value_above_expected_salary=("value_above_expected_salary", "median"),
    )
)

display(rookie_summary.round(3))

plt.figure(figsize=(8, 4))
sns.barplot(
    data=rookie_summary,
    x="position",
    y="median_value_above_expected_salary",
    hue="rookie_contract_proxy",
    order=["QB", "RB", "WR", "TE"],
)
plt.axhline(0, color="black", linewidth=1)
plt.title("Median Salary Efficiency by Rookie-Contract Proxy")
plt.xlabel("Position")
plt.ylabel("Median value above expected salary")
plt.legend(title="Years exp <= 3")
plt.show()
```

## Saved Outputs

The notebook saves salary-efficiency tables under `outputs/tables/`. I am saving the results because these are small, recruiter-friendly artifacts, while the raw contract data stays local and ignored by Git.


```python
for path in sorted(output_dir.glob("salary_efficiency_*.csv")):
    print(path)
```

## Methodological Notes

This is a useful first salary-efficiency version, but I would not call it final.

Main strengths:

- The merge uses `gsis_id`, so it avoids most name-matching problems.
- Contract cost is adjusted with `inflated_apy`, which is more comparable across seasons than raw APY.
- The residual approach is better than only using value-per-dollar ratios.

Main limitations:

- `inflated_apy` is not the same as exact season cap hit or cash paid.
- Active contract seasons are inferred from `year_signed` and `years`.
- Restructures, void years, trades, extensions, incentives, dead cap, and guarantees are not fully modeled.
- The residual model is descriptive, not causal.

The next improvement would be to add true season-level cap-hit or cash-paid data, then rerun the same framework with a more precise cost variable.
