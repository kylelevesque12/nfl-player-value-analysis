# Exploratory Analysis

This notebook checks whether the value metrics behave sensibly before I use them for modeling. The goal is not just to make charts; it is to understand what the metric rewards, where it may be biased, and which patterns need to be explained carefully.

I focus on total EPA value, per-game context, position differences, age and experience patterns, and relationships with supporting box-score statistics.


## Load Value Scores

This section loads the value-score dataset from Notebook 03. At this point, each row should represent one qualifying player-season, with traded-player stints already collapsed.

The path logic lets the notebook run from VS Code, Jupyter, or the project root without manually changing file paths.



```python
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def find_project_root(expected_file):
    """Find the repo root from common VS Code/Jupyter working directories."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / expected_file).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find " + expected_file + " from working directory " + str(Path.cwd())
    )

project_root = find_project_root("data/processed/player_value_scores_2016_2025.csv")
processed_dir = project_root / "data" / "processed"
figures_dir = project_root / "outputs" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

value_scores_path = processed_dir / "player_value_scores_2016_2025.csv"
value_scored = pd.read_csv(value_scores_path)

print(value_scored.shape)
value_scored.head()

```

## Quick Data Checks

Before interpreting any plots, I check the basic structure of the data. Duplicate player-season rows, missing metric columns, or unexpected row counts would make the later rankings and charts misleading.



```python
required_cols = [
    "season", "player_id", "player_display_name", "position", "team",
    "games_played", "value_epa_total", "value_epa_per_game",
    "value_score", "value_score_per_game",
    "position_season_rank", "position_season_percentile"
]

missing_cols = [col for col in required_cols if col not in value_scored.columns]
print("Missing columns:", missing_cols)

duplicate_rows = value_scored.duplicated(["season", "player_id"]).sum()
print("Duplicate player-season rows:", duplicate_rows)

print("Seasons:", sorted(value_scored["season"].dropna().unique()))
print("Positions:", value_scored["position"].value_counts().to_dict())

value_scored[[
    "value_epa_total", "value_epa_per_game", "value_score", "value_score_per_game"
]].describe()

```

## Add Diagnostic Columns

`value_score_gap` compares per-game standardized value to total-EPA standardized value. I use it as a diagnostic, not as a new value metric.

A large positive gap usually means a player looked better on a per-game basis than as a full-season contributor. That can happen because of missed games, changing roles, or small samples.



```python
value_scored = value_scored.copy()
value_scored["value_score_gap"] = value_scored["value_score_per_game"] - value_scored["value_score"]

value_scored[[
    "player_display_name", "season", "position", "team", "games_played",
    "value_epa_total", "value_epa_per_game", "value_score",
    "value_score_per_game", "value_score_gap"
]].sort_values("value_score_gap", ascending=False).head(10)

```

## Top Full-Season Value by Position

These tables use `value_epa_total`, the raw full-season EPA metric. This is the most direct way to see how much expected-point value a player added over the season.

I separate the tables by position because the project is intentionally position-aware. A QB and a WR should not be judged on the same raw scale without context.



```python
season_to_check = 2024

for pos in ["QB", "RB", "WR", "TE"]:
    print()
    print("Top", pos + "s", "by total EPA in", season_to_check)
    display(
        value_scored[
            (value_scored["season"] == season_to_check) &
            (value_scored["position"] == pos)
        ][[
            "player_display_name", "team", "games_played",
            "value_epa_total", "value_epa_per_game",
            "value_score", "position_season_rank"
        ]]
        .sort_values("value_epa_total", ascending=False)
        .head(10)
    )

```

## Top Per-Game Value by Position

These tables use `value_epa_per_game`. This rate view is useful, but it needs to be read next to `games_played`.

A high per-game number can identify a very efficient player, but it does not always mean that player created the most total season value.



```python
for pos in ["QB", "RB", "WR", "TE"]:
    print()
    print("Top", pos + "s", "by EPA per game in", season_to_check)
    display(
        value_scored[
            (value_scored["season"] == season_to_check) &
            (value_scored["position"] == pos)
        ][[
            "player_display_name", "team", "games_played",
            "value_epa_total", "value_epa_per_game",
            "value_score", "value_score_per_game"
        ]]
        .sort_values("value_epa_per_game", ascending=False)
        .head(10)
    )

```

## Total Value vs Per-Game Value

This is one of the most important methodology checks in the notebook. The chart compares standardized per-game EPA with standardized total EPA.

Players in the upper-right were strong by both views. Players far to the right but not high on the y-axis were productive when active but did not accumulate as much total season value. This is why I chose total EPA as the primary value metric and kept per-game EPA as context.



```python
plot_df = value_scored[value_scored["season"] == season_to_check].copy()

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=plot_df,
    x="value_score_per_game",
    y="value_score",
    hue="position",
    size="games_played",
    sizes=(30, 220),
    alpha=0.75
)
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.title("Total EPA Value vs EPA Per Game Value, " + str(season_to_check))
plt.xlabel("Standardized EPA per game")
plt.ylabel("Standardized total EPA")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

```

## Top Player-Seasons Overall

This table ranks the strongest player-seasons by raw total EPA across the full dataset.

Because QBs handle the ball on many more plays than other positions, raw total EPA is not position-neutral. I use this table for interpretation, but the standardized `value_score` is still the better cross-position comparison.



```python
value_scored[[
    "season", "player_display_name", "position", "team", "games_played",
    "value_epa_total", "value_epa_per_game", "value_score",
    "value_score_per_game", "position_season_rank"
]].sort_values("value_epa_total", ascending=False).head(25)

```

## Standardized Value Distribution by Position

This boxplot shows the spread of `value_score` by position. Since the metric is standardized within season-position groups, the center of each position should be near zero.

The main thing I am checking is whether any position has strange outliers or an unexpected distribution shape that would affect later modeling.



```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=value_scored, x="position", y="value_score", order=["QB", "RB", "WR", "TE"])
plt.axhline(0, color="black", linewidth=1)
plt.title("Distribution of Standardized Total EPA Value Score by Position")
plt.xlabel("Position")
plt.ylabel("Value score: total EPA z-score")
plt.show()

```

## Raw Total EPA Distribution by Position

This plot keeps the metric in raw EPA units, which is easier to explain to a nontechnical reader: higher values mean more expected points added.

The tradeoff is that raw EPA is not automatically fair across positions because roles and opportunity levels are different.



```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=value_scored, x="position", y="value_epa_total", order=["QB", "RB", "WR", "TE"])
plt.axhline(0, color="black", linewidth=1)
plt.title("Raw Total EPA Distribution by Position")
plt.xlabel("Position")
plt.ylabel("Total EPA")
plt.show()

```

## Age Curves by Position

This chart looks for broad age-related patterns in value. I am using it descriptively, not causally.

Age curves can be noisy because the players who remain in the league at older ages are usually the ones who were already good enough to survive roster cuts. That selection effect is important to keep in mind.



```python
age_curve = (
    value_scored
    .dropna(subset=["age", "value_score"])
    .assign(age=lambda df: df["age"].round().astype(int))
    .groupby(["position", "age"], as_index=False)
    .agg(
        avg_value_score=("value_score", "mean"),
        avg_total_epa=("value_epa_total", "mean"),
        n=("player_id", "count")
    )
    .query("n >= 5")
)

plt.figure(figsize=(11, 6))
sns.lineplot(data=age_curve, x="age", y="avg_value_score", hue="position", marker="o")
plt.axhline(0, color="black", linewidth=1)
plt.title("Average Standardized Total EPA Value by Age")
plt.xlabel("Age")
plt.ylabel("Average value score")
plt.show()

```

## Experience Curves by Position

This view uses `years_exp` to look at career-stage patterns. It helps separate age from NFL experience, although the two are obviously related.

I am mainly looking for whether value appears to rise, flatten, or decline at different points for each position.



```python
exp_curve = (
    value_scored
    .dropna(subset=["years_exp", "value_score"])
    .assign(years_exp=lambda df: df["years_exp"].round().astype(int))
    .groupby(["position", "years_exp"], as_index=False)
    .agg(
        avg_value_score=("value_score", "mean"),
        avg_total_epa=("value_epa_total", "mean"),
        n=("player_id", "count")
    )
    .query("n >= 5")
)

plt.figure(figsize=(11, 6))
sns.lineplot(data=exp_curve, x="years_exp", y="avg_value_score", hue="position", marker="o")
plt.axhline(0, color="black", linewidth=1)
plt.title("Average Standardized Total EPA Value by Years of Experience")
plt.xlabel("Years of experience")
plt.ylabel("Average value score")
plt.show()

```

## Per-Game Outliers

This table highlights players whose per-game score is much higher than their total-EPA score.

These are useful edge cases. Some players may have been excellent in limited time, while others may be small-sample outliers. Either way, they are exactly the kind of cases where a single metric needs context.



```python
value_scored[[
    "season", "player_display_name", "position", "team", "games_played",
    "value_epa_total", "value_epa_per_game", "value_score",
    "value_score_per_game", "value_score_gap"
]].sort_values("value_score_gap", ascending=False).head(25)

```

## Correlation With Supporting Statistics

The heatmap and correlation table show how the value metrics relate to supporting variables like games played, yards, touchdowns, age, experience, and efficiency.

I am not treating these correlations as causal. The goal is to understand which variables move with the value score and which variables may be redundant if used together in a model.



```python
supporting_cols = [
    "value_score", "value_score_per_game", "value_score_gap",
    "value_epa_total", "value_epa_per_game", "games_played",
    "yards_per_game", "tds_per_game", "scrimmage_yards_per_game",
    "scrimmage_touches_per_game", "yards_per_scrimmage_touch",
    "qb_yards_per_play", "qb_tds_per_game", "interceptions_per_game",
    "age", "years_exp", "draft_number"
]

available_cols = [col for col in supporting_cols if col in value_scored.columns]

corr = value_scored[available_cols].corr(numeric_only=True)
plt.figure(figsize=(11, 8))
sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.5)
plt.title("Correlation Heatmap for Value and Supporting Metrics")
plt.show()

corr["value_score"].sort_values(ascending=False)

```

## Methodological Notes

The current design separates raw production from standardized comparison.

- `value_epa_total` is the primary raw value metric because it measures full-season production.
- `value_score` is the main comparison metric because it standardizes total EPA within season-position groups.
- `value_epa_per_game` is kept as context, especially for players with limited games.
- Supporting stats help explain the value score, but they are not part of the value-score formula.

The main limitation is that this is still a production-based metric. It does not fully capture blocking, route quality, scheme, offensive line effects, injuries, teammate effects, or defensive attention. This matters most for tight ends because blocking value is not well represented in the current data.
