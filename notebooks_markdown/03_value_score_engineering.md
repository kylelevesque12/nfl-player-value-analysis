# Value Score Engineering

This notebook creates the core value metric for the project. I am using EPA because it connects player production to expected points instead of just counting yards or touchdowns.

The primary raw metric is `value_epa_total`. For QBs, that means passing EPA plus rushing EPA. For RBs, WRs, and TEs, it means rushing EPA plus receiving EPA. The primary comparison metric is `value_score`, which standardizes total EPA within each season-position group.

I also keep per-game EPA as supporting context, but I do not use it as the main value score because availability is part of season value.


## Load Cleaned Player-Season Data

The cleaned file can contain more than one row for a player in a season if he changed teams. Before scoring, I collapse those stints into one player-season row so a traded player is evaluated on his full season instead of being split into smaller samples.



```python
from pathlib import Path
import sys

import pandas as pd


def find_project_root(expected_file="data/processed/skill_player_seasons_2016_2025.csv"):
    """Find the repo root from common VS Code/Jupyter working directories."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / expected_file).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find " + expected_file + " from working directory " + str(Path.cwd())
    )


project_root = find_project_root()
processed_dir = project_root / "data" / "processed"

sys.path.insert(0, str(project_root / "src"))
from prediction_report import MIN_VALUE_GAMES, create_player_season_value_scores

skill_season_path = processed_dir / "skill_player_seasons_2016_2025.csv"
skill_player_team = pd.read_csv(skill_season_path)

print("Raw cleaned rows:", skill_player_team.shape)
print("Unique player-seasons before collapse:", skill_player_team[["season", "player_id"]].drop_duplicates().shape[0])
skill_player_team.head()

```

## Build Value Scores

This helper does the main methodological work in one place:

1. Collapse multi-team stints to one player-season.
2. Calculate the correct EPA definition for QBs and non-QBs.
3. Apply the minimum games-played filter after the collapse.
4. Standardize total EPA within each season-position group.
5. Add ranks, percentiles, and per-game context columns.

Keeping this logic in the shared source file helps prevent the notebooks and report from drifting into slightly different definitions.



```python
value_scored = create_player_season_value_scores(
    skill_player_team,
    min_games=MIN_VALUE_GAMES,
)

print("Value-scored player-season rows:", value_scored.shape)
print("Duplicate player-season rows:", value_scored.duplicated(["season", "player_id"]).sum())

display(
    value_scored[[
        "season", "player_display_name", "position", "team", "teams",
        "games_played", "value_epa_total", "value_epa_per_game",
        "value_score", "value_score_per_game", "position_season_rank",
        "position_season_percentile"
    ]].head(10)
)

```

## Method Checks

After standardizing, each season-position group should be centered around zero. This is a quick check that the z-score logic is working and that each player is being compared only to his peers at the same position in the same season.



```python
group_check = (
    value_scored
    .groupby(["season", "position"], as_index=False)
    .agg(
        players=("player_id", "count"),
        mean_value_score=("value_score", "mean"),
        std_value_score=("value_score", "std"),
        mean_total_epa=("value_epa_total", "mean"),
        mean_epa_per_game=("value_epa_per_game", "mean"),
    )
)

display(group_check.tail(16))

assert value_scored.duplicated(["season", "player_id"]).sum() == 0
assert value_scored["games_played"].ge(MIN_VALUE_GAMES).all()

```

## Sanity Check 2024 Rankings

I use 2024 as a familiar season to inspect the rankings. This is not formal validation, but it is useful for catching obvious problems. If the top players by position look completely unreasonable, the metric probably needs to be revisited before moving forward.



```python
season_to_check = 2024

for pos in ["QB", "RB", "WR", "TE"]:
    print(f"Top {pos}s in {season_to_check}")
    display(
        value_scored[
            (value_scored["season"] == season_to_check) &
            (value_scored["position"] == pos)
        ][[
            "player_display_name", "team", "games_played",
            "value_epa_total", "value_epa_per_game", "value_score",
            "value_score_per_game", "position_season_rank"
        ]]
        .sort_values("value_score", ascending=False)
        .head(10)
    )

```

## Per-Game Context

`value_epa_per_game` answers a different question than the main value score. It shows how productive a player was when active, while `value_score` rewards full-season production.

I keep both because the difference is informative. A player with a strong per-game score but a lower total score may have been excellent in a smaller sample, which matters for interpretation and later forecasting.



```python
gap_cols = [
    "season", "player_display_name", "position", "team", "games_played",
    "value_epa_total", "value_epa_per_game", "value_score",
    "value_score_per_game", "value_score_gap"
]

value_scored[gap_cols].assign(
    absolute_gap=value_scored["value_score_gap"].abs()
).sort_values("absolute_gap", ascending=False).head(20)

```

## Save Player Value Scores

The processed value-score file is saved locally so the analysis and modeling notebooks can use the same dataset. It is ignored by Git because it can be regenerated from the notebooks and source code.



```python
output_path = processed_dir / "player_value_scores_2016_2025.csv"
value_scored.to_csv(output_path, index=False)

print(f"Saved {len(value_scored):,} rows to {output_path}")

```
