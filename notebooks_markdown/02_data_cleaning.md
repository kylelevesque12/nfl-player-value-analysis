# Data Cleaning

The goal of this notebook is to turn weekly NFL player stats into a clean player-season dataset for QBs, RBs, WRs, and TEs. I keep the cleaning step separate from the value-score step so the project has a clear data pipeline: raw weekly data first, cleaned season-level data second, value metrics later.

The main methodological choices here are to filter to regular-season games, keep offensive skill positions, aggregate weekly stats to the season level, and merge roster information such as age, experience, draft slot, and college. I also avoid creating a universal yards-per-touch metric because passing yards and rushing/receiving touches are not the same kind of opportunity.


## Load Data

This section loads the locally saved raw CSVs. I keep these raw files out of GitHub because they can be regenerated and are larger than the project code itself.



```python
from pathlib import Path

import pandas as pd
import numpy as np


def find_project_root(expected_file):
    """Find the repo root from common VS Code/Jupyter working directories."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / expected_file).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find " + expected_file + " from working directory " + str(Path.cwd())
    )

project_root = find_project_root("data/raw/player_stats_2016_2025.csv")
raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

player_stats = pd.read_csv(raw_dir / "player_stats_2016_2025.csv")
rosters = pd.read_csv(raw_dir / "rosters_2016_2025.csv")
schedules = pd.read_csv(raw_dir / "schedules_2016_2025.csv")

print("Player stats:", player_stats.shape)
print("Rosters:", rosters.shape)
print("Schedules:", schedules.shape)

```


```python
# player_stats columns
[col for col in player_stats.columns]
```


```python
# Rosters columns
[col for col in rosters.columns]
```

## Aggregate Weekly Stats to Season Level

Here I collapse weekly rows into season-level production. Counting stats such as yards, touchdowns, attempts, carries, targets, and EPA are summed across weeks, while roster fields are merged in after the aggregation.

This step matters because every later metric assumes one row represents a player's season-level production. If a player changed teams, the cleaned output may still have team-specific rows at this stage; that is handled more carefully in the value-score and modeling notebooks.



```python
#filter to regular season skill players

skill_positions = ["QB", "RB", "WR", "TE"]

skill_weekly = player_stats[
    (player_stats["position"].isin(skill_positions)) &
    (player_stats["season_type"] == "REG")
].copy()

skill_weekly.shape

```


```python
#main stat columns available for skill players
[col for col in skill_weekly.columns if any(word in col for word in [
    "passing", "rushing", "receiving", "targets", "receptions",
    "carries", "attempts", "interceptions", "fantasy"
])]
```


```python
# Create season-level dataset by summing weekly counting stats

# Share/rate metrics such as target_share, air_yards_share, and wopr are not summed here.
# They should be recalculated or handled separately if needed for later analysis.
sum_cols = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "sack_yards_lost",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_first_downs",
    "passing_epa",
    "passing_2pt_conversions",

    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",
    "rushing_epa",
    "rushing_2pt_conversions",

    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_first_downs",
    "receiving_epa",
    "receiving_2pt_conversions",

    "fantasy_points",
    "fantasy_points_ppr"
]

# Keep only columns that actually exist in the nflreadpy export.
sum_cols = [col for col in sum_cols if col in skill_weekly.columns]

# Do not group by player_name because it can change slightly within a season
# for the same player_id, which would split one player-season-team into duplicates.
group_cols = ["season", "player_id", "player_display_name", "position", "team"]

skill_season = (
    skill_weekly
    .groupby(group_cols, as_index=False)[sum_cols]
    .sum()
)

# Keep one readable short player_name for convenience.
player_names = (
    skill_weekly
    .sort_values(["season", "player_id", "team", "week"])
    .groupby(["season", "player_id", "team"], as_index=False)["player_name"]
    .first()
)

skill_season = skill_season.merge(
    player_names,
    on=["season", "player_id", "team"],
    how="left"
)

# Put name columns near the front.
front_cols = ["season", "player_id", "player_name", "player_display_name", "position", "team"]
other_cols = [col for col in skill_season.columns if col not in front_cols]
skill_season = skill_season[front_cols + other_cols]

skill_season.head()

```


```python
# Add games played for each player-season-team row.
# Because this dataset is grouped by team, players who changed teams midseason remain
# team-specific rows and games_played is counted within that team stint.
games_played = (
    skill_weekly
    .groupby(["season", "player_id", "team"], as_index=False)["week"]
    .nunique()
    .rename(columns={"week": "games_played"})
)

skill_season = skill_season.merge(
    games_played,
    on=["season", "player_id", "team"],
    how="left"
)

skill_season.head()

```


```python
# Additional season level features

# -----------------------------
# General total production
# -----------------------------
skill_season["total_yards"] = (
    skill_season.get("passing_yards", 0) +
    skill_season.get("rushing_yards", 0) +
    skill_season.get("receiving_yards", 0)
)

skill_season["total_epa"] = (
    skill_season.get("passing_epa", 0) +
    skill_season.get("rushing_epa", 0) +
    skill_season.get("receiving_epa", 0)
)

skill_season["total_tds"] = (
    skill_season.get("passing_tds", 0) +
    skill_season.get("rushing_tds", 0) +
    skill_season.get("receiving_tds", 0)
)

skill_season["yards_per_game"] = (
    skill_season["total_yards"] / skill_season["games_played"]
)

skill_season["epa_per_game"] = (
    skill_season["total_epa"] / skill_season["games_played"]
)

skill_season["tds_per_game"] = (
    skill_season["total_tds"] / skill_season["games_played"]
)

# -----------------------------
# RB / WR / TE scrimmage features
# -----------------------------
skill_season["scrimmage_touches"] = (
    skill_season.get("carries", 0) +
    skill_season.get("receptions", 0)
)

skill_season["scrimmage_yards"] = (
    skill_season.get("rushing_yards", 0) +
    skill_season.get("receiving_yards", 0)
)

skill_season["scrimmage_tds"] = (
    skill_season.get("rushing_tds", 0) +
    skill_season.get("receiving_tds", 0)
)

skill_season["scrimmage_yards_per_game"] = (
    skill_season["scrimmage_yards"] / skill_season["games_played"]
)

skill_season["scrimmage_epa"] = (
    skill_season.get("rushing_epa", 0) +
    skill_season.get("receiving_epa", 0)
)

skill_season["scrimmage_epa_per_game"] = (
    skill_season["scrimmage_epa"] / skill_season["games_played"]
)

skill_season["scrimmage_touches_per_game"] = (
    skill_season["scrimmage_touches"] / skill_season["games_played"]
)

skill_season["yards_per_scrimmage_touch"] = (
    skill_season["scrimmage_yards"] /
    skill_season["scrimmage_touches"].replace(0, np.nan)
)


# -----------------------------
# QB-specific features
# -----------------------------
skill_season["qb_plays"] = (
    skill_season.get("attempts", 0) +
    skill_season.get("carries", 0)
)

skill_season["qb_total_yards"] = (
    skill_season.get("passing_yards", 0) +
    skill_season.get("rushing_yards", 0)
)

skill_season["qb_epa"] = (
    skill_season.get("passing_epa", 0) +
    skill_season.get("rushing_epa", 0)
)

skill_season["qb_total_tds"] = (
    skill_season.get("passing_tds", 0) +
    skill_season.get("rushing_tds", 0)
)

skill_season["qb_yards_per_play"] = (
    skill_season["qb_total_yards"] /
    skill_season["qb_plays"].replace(0, np.nan)
)

skill_season["qb_yards_per_game"] = (
    skill_season["qb_total_yards"] / skill_season["games_played"]
)

skill_season["qb_epa_per_game"] = (
    skill_season["qb_epa"] / skill_season["games_played"]
)

skill_season["qb_tds_per_game"] = (
    skill_season["qb_total_tds"] / skill_season["games_played"]
)


skill_season.head()
```


```python
# Retain only relevant columns from rosters
roster_cols = [
    "season",
    "gsis_id",
    "birth_date",
    "height",
    "weight",
    "years_exp",
    "entry_year",
    "rookie_year",
    "draft_club",
    "draft_number",
    "college"
]

rosters_trimmed = rosters[roster_cols].drop_duplicates()

# Merge season level stats with roster info

skill_season = skill_season.merge(
    rosters_trimmed,
    left_on=["season", "player_id"],
    right_on=["season", "gsis_id"],
    how="left"
)

skill_season.head()
```


```python
# Adding age
skill_season["birth_date"] = pd.to_datetime(skill_season["birth_date"], errors="coerce")

skill_season["age"] = (
    skill_season["season"] -
    skill_season["birth_date"].dt.year
)

skill_season[["player_display_name", "season", "position", "age", "years_exp"]].head()
```


```python
skill_season.head()

```


```python
# Check for duplicate player-season-team rows.
# This should be empty after aggregation. Multi-team players should appear once per team stint.
duplicates = (
    skill_season
    .groupby(["season", "player_id", "team"])
    .size()
    .reset_index(name="num_rows")
    .query("num_rows > 1")
    .sort_values("num_rows", ascending=False)
)

duplicates.head(20)

```


```python
# missing values in roster variables
# draft_number has a high missing value percentage, could be due to many undrafted players
skill_season[["birth_date", "age", "years_exp", "height", "weight", "draft_number"]].isna().mean()
```


```python
# Checking that features makes sense
# Should see mostly QBs at the top for total yards
skill_season[
    ["player_display_name", "season", "position", "team", "games_played",
     "total_yards", "total_tds", "scrimmage_yards", "scrimmage_touches",
     "qb_total_yards", "qb_plays"]
].sort_values("total_yards", ascending=False).head(20)
```


```python
# Should be Non-QBs at the top, specifically mostly RBs for scrimmage yards
skill_season[
    skill_season["position"].isin(["RB", "WR", "TE"])
][
    ["player_display_name", "season", "position", "team", "games_played",
     "scrimmage_yards", "scrimmage_touches", "yards_per_scrimmage_touch"]
].sort_values("scrimmage_yards", ascending=False).head(20)

```


```python
required_cols = [
    "season", "player_id", "player_display_name", "position", "team",
    "games_played",
    "scrimmage_epa_per_game", "scrimmage_yards_per_game",
    "scrimmage_touches_per_game", "yards_per_scrimmage_touch",
    "qb_epa_per_game", "qb_yards_per_game", "qb_yards_per_play",
    "passing_interceptions",
    "age", "years_exp", "draft_number"
]

[col for col in required_cols if col not in skill_season.columns]
```


```python
output_path = processed_dir / "skill_player_seasons_2016_2025.csv"

skill_season.to_csv(output_path, index=False)

print(f"Saved {len(skill_season):,} rows to {output_path}")

```
