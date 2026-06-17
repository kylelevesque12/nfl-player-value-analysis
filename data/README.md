# Data Notes

This folder holds local data used by the notebooks and scripts.

## Raw Data

Raw files are saved under `data/raw/` and are intentionally ignored by Git. They can be regenerated from the project notebooks or source URLs.

Current raw inputs:

- `player_stats_2016_2025.csv`
- `rosters_2016_2025.csv`
- `schedules_2016_2025.csv`
- `historical_contracts.csv`

The NFL stat and roster files come from `nflreadpy` / nflverse. The contract file comes from the nflverse historical contracts release, which is sourced from OverTheCap.

## Processed Data

Processed modeling files are saved under `data/processed/` and are also ignored by Git. These files are intermediate outputs from the notebooks, not hand-edited source files.

Important processed files:

- `skill_player_seasons_2016_2025.csv`
- `player_value_scores_2016_2025.csv`

## GitHub Policy

Raw and processed data are kept local because they can be large and can be regenerated. Final lightweight result tables that are useful for reviewing the project are saved under `outputs/tables/` and selected files are allowed into Git.

The salary-efficiency outputs use a season-specific cap hit reconstructed from contract terms (prorated signing bonus + backloaded base; see `src/cap_hit_reconstruction.py`), carried in `salary_millions` with a `salary_source` flag. It is a principled estimate built from the available contract fields, not exact season-level cap accounting or cash paid — the source contracts have no year-by-year cap breakdown.
