# Salary Efficiency Analysis

This report summarizes the first salary-efficiency phase of the NFL player value project.

## Data Source

The analysis uses nflverse historical contract data from OverTheCap. The local raw file is:

`data/raw/historical_contracts.csv`

Raw data is ignored by Git, but the salary-efficiency outputs are saved under `outputs/tables/`.

## Method

The value-score dataset is merged to historical contracts using `gsis_id`. Contract rows are expanded to player-seasons using `year_signed` and contract length. If more than one contract is active for a player-season, the latest signed contract is used.

The salary metric is a season-specific cap hit reconstructed from contract terms (prorated signing bonus + backloaded base), carried in `salary_millions` (inflation-adjusted millions) with a `salary_source` quality flag. It is a principled estimate, not exact cap hit or cash paid — the source contracts have no year-by-year cap breakdown.

The main salary-efficiency metric is:

`value_above_expected_salary = actual value_score - expected value_score`

Expected value is estimated with salary, position, age, experience, draft slot, and games played.

## Outputs

- `outputs/tables/salary_efficiency_2016_2025.csv`
- `outputs/tables/salary_efficiency_merge_diagnostics.csv`
- `outputs/tables/salary_efficiency_by_position.csv`
- `outputs/tables/salary_efficiency_by_season.csv`
- `outputs/tables/salary_efficiency_top_players.csv`
- `outputs/tables/salary_efficiency_lowest_players.csv`
- `report/salary_efficiency_findings.md`
- `outputs/tables/salary_findings_team_season.csv`
- `outputs/tables/salary_findings_top_surplus_players.csv`

## Validation Snapshot

- 4,753 value-score rows analyzed.
- 4,569 rows matched to inferred contract data.
- Overall contract match rate: 96.1%.
- 3,531 matched player-seasons remain after applying the findings filter of at least 8 games played.

## Interpretation

This is a first-pass contract-efficiency analysis. It is useful for finding possible surplus-value players and underperforming contracts, but it should not be treated as final cap accounting.

The next improvement is to add true season-level cap hit or cash-paid data, then rerun the same framework with a more precise cost variable.
