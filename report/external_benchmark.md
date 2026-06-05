# External Benchmark

**Sources**: `draftkings_implied_via_rotoguru`, `vegas_team_environment_implied`

## What this benchmarks

Head-to-head RMSE/MAE/win-rate vs externally derived market projections.
Two sources are wired in:

- `draftkings_implied_via_rotoguru` — the strongest free fantasy
  benchmark. DK sets salaries pregame; the per-(season, position)
  regression of actual PPR on DK salary recovers the salary→points
  conversion the market is using. RotoGuru's free archive covers
  through 2021 only.
- `vegas_team_environment_implied` — a weaker but longer-coverage
  benchmark. Per-(season, position) OLS of PPR on implied team total,
  spread, and home/away. Encodes only team-environment information
  (no player-specific signal), but extends through 2025.

Because both conversions are fit on the season's actuals, the implied
projections are *strong* benchmarks — stronger than real-time
implementations would be. Beating them on this setup is conservative.

## Per-source overall

| Source | n | Model RMSE | External RMSE | Skill vs external |
| --- | ---: | ---: | ---: | ---: |
| `draftkings_implied_via_rotoguru` | 11,191 | 6.386 | 6.493 | +1.650% |
| `vegas_team_environment_implied` | 34,906 | 6.147 | 7.612 | +19.249% |

## By position (per source)

### `draftkings_implied_via_rotoguru`

| Position | n | Model RMSE | External RMSE | Skill vs external |
| --- | ---: | ---: | ---: | ---: |
| QB | 1,262 | 7.695 | 7.842 | +1.875% |
| RB | 2,905 | 6.586 | 6.707 | +1.805% |
| TE | 2,302 | 5.102 | 5.137 | +0.681% |
| WR | 4,722 | 6.438 | 6.553 | +1.756% |

### `vegas_team_environment_implied`

| Position | n | Model RMSE | External RMSE | Skill vs external |
| --- | ---: | ---: | ---: | ---: |
| QB | 3,901 | 7.466 | 8.930 | +16.400% |
| RB | 9,146 | 6.255 | 7.987 | +21.694% |
| TE | 7,312 | 4.953 | 5.944 | +16.672% |
| WR | 14,547 | 6.228 | 7.731 | +19.449% |


## By season (per source)

### `draftkings_implied_via_rotoguru`

| Season | n | Model RMSE | External RMSE | Skill |
| --- | ---: | ---: | ---: | ---: |
| 2020 | 5,436 | 6.518 | 6.631 | +1.701% |
| 2021 | 5,755 | 6.259 | 6.360 | +1.598% |

### `vegas_team_environment_implied`

| Season | n | Model RMSE | External RMSE | Skill |
| --- | ---: | ---: | ---: | ---: |
| 2020 | 5,530 | 6.504 | 7.901 | +17.685% |
| 2021 | 5,856 | 6.236 | 7.675 | +18.759% |
| 2022 | 5,818 | 6.087 | 7.447 | +18.258% |
| 2023 | 5,811 | 6.016 | 7.483 | +19.615% |
| 2024 | 5,848 | 6.079 | 7.654 | +20.570% |
| 2025 | 6,043 | 5.967 | 7.518 | +20.635% |

## Per-player-week win rate (per source)

Share of player-weeks where the model's projection landed closer
to the actual PPR than the external projection did.

### `draftkings_implied_via_rotoguru`

| Position | n | Model win rate |
| --- | ---: | ---: |
| QB | 1,262 | 0.516 |
| RB | 2,905 | 0.550 |
| TE | 2,302 | 0.546 |
| WR | 4,722 | 0.548 |

### `vegas_team_environment_implied`

| Position | n | Model win rate |
| --- | ---: | ---: |
| QB | 3,901 | 0.581 |
| RB | 9,146 | 0.672 |
| TE | 7,312 | 0.662 |
| WR | 14,547 | 0.657 |

## Honest reading of this result

Public DFS analytics shops sell projections claiming a 1-3% edge over
the DK salary line. A calibrated positive edge after honest rolling
backtesting is the qualifying bar for a fantasy-projection portfolio
piece. Where the model beats DK, the beat is real; where it loses or
barely ties, the gap is reported as-is rather than hidden.

Vegas-team-environment-implied is a *weaker* benchmark than DK closing
line because it has no player-specific signal — it can only say 'WRs
on this offense are expected to score higher because the implied team
total is higher.' Beating it by larger margins is therefore less
impressive than beating DK by smaller ones; both numbers belong in the
table side-by-side so the reviewer can weight them appropriately.

## Coverage gap

RotoGuru's free DK salary archive currently ends in 2021. The DK
comparison is therefore restricted to 2020-2021 (the overlap with
the weekly model's rolling backtest). Vegas-team-environment-implied
extends to 2025 because schedule lines are local. Extending DK-style
coverage to 2022-2025 requires a different (likely paid) source —
see `PORTFOLIO_ROADMAP.md` Tier 1 item #1 for options (Stokastic,
FantasyData, FantasyPros MVP archives, or the `ffanalytics` R
package). The scaffolding accepts any CSV at
`data/raw/external_projections*.csv` matching the documented schema.
