# External Benchmark

**Sources**: `draftkings_implied_via_rotoguru`

## What this is

Head-to-head: weekly fantasy model vs the DraftKings closing-line
implied projection. DK sets salaries pregame, so a per-(season,
position) regression of actual PPR on DK salary recovers the
salary→points conversion DK is implicitly using. The fitted value
is the market's implied projection for each player-week.

Why this is a strong benchmark: the salary conversion is fit on the
season's actual production, so the implied projection has access to
information a real-time DK projection would not. Beating this version
is harder than beating a live DK projection would be.

Coverage limit: RotoGuru's free DK archive stops at the 2021 season,
so the head-to-head sample is 2020-2021 only. The 2022-2025 stability
is shown in `report/weekly_fantasy_projection_summary.md` against an
internal baseline.

The loader at `src/external_benchmark.py` globs every
`data/raw/external_projections*.csv` file, so dropping in a paid
source (Stokastic, FantasyData, FantasyPros archives) extends the
benchmark with no code changes.

## Per-source overall

| Source | n | Model RMSE | External RMSE | Skill vs external |
| --- | ---: | ---: | ---: | ---: |
| `draftkings_implied_via_rotoguru` | 11,191 | 6.386 | 6.493 | +1.650% |

## By position (per source)

### `draftkings_implied_via_rotoguru`

| Position | n | Model RMSE | External RMSE | Skill vs external |
| --- | ---: | ---: | ---: | ---: |
| QB | 1,262 | 7.695 | 7.842 | +1.875% |
| RB | 2,905 | 6.586 | 6.707 | +1.805% |
| TE | 2,302 | 5.102 | 5.137 | +0.681% |
| WR | 4,722 | 6.438 | 6.553 | +1.756% |


## By season (per source)

### `draftkings_implied_via_rotoguru`

| Season | n | Model RMSE | External RMSE | Skill |
| --- | ---: | ---: | ---: | ---: |
| 2020 | 5,436 | 6.518 | 6.631 | +1.701% |
| 2021 | 5,755 | 6.259 | 6.360 | +1.598% |

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

## Honest reading of this result

Public DFS analytics shops sell projections claiming a 1-3% edge over
the DK salary line. A calibrated positive edge after honest rolling
backtesting is the qualifying bar for a fantasy-projection portfolio
piece. Where the model beats DK, the beat is real; where it loses or
barely ties, the gap is reported as-is rather than hidden.

For temporal-stability evidence across the full 2020-2025 rolling
validation window (where DK coverage is unavailable), see the
by-season skill scores against internal baselines in
`report/weekly_fantasy_projection_summary.md`. Those baselines are
weaker than DK but the *consistency* of the beat across six seasons
is the relevant signal there, not the absolute margin.

## Coverage gap

RotoGuru's free DK salary archive currently ends in 2021. Extending
DK-style coverage to 2022-2025 requires a different (likely paid)
source — see `PORTFOLIO_ROADMAP.md` Tier 1 item #1 for options
(Stokastic, FantasyData, FantasyPros MVP archives, or the
`ffanalytics` R package). The scaffolding accepts any CSV at
`data/raw/external_projections*.csv` matching the documented schema,
so swapping in a richer source is purely a data-acquisition step.
