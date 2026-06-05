# External Benchmark

**Sources**: `draftkings_implied_via_rotoguru`

## What this benchmarks

Head-to-head RMSE/MAE/win-rate vs externally derived market projections.
The multi-source loader globs `data/raw/external_projections*.csv`, so
adding a richer source (FantasyPros MVP archives, Stokastic, or any
other paid feed) is purely a data-acquisition step — drop a CSV in the
documented schema and rerun.

The currently active source is `draftkings_implied_via_rotoguru` — the
strongest free fantasy benchmark available. DK sets salaries pregame;
the per-(season, position) regression of actual PPR on DK salary
recovers the salary→points conversion the market is implicitly using.
The fitted value is the market's implied PPR projection for each
player-week. RotoGuru's free archive ends in 2021, so the comparison
is restricted to 2020-2021 — the overlap with the weekly model's
rolling backtest window.

Because the conversion is fit on the season's actuals, the implied
projection is a *strong* benchmark — stronger than a real-time
implementation would be. Beating it on this setup is therefore a
conservative claim.

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
