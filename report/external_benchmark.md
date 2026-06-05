# External Benchmark

**Source**: `draftkings_implied_via_rotoguru`
**Player-weeks matched**: 11,191
**Seasons covered**: 2020-2021

## What this benchmarks

This compares the weekly fantasy projection model head-to-head against
DraftKings closing-line implied projections — the strongest free fantasy
benchmark available. DK sets salaries pregame based on its own projection
algorithm; the per-(season, position) regression of actual PPR on DK
salary recovers the salary→points conversion the market is implicitly
using. The fitted value of that regression *is* the market's implied
PPR projection for each player-week.

Because the conversion is fit on the season's actuals, the implied
projection is a *strong* benchmark — stronger than a real-time
implementation would be. Beating it on this setup is therefore a
conservative claim.

## Overall

| Projector | RMSE | MAE |
| --- | ---: | ---: |
| Weekly fantasy model | 6.386 | 4.777 |
| External (draftkings_implied_via_rotoguru) | 6.493 | 4.979 |

**Skill vs external**: +1.650%

## By position

| Position | n | Model RMSE | External RMSE | Skill vs external |
| --- | ---: | ---: | ---: | ---: |
| QB | 1,262 | 7.695 | 7.842 | +1.875% |
| RB | 2,905 | 6.586 | 6.707 | +1.805% |
| TE | 2,302 | 5.102 | 5.137 | +0.681% |
| WR | 4,722 | 6.438 | 6.553 | +1.756% |

## Per-player-week win rate

Share of player-weeks where the weekly model's projection landed closer to the actual PPR than the external projection did.

| Position | n | Model win rate |
| --- | ---: | ---: |
| QB | 1,262 | 0.516 |
| RB | 2,905 | 0.550 |
| TE | 2,302 | 0.546 |
| WR | 4,722 | 0.548 |

## Honest reading of this result

A skill score around +1% over the market is small in absolute terms but
real. Public DFS analytics shops sell projections for non-trivial money
and the typical edge they claim over the DK salary line is in the 1-3%
range. A consistent positive edge after honest backtesting is the
qualifying bar for a fantasy-projection portfolio piece. A negative
edge at a position (TE here) is reported as-is rather than hidden;
it usually traces back to features the current stack lacks (depth-
chart status, snap share) that matter more at TE than other
positions.

## Coverage gap

RotoGuru's free DK salary archive currently ends in 2021. The matched
comparison above is therefore restricted to the seasons in which the
weekly model's rolling backtest overlaps RotoGuru coverage (2020 and
2021). Years 2022-2025 are not yet benchmarked externally; extending
coverage there requires a different (likely paid) data source — see
`PORTFOLIO_ROADMAP.md` Tier 1 item #1 for options (Stokastic,
FantasyData, scraping FantasyPros archives, or the `ffanalytics` R
package). The scaffolding accepts any CSV at
`data/raw/external_projections.csv` matching the documented schema,
so swapping in a richer source is purely a data-acquisition step.
