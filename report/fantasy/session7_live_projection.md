# Session 7 — Live weekly projections + per-position conformal intervals

## What I built

Up to now the weekly model could only score games that had already happened —
every modeling row carried its realized PPR target. That's fine for a backtest
but useless if you actually want to know what to expect *this* Sunday. So this
session adds the infrastructure to project the upcoming week before it's played,
plus the per-position interval fix that Session 6 pointed at.

Two pieces:

1. `src/live_weekly_projection.py` — synthesizes one feature row per active
   player for the next week and scores it with the production model.
2. Per-position conformal intervals — replacing the single global halfwidth that
   Session 6 showed badly under-covers quarterbacks.
3. A minimal "Current / next week" mode on the Fantasy Player Board (no redesign).

## How a future row is synthesized

There is no box score for a game that hasn't happened, so I can't just look the
row up — I have to build it. For each active player (fantasy-relevant position,
appeared in the last three weeks), the row is assembled from two clearly separated
sources:

- **Player-history features** (rolling PPR, target/carry/yardage averages, snap
  share, PBP depth-chart rank, availability) are *carried forward* from the
  player's latest completed week. Those features are already computed from games
  strictly before that week, so they contain nothing about the upcoming game.
- **Game-context features** (opponent, home/away, rest, spread/total/implied
  total, roof/temp/wind) come from the upcoming game's **schedule** row, which is
  known before kickoff. Opponent PPR-allowed is looked up for the upcoming
  opponent's defense.

The whole thing reuses the production `build_modeling_frame`, so the live row
carries exactly the same `WEEKLY_FANTASY_FEATURES` columns as the trained model
expects — a test asserts none are missing.

## How leakage is avoided

The guardrail for this session is strict: a future projection may use only
information available before the game. Concretely:

- The carried-forward player features are as-of the latest *completed* week, so
  by construction they exclude the projected game.
- The outcome column is dropped entirely from the live frame — there is no
  target to leak, and scoring never references one (a test scores a frame that
  has no target column at all).
- No same-week PBP, NGS, or PFR touches the row. NGS/PFR were already rejected in
  Session 2; a test re-asserts none appear here.

The one honest approximation: carry-forward means the rolling features are "as of
the last completed week" rather than re-rolled to include that week's result — a
one-game lag on a multi-game window. The plan explicitly chose carry-forward for
this reason, and it keeps the feature vector complete (no imputation holes) at the
cost of being very slightly stale.

## Schedule and weather

The static schedule file carries roof, temperature, and wind for every game, so
the live rows pull real values here. In a true production deployment those would
be unavailable for a game days out, so the module degrades gracefully: any missing
temp/wind is filled with neutral defaults (70°F, 0 mph) and the row is flagged
`weather_is_default` so the UI can mark it as projected context. On the 2025 data
no rows needed the fallback.

A data-boundary note: the dataset ends on a completed 2025 regular season, so the
literal "next week" is the playoffs (not a regular-season slate) and the live
builder correctly returns nothing for it. To demonstrate the capability I point
the builder "as of" week 17 and project week 18 — 443 players, one row each. The
top of that board is sensible (Trevor Lawrence 21.2, Gibbs / Bijan ~20.7, Ja'Marr
Chase 19.3), which is the right smell test for a pregame model.

## Per-position conformal intervals

Session 6's finding: one global symmetric halfwidth can't represent QB variance,
so QBs fall outside their intervals far too often. The fix is to calibrate the
conformal halfwidth *within each position*, with a fallback to the global value
for any position too thin to calibrate on its own (< 200 calibration rows).

Trained on seasons < 2025, calibrated on the last 20% of training rows, measured
on the 2025 hold-out:

**80% intervals — empirical coverage (target 0.80) / mean width**

| Position | Global coverage | Per-position coverage | Global width | Per-position width |
|---|---:|---:|---:|---:|
| QB | 0.575 | **0.730** | 12.96 | 17.52 |
| RB | 0.790 | 0.788 | 11.76 | 11.68 |
| WR | 0.804 | 0.810 | 11.67 | 11.87 |
| TE | 0.862 | 0.790 | 11.27 | 9.07 |

**50% intervals**

| Position | Global coverage | Per-position coverage |
|---|---:|---:|
| QB | 0.294 | **0.438** |
| RB | 0.504 | 0.515 |
| WR | 0.503 | 0.510 |
| TE | 0.604 | 0.503 |

Per-position conformal does exactly what it should. It pulls QB coverage up
sharply (0.575 → 0.730 at 80%, 0.294 → 0.438 at 50%) by giving QBs the wider band
their volatility demands, and it tightens TE — which the global halfwidth was
*over*-covering — from 0.862 to a near-target 0.790 with a narrower interval (9.07
vs 11.27). RB and WR, already close to the global, barely move. No position gets
an unreasonable width; QBs are wide (17.5) because QB scoring genuinely is.

Honest caveat: even per-position, QB still under-covers (0.730 vs 0.80 target).
QB weekly PPR is heavy-tailed and the 2025 QB sample is small, so a symmetric
halfwidth can't fully reach nominal coverage. It's a clear improvement, not a cure
— quantile or asymmetric bands (Session 6) would push further but at a width cost
that wasn't worth it. This is the better of the two interval upgrades.

## Final decision

**Live projections are app-ready as a preview**, and **per-position conformal
intervals are kept** for the live projections. Production historical backtest
behavior is unchanged — the per-position halfwidths are applied in the live
scoring path only, so nothing about the existing backtest tables moves. The
Fantasy Player Board gets a minimal mode toggle (historical vs upcoming week) with
position/team/min-points filters; the page is otherwise untouched.

What keeps this honest about "app-ready": the projections are only as fresh as the
latest completed week of data, the weather is schedule-static rather than a live
forecast, and the carry-forward lag means a player coming off an outlier game is
nudged, not fully updated, until the next data refresh. Good enough to show; worth
a banner noting it's a model preview, not a betting tool.

## Reproduce

```
python -m scripts.eval_session7_live_projection
```
