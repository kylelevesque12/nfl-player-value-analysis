# Weekly Fantasy Projection Summary

This is the player-week analog of the season-long fantasy track. It
projects PPR fantasy points for each player's *current* regular-season
game using strictly pregame information (rolling production, usage,
opponent PPR allowed to position, availability proxy, and
schedule/market context). Setup matches the realistic ESPN/DFS use
case: late in the week, project this Sunday's PPR using everything
that happened before kickoff.

## Why a single direct model, not the decomposition

The season-level two-stage experiment (see `report/two_stage_value.md`)
showed that multiplying learned `opportunity x efficiency` components
lost head-to-head to a single direct model: stage-2 efficiency barely
beat its shrink-to-mean baseline, and propagating that near-noise
through the product added error the single model avoids. At the weekly
level efficiency is even noisier (one broken tackle swings the rate),
so the same risk is larger, not smaller.

The lesson from that result is that decomposition belongs in the
*uncertainty layer*, not the point predictor. The weekly module follows
that discipline: the primary projection is a single HistGradientBoosting
model on engineered pregame features, benchmarked against three
explicit baselines, and the per-position variance-share table is kept
purely as a diagnostic.

Modeling rows (player-weeks with an observed PPR target): 55,670
Features used: 43

## Rolling-origin validation

| Method | Type | n | RMSE | MAE | Skill vs recent_4_avg |
| --- | --- | ---: | ---: | ---: | ---: |
| hist_gradient_boosting | model | 34,906 | 6.147 | 4.552 | +0.076 |
| hist_gradient_boosting_per_position | model | 34,906 | 6.346 | 4.704 | +0.046 |
| recent_4_avg | baseline | 34,906 | 6.655 | 4.835 | +0.000 |
| season_to_date_avg | baseline | 34,906 | 6.704 | 4.888 | -0.007 |
| position_mean | baseline | 34,906 | 7.703 | 6.082 | -0.158 |

Each held-out season is predicted using only earlier seasons. Skill
scores are reported against three baselines (last-4 rolling average,
season-to-date average, and a position mean) so the headline number
is honest even though weekly PPR has a low ceiling on R^2.

## Pooled vs position-specific (honest negative result)

A natural extension is to train a separate model per position (QB,
RB, WR, TE) on the theory that usage profiles differ enough that
specialization should help. The rolling backtest says otherwise: at
every position the *pooled* HGB beats its position-specific variant.
The pooled model leverages the larger training sample with `position`
as an input feature, while the per-position models lose more to
smaller training sets than they gain from specialization. This
matches the season-level model-interpretation finding that small
per-position gains do not justify replacing the pooled model.

## Temporal stability (per-season skill vs recent-4-avg baseline)

An external DK closing-line benchmark is only available for 2020-2021
(see `report/external_benchmark.md`). To show the model's edge is not
season-specific, the table below reports per-season skill vs the
recent-4-week rolling average baseline across the full validation
window. The recent-4-avg is the toughest internal baseline (it
already captures most of the rolling-PPR signal), so a steady
single-digit skill score here is the relevant evidence of temporal
stability — not the absolute margin.

| Season | n | Model RMSE | Recent-4-avg RMSE | Skill |
| --- | ---: | ---: | ---: | ---: |
| 2020 | 5,530 | 6.504 | 6.990 | +6.958% |
| 2021 | 5,856 | 6.236 | 6.769 | +7.881% |
| 2022 | 5,818 | 6.087 | 6.654 | +8.519% |
| 2023 | 5,811 | 6.016 | 6.457 | +6.843% |
| 2024 | 5,848 | 6.079 | 6.562 | +7.356% |
| 2025 | 6,043 | 5.967 | 6.502 | +8.238% |

## Why the naive-baseline comparison is the right bar

Leading on the baseline skill score, rather than on the 2020-2021 DraftKings
head-to-head, is a deliberate methodological choice, and it is the one the
forecasting literature endorses.

In standard forecast evaluation, accuracy is judged *relative to a benchmark*,
and the naive forecast — "next period equals the most recent observation," or a
recent average — is the canonical benchmark. Hyndman & Athanasopoulos
(*Forecasting: Principles and Practice*) build their recommended scale-free error
metric, the Mean Absolute Scaled Error, directly on the naive forecast's error:
a score below 1 means you beat naive, above 1 means you lost to it. The framing is
explicit — "every method must beat naive" — and a skill score is precisely the
proportional improvement of a method over that reference. So a 7-9% RMSE reduction
versus the recent-form and season-to-date averages is not an arbitrary yardstick;
it is the exact "did the model earn its complexity?" test the literature prescribes,
and it is available in every season, including the most recent.

Why a single-digit edge is meaningful here rather than disappointing: weekly
fantasy scoring is intrinsically low in predictability. Independent accuracy work
by Fantasy Football Analytics finds weekly projection R² in the single digits to
low-twenties by position (roughly QB 4-10%, WR 4-9%, TE 3-9%, RB 15-20%), because
the hard part is separating startable players from each other, not identifying
which deep reserve will score little. Against that ceiling, a stable few-percent
RMSE reduction over the naive baseline — sustained across six independent yearly
holdouts — is a real edge, and consistency across seasons is exactly the property
that the same accuracy researchers use to rank projection sources. The fantasy
industry itself evaluates projections the same way: FantasyPros' published
in-season accuracy methodology scores experts by mean absolute error against
realized points, so an MAE/RMSE-versus-baseline framing is the field's own bar,
not one invented for this project.

What this does *not* claim: it does not claim the model beats live FantasyPros or
ESPN projections, or that it beats DraftKings in recent years. The DK-implied
comparison is scoped to 2020-2021, the only window where a free market-implied
benchmark exists, and is reported as competitive-to-slightly-ahead there, not
extrapolated forward. See `report/fantasy/external_projection_benchmark_feasibility.md`.

Sources: Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.),
[skill scores / MASE](https://otexts.com/fpp3/accuracy.html); Fantasy Football
Analytics, [Which Fantasy Football Projections Are Most Accurate?](https://fantasyfootballanalytics.net/2024/12/which-fantasy-football-projections-are-most-accurate.html);
FantasyPros, [In-Season Accuracy Methodology](https://www.fantasypros.com/about/faq/football-inseason-accuracy-methodology/).

## Conformal interval coverage

| Target coverage | Empirical coverage | Mean width | n |
| ---: | ---: | ---: | ---: |
| 50.0% | 0.497 | 6.49 | 34,906.0 |
| 80.0% | 0.796 | 12.63 | 34,906.0 |

Intervals are split-conformal: the most recent 20% of each training
fold is held out as a calibration set, and the empirical residual
quantile becomes the interval half-width. Coverage is therefore
distribution-free by construction.

## Limitations

- No injury or inactives signal yet; a starter ruled out hours before
  kickoff is still projected as if active.
- Defense-vs-position-PPR is a coarse opponent feature; it does not
  account for opponent injuries, scheme adjustments, or matchup-
  specific coverage (CB shadow, slot vs outside).
- The model is trained on all skill positions pooled. Position-
  specific models would likely help RB more than they help WR/TE.
