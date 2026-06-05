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
Features used: 28

## Rolling-origin validation

| Method | Type | n | RMSE | MAE | Skill vs recent_4_avg |
| --- | --- | ---: | ---: | ---: | ---: |
| hist_gradient_boosting | model | 34,906 | 6.187 | 4.593 | +0.070 |
| hist_gradient_boosting_per_position | model | 34,906 | 6.381 | 4.741 | +0.041 |
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

## Conformal interval coverage

| Target coverage | Empirical coverage | Mean width | n |
| ---: | ---: | ---: | ---: |
| 50.0% | 0.500 | 6.63 | 34,906.0 |
| 80.0% | 0.794 | 12.69 | 34,906.0 |

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
