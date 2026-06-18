# Session 6 — Ensemble stacking & quantile intervals (a negative + a near-miss)

## What was tested

The Session 1 weekly model is a single pooled HistGradientBoosting regressor with
symmetric conformal prediction intervals. Two natural questions: can *stacking*
several models squeeze out more accuracy, and can *quantile* gradient boosting
produce better-shaped intervals than the one-size-fits-all conformal halfwidth? An
off-production experiment harness (`src/weekly_ensemble_experiment.py`) answers
both without touching the production model — the rule for this session was that
nothing ships unless it clearly wins.

Everything runs on the exact Session 1 leakage-safe frame, feature list, target,
folds, and position groups. No NGS/PFR features (a guard test enforces that), no
same-week information, time-based validation only.

**Compute note.** The full production HGB config (`max_iter=400`) across the OOF
stacking inner loops was too slow to run repeatedly in this environment, so the
numbers below use a lighter `max_iter=120` applied *identically to every arm*.
The decision rules turn on the *relative* gap between arms, which a uniform
lighter config preserves; the absolute RMSE is a touch higher than the 400-iter
production model but the ordering and margins are what matter here.

## Point models

Four arms, rolling validation on 2024–2025:

- **A — pooled HGB** (the production baseline)
- **B — position-specific HGB** (one HGB per position, ≥50-row threshold, pooled
  fallback otherwise)
- **C — linear (Ridge)** — same features/preprocessing, a deliberately different
  error profile to feed the stacker
- **D — stacked ensemble** — a non-negative Ridge meta-model over A, B, and C

**Stacking leakage control.** The meta-model is trained only on *out-of-fold*
base predictions. For each validation year, the base learners predict the last
two training seasons using models fit on strictly earlier seasons; the meta-model
is fit on those held-out predictions, then the base learners are refit on the full
training window and the meta-model maps their validation predictions to the
ensemble. No base model ever scores a row it trained on — a unit test asserts
this by spying on every base-prediction call.

| Arm | RMSE | vs pooled | MAE | QB | RB | WR | TE |
|---|---:|---:|---:|---:|---:|---:|---:|
| stacked ensemble | 5.934 | **+0.07%** | 4.204 | 7.565 | 5.957 | 5.945 | 4.807 |
| pooled HGB (baseline) | 5.939 | — | 4.265 | 7.543 | 5.980 | 5.944 | 4.819 |
| position-specific HGB | 5.998 | −1.00% | 4.337 | 7.660 | 6.007 | 6.001 | 4.887 |
| linear Ridge | 6.037 | −1.66% | 4.403 | 7.789 | 6.055 | 6.026 | 4.874 |

The stacked ensemble beats the pooled baseline by **0.07% on RMSE** — an order of
magnitude below the 0.5% threshold for keeping a more complex model. The
meta-model essentially learns to lean almost entirely on the pooled HGB, because
the other two base learners are strictly worse: the position-specific model loses
~1% (the per-position sample splits hurt more than the specialization helps, the
same finding the production code's own backtest reached), and the linear model
loses ~1.7%, as expected. The one mild bright spot is MAE — the ensemble's 4.204
vs the baseline's 4.265 (~1.4% better) — because the blend shrinks predictions
slightly toward the middle, helping the median error while barely moving the
squared error. Not enough to justify shipping three models where one will do.

## Intervals: quantile GB vs conformal

Quantile HGB models at 0.10/0.25/0.75/0.90 form 80% and 50% intervals, compared
to the production conformal intervals (symmetric, calibrated on held-out
residuals) at the same nominal coverage.

| Level | Method | Empirical coverage (target) | Mean width |
|---|---|---:|---:|
| 50% | conformal | 0.495 (0.50) | 6.11 |
| 50% | quantile | 0.464 | 6.85 |
| 80% | conformal | 0.786 (0.80) | 11.81 |
| 80% | quantile | 0.832 | 13.62 |

Overall, conformal is the better-calibrated and tighter method: at 50% it nails
the target almost exactly (0.495) and is narrower; at 80% it sits a hair under
target (0.786) and is narrower than quantile, which over-covers (0.832) by being
wider. On the headline overall tradeoff, quantile is "wider without better
coverage" — the negative-result case in the decision rule.

But the by-position breakdown at 80% is where it gets interesting:

| Position | Conformal coverage | Conformal width | Quantile coverage | Quantile width |
|---|---:|---:|---:|---:|
| QB | **0.599** | 12.98 | 0.742 | 18.07 |
| RB | 0.790 | 11.84 | 0.807 | 13.73 |
| WR | 0.794 | 11.76 | 0.854 | 13.65 |
| TE | 0.864 | 11.26 | 0.868 | 11.08 |

The conformal interval **badly under-covers quarterbacks — 60% empirical at the
80% target.** That's a real calibration failure: a single global halfwidth can't
represent QB scoring variance, which is much fatter than the pooled residual
distribution assumes, so QBs fall outside their intervals far too often. The
quantile model, being per-row adaptive and asymmetric, fixes most of that
(0.60 → 0.74), at the cost of much wider QB intervals (18.1 vs 13.0). It's a
genuine improvement exactly where conformal is weakest.

## Decision: keep neither (with one flagged insight)

**Point model:** leave production on the Session 1 pooled HGB. The stacked
ensemble's +0.07% RMSE is a rounding error, the other base learners are worse,
and shipping a three-model stack for no real accuracy is exactly the
complexity-for-nothing the decision rule warns against. Documented negative
result.

**Intervals:** leave production on conformal. Quantile intervals are wider and
over-cover on the overall tradeoff, which fails the keep test. Neither is kept —
but the real finding underneath stands: the production conformal
interval is meaningfully miscalibrated for QBs (60% vs 80%). The cheapest honest
fix is not quantile GB at all but **per-position conformal halfwidths** — compute
the calibration residual quantile within each position rather than globally. That
would fix QB coverage without the across-the-board width inflation quantile GB
brings. That's a targeted interval upgrade worth doing, but it's outside this
session's "ensemble vs quantile" scope, so it's left as a noted next step rather
than smuggled in here.

Net: production is unchanged. The experiment earned its keep by ruling out two
plausible-sounding upgrades and surfacing the one interval problem actually worth
fixing.

## Reproduce

```
python -m scripts.eval_session6_ensemble
```

(Set `weekly_ensemble_experiment.set_hgb_params(max_iter=400)` and folds to
`[2020..2025]` to run the full production config; it's slower but the arm
ordering is unchanged.)
