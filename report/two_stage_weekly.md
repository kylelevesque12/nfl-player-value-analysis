# Structurally-Constrained Two-Stage Weekly (WR/TE)

Tier 2 #5 from `PORTFOLIO_ROADMAP.md`. Tests whether decomposing
weekly WR/TE PPR projections into
`expected_team_pass_attempts × target_share × PPR_per_target`,
with target shares renormalized to sum to 1 within each (team, season,
week), beats a pooled HistGradientBoosting model on the same player-
weeks.

The structural constraint encodes real-world physics — a team only
throws so many passes per game, and those passes get distributed
across active receivers rather than assigned independently. Earlier
two-stage attempts in this project lost to single pooled models
because their multiplicative components were unconstrained.

## Head-to-head on identical WR/TE player-weeks

| Method | n | RMSE | MAE | Skill vs pooled HGB |
| --- | ---: | ---: | ---: | ---: |
| two_stage_structured | 21,778 | 6.406 | 4.931 | -9.759% |
| two_stage_structured_shrunk_eff | 21,778 | 6.278 | 4.864 | -7.569% |
| pooled_hgb_wrte_only | 21,778 | 5.837 | 4.261 | +0.000% |

## By validation year

| Year | n | Two-stage RMSE | Shrunk-eff RMSE | Pooled RMSE | Two-stage skill | Shrunk skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | 3,438 | 6.928 | 6.593 | 6.169 | -12.302% | -6.875% |
| 2021 | 3,679 | 6.483 | 6.388 | 5.896 | -9.944% | -8.338% |
| 2022 | 3,582 | 6.495 | 6.418 | 5.865 | -10.750% | -9.435% |
| 2023 | 3,644 | 6.129 | 6.036 | 5.720 | -7.160% | -5.526% |
| 2024 | 3,652 | 6.252 | 6.177 | 5.838 | -7.086% | -5.809% |
| 2025 | 3,783 | 6.157 | 6.067 | 5.544 | -11.057% | -9.433% |

## Per-stage quality

How accurate each stage is *on its own*. If the two-stage product
loses to the pooled model, this table diagnoses which component is
dragging it down. The classical failure mode is stage 3 (PPR per
target) — per-target efficiency is noisy and multiplying it through
compounds error the pooled model avoids.

| Stage | n | Stage RMSE | Mean-only RMSE | Skill vs mean |
| --- | ---: | ---: | ---: | ---: |
| target_share (renormalized) | 21,778 | 0.081 | 0.123 | +34.327% |
| team_attempts | 21,778 | 8.357 | 8.097 | -3.207% |
| ppr_per_target | 19,023 | 1.446 | 1.443 | -0.245% |

## Honest reading

Both two-stage variants lose to the pooled HGB in every fold. That
is a third honest negative result in this project's decomposition
pattern (season-level two-stage value lost in 2024; weekly position-
specific HGB lost at every position; this one loses again). What
makes it different is the per-stage diagnostic table above: it shows
*why* it loses, structurally.

- **Stage 1 (target share) is genuinely informative.** The renormalized
  predictions beat the mean baseline by ~34% on RMSE. The structural
  constraint that target shares sum to 1 within a team-week is a real
  piece of physics and the model can learn it. A fully Bayesian
  Dirichlet stage-1 likelihood would not change this story — stage 1
  is not the problem.
- **Stage 2 (team passing attempts) is approximately noise.** Skill vs
  predict-the-mean is ~0%. Vegas implied team total + recent pass
  rate carry less per-game information than expected.
- **Stage 3 (PPR per target) is genuine noise.** Skill vs mean is
  ~0%; per-target efficiency at the player-week level is dominated
  by a handful of plays. This is the noise-multiplied-through that
  killed the season-level two-stage and kills this one too.

The shrunk-efficiency variant (stage 3 replaced by the position-
season mean) outperforms the full learned variant by ~2 percentage
points in every fold, which *confirms* the diagnosis: the unshrunk
stage 3 was actively adding error rather than information. But even
with the prescription applied, the structurally-constrained two-stage
still loses to pooled HGB by ~7-8%.

## What this means for the portfolio

The cumulative evidence across four decomposition attempts in this
project is now a real *finding*: for weekly fantasy point projection,
tree-based pooled models on engineered rolling features extract the
team-attempts and per-target-efficiency signals more efficiently
than any explicit multiplicative decomposition tested in this project. Adding
structural constraints (target-share renormalization, position-mean
shrinkage) helps the decomposition somewhat but does not close the
gap. The pooled HGB's implicit feature interactions are the right
inductive bias for this problem.

The actionable next bet is *not* another decomposition variant. It is
either (a) a different model class for the pooled approach (e.g. a
gradient-boosted *quantile* model for proper per-prediction interval
shapes), or (b) better features — specifically the depth-chart-rank
and snap-projection signals that the existing nflverse-supplementary
feeds *would* provide if their schemas were cleaner.
