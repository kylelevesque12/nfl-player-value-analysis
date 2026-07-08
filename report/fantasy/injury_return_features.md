# Injury-return features: a documented negative result

## The problem

The season model projects a next-season PPR **total**. The linear
candidates that win overall lean on the current season's totals, which
collapse for a player who missed half the year to injury. So a receiver
who averaged 14 PPR/game across 4 games (Malik Nabers, 2025) projects
like a backup, because the model reads his low total, not his healthy
rate. This is the 'injury blindness' the roadmap set out to fix.

## What was tried

Four features meant to separate 'hurt' from 'washed', all computed on
the current-season row (strictly pre-target, leakage-tested):

- `games_missed` — season length minus games played.
- `injury_report_weeks` — weeks the player appeared on the injury report.
- `injury_out_weeks` — weeks he was formally ruled Out or Doubtful.
- `ppr_per_game_x_games_missed` — the interaction a linear model cannot
  construct for itself, meant to let it add back the points a healthy
  player would have scored.

## The result: they do not help

Rolling-origin validation with vs. without the block, measured on the
injury-return cohort (this-season games_played <= 8 with a healthy
per-game rate >= 8):

| Feature set | Model | Cohort RMSE | Overall RMSE |
| --- | --- | ---: | ---: |
| without | elastic_net_total | 86.27 | 59.09 |
| without | two_stage_hist_gradient_boosting | 82.40 | 60.22 |
| without | random_forest_total | 83.71 | 59.61 |
| with | elastic_net_total | 85.94 | 59.12 |
| with | two_stage_hist_gradient_boosting | 82.45 | 60.36 |
| with | random_forest_total | 83.76 | 59.63 |

For the production (elastic net) model the cohort RMSE moves just
**86.27 -> 85.94
(+0.4%)** — below the project's ~0.2% ablation threshold. The
injury-return cohort is intrinsically high-variance (a player coming
off a lost season may bounce back, re-injure, or lose his job), and the
signal the new features carry is already largely present in the
existing per-game and games-played features.

## A model switch is not a fix either

The two-stage games x rate model does better on the cohort in the table
above, which is tempting. But look at what it actually projects for the
marquee 2026 cases:

| Player | 2025 games | Elastic net | Two-stage |
| --- | ---: | ---: | ---: |
| Tyreek Hill | 4 | 87 | 63 |
| Malik Nabers | 4 | 128 | 225 |
| Joe Burrow | 8 | 172 | 184 |
| Christian McCaffrey | 17 | 269 | 178 |
| Justin Jefferson | 17 | 168 | 238 |

The two-stage model does rescue Nabers (a sensible ~200 instead of
~125), but it badly under-projects a healthy elite in Christian
McCaffrey — its noisy games-played sub-model only gives him ~11 games.
It doesn't fix injury blindness; it trades a visible error on injured
players for a worse one on healthy stars, which is exactly why it lost
overall. There is no free lunch here.

## What ships instead

Two honest outcomes, not a fake point fix:

1. The engineered features are **kept on the record but out of
   production** — tested, documented, and pruned for not clearing the
   ablation bar, the same discipline applied to the NGS and ensemble
   experiments.
2. The app **flags players whose projection rests on an injury-shortened
   prior season**, so the reader sees the wide honest range and the
   caveat rather than trusting a false-precision point estimate. The
   right answer to an intrinsically uncertain projection is to show the
   uncertainty, not to manufacture confidence.

