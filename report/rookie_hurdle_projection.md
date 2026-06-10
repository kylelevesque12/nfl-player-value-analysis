# Rookie Projection (Hurdle Model)

Two coupled Bayesian models give a calibrated projection of a rookie's
expected season-long fantasy production. The original PPR/game-only
model silently dropped rookies who didn't play enough games — Jordan
Love in 2020 (behind Rodgers) ended up with a NaN target and was
never learned from. This version handles those cases explicitly.

**Stage 1** (logistic). P(plays >= 4 games in rookie year) given draft
slot, age, height, weight, position. Trained on every rookie in the
frame, regardless of whether they ended up playing.

**Stage 2** (Normal). PPR/game *conditional on having played*, with
the same features. Trained only on rookies who cleared the 4-game
threshold.

**Combined projection**: P(plays) * E[PPR/game | plays]. For Jordan
Love at draft time, this answers: low expected rookie-year production
because he was unlikely to play, not because he was projected as a
bad player.

2,265 rookie player-seasons in the frame; 750 cleared
the 4-game threshold (33%).

## Rolling-origin validation against the full target

The 'full target' is rookie-year PPR/game with non-players treated as
zero rather than dropped from the evaluation.

| Rookie class | n | Played n | RMSE | MAE | 50% cov | 80% cov | Plays Brier | Plays actual | Plays pred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | 202 | 79 | 3.20 | 1.89 | 19% | 71% | 0.125 | 39% | 40% |
| 2021 | 158 | 63 | 2.68 | 1.79 | 19% | 77% | 0.115 | 40% | 44% |
| 2022 | 239 | 79 | 2.52 | 1.61 | 21% | 74% | 0.126 | 33% | 34% |
| 2023 | 225 | 78 | 2.72 | 1.64 | 21% | 75% | 0.120 | 35% | 34% |
| 2024 | 228 | 70 | 2.82 | 1.60 | 25% | 78% | 0.109 | 31% | 35% |
| 2025 | 240 | 78 | 2.53 | 1.60 | 23% | 77% | 0.105 | 32% | 34% |

## Spot-check: Jordan Love

- P(plays >= 4 games as a rookie): 0.76
- Conditional E[PPR/game if plays]: 11.9
- Combined expected rookie-year PPR/game: 9.1
- 80% interval: 3.3 to 15.1
- Actual rookie year (2020): 0.0 PPR/game (0 games)
