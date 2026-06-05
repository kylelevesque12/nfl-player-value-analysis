# Bayesian Rookie Projection

Hierarchical Normal regression on rookie-season PPR per game, partial-
pooled across the four skill positions. Solves the cold-start problem
the existing HGB stack cannot: projecting a rookie before they have any
NFL snaps. Features: log draft pick, age at draft, height, weight (and
an optional college-production score). The model class is the Tier 2 #4
deliverable from `PORTFOLIO_ROADMAP.md`.

PyMC is isolated in `src/rookie_bayes.fit_rookie_model` because its ABI
conflicts with the rest of the project's pins. Run this section in a
dedicated venv built from `requirements-bayes.txt`.

Rookie modeling rows: 2,265

## Validation (rolling-origin by rookie class)

| Rookie class | n | RMSE | MAE | Bias | 50% coverage | 80% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | 105 | 4.143 | 3.194 | +0.639 | 0.448 | 0.810 |
| 2021 | 91 | 3.743 | 2.960 | +1.517 | 0.451 | 0.857 |
| 2022 | 101 | 3.227 | 2.646 | +0.779 | 0.525 | 0.881 |
| 2023 | 94 | 3.756 | 2.939 | +0.200 | 0.479 | 0.872 |
| 2024 | 93 | 3.804 | 3.092 | +0.795 | 0.473 | 0.763 |
| 2025 | 101 | 3.325 | 2.702 | +0.795 | 0.545 | 0.871 |

## Honest caveats

- College production is not yet wired in. The current model uses draft
  capital and physical features only. Adding a college-translation
  score is the obvious next data-acquisition step (see
  `scripts/fetch_college_production.py` for the stub).
- The Normal likelihood will under-cover for elite rookies (right-
  tailed PPR distributions). A Student-T likelihood is a one-line
  upgrade if calibration looks off.
- Position-specific submodels could plausibly help QB the most (small
  sample, very different production scale). Partial pooling is the
  bet that shared structure outweighs that.
