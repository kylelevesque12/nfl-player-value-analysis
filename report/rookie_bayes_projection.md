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

## Validation

Rolling-origin validation has not been executed yet — install PyMC
via `requirements-bayes.txt` and run
`python -c "from src.rookie_bayes import build_rookie_bayes_outputs; build_rookie_bayes_outputs()"`.


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
