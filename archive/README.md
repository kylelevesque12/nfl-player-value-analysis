# Archive

Code, reports, and outputs that were part of earlier phases of the project and aren't part of the final story. I kept them here rather than deleting them outright because they document the path I took and a few of them might come back later.

## What's in here and why

**`src/weekly_win_projection.py`** — A logistic-regression home-win-probability model with rolling backtest. I built it as an additional "perspective" but it never got past draft-quality, and the final project focuses on the fantasy and front-office angles. Dropping it tightened the narrative.

**`src/advanced_modeling.py`** — An experimental module that wired in Optuna for hyperparameter search, SHAP for explanation, and MLflow for tracking. The Optuna search didn't find anything meaningfully better than the hand-tuned HGB, the SHAP plots duplicated information already in the permutation-importance reports, and MLflow ended up being noise for a one-developer project. Outputs are in `outputs/tables/advanced_modeling_*`.

**`src/context_features.py`** + **`src/feature_impact.py`** — A side-track investigating whether "context" feature groups (usage context, team environment, schedule context) added value when bundled. The rolling validation comparisons were inconclusive; I folded the genuinely useful pieces (snap share, opponent strength) directly into the weekly fantasy model and the rest didn't earn a permanent spot.

**`outputs/tables/position_model_comparison_*.csv`** — Per-fold tables from the early position-specific HGB experiment. The pooled model won at every position; the lesson lives in the prose of the weekly fantasy and two-stage reports, so the tables themselves don't need to ship.

**`mlruns/`** — MLflow tracking directory from `advanced_modeling.py`. Mostly empty; kept for completeness.

## Why keep this around

For a research project, the path matters. If a reader wants to see what I tried that didn't work, this is where it lives. The active code in `src/` is the version I'd defend; the archive is the version I wrote on the way there.
