# Share-normalized opportunity features — and a 44 → 13 feature simplification

Step 1 of the 2026-07 review (`report/project_review_2026_07.md` §1B.7).
Reproduced by `scripts/eval_opportunity_features.py`; numbers in
`outputs/tables/opportunity_feature_ablation.csv`. Baseline frozen at
`outputs/baselines/v1_fantasy_model_comparison.csv`.

## The question

The season fantasy model used raw counts — `targets`, `receptions`, `carries`
and their per-game and lagged variants. A raw count conflates a player's **role**
with his team's **play volume**: 120 targets in a 700-attempt offense is a
different thing from 120 targets in a 520-attempt offense.

nflverse already ships the share-normalized versions (`target_share`,
`air_yards_share`, `wopr`), and they were sitting unused — `air_yards_share` had
zero references anywhere in `src/`. So: do they earn a place?

## How the features are built

Not by averaging the weekly ratios — that weights a 2-target game the same as a
12-target game. Numerator and denominator are aggregated separately:

```
season target_share = player season targets / team season targets
```

with team totals summed over the skill-position frame, shares computed at the
player-season-**team** level and combined with a games-weighted mean so traded
players keep the share they actually held. Receiving shares are NaN (not 0) for
quarterbacks. See `src/opportunity_features.py`.

Coverage: `target_share` / `air_yards_share` / `wopr` 88.5% non-null (the gap is
QBs, by design), `carry_share` 100%, prior-season lags 61.9%.

## Results

Four arms, identical folds (2020–2024), identical models, target
`next_fantasy_points_ppr`. RMSE, best model per arm:

| arm | n features | ElasticNet RMSE | vs V1 |
|-----|-----------|-----------------|-------|
| **V1 baseline** | 44 | 59.094 | — |
| V1 + shares | 50 | 59.070 | +0.041% |
| lean, raw counts | 13 | 59.330 | −0.400% |
| **lean, shares** | **13** | **59.127** | **−0.055%** |

### Finding 1 — bolting shares onto the existing 44 does nothing

+0.041%, far below the project's 0.2% bar. **Not because the features are
useless** — because the existing set already contains `targets`,
`targets_per_game`, `targets_prev` and `targets_last2_avg`, so a regularized
model has almost nothing left for a share term to explain. This is a redundancy
result, not a feature result.

### Finding 2 — 13 features match 44

The lean set lands at 59.127 vs V1's 59.094: **−0.055%, comfortably inside the
0.2% noise bar**, with a *slightly better* rank correlation (Spearman 0.7229 vs
0.7222). Rank order is what a draft board actually consumes, so that is the
metric that matters most here.

**70% fewer features, same accuracy, marginally better ranking, and every
remaining feature has a stated reason to exist.**

### Finding 3 — share normalization itself clears the bar

Isolating the one design change (same 13-feature structure, shares vs raw
per-game counts):

| model | raw | shares | improvement |
|-------|-----|--------|-------------|
| Elastic Net | 59.330 | 59.127 | **+0.343%** |
| Ridge | 59.339 | 59.135 | **+0.344%** |
| Random Forest | 59.506 | 59.374 | **+0.223%** |
| HistGradientBoosting | 61.325 | 61.188 | **+0.224%** |
| Two-Stage HGB | 59.764 | 60.429 | **−1.113%** |

On four of five models the gain **clears the 0.2% bar**. Share normalization
earns its place — but only in a feature set lean enough for it to matter.

## Honest caveats

- **The two-stage HGB gets worse** (−1.113%). It models games × points-per-game
  separately, and the share block appears to interact badly with its rate stage.
  Reported, not hidden. It is not the selected model.
- **The lean set was hand-specified** from domain reasoning (profile, durability,
  scoring rate, short history, efficiency, role), not searched. A tuned subset
  might do better; the claim here is parity-with-44, not optimality.
- **Finding 2 changes several things at once** (fewer features, per-game instead
  of totals, shares instead of counts). Finding 3 is the controlled comparison
  that isolates the share effect; Finding 2 should be read as "a small
  well-motivated set suffices," not as attributing parity to any one change.

## Recommendation

Adopt the 13-feature share-based set as the fantasy model's feature list, keeping
`FANTASY_FEATURES_V1` frozen in the module and the V1 metrics in
`outputs/baselines/` so the comparison stays reproducible. **The front-office /
value model (`ENHANCED_FEATURES`, target `next_value_score`) is untouched by
this work** — it is a separate list feeding the surplus-vs-cap analysis.

Next experiments, in order (review §1B.7): the `value_epa` vs PPR-history
head-to-head, vacated opportunity + incoming draft capital, red-zone
opportunity from play-by-play, and position-aware age.
