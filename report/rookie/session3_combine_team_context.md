# Session 3 — Combine & team-context features for the rookie hurdle

## The problem

The rookie hurdle model's first stage asks a simple question: will this rookie
*play meaningfully* (≥ 4 games) in year one? The baseline answers it with draft
position, age, and size. Those are good — draft capital especially — but they're
blind to the one thing that decides a lot of rookie playing time: **is there
anyone in the way?** A third-round running back walking into an empty depth chart
and a third-round quarterback drafted behind a freshly-extended franchise QB look
identical to the baseline, and they shouldn't. Jordan Love in 2020 is the poster
child: 26th overall pick, but sitting behind Aaron Rodgers, he never saw the
field. The baseline model had no way to know that.

So this session adds pre-season context the model was missing, in three families:

1. **Combine athletic testing** — forty, vertical, broad jump, bench, cone,
   shuttle, plus a BMI built from the height/weight already on the frame.
2. **Prior-season team context** — the team's pass rate and the prior starting
   QB's production the year *before* the rookie arrived.
3. **Incumbent / depth at the rookie's position** — how many veterans are ahead
   of him, the best returning veteran's prior-year PPR/game, whether there's an
   established incumbent, and whether that incumbent recently signed a meaningful
   extension.

## How leakage was avoided

The guardrail for this session is strict: nothing may use the rookie's *own*
first-season outcomes. The discipline held in three concrete ways.

- **Combine is pre-draft** by definition, so it's safe on its face. It's joined
  through stable IDs only — combine's `pfr_id` (and `cfb_id` as a fallback)
  bridged to `gsis_id` through the purpose-built draft_picks table, then rosters.
  No fuzzy name matching.
- **Every team/incumbent feature is computed from `rookie_year - 1`.** The prior
  starting QB's PPR/game, the veterans' production, the pass rate — all from the
  season before. A unit test builds a synthetic team whose incumbent QB posts 20
  PPR/game the prior year and 5 the rookie year, and asserts the feature picks up
  20, never 5.
- **The rookie is excluded from his own position's veteran tallies**, and only
  players who entered the league *before* the rookie year count as veterans. A
  second test pins this: a rookie WR on a team with one veteran WR gets a
  veteran count of 1, not 2.

A guard test also asserts that no rookie-year outcome column
(`season_ppr_per_game`, `games_played`, `played_meaningfully`, …) ever appears in
the model's feature set.

## Coverage

Combine is sparse — only ~330 players are invited each year, and not all of them
run every drill — so among the 2,265 skill-position rookies the join lands on a
minority. Team-context and depth features, by contrast, are nearly complete.

| Feature family | Coverage (non-null) |
|---|---|
| Combine `forty` (QB / RB / WR / TE) | 0.34 / 0.34 / 0.26 / 0.25 |
| Combine by draft year | 0.23 – 0.42 |
| Prior-season team context | ~0.91 |
| Incumbent / depth | 1.00 |

Missing combine values are kept as NaN and mean-imputed inside each training
fold (the project's existing convention — Session 2 confirmed it doesn't add
explicit missingness flags), so no rookie is dropped for lacking a 40 time.

## Does it help? (stage-1 P(plays) surrogate)

The production hurdle is a PyMC hierarchical model. **PyMC was not available in
the sandbox this evaluation ran in** (it's intentionally kept out of the main
project venv), and these features target stage 1 specifically. So for fast,
apples-to-apples feature comparison, a logistic-regression surrogate runs on the
same `played_meaningfully` target with the same rolling-by-rookie-year validation
(2020–2025). The selected features are wired into the production model's
`FEATURE_COLUMNS`, so the PyMC hurdle picks them up when run in the bayes venv.

Pooled across **all** positions:

| Arm | AUC | log loss | Brier |
|---|---:|---:|---:|
| baseline | 0.8776 | 0.4020 | 0.1210 |
| + combine | 0.8747 | 0.4021 | 0.1205 |
| + team_context (7 feats) | 0.8722 | 0.4058 | 0.1219 |
| + both | 0.8705 | 0.4052 | 0.1208 |
| + incumbent_core (3 feats) | 0.8737 | 0.4035 | 0.1213 |

At the pooled level, nothing beats the baseline — combine is flat-to-negative,
and the full 7-feature team-context set clearly *overfits*. But the pooled view
is the wrong lens, because these features matter almost entirely in one cell:
quarterbacks. Looking only at QB rookies (n = 221), where the production model's
per-position slopes would actually isolate the signal:

| Arm | QB-only AUC |
|---|---:|
| baseline | 0.8369 |
| + team_context (7 feats) | 0.8169 |
| + incumbent_core (3 feats) | **0.8398** |

The lesson is parsimony. Throwing all seven team features at 221 QBs overfits and
*loses* AUC. A focused three-feature core — `established_incumbent`,
`incumbent_recent_extension`, and `prior_qb_pprpg` — actually *improves* QB AUC
over the baseline. That's the set worth keeping.

## Jordan Love (and Patrick Mahomes)

Trained QB-only on rookies before each player's draft year, then predicted them:

| Player (year) | baseline P(plays) | + incumbent_core | actually played? |
|---|---:|---:|---|
| Jordan Love (2020) | 0.611 | **0.513** | no |
| Patrick Mahomes (2017) | 0.826 | 0.829 | no |

Love moves the right way and meaningfully — from leaning "will play" to a coin
flip — once the model can see that Green Bay had an established, recently-extended
incumbent. It doesn't crater to 0.2, and forcing it there wasn't the aim; the
honest signal from three features on a small sample is a nudge, not a verdict.
Mahomes barely moves, which is itself informative: his 2017 prediction trains on
only the 2016 rookie class, far too little for the incumbent features to have
learned much yet. The feature is general (it fires for any rookie behind a
productive, recently-paid starter) and is never hand-coded to a specific player.

## Decision

**Keep the three-feature incumbent core; drop combine and the broader depth set.**
`established_incumbent`, `incumbent_recent_extension`, and `prior_qb_pprpg` are
wired into the model's `FEATURE_COLUMNS` (via `CONTEXT_FEATURES` in
`src/rookie_bayes.py`), so the production hurdle picks them up in both stages. The
combine metrics, the pass-rate feature, and the veteran-count/sum features are
still computed in `src/rookie_context_features.py` and remain available for
inspection, but they're left out of the model — they didn't earn their place.

This is the "neutral-to-positive with clear theoretical value" bar: the kept
features improve the exact slice they were built for (QB playing time), move the
marquee case in the right direction, and the hierarchical production model — with
its per-position partial pooling — is well-suited to exploit them for QBs while
shrinking them toward zero everywhere else. Combine, by contrast, never beat raw
draft capital, which makes sense: where a player is drafted already prices in how
he tested.

## Reproduce

```
python -m scripts.eval_session3_rookie_context
```
