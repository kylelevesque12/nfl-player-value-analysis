# Session 2 — Next Gen Stats & PFR weekly features (a leakage story)

## Goal

Session 1 left the weekly fantasy model with play-by-play usage, depth-chart
rank, and weather. The natural next move was to reach for richer per-game data:
Next Gen Stats (separation, cushion, air-yards share, rushing efficiency, time
to throw) and Pro-Football-Reference weekly charting (broken tackles, drop
rate, passing drops). These describe *how* a player performed, not just how
much, so the hope was that they'd sharpen the projections.

The whole game here is timing. NGS and PFR are **post-game** measurements —
they only exist because the game already happened. Using week-t separation to
predict week-t fantasy points would be looking at the answer sheet. So every
metric had to be turned into a strictly prior-game feature before the model was
allowed near it. `src/external_player_features.py` does that: load
each feed, key it to one row per player-week, and expose only
`groupby(player_id).shift(1)` lag-1 and rolling-3 versions.

NGS files carry `player_gsis_id`, so they join straight onto the model's
`player_id`. PFR is keyed on `pfr_player_id`, so it goes through the rosters
table (`pfr_id → gsis_id`), deduplicated on `(season, pfr_id)` so a mid-season
trade can't fan one player-week into several. All requested columns existed in
the local data, so nothing had to be invented.

## The result that looked too good

Adding the lagged NGS values dropped validation RMSE from 5.876 to 5.333 — a
**9.2% improvement** on the 2025 hold-out. That is a suspiciously large jump for
prior-game receiving metrics, and a large jump is exactly the point to stop
celebrating and start auditing.

The first red flag: the individual lagged NGS features barely correlate with
the target (≈0.02–0.09 for WRs), nowhere near the 0.5+ expected if they
were doing the heavy lifting. So where was the 9% coming from?

A permutation test settled it. Shuffling the NGS values among the rows that
had them, keeping the missing-value pattern exactly intact, and re-running the
model:

| Arm | RMSE (validate 2025) | vs baseline |
|---|---:|---:|
| Session 1 baseline | 5.876 | — |
| + NGS lagged values | 5.333 | +9.24% |
| + NGS values **shuffled** (NaN pattern kept) | 5.330 | +9.30% |
| + NGS values shuffled gave the *same* gain | | |

Scrambling the numbers changed nothing. The model wasn't using the values at
all — it was using the **missingness pattern**.

## Why the missingness was a leak

Here's the mechanism. The lagged NGS table only has rows for player-weeks the
player was actually tracked, and it's left-joined onto the modeling frame on the
*current* week. So a model row gets a non-null NGS value if and only if the
player was a tracked contributor in that same week:

- P(NGS value present | player tracked this week) = **0.948**
- P(NGS value present | player not tracked this week) = **0.000**
- corr(value-present, same-week tracking) = **0.966**
- corr(value-present, **same-week** fantasy points) = **0.359**

The median imputer then fills the missing rows with a single constant, creating
a point-mass the gradient booster happily splits on — effectively recovering a
"was this player active and used *this* week" indicator. That indicator is
contemporaneous with the outcome. It is precisely the same-week-availability
leak the Session 2 guardrail warned against, sneaking in through the back door
of the join's missingness rather than through a raw value.

An early attempt at a "clean" coverage flag — `ngs_value_lag1.notna()` — was
derived from that very pattern, so it inherited the leak and also showed ~+9%.
Catching that was the real lesson of the session.

## The leak-free version, and the honest result

The correct way to ask "was this player a tracked contributor *last* game" is
to mark same-week NGS presence against the player's actual game sequence and
then `shift(1)` — so week-t's own status never enters week-t. That flag behaves
sensibly (corr 0.204 with the target, a genuine prior-game role signal) and is
*not* identical to current-week tracking. Adding it to the model:

| Arm | RMSE (validate 2025) | vs baseline |
|---|---:|---:|
| Session 1 baseline | 5.876 | — |
| + NGS coverage flags (leak-free) | 5.878 | **−0.03%** |
| + PFR lagged values | 5.879 | −0.05% |
| + PFR values shuffled | 5.873 | +0.06% |
| + NGS + PFR coverage flags | 5.877 | −0.01% |

Once the leak is removed, the signal is gone. Properly lagged NGS coverage adds
nothing measurable, because Session 1's usage features (PBP targets/touches,
snap share, depth rank, the active-last-game flags) already encode prior-game
role. PFR is neutral-to-negative on every cut, on top of only covering 2018+
through a fragile id bridge.

## Coverage by position

For completeness, here's how often each feed actually tracks a player in a given
week, by modeling-frame position:

| Feed | QB | RB | WR | TE |
|---|---:|---:|---:|---:|
| NGS receiving | 0.00 | 0.00 | 0.42 | 0.25 |
| NGS rushing | 0.00 | 0.36 | 0.00 | 0.00 |
| NGS passing | 0.84 | 0.00 | 0.00 | 0.00 |
| PFR (any, 2018+) | 0.77 | 0.68 | 0.67 | 0.68 |

The coverage is exactly as expected — NGS tracks each position only in its
relevant phase — which is also why "is this player NGS-tracked" is such a clean
proxy for role, and why it leaks so cleanly when joined on the wrong week.

## Decision

**Keep no NGS or PFR features in the production model.** This is a negative
result, and a deliberate one. The `external_player_features.py` module and the
diagnostic script are kept as the documented investigation — the way to re-test
this if a future feature ever looks too good — but nothing from this
session is registered in `WEEKLY_FANTASY_FEATURES`. A guard test
(`test_weekly_feature_list_excludes_ngs_and_pfr`) fails if any `ngs_`/`pfr_`
column sneaks back in, and `test_value_join_leaks_same_week_availability`
pins the leak so the reasoning isn't lost.

The takeaway: a 9% RMSE drop is not automatically a win. Here it
was the model quietly learning who suited up, and the right outcome was to
throw it away rather than ship it.

## Reproduce

```
python -m scripts.eval_session2_diagnostic
```

Runs the baseline, value, shuffled-value, and leak-free-flag arms for NGS and
PFR on the 2025 hold-out.
