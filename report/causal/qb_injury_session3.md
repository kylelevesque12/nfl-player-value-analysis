# Causal Session 3: first injury-report appearance as treatment

## Why move the treatment earlier

Sessions 1-2 defined treatment as the formal QB injury event — the week a
starting QB was ruled Out and replaced — and landed on an honest null: WR
PPR didn't drop after the designation. The mechanism I proposed there was
endogenous timing. By the time a QB is formally Out, he has usually been
playing hurt for weeks, and his receivers have already been sliding. The
formal Out is a *lagging* indicator. So the obvious follow-up, and the
point of this session, is to move treatment to the first sign of trouble:
the first week the established starter shows up on the injury report at
all — a Questionable tag, a limited practice, anything — even if he still
starts that Sunday.

## Sample construction and eligibility

One candidate event per (team, season): the first week the modal starting
QB appears on the injury report within his starting tenure. Eligibility
then requires at least 3 pre weeks and 2 post weeks. Drop counts:

| Stage | Count |
| --- | ---: |
| candidates | 320 |
| no_injury_report | 75 |
| event_week_le_min_pre | 62 |
| insufficient_pre_starts | 73 |
| insufficient_post_weeks | 6 |
| eligible | 104 |

**Treated events under first-report: 104.** Under the old
Out-only trigger (same eligibility): **19** — the earlier
definition yields far more events because most starting QBs hit the injury
report long before (or without ever) being formally ruled Out. By
construction the first-report week is on or before the first Out week for
every team-season where both exist.

First-report status mix (NaN = practice-report-only, no game designation): {nan: 70, 'Questionable': 20, 'Out': 11, 'Doubtful': 3}.

## Design and controls

Outcome is WR PPR/game, the same family as sessions 1-2. The panel runs
event time {-3..-1} (pre), 0 (first report), {+1..+2}
(post). Controls are receivers on teams whose starting QB was both
*stable* and *injury-report-free* across the whole window — a would-be-
treated team can't leak into another event's control pool. Same-calendar-
week matching absorbs league-wide trends; nothing post-treatment is used
to pick controls.

- Treated WRs: **207**, control WRs: **349**, events in panel: **104**, panel rows: 20,820.

## Parallel trends

Player-fixed-effect pre-period interaction coefficients (vs offset -1):

| Pre offset | Coef | SE | t | p |
| ---: | ---: | ---: | ---: | ---: |
| -3 | +0.786 | 0.624 | +1.259 | 0.208 |
| -2 | +0.239 | 0.620 | +0.385 | 0.701 |

No pre-period interaction is significant at 5%, so the fixed-effect
parallel-trends test **passes** — cleaner than session 1, where it
failed. One honest caveat: the cell-mean event study below shows the
treated-minus-control gap is already a touch elevated at -3, so the
pre-period isn't perfectly flat. I read trends as plausible but not
pristine.

## Treatment effects

Event-study coefficients (cell-mean DiD, reference = offset -1):

| Offset | Coef | SE | p |
| ---: | ---: | ---: | ---: |
| -3 (pre) | +1.028 | 0.454 | 0.023 |
| -2 (pre) | +0.399 | 0.541 | 0.460 |
| 0 (event) | -0.128 | 0.470 | 0.785 |
| 1 (post) | -1.085 | 0.479 | 0.023 |
| 2 (post) | -0.537 | 0.520 | 0.302 |

- **Pooled post-period ATT (event study): -0.584 PPG** (SE 0.283, p ≈ 0.039)
- **Simple 2x2 DiD: -1.014 PPG** (SE 0.279, p ≈ 0.000)
- **Matched-panel pooled ATT: -1.477 PPG** (p ≈ 0.000)
- Treated events: 104, treated WRs: 207, control WRs: 349.

## Interpretation

Moving the treatment earlier surfaces a measurable negative effect that the Out-only
design missed. The drop is concentrated at offset +1 (the first game after
the QB first appears on the report), the treatment-week effect itself is
near zero, and the pooled post-period estimate is roughly −0.6 PPG in the
event study and around −1 PPG in the 2x2 and matched specifications. That
is consistent with the mechanism session 2 hypothesized: the causal damage
clusters around when a QB's health first becomes shaky, not around the
formal Out weeks later. It is real but modest — a fraction of a fantasy
point per receiver per week — and the marginal p-values plus the slightly
elevated -3 pre-period gap mean I would not sell this as a clean headline
causal estimate.

## Limitations

- WR PPR is noisy; with ~100 events the design is moderately powered and
  the post-period estimate sits near the 5% significance border.
- 'First report' lumps a season-ending injury in with a Wednesday
  limited-practice rest day; the treatment is heterogeneous by design.
- Some pre-period drift remains, so part of the post drop may be the
  continuation of an already-declining trajectory rather than pure causal
  effect — the same endogenous-timing problem, pushed one step earlier.

## Verdict

**Underpowered-but-suggestive negative effect, not a clean headline.**
Re-timing treatment to the first injury-report appearance does move the
result off the Out-only null toward a small (~0.6–1.0 PPG) post-period WR
decline, in the direction the session-2 mechanism predicted. I report it as
suggestive evidence that limited QB availability matters before formal
absence — while being explicit that the effect is modest, the design is
only moderately powered, and the pre-period is plausible rather than
pristine. No QBs or teams were hand-selected; the construction is general.
