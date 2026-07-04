# Portfolio Roadmap & Build Log

This document has two parts. The first part is the build log for version 1, which
is complete: the modeling work, the app, and the write-ups. The second part is the
plan for version 2, which turns the app from a research dashboard into a genuine
fantasy football product while the repo continues to carry the research depth.

The guiding split for version 2: **the app is a fantasy product; the repo is the
research portfolio.** A league-mate should be able to draft with the app in August
without any explanation, a recruiter reading the repo in October should see the
rigor, and every number on screen should be honest about what it does not know.

Status legend: `[x]` complete · `[~]` complete with a documented limitation ·
`[ ]` planned · `[–]` deliberately out of scope.

---

## Version 1 build log (complete)

### Phase 1 — Modeling

**Stage 1 — Weekly model: PBP role + weather features.** `[x]`
Rebuilt depth-chart rank from play-by-play (nflverse dropped the numeric
`list_rank` field ~2024) and added game weather. Weekly RMSE fell **6.020 → 5.944
(−1.27%)**, every position improving. Leakage-safe (`shift(1)`) and tested.
→ `report/session1_pbp_weather.md`

**Stage 2 — NGS / PFR features: rejected.** `[x]`
NGS and PFR weekly stats looked like a +9% gain, but a permutation test showed the
signal was entirely a same-week *availability* leak through the join's missingness
pattern. Nothing kept in production; the investigation is the deliverable.
→ `report/fantasy/session2_ngs_pfr_features.md`

**Stage 3 — Rookie incumbent context.** `[x]`
Added a focused 3-feature incumbent core (established incumbent, recent extension,
prior-year starting-QB production) to the rookie hurdle gate. Jordan Love's modeled
P(plays) moves **0.611 → 0.513**. Combine and broad-depth features were tested and
dropped — they did not beat draft capital.
→ `report/rookie/session3_combine_team_context.md`

**Stage 4 — Reconstructed cap hits.** `[x]`
Replaced the flat `inflated_apy` proxy with a season-specific cap hit reconstructed
from contract terms (prorated signing bonus + backloaded base), flagged per
player-season. The surplus framework now prices stars in the early years of
mega-extensions realistically; Brock Purdy 2023 remains the top surplus season.
An estimate, not exact cap accounting — stated plainly.
→ `report/salary/session4_cap_hit_reconstruction.md`

**Stage 5 — Causal: first injury-report treatment.** `[x]`
Re-timed the QB-injury treatment from the formal "Out" designation (Stages 1-2
null) to the first injury-report appearance, expanding the event set **19 → 104**
and surfacing a small post-period WR decline (ATT ≈ **−0.58 PPG, p ≈ 0.04**).
Reported as suggestive and underpowered, with the parallel-trends caveat.
→ `report/causal/qb_injury_session3.md`

**Stage 6 — Ensemble & quantile intervals: rejected, with one insight.** `[x]`
Stacking improved RMSE by ~0.07% (below threshold) and quantile intervals were
wider without better calibration — both left out of production. The useful finding:
the global conformal interval badly under-covers QBs.
→ `report/fantasy/session6_ensemble_quantile.md`

**Stage 7 — Live weekly projections + per-position conformal.** `[x]`
Built the infrastructure to score the upcoming week from carried-forward player
state and schedule context (leakage-safe), and replaced the global conformal
halfwidth with per-position halfwidths — QB 80% coverage rises **0.575 → 0.730**.
→ `report/fantasy/session7_live_projection.md`

### App and presentation

**Stages 8–10, 12 — Landing page, component migration, player search, CI.** `[x]`
A real entry point, one scaffold for every detail page, global player search over
a 2,721-player index, GitHub Actions CI on the data-independent tests, and a
cleanup pass that synced the committed data with the code.
→ `report/app/`

**Stage 13 — Fantasy-first redesign.** `[x]`
Rebuilt the app around its actual audience: product-first navigation (Draft Board,
Rookies, Player Detail, Front Office, Methodology & Research), nav rows instead of
radio circles, a card-based Home with top-projected player tiles and team colors,
research pages reduced to summary cards linking to the repo reports, the
methodology reference rewritten in plain sequential style, and app copy translated
from research voice to fantasy voice. Also fixed in this pass: a cloud ImportError
from Streamlit's hot-reload, a broken multi-Python local venv (rebuilt on 3.12 /
Streamlit 1.58 to match the cloud), and migration off deprecated Streamlit APIs.

---

## Version 2 plan

Version 2 is organized around three calendar anchors: August is fantasy draft
season (when the app is most useful), the NFL season starts around September 10
(the app must update itself by then), and internship applications open August
through October (when the portfolio should be at its best).

The centerpiece is the **Draft Room**: a draft assistant that does not just answer
"who is the best player available," but plans the whole rest of the draft. Two
concepts drive it. The first is **fantasy VORP** (value over replacement player):
a player's projected points above what a freely available player at the same
position would score, which is what makes players comparable across positions.
The second is **positional dropoff**: the cost of waiting. If passing on a running
back now means a much worse running back at the next pick, while receiver holds
its value for three more rounds, the right pick is the running back — and the
engine should say so with a number.

### July block — draft-ready (now → ~Aug 1)

- `[~]` **App rebuild around player content.** Done: Home now leads with top
  projected players, projected risers, and a regression watch (efficiency-driven
  production from the decomposition outputs, RB/WR/TE only since QB efficiency
  genuinely repeats); the Draft Board gained uncertainty-scaled tiers and
  stable/shaky role badges; Front Office and Rookies left the navigation and
  became research summary cards. Final navigation: Home · Draft Board · Draft
  Room · Player Detail · Methodology & Research. The remaining piece is the
  rookie-fliers Home module and Draft Board rookie view, which wait on the 2026
  rookie class being scored (the "rookies into the season rankings" item).
- `[x]` **ADP data.** `scripts/fetch_adp.py` pulls a snapshot from Fantasy
  Football Calculator's free public API (943 real 12-team PPR drafts at first
  fetch) into `data/external/`, and `src/adp.py` matches it to the projection
  table by normalized name + position with a reported match rate (92.6%; every
  unmatched top-100 pick is a 2026 rookie, which is the next item's job).
- `[x]` **Fantasy VORP + overall board.** `src/fantasy_vorp.py` computes
  replacement level by actually filling all 12 starting lineups including flex,
  then ranks all positions together by value over replacement, with auction
  values and an edge-vs-ADP column. Written to
  `outputs/tables/draft_board_2026.csv`, documented in
  `report/draft_board_vorp.md`, and shipped as the Draft Board's Overall view
  plus a draft-day values module on Home.
- `[x]` **Positional scarcity chart.** Projected points by positional rank, one
  line per position, so the running back cliff and the tight end cliff are visible
  before anyone drafts. Live as the first content on the Draft Room page, with a
  "cost of drafting the 12th-best instead of the best" table beneath it.
- `[ ]` **Draft Room v1 — the whole-draft planner.** League setup (teams, slot,
  snake, roster spec), one-click pick tracking, and a deterministic planner: first
  project the "shelf" (the expected best available player at each position at each
  of the user's future picks, assuming opponents draft by ADP), then a small
  dynamic program assigns positions to the remaining picks to maximize total
  starter value. The recommendation for the current pick is the first move of the
  best full plan, shown with a cost-of-waiting table and tier-aware language. The
  engine is projection-agnostic from day one: it consumes any values table, which
  is what later allows user-supplied rankings.
- `[ ]` **Rookies into the season rankings.** The Bayesian rookie projections feed
  the Draft Board with honestly wide ranges. A 2026 draft board without rookies is
  broken from a user's point of view.
- `[ ]` **Injury-blindness fix.** The season model currently treats an
  injury-shortened season as pure decline (Nabers, Hill, Burrow). Add features for
  *why* games were missed (IR flags, injury-report weeks, healthy per-game rates,
  and a rate × games-missed interaction), evaluate on the cohort of players with
  ≤8 games in the prior season, report before/after, and flag injury-return
  players in the UI.

### August block — peak portfolio

- `[ ]` **Quantile / CQR intervals.** Replace the constant-width intervals with
  player-conditional, asymmetric floors and ceilings (conformalized quantile
  regression), evaluated with pinball loss and coverage by segment. This is what
  makes the tiers and boom/bust identity real, and it is the strongest single
  research artifact in the plan.
- `[ ]` **Draft Room v2 — Monte Carlo.** Simulate the rest of the draft hundreds
  of times (opponent picks sampled around ADP with noise and roster caps), solve
  the plan inside each simulated future, and report robustness: "taking a running
  back now leads to the best final roster in 71% of simulated drafts."
- `[ ]` **Draft strategy tournament.** Validate the planner against
  best-player-available, an ADP follower, and static VORP over hundreds of
  simulated drafts; publish the final-roster results whether the planner wins or
  loses.
- `[ ]` **Bring-your-own-board.** Users upload a rankings CSV from any source
  (ESPN export, FantasyPros, Underdog); the app matches names (with a reported
  match rate), lets the user pick the value source (our model / their board /
  blend), wraps our calibrated uncertainty around their point estimates, and shows
  a disagreement view. Nothing third-party is stored or redistributed — the board
  lives only in the user's session.
- `[ ]` **Historical ADP backtest.** Model rank versus ADP rank for 2020–2025,
  extending the market comparison past the 2021 free-data wall that is currently
  the project's biggest evaluation limitation.
- `[ ]` **Decision-grade evaluation.** Tier accuracy, start/sit regret, and the
  probability the model's pick outscores the baseline's pick — the models judged
  on the decisions they drive, not just RMSE.

### Late August → September 10 — alive in-season

- `[ ]` **This Week board.** The live weekly projections surfaced as a flagship
  page (they currently exist only inside Player Detail), with staleness detection
  and an offseason mode.
- `[ ]` **Automated weekly refresh.** A GitHub Actions cron that fetches data,
  rebuilds the live projections, runs sanity guardrails (row counts, no missing
  projections, week strictly advancing), and commits only when everything passes.
  The deployed app then updates itself all season at zero hosting cost.
- `[ ]` **Start/Sit comparator.** Two players side by side, full distributions,
  and "the model favors X in about N% of outcomes" — straightforward once the
  quantile intervals exist.

### Fall, opportunistic

- `[ ]` Player headshots and a richer Player Detail card (nflverse rosters carry
  headshot URLs).
- `[ ]` Feature-group ablation for the weekly model; prune any group earning less
  than ~0.2% skill before the automation hardens around it.
- `[ ]` Consistency and boom/bust profiles from the weekly backtest residuals
  already on disk.
- `[ ]` Scoring-format toggle (PPR / half / standard), which requires projecting
  stat components rather than PPR directly.
- `[ ]` Sleeper live-sync for the Draft Room (Sleeper's API is genuinely open, so
  picks could appear in the room without clicking).
- `[ ]` Season simulation: Monte Carlo over weekly distributions to answer
  questions like "what is the chance he finishes as the QB1?"

### Standing practices, every block

Tests stay green and grow with each feature. UI changes are verified with rendered
screenshots before pushing. Each modeling change ships with an honest write-up
(problem → fix → evidence → what is still unsolved). The README and
`PROJECT_REFERENCE.md` stay in sync with what is actually built.

The dependency spine: **ADP → VORP → Draft Room v1 → Monte Carlo → tournament
write-up**, and **quantile intervals → tiers and boom/bust → start/sit**.
Everything else hangs off those two chains.

---

## Scope boundaries (intentionally not done)

These were considered and left out on purpose, not forgotten:

- **Paid external projections (FantasyPros / ESPN historical / OverTheCap
  premium).** The DraftKings-implied benchmark is free but covers only 2020–2021;
  the cap-hit reconstruction approximates from contract terms. Both gaps are
  documented and would close with paid data. The historical-ADP backtest in the
  August block is the free-data answer to the same question. See
  `report/fantasy/external_projection_benchmark_feasibility.md`.
- **Redistributing third-party projections in the deployed app.** ESPN has no
  official projections API, and shipping their numbers publicly is a terms-of-use
  problem. The bring-your-own-board upload is the deliberate workaround.
- **DFS lineup optimizer, NLP on injury reports, Streamlit→React rewrite.** Narrow
  payoff or out of scope for a research portfolio. Hosting alternatives were
  evaluated (Hugging Face Spaces, Render, static-site rebuilds); Streamlit
  Community Cloud remains the right call until the app has real users.

## What "finished" looks like for version 2

A league-mate opens the app in August and drafts with it: an overall board with
tiers and honest ranges, a Draft Room that plans their whole draft, and rookie and
injury-return players handled sensibly. During the season the app updates itself
every week and answers start/sit questions with probabilities. A hiring manager
opening the repo finds the same rigor as version 1 — every claim graded against a
hard baseline, every failure on the record — plus three new research artifacts:
calibrated quantile intervals, a validated draft strategy, and a decision-grade
evaluation of the projections.
