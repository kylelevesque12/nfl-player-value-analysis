# Portfolio Roadmap & Build Log

This started as a sequenced plan to take the project from "clean engineering" to a
credible NFL-front-office / fantasy data-science portfolio. Phase 1 (the modeling
work) and the app-polish phase are now complete; this document is the finished
build log. It reads as one unified product — a weekly fantasy projector, a
contract-adjusted surplus framework, a rookie opportunity model, and a causal
study of QB injury reports — with the negative results kept on the record because
they are part of what makes the rest credible.

Status legend: `[x]` complete · `[~]` complete with a documented limitation ·
`[ ]` deliberately out of scope.

---

## Phase 1 — Modeling (complete)

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

## App polish (complete)

**Stage 8 — Landing page.** `[x]`
A real entry point: hero, four findings cards (one per research thread) with
drill-in navigation, a methodology strip, and a "how to use this" guide.
→ `report/app/session8_landing_page.md`

**Stage 9 — Detail-page component migration.** `[x]`
Every routed page pulled onto one scaffold (title → summary → KPIs → visual →
caveat → expanders → footer), with stale pre-Stage-4/5 copy corrected.
→ `report/app/session9_component_migration.md`

**Stage 10 — Global player search + unified detail view.** `[x]`
Sidebar search over a 2,721-player index keyed on stable gsis ids; one detail page
assembles every output for a player (weekly, live, surplus, rookie, causal) with
clean "not available" states for missing modules.
→ `report/app/session10_player_search.md`

**Stage 12 — Cleanup, CI, prose pass.** `[x]`
Regenerated the salary outputs so the committed data matches the Stage 4 code,
corrected stale `inflated_apy` wording across the README and reports, added GitHub
Actions CI (compile check + data-independent tests), and tightened `.gitignore`.
→ `report/app/session12_cleanup_ci.md`

## Remaining

**Stage 11 — Mobile / README visuals.** `[ ]`
The one cosmetic item left: a narrow-viewport responsive pass on the `st.columns`
layouts and recorded screenshots / a GIF for the README. Deliberately deferred —
it does not affect any result.

---

## Scope boundaries (intentionally not done)

These were considered and left out on purpose, not forgotten:

- **Paid external projections (FantasyPros / ESPN / OverTheCap premium).** The
  DraftKings-implied benchmark is free but covers only 2020-2021; the cap-hit
  reconstruction approximates from contract terms. Both gaps are documented and
  would close with paid data. See `report/fantasy/external_projection_benchmark_feasibility.md`.
- **DFS lineup optimizer, NLP on injury reports, Streamlit→React rewrite.** Narrow
  payoff or out of scope for a research portfolio.

## What "finished" looks like here

A hiring manager opening the repo finds: a README that states four results in plain
language with honest scope, a Streamlit app with a landing page and consistent
detail pages, global player search, a green CI badge on the data-independent test
suite, reports written as research notes (including the experiments that failed),
and a salary track whose committed data matches its code.
