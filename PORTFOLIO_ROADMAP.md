# Portfolio Roadmap

This file is the sequenced plan for taking the project from "clean engineering portfolio" to "credible NFL team / ESPN DS portfolio." It is opinionated about order — items at the top buy the most credibility per week of work.

The goal is **not** to ship every item below. It is to make targeted, high-impact moves that close the gap a hiring manager would notice on a first read of the README.

Status legend:
- `[ ]` Not started
- `[~]` Scaffolding only / partial — needs data or a follow-up pass
- `[x]` Complete

---

## Tier 1 — Required for credibility

These are the non-negotiables. Until all three land, the project reads as "well-engineered but doesn't compete with what's already public."

### 1. External benchmark evaluation

**Why it matters.** Beating an internal baseline is the floor. The ceiling reference is: do we beat what a fantasy player could read for free? Without that comparison, every skill score in the project is a number without a market.

**What to build.**
- [x] Scaffolding: `src/external_benchmark.py` loads `data/raw/external_projections.csv` and runs head-to-head vs the weekly fantasy model with MAE / RMSE / win-rate / by-position output.
- [x] Acquire DraftKings closing-line implied projections via RotoGuru. `scripts/fetch_rotoguru_salaries.py` downloads the SCSV archives; `scripts/build_external_projections_from_dk.py` fuzzy-matches names to nflverse gsis_ids and fits per-(season, position) salary→points conversions. RotoGuru's free archive covers **through 2021 only**.
- [x] Run head-to-head on the 2020-2021 overlap. **Current result: +1.1% RMSE skill vs market overall; +1.7% QB, +1.4% RB, +1.2% WR, −0.4% TE (honest negative); 52-55% win rate by position.** Documented in `report/external_benchmark.md`.
- [x] Add the benchmark table to README. Now in the main results section near the top of the document.
- [ ] **Extend coverage to 2022-2025.** Requires a paid source: Stokastic (~$50/mo), FantasyData (API pricing), or scraping FantasyPros archives (MVP ~$8/mo gets historical access). Or use the `ffanalytics` R package (free, multi-source). The scaffolding accepts any CSV matching the documented schema, so this is purely a data-acquisition step.
  - **Considered and rejected**: a free Vegas-team-environment-implied benchmark (per-(season, position) OLS of PPR on implied_team_total, spread, is_home). It extended coverage to 2022-2025 and the model beat it by ~19% every season. Pulled because Vegas-implied has *no player-specific signal* — the +19% beat invites the read "this model is just better at team-environment work" and dilutes the genuinely-player-level +1.6% beat against DK. Per-season *internal* baseline stability (in `report/weekly_fantasy_projection_summary.md`) covers the temporal-stability story cleanly without that risk.

**Effort:** 1–2 weeks once data is available. Data acquisition is the bottleneck.

**Definition of done:** A committed `external_benchmark.md` that says, in plain language, "our model beats / loses to FantasyPros consensus by X% at position Y on player-weeks Z, with the following diagnostic pattern." If you lose everywhere, say so. If you lose only on rookies and high-injury-risk veterans, that itself is a strong analysis.

---

### 2. The signals that actually move weekly projections

**Why it matters.** Most of the gap between your current weekly model and public projectors lives in *features you don't have*, not in model class. Public projectors have known these for years. Adding them is data engineering, not ML — but it's what closes the skill gap.

**What to build.**
- [ ] Install `nfl_data_py` (lightweight wrapper around nflverse). Add to `requirements.txt`.
- [~] `scripts/fetch_nflverse_data.py` — one-shot fetcher for snap counts, injury reports, depth charts. Saves to `data/raw/`. **Run this once between sessions.**
- [x] **Snap counts.** Now live at 82.6% coverage via the `pfr_id → gsis_id` roster hop. Adding rolling snap-share features (`offense_snap_pct_last1`, `offense_snap_pct_last4_avg`) moved the external benchmark from +1.1% to +1.7% overall and flipped TE from −0.4% to +0.7%. Snap share is confirmed as the single most predictive non-stat weekly feature.
- [~] **Injury reports.** Wired and joined at 17.1% coverage (only injured players are reported by definition). The current attach uses *worst* practice status during the week. **Open improvement**: use Friday-specific practice status only, since Friday is the most predictive day. Rebuilding `_attach_injury_status` to filter by `date_modified` and select the Friday row is the next refinement.
- [~] **Depth chart position.** Attach scaffolding is in place but coverage is effectively 0% because nflverse dropped the numeric `list_rank` field around 2024; the `depth_position` column carries only a string position label ("WR", "RB"). **Open improvement**: derive rank within each `(season, week, team, depth_position)` group via row order, and audit per-season schema variation. Likely material lift for RB1 vs RB2 differentiation.
- [x] **Vegas interaction features.** Position × implied-total and position × spread interactions live; tree model already captured most of this through one-hot position, so the additional lift was small.
- [x] Re-train the weekly model with the available new feature groups. Skill-vs-recent-4-avg went from +7.1% to +7.6%; skill vs DK closing-line went from +1.1% to +1.7%.

**Effort:** 2–3 weeks if you parallelize, 4–5 weeks if you do them one at a time.

**Definition of done:** Weekly model RMSE improves and the `report/weekly_fantasy_projection_summary.md` table includes a "with new signals" row. If a signal doesn't help, drop it and document the negative result.

---

### 3. Replace APY with real cap-hit accounting

**Why it matters.** Front-office reviewers will spot APY-as-cap-cost in 30 seconds and discount the whole salary track. Real cap analysis is year-by-year cap hits, dead money, void years, and replacement-level surplus.

**What to build.**
- [ ] Acquire year-level cap data. Two paths:
  1. **OverTheCap premium** (paid). Has true year-by-year cap hits, cash spent, dead money, void year accounting.
  2. **Manual reconstruction from contract terms** (free but tedious). Each contract has a structure (signing bonus, base salary by year, roster bonuses, option bonuses). The cap hit each year is base + prorated signing bonus + roster bonus. Can be approximated for analysis purposes from the contract value, years, and guaranteed money columns already in `historical_contracts.csv`.
- [ ] `src/cap_accounting.py` — implements the reconstruction logic. Takes `historical_contracts.csv` → produces `data/processed/cap_hit_by_year.csv` with `(player, season, cap_hit, dead_money, cash_spent)`.
- [ ] Rewrite `src/salary_efficiency.py` to use `cap_hit` instead of `inflated_apy`.
- [~] **Replacement-level surplus.** Adds the framing front offices actually use: for each player-season, compute the surplus of `predicted value × games` minus the cap hit of the cheapest *replacement-level player* at that position. The replacement-level player is the median PPR of player-seasons with cap hit at or below the league veteran minimum. This becomes the "true value" output and would replace the current efficiency residual as the headline. *Starting this session with APY as cap hit proxy; needs cap-hit data to be honest.*
- [ ] Add a "**team cap-allocation efficiency**" table: for each team-season, sum the replacement-level-adjusted surplus across all players. Rank teams by how efficiently they spend cap dollars. This is the kind of analysis NFL analytics departments actually produce.

**Effort:** 1 week with OTC data, 2–3 weeks with manual reconstruction.

**Definition of done:** `report/salary_efficiency_findings.md` is rewritten with cap hit as the cost variable, includes the replacement-level surplus table, and the README's "Salary-efficiency findings" link points to the new version. The old APY-based analysis can be retained as an appendix.

---

## Tier 2 — Pick one or two to differentiate

Tier 1 makes you credible. Tier 2 makes you stand out. Don't start any of these until Tier 1 is complete.

### 4. Bayesian hierarchical model for rookies (and small-sample veterans)

**Why it matters.** Your current system literally cannot project a rookie's Week 1 — they have no rolling features. Public projectors handle this with cold-start models that translate college production to NFL. Building a proper hierarchical Bayesian version of this is the methodology piece reviewers will notice. It also gives you posterior intervals that are theoretically cleaner than split-conformal.

**What to build.**
- [x] Architecture, modeling frame, and validation harness in `src/rookie_bayes.py`. Hierarchical Normal with partial pooling across positions, non-centered parameterization for clean sampling.
- [x] Bayes-specific dependency manifest at `requirements-bayes.txt`. Use `python3.12 -m venv .venv-bayes` (PyMC 6 requires Python 3.10+).
- [x] **PyMC sampling pass ran live.** Rolling-origin validation over 2020-2025 rookie classes: RMSE 3.2-4.1 PPR/game, 80% posterior coverage 76-88%. Results in `outputs/tables/rookie_bayes_validation_metrics.csv` and `report/rookie_bayes_projection.md`.
- [ ] Acquire college production data. `cfbd-py` (CollegeFootballData API) is free with rate limits. Pull receiving yards, target share, breakout age, conference quality. Acquisition stub at `scripts/fetch_college_production.py` documents the path.
- [ ] Model spec (PyMC or NumPyro):
  ```
  PPR_per_game[p] ~ Normal(mu[p], sigma[p])
  mu[p] = alpha[position[p]] + beta_age * (age[p] - 27) +
          beta_draft * draft_capital[p] + beta_college * college_score[p] +
          player_effect[p]
  player_effect[p] ~ Normal(0, tau[position[p]])  # partial pooling
  ```
- [ ] Rolling-origin validation: predict 2024 / 2025 rookies' season-long PPR/game using the prior alone (no in-season updates), compare to FantasyPros rookie projections.
- [ ] Output: `report/rookie_projection.md` and `outputs/tables/2026_rookie_projections.csv`.

**Effort:** 4–6 weeks if you're new to PyMC, 2–3 weeks if you're not.

**Definition of done:** A committed rookie projection model with rolling-validation comparison to public rookie projections, posterior intervals visible in the table, and the methodology written up.

---

### 5. Two-stage opportunity × efficiency that actually wins

**Why it matters.** You've lost this fight twice now because you were multiplying noisy components. The next attempt has to be structurally different — not just "two HGBs in sequence."

**What was built and the result.**
- [x] Stage 1: target-share predictor with **team-week renormalization** (the lighter version of the Dirichlet constraint — same structural property, no PyMC dependency for the headline run). HGB on rolling features.
- [x] Stage 2: team-week expected pass attempts (HGB on team-environment features).
- [x] Stage 3: per-target PPR efficiency (HGB on rolling efficiency features).
- [x] Heavy-shrinkage variant of stage 3 (replace with position-season mean) to test the documented prescription.
- [x] Honest head-to-head on identical WR/TE player-weeks vs the pooled HGB across all six rolling-validation seasons. **Result: pooled HGB wins in every fold. Two-stage loses by 9.8%; shrunk-efficiency variant loses by 7.6%.**
- [x] **Per-stage quality diagnostic.** Stage 1 (target share renormalized) beats predict-the-mean by +34.3% — the structural constraint genuinely works. Stages 2 and 3 are essentially noise (≈0% skill over mean). The shrunk variant outperforming the full learned variant *confirms* the diagnosis: unshrunk stage 3 was actively adding error.

**Portfolio-level takeaway.** This is the fourth decomposition attempt in the project. The cumulative evidence is now a *finding*: for weekly fantasy point projection, tree-based pooled models on engineered rolling features extract the team-attempts and per-target-efficiency signals more efficiently than any explicit multiplicative decomposition we have tried. A Dirichlet stage-1 upgrade would not help here — stage 1 is not the problem; stages 2-3 are noise. The actionable next move is not another decomposition variant.

**Open path forward (not Tier 2 #5 anymore).** Either (a) a different pooled model class — gradient-boosted *quantile* regression for proper per-prediction interval shapes — or (b) better features, specifically depth-chart rank and projected snap share, currently blocked on nflverse-supplementary schema cleanup.

**Effort:** 3–5 weeks.

**Definition of done:** A `report/weekly_two_stage_v2.md` that either reports a *real* head-to-head win and explains why the structural constraint made the difference, or reports a third negative result with forensic detail on what specifically about the data-generating process makes decomposition the wrong frame here.

---

### 6. A causal analysis with a sharp question (IN PROGRESS, session 1 complete)

**Session 2 status (`report/causal/qb_injury_session2.md`)**: complete with an honest portfolio-grade finding.

| Estimator | Reference | ATT (PPG) | p-value |
| --- | --- | ---: | ---: |
| Event-study pooled post-period | offset -1 | **+0.60** | 0.001 |
| Simple 2×2 DiD | full pre-period avg | **+0.03** | 0.88 |

**The formal "QB ruled Out" designation does not cause a measurable drop in WR PPR.** Both estimators agree on a null or slightly positive effect. The mechanism revealed by the event-study pre-period coefficients (also significantly positive — meaning treated did better at offsets -4 / -3 than at offset -1): WR production declines weeks before the formal QB injury designation, and bottoms out at offset -1. The QB Out designation is a *lagging indicator* of QB health, not the moment causal damage begins.

The level-matching mitigation failed on its own — restricting controls by baseline PPR introduced regression-to-the-mean bias that widened rather than narrowed the pretrend. Notably, the headline finding (null/positive ATT) survives this mitigation failure because both the matched and unmatched estimators agree.

**Open path forward (session 3)**: re-run treatment identification using *first week of injury report appearance* (Questionable, Limited Practice — anything) instead of *first week ruled Out*. Shifts the treatment moment earlier to capture the actual causal decline. The infrastructure (treatment identification module, control matching, event-study estimator) is already in place — session 3 is a single edit to the classifier and a re-run of sessions 1-2 logic on the new event set.

**Session 1 status (`report/causal/qb_injury_session1.md`)**: complete.
Treatment identification module captures 213 QB-injury events across
2016-2025, validated against hand-checked cases (Burrow 2023, Lawrence
2024, Wentz 2017). Control panel constructed via same-calendar-week
matching with stable-QB filter. **Parallel trends fail** (significant
pretrend at p ≈ 0.034) — exactly the diagnostic finding session 1 is
designed to surface. The honest read: treated WRs were on a declining
trajectory before the formal QB injury (likely endogenous-timing
confounding), so a naive DiD would overstate the effect. Session 2 starts
with two mitigations (PSM on pre-period level + TWFE with differential
trends) and re-checks parallel trends before estimating.



**Why it matters.** NFL analytics departments do causal work — DiD on coordinator changes, synthetic control on rule changes, instrumental variables for player movement effects. A clean causal piece is the kind of analysis that gets shared internally and externally.

**What to build (pick one).**
- [ ] **QB injury impact on WR1 PPR.** DiD: WR-weeks where the starting QB is replaced (treated) vs matched WR-weeks where the QB started (control). Parallel-trends check, placebo on randomly-chosen weeks. Effect estimate with confidence interval. Headline question: "How much PPR does a WR1 lose when their QB goes down?"
- [ ] **Coordinator change effect.** Same template, treatment = OC change between seasons. Tests whether scheme matters for fantasy production.
- [ ] **Rule changes** (e.g., kickoff rules, hip-drop tackle). Synthetic control or interrupted time series.

**Effort:** 2–4 weeks per question depending on how clean the design is.

**Definition of done:** `report/causal/{question}.md` with a clearly stated question, identification strategy, parallel-trends evidence, placebo test, point estimate with confidence interval, and a clear discussion of what the result *does* and *does not* say.

---

## Tier 3 — Role-specific specialization

Don't start any of these until at least one Tier 2 item is complete. They specialize you toward one direction — pick based on which role you're actually targeting.

### 7. (ESPN / DFS) Lineup optimizer with correlation handling

Mixed integer program with salary-cap, position, and team-stacking constraints. Explicit QB-receiver correlation in the objective. Ownership-leverage adjustments. The realistic version uses `pulp` or `mip` and produces top-N lineups for a given slate. **Effort: 2-3 weeks.**

### 8. (NFL team) Modern draft value chart

Update Massey-Thaler 2005 / Stuart 2008 with current data. Position-specific value curves. Analysis of which teams systematically over- or under-pay in trades. **Effort: 2-3 weeks.**

### 9. (ML-heavy roles) NLP on injury reports

Fine-tune a small LM to convert verbal injury text into structured availability probabilities. Hugging Face stack. Demonstrates modern ML fluency. **Effort: 3-4 weeks.**

---

## Sequencing recommendation (the order I'd actually do it)

1. **Weeks 1–2**: Tier 1 #2 — fetch the new signals (snap counts, injuries, depth charts, Vegas interactions). Pure execution. Re-train weekly model.
2. **Week 3**: Tier 1 #1 — external benchmark. Do this *after* #2 so the comparison is fair.
3. **Week 4**: Tier 1 #3 — cap hit + replacement-level. Mechanical but high-impact for front-office credibility.
4. **Decision point.** Tier 1 done. Project is now credible.
5. **Weeks 5–9**: Tier 2 #4 (Bayesian hierarchical rookies). Highest-ceiling methodology piece. Crossover value for both NFL and ESPN roles.
6. **Weeks 10–13**: One Tier 3 item, chosen based on role direction.

Total to a strong portfolio: ~10–13 focused weeks.

---

## Standing rules across all items

- **No new sklearn HGBs on new framings.** Every new HGB at this point is diminishing returns. The next thing built should either use a model class you haven't used or be an analysis without a model.
- **Honest evaluation is the differentiator.** Every new model gets a rolling-origin backtest, a meaningful baseline, and an external benchmark when possible. If a new piece doesn't beat what existed before, that's a negative result *with forensic detail*, not silence.
- **Documentation lives with the code.** Every new module gets a `report/*.md` and a README pointer. The current README has become the project's actual UI — keep it that way.
- **Don't add complexity without a reason it earns its keep.** The Tier 2 items are intentionally more complex than HGB. They're justified because each solves a problem the current stack literally cannot. Adding complexity for its own sake is the same anti-pattern as adding more HGBs.
