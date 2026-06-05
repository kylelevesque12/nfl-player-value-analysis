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
- [~] Scaffolding: `src/external_benchmark.py` that loads a CSV of external projections and runs head-to-head vs the weekly fantasy model on matched (player, season, week) rows. Outputs MAE / RMSE / Spearman / win-rate-by-position. Code in place but needs data.
- [ ] Acquire external projections. Two viable sources, ordered by impressiveness:
  1. **DraftKings closing-line implied projections (best).** DK salary / (DK salary-to-points conversion factor for the position) ≈ market-implied projection. Closing salaries are scraped from RotoGrinders, FantasyData, or DK's own download. Beating the market is the strongest possible claim.
  2. **FantasyPros consensus projections (easier).** Public weekly projections at fantasypros.com/nfl/projections/. Historical archives require scraping; the `fpros` Python package wraps this. Less impressive than beating DK but enough to be credible.
- [ ] Run head-to-head on 2022–2025. Expect to lose initially. **Report exactly where you lose** (which positions, which player tiers, which weeks) — that diagnostic is the most useful single output the project can produce right now.
- [ ] Add `report/external_benchmark.md` at the *top* of the README. This becomes the headline section.

**Effort:** 1–2 weeks once data is available. Data acquisition is the bottleneck.

**Definition of done:** A committed `external_benchmark.md` that says, in plain language, "our model beats / loses to FantasyPros consensus by X% at position Y on player-weeks Z, with the following diagnostic pattern." If you lose everywhere, say so. If you lose only on rookies and high-injury-risk veterans, that itself is a strong analysis.

---

### 2. The signals that actually move weekly projections

**Why it matters.** Most of the gap between your current weekly model and public projectors lives in *features you don't have*, not in model class. Public projectors have known these for years. Adding them is data engineering, not ML — but it's what closes the skill gap.

**What to build.**
- [ ] Install `nfl_data_py` (lightweight wrapper around nflverse). Add to `requirements.txt`.
- [~] `scripts/fetch_nflverse_data.py` — one-shot fetcher for snap counts, injury reports, depth charts. Saves to `data/raw/`. **Run this once between sessions.**
- [ ] **Snap counts.** `nfl_data_py.import_snap_counts(years)` → save as `data/raw/snap_counts_2016_2025.csv`. Add features: `offense_snap_share_last4_avg`, `snap_count_last1`. Opportunity in fantasy is dominated by snap share. This is the single most predictive non-stat weekly feature and you don't have it.
- [ ] **Injury reports.** `nfl_data_py.import_injuries(years)` → save as `data/raw/injuries_2016_2025.csv`. Add features: `practice_status_last_friday` (Full / Limited / DNP / Out / Questionable), one-hot encoded. Players Questionable with Limited Friday practice score meaningfully less than healthy players — this is exactly the public-projector edge.
- [ ] **Depth chart position.** `nfl_data_py.import_depth_charts(years)` → save as `data/raw/depth_charts_2016_2025.csv`. Add `depth_chart_rank` (1, 2, 3, ...). RB1 vs RB2 is the single most important feature for RB projections, and you currently approximate it with last-4 carries.
- [x] **Vegas interaction features.** Already have `implied_team_total`. Add `implied_team_total × position` interactions so QBs in high-total games get the boost they should. *In progress this session.*
- [ ] Re-train the weekly model with all four new feature groups. Compare to the current model in `weekly_fantasy_method_summary.csv`. Expected gain: 1–3 percentage points of skill score.

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
- [ ] Acquire college production data. `cfbd-py` (CollegeFootballData API) is free with rate limits. Pull receiving yards, target share, breakout age, conference quality.
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

**What to build.**
- [ ] Stage 1: `target_share | snap_share, depth_chart_position, opponent_pass_rate` — modeled with a **Dirichlet** likelihood so target shares sum to 1.0 within a team-week. This is the structural constraint that prevents the model from over-allocating targets across a team's receivers.
- [ ] Stage 2: `PPR_per_target | depth_chart, qb_quality, weather, opponent_secondary_quality`.
- [ ] Recombine via the constraint `expected_PPR = expected_targets × expected_PPR_per_target`.
- [ ] Honest comparison: this two-stage vs the single pooled model on the same rows. Report the win/loss with detail on which player-weeks each wins on.
- [ ] Variance decomposition: for each prediction, what share of the interval width comes from target-share uncertainty vs efficiency uncertainty? This is the *actual* opportunity-vs-efficiency interval decomposition — and it's the analysis the project has been gesturing at for two iterations now.

**Effort:** 3–5 weeks.

**Definition of done:** A `report/weekly_two_stage_v2.md` that either reports a *real* head-to-head win and explains why the structural constraint made the difference, or reports a third negative result with forensic detail on what specifically about the data-generating process makes decomposition the wrong frame here.

---

### 6. A causal analysis with a sharp question

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
