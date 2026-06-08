# NFL Player Value Analysis

A comprehensive NFL analytics suite spanning three perspectives on the same underlying data — front-office cap allocation, weekly fantasy projection, and causal-methodology research — built on 10 years of nflverse play-by-play, schedules, rosters, contracts, and supplementary feeds.

The project tests one core question across three frames: **how do we measure, predict, and compare NFL player value in ways that are transparent, position-aware, and useful for real decisions?**

## Headline findings

- **Fantasy projection beats DraftKings closing-line implied projections by +1.6% RMSE** on 11,191 matched player-weeks (2020-2021). Public DFS analytics shops claim 1-3% edges as their core selling point. ([External benchmark](report/external_benchmark.md))
- **Brock Purdy's 2023 season delivered ~$37M of surplus over replacement-level QB cost** on a rookie contract — the largest single-season surplus in the 2016-2025 sample. ([Replacement-level findings](report/salary_efficiency_findings.md))
- **The formal "QB ruled Out" designation does not cause a measurable drop in WR PPR.** Two-session causal DiD with parallel-trends checks finds a null/positive ATT. Mechanism: the QB plays through a developing injury for weeks before being formally ruled out, so by the time the Out designation hits, the receivers' production has already declined. The Out flag is a lagging indicator, not a leading feature. ([Causal session 1](report/causal/qb_injury_session1.md) / [session 2](report/causal/qb_injury_session2.md))
- **Bayesian hierarchical rookie projections hit nominal posterior-interval coverage** (PyMC, non-centered, calibrated 80% intervals across six rolling-validation rookie classes). Solves the cold-start problem the headline HGB projector cannot. ([Rookie Bayes](report/rookie_bayes_projection.md))

## Three perspectives

The project is best read by perspective rather than by module. Each one stands on its own, with its own headline result, its own methodology decisions, and its own honest limitations.

| Perspective | Primary audience | Headline | Where to look |
| --- | --- | --- | --- |
| Front office | NFL team analytics, cap analysts | $37M Brock Purdy surplus over replacement | [Final project report](report/final_project_report.md), [salary findings](report/salary_efficiency_findings.md) |
| Fantasy / DFS | ESPN, DraftKings, FantasyPros | +1.6% RMSE skill score vs DK closing line | [Weekly fantasy projection](report/weekly_fantasy_projection_summary.md), [external benchmark](report/external_benchmark.md) |
| Methodology / research | Research labs, methodology-conscious reviewers | Causal null + structural negative result with diagnostic | [Causal sessions](report/causal/), [two-stage weekly](report/two_stage_weekly.md), [rookie Bayes](report/rookie_bayes_projection.md) |

## How to review this project

Recommended reading order:

1. **Start here**: [Final project report](report/final_project_report.md) for the whole story across all three perspectives.
2. **For NFL team / cap analytics reviewers**: skip to the [front-office section below](#front-office-perspective). Headline is replacement-level surplus.
3. **For ESPN / fantasy / DFS reviewers**: skip to the [fantasy section below](#fantasy--dfs-perspective). Headline is the DK benchmark beat.
4. **For methodology-focused reviewers**: skip to the [methodology section below](#methodology--research-perspective). Headline is the causal QB-injury null and the four-attempt decomposition finding.
5. **For deeper detail on any one piece**: every section links to its own dedicated report under `report/`.

## Front office perspective

> **How do we identify under-priced player-seasons relative to replacement-level cap cost?**

### Replacement-level surplus framework

For each `(season, position)` we estimate two baselines from the data: `replacement_salary_millions` (median bottom-quartile veteran-starter cost — the price of "next man up") and `replacement_value_score` (the value those replacement-level players actually deliver). For each player-season we compute the cap premium paid above replacement, the value delivered above replacement, and the dollar surplus — converting above-replacement value to dollars via the within-(season, position) slope of salary on value.

**Top 5 replacement-level surplus seasons, 2016-2025**:

| Season | Player | Pos | Team | Cap over replacement ($M) | Surplus ($M) |
| --- | --- | --- | --- | ---: | ---: |
| 2023 | Brock Purdy | QB | SF | 0.0 | **+37.4** |
| 2024 | Brock Purdy | QB | SF | -0.1 | **+29.3** |
| 2024 | Jayden Daniels | QB | WAS | 9.7 | **+25.6** |
| 2025 | Puka Nacua | WR | LA | 0.3 | **+17.4** |
| 2023 | Jake Browning | QB | CIN | -0.3 | **+15.1** |

The framework also surfaces **position-level market irrationality** — running back occasionally shows a negative implicit value-per-dollar slope at the position-season level, consistent with the well-documented RB-market inefficiency.

### Value scoring

EPA-based player value scores, z-scored within `(season, position)`, with multi-year history features and an availability sub-model. The headline deliverable is [the 2026 Excel report](outputs/tables/2026_player_value_predictions.xlsx) with 505 player projections, calibrated 80% intervals, plain-English prediction drivers, and team / position summaries.

### Salary efficiency

4,569 of 4,753 player-seasons matched to historical contracts (96.1% match rate). Top 25 surplus players, high-cost underperformers, rookie-contract proxy surplus, veteran values, and team-season leaderboards.

### Methodology audit and limitations

A 26-check methodology audit covers leakage safety, standardization correctness, interval calibration, and missing-target detection ([methodology checks](report/methodology_checks.md)). The honest limitation: the cost variable is `inflated_apy`, not year-by-year cap hit. The salary track is **contract efficiency, not cap accounting**. Replacing APY with true cap hit (OverTheCap premium or manual reconstruction) is the next data-acquisition step.

## Fantasy / DFS perspective

> **Can we project weekly PPR fantasy points more accurately than the DraftKings betting market?**

### The headline result

**+1.6% RMSE skill score over DraftKings closing-line implied projections**, 11,191 player-weeks across 2020-2021:

| Position | Model RMSE | DK-implied RMSE | Skill vs market |
| --- | ---: | ---: | ---: |
| QB | 7.70 | 7.84 | **+1.9%** |
| RB | 6.59 | 6.71 | **+1.8%** |
| WR | 6.44 | 6.55 | **+1.8%** |
| TE | 5.10 | 5.14 | **+0.7%** |
| **Overall** | **6.39** | **6.49** | **+1.6%** |

**Temporal stability** (per-season skill vs the recent-4-avg internal baseline, full 2020-2025 window — covering years where DK data isn't free):

| Season | n | Skill vs recent-4-avg |
| --- | ---: | ---: |
| 2020 | 5,530 | +7.0% |
| 2021 | 5,856 | +7.9% |
| 2022 | 5,818 | +8.5% |
| 2023 | 5,811 | +6.8% |
| 2024 | 5,848 | +7.4% |
| 2025 | 6,043 | +8.2% |

Consistent single-digit edge across every season — the DK beat isn't a 2020-2021 fluke.

### How the model works

A pooled HistGradientBoosting regression on engineered pregame features: rolling production/usage (last-1, last-4, last-8, season-to-date PPR; targets, receptions, carries, passing attempts), Vegas market signals with position × market interactions, availability proxy from team-schedule × player-appearance crosscheck, opponent PPR-allowed-to-position, **snap share from nflverse**, and schedule context. Split-conformal 50% / 80% prediction intervals calibrated on held-out folds (empirical coverage 50.0% / 79.4%).

### Supporting investigations

The fantasy model rests on three methodology decisions documented as their own reports:

- **Bayesian hierarchical rookie cold-start** — Solves the rookie-Week-1 problem the HGB cannot. Hierarchical Normal with partial pooling across positions, non-centered parameterization, calibrated posteriors. ([Rookie Bayes report](report/rookie_bayes_projection.md))
- **Two-stage decomposition experiment** — Tested whether structurally-constrained `team_attempts × target_share × PPR_per_target` beats the pooled HGB. It doesn't. The per-stage diagnostic explains why: stage 1 (target share renormalized) is +34% over mean, stages 2 and 3 are noise. ([Two-stage weekly](report/two_stage_weekly.md))
- **Causal investigation of QB injury as a feature** — Built a DiD to test whether QB-injury status is a usable leading feature. It isn't — the formal Out designation is a lagging indicator. ([Causal sessions 1 + 2](report/causal/))

### Honest limitations

- RotoGuru's free DK archive ends in 2021, so the head-to-head benchmark covers 2020-2021 only. Extending to 2022-2025 requires a paid source (documented in [`PORTFOLIO_ROADMAP.md`](PORTFOLIO_ROADMAP.md)).
- Depth-chart rank is broken at the nflverse data level around 2024 (numeric rank field dropped).
- Injury reports attach at 17.1% coverage (only injured players are reported by definition).

## Methodology / research perspective

> **What are the right modeling architectures for NFL player projection — and what do honest negative results teach us?**

### Causal DiD: QB injury → WR PPR

Two-session causal analysis testing the conventional-wisdom claim that QB injury causes WR PPR to crater.

**Session 1**: identifies 213 QB-injury treatment events 2016-2025 from `injuries × player_stats × schedules`. Validates against hand-checked cases (Burrow 2023, Lawrence 2024, Wentz 2017). Builds matched-control panels using same-calendar-week receivers on stable-QB teams. Runs the parallel-trends check — **and finds a violation** (p ≈ 0.034 at offset -3).

**Session 2**: implements pre-registered mitigations. Level matching fails (regression-to-the-mean widens the pretrend, p drops to 0.005). The event-study + 2×2 DiD on the unmatched panel — both estimators agree:

| Estimator | Reference | ATT | p-value |
| --- | --- | ---: | ---: |
| Event-study pooled post-period | offset -1 | **+0.60 PPG** | 0.001 |
| Simple 2×2 DiD | full pre-period avg | **+0.03 PPG** | 0.88 |

Both null or *positive*. The pre-period coefficients (also significantly positive) revealed the mechanism: treated WRs hit their absolute low at offset -1, the week immediately before the formal QB switch. The Out designation is a lagging indicator, not a leading feature.

This is the kind of finding that distinguishes a careful causal analysis from a regression-with-a-causal-interpretation. Hypothesis tested, parallel trends checked, mitigations attempted, null finding survived, mechanism named.

### Four-decomposition finding

Across four independent attempts in this project, **explicit multiplicative decompositions of player value have consistently lost to pooled tree-based models on engineered rolling features**:

1. Season-level two-stage value (opportunity × efficiency × games) — lost to single model on RMSE
2. Season-level position-specific HGB — lost to pooled HGB at every position
3. Weekly position-specific HGB — lost to pooled HGB at every position
4. Weekly WR/TE two-stage with structural constraint (target shares renormalized within team-week) — lost to pooled HGB by 7-8% even with shrinkage on the efficiency stage

The cumulative evidence is a *finding*, not four separate anecdotes: for NFL fantasy and value projection, pooled tree-based models with engineered rolling features extract the relevant signals more efficiently than any explicit decomposition we've tried. Reports for each attempt include per-stage diagnostic detail explaining the mechanism. ([Season-level](report/two_stage_value.md) / [weekly WR/TE](report/two_stage_weekly.md))

### Bayesian hierarchical methodology

A hierarchical Normal regression on rookie-season PPR/game with partial pooling across positions on both intercept and slopes. Non-centered parameterization brought divergences from 22-32 down to 1. Posterior coverage at the 50% and 80% levels is close to nominal in every rookie class:

| Rookie class | n | RMSE | 50% coverage | 80% coverage |
| --- | ---: | ---: | ---: | ---: |
| 2020 | 105 | 4.14 | 44.8% | 81.0% |
| 2021 | 91 | 3.74 | 45.1% | 85.7% |
| 2022 | 101 | 3.23 | 52.5% | 88.1% |
| 2023 | 94 | 3.76 | 47.9% | 87.2% |
| 2024 | 93 | 3.80 | 47.3% | 76.3% |
| 2025 | 101 | 3.32 | 54.5% | 87.1% |

PyMC has a numpy/pandas dependency conflict with the main project stack, so the sampling pass runs from a dedicated venv (`.venv-bayes`). See [`requirements-bayes.txt`](requirements-bayes.txt) and the [rookie Bayes report](report/rookie_bayes_projection.md).

## Interactive dashboard

The Streamlit dashboard surfaces all three perspectives:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Pages currently include front-office (value, salary, predictions, validation), fantasy (season-long, weekly with conformal intervals, model comparison), weekly win projections, and methodology / reports. A v2 rebuild is in progress to surface the newer methodology pieces (replacement-level surplus leaderboard, Bayesian rookie posteriors, causal QB-injury event study) — see [`PORTFOLIO_ROADMAP.md`](PORTFOLIO_ROADMAP.md) for the plan.

## Reproducing the pipeline

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

Pipeline steps run in dependency order: clean raw weekly data → rebuild value scores → rebuild value decomposition → rebuild 2026 prediction tables → rebuild salary efficiency → rebuild salary findings (including replacement-level surplus) → rebuild weekly fantasy projections → rebuild external benchmark → rebuild rookie modeling frame → rebuild two-stage weekly experiment → rebuild causal sessions 1+2 → rebuild weekly win projections → rebuild methodology checks → rebuild model interpretation → rebuild benchmark → rebuild season-level two-stage value.

Selective runs:

```bash
python scripts/run_pipeline.py --steps weekly_fantasy,external_benchmark
python scripts/run_pipeline.py --steps findings,causal_session2
```

Supplementary data fetches (run between pipeline calls when data goes stale):

```bash
pip install nfl_data_py
python scripts/fetch_nflverse_data.py --years 2016-2025
python scripts/fetch_rotoguru_salaries.py --years 2014-2021
python scripts/build_external_projections_from_dk.py
```

PyMC rookie sampling (separate venv):

```bash
python3.12 -m venv .venv-bayes
.venv-bayes/bin/python -m pip install -r requirements-bayes.txt
PYTHONPATH=. .venv-bayes/bin/python -c "from src.rookie_bayes import build_rookie_bayes_outputs; build_rookie_bayes_outputs()"
```

## Project layout

```
src/
  config.py / load_data.py / models.py    # shared infrastructure
  clean_data.py / features.py             # data cleaning, feature engineering
  prediction_report.py                    # 2026 Excel report
  value_decomposition.py                  # efficiency × opportunity decomp
  two_stage_value.py                      # season-level negative result
  salary_efficiency.py                    # contract efficiency analysis
  salary_findings.py                      # leaderboards + replacement-level
  replacement_level.py                    # the front-office surplus framework
  weekly_fantasy_projection.py            # weekly model + nflverse signals
  external_benchmark.py                   # DK closing-line head-to-head
  rookie_bayes.py                         # hierarchical Bayes for rookies
  two_stage_weekly.py                     # WR/TE decomp experiment
  causal/                                 # QB-injury DiD investigation
  fantasy_projection.py                   # season-long fantasy
  weekly_win_projection.py                # game-winner projection (draft)
  methodology_checks.py                   # leakage + interval audit
  model_benchmark.py                      # skill scores + conformal intervals
  model_interpretation.py                 # permutation importance + position FE
  context_features.py / feature_impact.py # context-feature audit
  advanced_modeling.py                    # Optuna + SHAP + MLflow diagnostics
  pipeline.py                             # orchestration

scripts/
  run_pipeline.py
  fetch_nflverse_data.py
  fetch_rotoguru_salaries.py
  build_external_projections_from_dk.py
  fetch_college_production.py             # cfbd-py stub
  export_notebooks_to_markdown.py
  prepare_notebooks_for_github.py

tests/
  72 tests covering leakage safety, feature engineering, model benchmark
  math, replacement-level, two-stage value, value decomposition, weekly
  fantasy, two-stage weekly, rookie Bayes, and causal treatment ID.

report/
  final_project_report.md                 # full narrative
  salary_efficiency_findings.md           # front-office headline
  weekly_fantasy_projection_summary.md    # fantasy headline
  external_benchmark.md                   # DK head-to-head
  rookie_bayes_projection.md              # Bayesian methodology
  two_stage_weekly.md                     # decomposition diagnostic
  causal/qb_injury_session1.md            # causal foundation
  causal/qb_injury_session2.md            # causal verdict
  two_stage_value.md / value_decomposition.md / ...
```

## Testing

```bash
pip install pytest
python -m pytest tests/ -q
```

72/72 tests passing. Coverage spans leakage-safety in feature engineering, benchmark math, replacement-level surplus, two-stage value math, value-decomposition arithmetic, weekly-fantasy structural invariants, rookie-Bayesian data prep, and causal treatment identification (including hand-checked Burrow / Lawrence / Wentz cases).

## Limitations and honest gaps

- **Salary track uses APY, not year-by-year cap hit.** Real cap analysts will treat this as contract efficiency. True cap-hit data (OTC premium or reconstruction from contract terms) is the next acquisition.
- **DK benchmark coverage stops in 2021.** RotoGuru's free archive doesn't go later. Extending requires a paid source.
- **No depth-chart rank.** nflverse dropped the numeric `list_rank` field around 2024. Deriving rank from row-order is the next feature-engineering target.
- **Injury attach at 17% coverage.** Only injured players are reported. The QB-injury causal study handles this carefully; the headline fantasy model uses injury indicators with caution.
- **Streamlit dashboard is currently a draft layer.** It surfaces the older modules but hasn't kept pace with replacement-level surplus, Bayesian rookie projections, the causal QB-injury findings, or the two-stage weekly experiment. A full rebuild is the next infrastructure investment — see [`PORTFOLIO_ROADMAP.md`](PORTFOLIO_ROADMAP.md).
- **No causal session 3.** The session 1+2 finding (QB Out is a lagging indicator) implies the *real* causal moment is the first week of injury-report appearance. Re-running the DiD on that treatment definition is the open follow-up.

## Next phase

Three concrete moves, in priority order:

1. **Full Streamlit dashboard rebuild** to surface all three perspectives — the v2 plan replaces the current draft layer.
2. **Causal session 3** — re-identify treatment as first-injury-report-appearance (not formal Out). Reuses all session-1/2 infrastructure.
3. **Real cap hit replacing APY** in the salary track — unblocks audit-grade front-office findings.

The full multi-session roadmap (including Tier 3 specializations) lives in [`PORTFOLIO_ROADMAP.md`](PORTFOLIO_ROADMAP.md).
