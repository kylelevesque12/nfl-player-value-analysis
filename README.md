# NFL Player Value Analysis

Ten years of nflverse data, three questions: how much is a player actually worth to a front office, can a weekly fantasy model consistently beat the naive baselines every forecast is measured against, and what do the negative results teach us about model architecture.

## Four results worth knowing

The weekly fantasy projector cuts RMSE 7-9% below the naive forecasting baselines — a player's recent-form and season-to-date averages — the standard bar any forecast must clear ([Hyndman & Athanasopoulos](https://otexts.com/fpp3/accuracy.html)) — and that edge holds in every season from 2020 through 2025, including the most recent. That margin is meaningful because weekly fantasy scoring is intrinsically low-predictability (single-digit to low-twenties R² by position, per [Fantasy Football Analytics](https://fantasyfootballanalytics.net/2024/12/which-fantasy-football-projections-are-most-accurate.html)). On the 2020-2021 window where a free market-implied benchmark exists, the model is also competitive-to-slightly-ahead of a DraftKings-implied projection (6.386 vs 6.493 RMSE on 11,191 matched player-weeks) — though that comparison can't be extended to recent years without paid projection data, so it is not the headline claim. Snap-share features did most of the work; the TE position flipped from a negative to a positive skill score the moment they were added. ([External benchmark](report/external_benchmark.md) · [feasibility note](report/fantasy/external_projection_benchmark_feasibility.md))

Brock Purdy's 2023 season produced about $37M of surplus over a replacement-level QB on a rookie deal — the largest single-season cap surplus in 2016-2025. Three of the top ten surplus seasons are rookie-deal QBs (Purdy 2023, Purdy 2024, Jayden Daniels 2024). The RB market shows a negative implicit price for value at the position level, consistent with the long-documented RB market inefficiency. ([Replacement-level findings](report/salary_efficiency_findings.md))

The "QB1 goes down, WR1 craters" story does not survive a careful DiD. Two-session causal analysis with parallel-trends checks finds a null effect when treatment is defined as the formal Out designation. The reason matters: the QB plays through a developing injury for weeks before being ruled out, so by the time the Out flag triggers, receiver production has already declined. The Out flag lags the actual onset. The session-3 follow-up is to re-define treatment as the first week of any injury-report appearance. ([Causal session 1](report/causal/qb_injury_session1.md) · [session 2](report/causal/qb_injury_session2.md))

Bayesian hierarchical rookie projections (PyMC, non-centered, partial pooling across positions) hit near-nominal posterior interval coverage in every rookie class. The first version had a Jordan Love problem: rookies drafted behind a veteran starter had NaN targets and disappeared from training. The current hurdle model handles this. Stage 1 predicts whether a rookie plays meaningfully; stage 2 predicts production conditional on playing; the product is the projection. Late-round QBs now correctly project as low-volume rather than getting silently excluded. ([Rookie Bayes](report/rookie_bayes_projection.md) · [Hurdle model](report/rookie_hurdle_projection.md))

## Where to start

The project covers three audiences. Pick the section that matches yours:

| Audience | Section | Top deliverable |
| --- | --- | --- |
| NFL team analytics / cap analysts | [Front office](#front-office-perspective) | Replacement-level surplus framework with Brock Purdy at #1 |
| ESPN / DraftKings / FantasyPros | [Fantasy / DFS](#fantasy--dfs-perspective) | Weekly projector: 7-9% RMSE edge over naive baselines every season 2020-2025; competitive with a market-implied DK benchmark on 2020-2021 |
| Research labs / methodology reviewers | [Methodology](#methodology--research-perspective) | Causal DiD with a null finding, rookie hurdle Bayes, four decomposition experiments |

If you want the whole story start to finish, read the [final project report](report/final_project_report.md). If you want to play with the live numbers, run the [Streamlit dashboard](#interactive-dashboard).

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

A 26-check methodology audit covers leakage safety, standardization correctness, interval calibration, and missing-target detection ([methodology checks](report/methodology_checks.md)). The biggest open limitation is that the cost variable is `inflated_apy`, not year-by-year cap hit. The salary track is contract efficiency, not cap accounting. Switching to OverTheCap year-by-year cap hits is the next data investment.

## Fantasy / DFS perspective

> **Can a weekly PPR model consistently beat the naive baselines every forecast is measured against — and how does it stack up against the market where that comparison is possible?**

### The headline result

**A 7-9% RMSE reduction versus the naive forecasting baselines — a player's recent-form and season-to-date averages — sustained in every season 2020-2025.** Beating the naive forecast is the standard bar in forecast evaluation ([Hyndman & Athanasopoulos](https://otexts.com/fpp3/accuracy.html)); reporting it per season, across six independent yearly holdouts, is the recency-proof evidence that the edge is real and not a one-year artifact:

| Season | n | Skill vs recent-4-avg |
| --- | ---: | ---: |
| 2020 | 5,530 | +7.0% |
| 2021 | 5,856 | +7.9% |
| 2022 | 5,818 | +8.5% |
| 2023 | 5,811 | +6.8% |
| 2024 | 5,848 | +7.4% |
| 2025 | 6,043 | +8.2% |

A single-digit edge is meaningful here because weekly fantasy scoring is intrinsically low-predictability — single-digit to low-twenties R² by position ([Fantasy Football Analytics](https://fantasyfootballanalytics.net/2024/12/which-fantasy-football-projections-are-most-accurate.html)).

### Scoped market check (2020-2021 only)

Where a *free* market-implied benchmark exists — DraftKings closing-line salaries, available via RotoGuru's free archive only through 2021 — the model is competitive-to-slightly-ahead of the DK-implied projection on 11,191 matched player-weeks:

| Position | Model RMSE | DK-implied RMSE | Skill vs market |
| --- | ---: | ---: | ---: |
| QB | 7.70 | 7.84 | **+1.9%** |
| RB | 6.59 | 6.71 | **+1.8%** |
| WR | 6.44 | 6.55 | **+1.8%** |
| TE | 5.10 | 5.14 | **+0.7%** |
| **Overall** | **6.39** | **6.49** | **+1.6%** |

This is a scoped, secondary result, not a claim of beating live DraftKings, FantasyPros, or ESPN in recent years — that would require paid historical projection data the project doesn't have. The DK regression is also fit on in-season actuals, making it a deliberately tough bar. See the [feasibility note](report/fantasy/external_projection_benchmark_feasibility.md).

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

> **What are the right modeling architectures for NFL player projection, and what do the negative results teach us?**

### Causal DiD: QB injury → WR PPR

Two-session causal analysis testing the conventional-wisdom claim that QB injury causes WR PPR to crater.

**Session 1**: identifies 213 QB-injury treatment events 2016-2025 from `injuries × player_stats × schedules`. Validates against hand-checked cases (Burrow 2023, Lawrence 2024, Wentz 2017). Builds matched-control panels using same-calendar-week receivers on stable-QB teams. Runs the parallel-trends check — **and finds a violation** (p ≈ 0.034 at offset -3).

**Session 2**: implements pre-registered mitigations. Level matching fails (regression-to-the-mean widens the pretrend, p drops to 0.005). The event-study + 2×2 DiD on the unmatched panel — both estimators agree:

| Estimator | Reference | ATT | p-value |
| --- | --- | ---: | ---: |
| Event-study pooled post-period | offset -1 | **+0.60 PPG** | 0.001 |
| Simple 2×2 DiD | full pre-period avg | **+0.03 PPG** | 0.88 |

Both null or *positive*. The pre-period coefficients (also significantly positive) revealed the mechanism: treated WRs hit their absolute low at offset -1, the week immediately before the formal QB switch. The Out designation is a lagging indicator, not a leading feature.

The pretrend failure was found by the session-1 diagnostic, not papered over. The session-2 mitigations were pre-registered before running. The null result survived both estimators and both panel specifications. The mechanism (the QB plays through a developing injury for weeks before being formally ruled out) was named, and the follow-up — re-defining treatment as the first week of any injury-report appearance — is on the roadmap.

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
  config.py / load_data.py / models.py    # shared utilities
  clean_data.py / features.py             # cleaning + feature engineering
  methodology_checks.py                   # leakage + interval audit
  model_benchmark.py                      # skill scores + conformal intervals
  model_interpretation.py                 # permutation importance + position FE

  # Front office
  prediction_report.py                    # 2026 Excel report
  value_decomposition.py                  # efficiency × opportunity decomp
  two_stage_value.py                      # season-level decomposition result
  salary_efficiency.py                    # contract efficiency
  salary_findings.py                      # leaderboards + replacement-level
  replacement_level.py                    # replacement-level surplus framework

  # Fantasy
  fantasy_projection.py                   # season-long projections
  weekly_fantasy_projection.py            # weekly model + nflverse signals
  external_benchmark.py                   # DK closing-line head-to-head
  rookie_bayes.py                         # hierarchical Bayes + hurdle stage
  two_stage_weekly.py                     # WR/TE decomp experiment
  causal/                                 # QB-injury DiD

  pipeline.py                             # orchestration

archive/                                  # earlier experiments retired with a brief writeup

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

## Limitations and gaps

- **Salary track uses APY, not year-by-year cap hit.** Real cap analysts will treat this as contract efficiency. True cap-hit data (OTC premium or reconstruction from contract terms) is the next acquisition.
- **DK benchmark coverage stops in 2021.** RotoGuru's free archive doesn't go later. Extending requires a paid source.
- **No depth-chart rank.** nflverse dropped the numeric `list_rank` field around 2024. Deriving rank from row-order is the next feature-engineering target.
- **Injury attach at 17% coverage.** Only injured players appear on the report. The QB-injury causal study works around this by joining at the team-week level; the fantasy model treats missing injury status as "healthy" and tolerates the resulting noise.
- **Streamlit dashboard is currently a draft layer.** It surfaces the older modules but hasn't kept pace with replacement-level surplus, Bayesian rookie projections, the causal QB-injury findings, or the two-stage weekly experiment. A full rebuild is the next infrastructure investment — see [`PORTFOLIO_ROADMAP.md`](PORTFOLIO_ROADMAP.md).
- **No causal session 3.** The session 1+2 finding (QB Out is a lagging indicator) implies the *real* causal moment is the first week of injury-report appearance. Re-running the DiD on that treatment definition is the open follow-up.

## Next phase

Three concrete moves, in priority order:

1. **Full Streamlit dashboard rebuild** to surface all three perspectives — the v2 plan replaces the current draft layer.
2. **Causal session 3** — re-identify treatment as first-injury-report-appearance (not formal Out). Reuses all session-1/2 infrastructure.
3. **Real cap hit replacing APY** in the salary track — unblocks audit-grade front-office findings.

The full multi-session roadmap (including Tier 3 specializations) lives in [`PORTFOLIO_ROADMAP.md`](PORTFOLIO_ROADMAP.md).
