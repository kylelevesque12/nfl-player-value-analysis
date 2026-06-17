# Measuring and Predicting NFL Offensive Player Value

## Executive summary

This project measures the value of NFL offensive players, predicts how that value
will carry into the next season, and compares value to contract cost. It covers
quarterbacks, running backs, wide receivers, and tight ends from 2016 through 2025.

The work began with a conventional approach — score each player-season by
standardized EPA, then train a model to predict next season's score — and then
interrogated that approach honestly. Three findings, in order, shaped the project:

1. **A simple baseline is hard to beat.** When the prediction target is
   standardized within each season-position group, its standard deviation is
   about 1.0, so predicting the group mean already yields an RMSE near 1.0.
   Measured against a properly shrunken persistence baseline rather than against
   zero, a tuned Random Forest improves RMSE by only about 4.3% (and conformal
   intervals hit 81.1% coverage against an 80% target). That is the honest size
   of the signal, and it pointed toward understanding *what* in value is even
   predictable.

2. **Value is two different things wearing one number.** Player value factors
   exactly into *efficiency* (production per opportunity) times *opportunity*
   (how much a player is used). These factors behave completely differently over
   time: opportunity is highly persistent year to year, while efficiency — for
   skill positions — is close to noise. This decomposition is the project's
   central analytical insight.

3. **But separating the axes did not improve prediction.** A natural hypothesis
   followed from finding 2: model the two factors separately and recombine them.
   It was built, validated honestly head-to-head, and **it lost.** On identical
   rows, a single model predicting value directly scored RMSE 2.318 (R² 0.203)
   while the two-stage recombination scored 2.417 (R² 0.134) — about 4% worse.
   The reason is coherent with the decomposition itself: the efficiency stage
   adds almost nothing over a shrink-to-mean baseline, and multiplying a noisy
   efficiency estimate into the product injects error the single model avoids. The
   decomposition was *diagnostically right about the world* and *wrong as a bet
   that separation improves accuracy.* Reporting that plainly is the point.

So the project ships two complementary things rather than one. **The single
model is the accuracy engine** for point predictions. **The two-stage
decomposition is an interpretability and uncertainty layer**: it produces a
calibrated, *asymmetric* prediction interval (validated at 80.9% coverage) that is
wide along the axis the model genuinely cannot predict, and it labels each player
as role-driven or efficiency-driven — distinguishing "we are unsure about this
player's role" from "we are unsure about his per-play quality." That distinction
is real, validated, and useful even though it did not lower RMSE.

The repository delivers cleaned datasets, reusable `src/` modules, a reproducible
command-line pipeline, validation and interpretation reports, a salary-efficiency
analysis, and player-value projections with calibrated, axis-aware uncertainty.

## The question

How can NFL offensive player value be measured, predicted, and compared to salary
in a way that is transparent, position-aware, and useful for football decisions?

The answer is built in layers: define a defensible value metric, understand what
in that metric is actually predictable, model the predictable parts well, and
attach uncertainty that a decision-maker can act on.

## Data

The project uses nflverse data loaded with `nflreadpy`: weekly player statistics,
rosters, and schedules, from 2016 to 2025. The salary-efficiency section adds
nflverse historical contract data sourced from OverTheCap. The cost metric is a
season-specific cap hit reconstructed from contract terms (prorated signing bonus
+ backloaded base; see `src/cap_hit_reconstruction.py`), replacing the flat
average-annual-value (APY) proxy used earlier in the project.
Raw and processed data are excluded from version control; lightweight output
tables and reports are committed so the project can be reviewed without large
files.

Weekly rows are filtered to regular-season QB/RB/WR/TE and aggregated to
player-season rows. Multi-team stints are collapsed to a single player-season
before scoring, so a traded player is not split into misleading partial samples.
Position-specific production is kept separate throughout: quarterbacks are
measured by passing-plus-rushing production, while running backs, receivers, and
tight ends are measured by rushing-plus-receiving production.

## The value metric, and its hidden flaw

The headline metric, `value_score`, is total value EPA standardized within each
season-position group:

- QB value EPA = passing EPA + rushing EPA
- RB/WR/TE value EPA = rushing EPA + receiving EPA
- `value_score` = z-score of value EPA within (season, position)

So `0.0` is positional average for that season, `+1.0` is one standard deviation
above peers, and players at different positions become comparable only after each
is first judged against their own group. This is a reasonable, position-aware
comparison metric, and it remains the project's reference scale.

But it has a flaw that matters for talent evaluation: **total EPA rewards
opportunity as much as quality.** A high-volume, average-efficiency player can
out-score a low-volume, highly efficient one. For a front office, that conflates
two distinct questions — *how good is this player per play?* and *how much is this
player used?* — into a single number. The rest of the project is, in large part,
about pulling those two questions apart.

## Decomposing value: what is actually predictable

Because value EPA per game equals efficiency per opportunity times opportunities
per game, each player-season can be split into two standardized axes:

- **efficiency** = value EPA per opportunity (per dropback for QBs, per
  carry-or-target for skill players)
- **opportunity** = opportunities per game (usage / role)

The decisive question is how *repeatable* each axis is from one season to the
next, because a repeatable signal is more likely to reflect stable ability than
luck or a one-year role. Measuring lag-1 year-over-year correlation on the real
data gives a clear answer:

| Segment | Total value | Efficiency | Opportunity |
| --- | ---: | ---: | ---: |
| Overall | 0.42 | 0.26 | 0.76 |
| QB | 0.49 | 0.47 | 0.53 |
| RB | 0.21 | 0.22 | 0.78 |
| WR | 0.49 | 0.18 | 0.79 |
| TE | 0.50 | 0.25 | 0.77 |

(Efficiency is measured only on player-seasons with enough volume to be
meaningful — a position-specific minimum opportunity load — so a one-target
receiver's noise does not pollute the signal.)

The pattern is stark and is the analytical heart of the project. **Opportunity
persists strongly (~0.76); efficiency, for skill positions, barely persists at
all (0.18–0.25). Quarterbacks are the exception — their efficiency is genuinely
sticky (0.47).** Much of what the total-value score appears to "predict" from
season to season is therefore role stability, not per-play ability. That insight
motivated the next experiment — and survives it even though the experiment itself
did not pan out as a prediction improvement.

## The two-stage experiment, and its honest result

The decomposition suggested a hypothesis: predict each axis with the tool suited
to its signal, recombine them, and beat the blended single model. The hypothesis
was implemented in full, validated head-to-head, and **did not hold.** It is
reported here because a negative result, honestly shown, is more valuable than a
flattering one — and because the reasoning behind it is itself the insight.

**Stage 1 — opportunity.** Predict next-season opportunity per game (Random Forest
and Histogram Gradient Boosting, against a persistence baseline). This is the
high-signal half: a persistence baseline alone explains about 83% of next-season
opportunity variance for skill positions. The instructive exception is
quarterbacks, where persistence explains almost nothing (R² ≈ 0.03) because QB
"opportunity" — dropbacks — swings violently with starter-versus-backup status.
That is precisely where role information (depth chart, contract) has the most to
add.

| Position | Persistence R² (next-season opportunity) |
| --- | ---: |
| Overall | 0.83 |
| TE | 0.62 |
| RB | 0.59 |
| WR | 0.58 |
| QB | 0.03 |

**Stage 2 — efficiency.** Predict next-season efficiency on efficiency-qualified
seasons only, against a shrink-to-mean baseline (predict the positional mean) —
the correct null when efficiency barely autocorrelates. The talent-isolating rate
features (catch rate, yards per target, air yards per target, YAC per reception,
RACR, yards per carry, completion percentage, yards per attempt, passing aDOT,
PACR) feed this stage. The result confirms the decomposition's thesis: efficiency
is learnable for quarterbacks and almost pure regression to the mean for everyone
else.

| Position | Efficiency skill over the positional mean |
| --- | ---: |
| QB | +12.7% |
| TE | +1.8% |
| WR | +1.7% |
| RB | +1.6% |

This is itself a front-office insight, not a disappointment: for skill positions,
the honest forecast of next-year efficiency is "close to the positional average,"
and confidently projecting otherwise is usually overfitting.

**Recombination — the verdict.** The two stages multiply back into a per-game
value projection (standardized to the `value_score` scale with *frozen*
training-season statistics, so the future cross-section is never used) and are
compared head-to-head, on identical rows, against a single model predicting value
directly from the same feature union:

| Method | RMSE | R² | Skill vs single model |
| --- | ---: | ---: | ---: |
| single model | 2.318 | 0.203 | — |
| two-stage | 2.417 | 0.134 | −4.2% |
| persistence | 2.482 | 0.087 | −7.1% |

**The single model wins.** The two-stage recombination is about 4% worse on RMSE,
and the reason traces directly to the decomposition: Stage 2 efficiency barely
beats a shrink-to-mean baseline (and trails shrunken persistence), so multiplying
that near-noise estimate into the product adds error the single model sidesteps.
Stage 1 opportunity only beats persistence at quarterback; for RB/WR/TE,
persistence is already so strong (R² ≈ 0.58–0.62) that the model mostly adds
noise. The conclusion is therefore clean and was decided by the data: **use the
single model for point predictions.** The decomposition's payoff is not accuracy —
it is the uncertainty layer below.

## Asymmetric, axis-aware uncertainty — the part that worked

The two-stage structure did not win on accuracy, but it earns its place by
producing something a blended model structurally cannot: an interval that knows
*which axis* it is uncertain about. Because value is a product, independent stage
errors propagate as:

`Var(E·O) = O²·σ_E² + E²·σ_O² + σ_E²·σ_O²`

The first term is the uncertainty contributed by the efficiency axis, the second
by opportunity. This both sets the interval width and *decomposes* each player's
uncertainty into an efficiency share and an opportunity share, which becomes a
plain-language label — **efficiency-driven** or **role-driven** — telling a scout
*why* a projection is uncertain, not just how much.

This was validated with rolling origin (train both stages on a proper-training
set, set per-position sigmas from a held-out calibration season, check coverage of
the resulting interval), and it holds up:

| Segment | Coverage (target 80%) | Efficiency share of uncertainty |
| --- | ---: | ---: |
| Overall | 80.9% | 95% |
| QB | 80.0% | 98% |
| RB | 84.5% | 93% |
| TE | 80.8% | 83% |
| WR | 78.4% | 80% |

Coverage lands essentially on target, and the variance attribution is the
genuinely novel output: for skill positions, 80–98% of a player's value
uncertainty comes from the efficiency axis the model cannot pin down. A front
office reading "this projection is wide because we can't predict his efficiency,
not his role" is getting information a single point estimate — or a single
symmetric error bar — cannot convey. The independent benchmark stage corroborates
the calibration story: distribution-free conformal intervals on the single model
hit 81.1% coverage against the same 80% target.

## The player-value deliverable

The project ships two complementary deliverables, with a clear division of labor
decided by the validation results.

The **single-model Excel workbook**
(`outputs/tables/2026_player_value_predictions.xlsx`, 505 players with
plain-English drivers, confidence levels, and availability risk) is the **primary
point-prediction deliverable** — it is the more accurate model and the polished
presentation layer.

The **two-stage projection table**
(`outputs/tables/two_stage_2026_projection.csv`, also 505 players) is the
**uncertainty and interpretability layer**. Each row carries the asymmetric
interval and the role-driven-versus-efficiency-driven label. Of the 505 players,
261 are efficiency-qualified (enough volume for a reliable efficiency signal); the
rest lean on the positional efficiency prior and carry visibly wider intervals,
with the driver label making that lower confidence explicit rather than hiding it.
Read together, the workbook answers "how good?" and the two-stage table answers
"how sure, and why?"

## Honest model evaluation

A recurring theme deserves to be stated plainly, because it is the most important
methodological point in the project: **on a within-group standardized target,
RMSE alone overstates model quality.** A benchmark stage
(`src/model_benchmark.py`) reports a *skill score* — percentage RMSE reduction
versus strong baselines (season mean, raw persistence, shrunken persistence, age
curve) — for both Random Forest and Histogram Gradient Boosting under rolling
validation, plus distribution-free conformal intervals. The takeaway is that a
one-line shrunken-persistence baseline is genuinely competitive on the blended
target, which is exactly what motivated the decomposition. Reporting that honestly
is more valuable than a headline number that flatters the model.

Supporting analyses reinforce rather than inflate this view. Position-specific
models offer only small gains over the pooled model and do not justify the added
complexity. Adding contextual football features (usage, team environment,
schedule) moves rolling-validation error only marginally, so context is treated as
roughly neutral rather than a breakthrough. An optional advanced-modeling track
(Optuna tuning, SHAP explanation, Polars profiling, MLflow tracking) tightens
RMSE by a small amount and confirms that current production and recent value
history are the dominant signals — useful as methodological diligence, not as a
reason to ship a more complex model.

## Reproducibility and quality control

Every result is rebuildable from one command:

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

The pipeline runs in dependency order: clean data, build value scores, decompose
value, generate predictions and the Excel workbook, run salary efficiency and
findings, fantasy projections, the weekly win backtest, methodology checks, model
interpretation, the benchmark, and the two-stage model. Individual stages can be
run with `--steps` (for example `--steps decompose,two_stage`).

A methodology-check report (`report/methodology_checks.md`) audits the project's
core assumptions — raw/processed data are untracked, aggregated rows are unique at
the intended grain, value scores standardize correctly within season-position
groups, prediction intervals are ordered, and no feature contains a next-season
target column. A unit-test suite under `tests/` independently verifies the
leakage-safety of the lag/rolling/target construction, the benchmark and
two-stage metric math, the interval-propagation formula, and cross-module config
consistency.

## Salary efficiency

The salary-efficiency analysis merges value scores with historical contract data
(96.1% match rate, 4,569 of 4,753 rows) and defines surplus as value above what a
regression on salary, position, age, experience, draft slot, and games played
would expect. On a cleaner sample of 3,531 player-seasons with at least eight
games, the top individual surplus season is 2025 Puka Nacua, and the top
team-season is 2018 Kansas City. High-cost running-back seasons show negative
average residuals, consistent with the well-known risk in the timing and decline
profile of veteran RB contracts.

These results are framed as efficiency findings, not exact cap accounting: the
cost variable is a season-specific cap hit reconstructed from contract terms (a
principled estimate, source-flagged, since the contracts carry no year-by-year
cap breakdown), and the residuals are descriptive rather than causal. The clearest
next improvement to this section is true OverTheCap season-level cap data.

## Limitations

The value metric is production-based and does not fully isolate scheme, offensive
line, quarterback quality for receivers, play-calling, injuries, coaching changes,
teammate effects, or defensive attention. Tight ends are hardest, because blocking
value is not well captured in the available data. The two-stage interval assumes
the two stages' errors are roughly independent — reasonable given they are modeled
on different feature sets, and the interaction term is retained rather than dropped
to stay honest about mild dependence. The salary analysis carries the
contract-data caveats above.

## What this project demonstrates

Beyond the football results, the project is meant to show a particular way of
working: define a metric, then *stress-test the assumption that it is
predictable*; benchmark against baselines strong enough to be embarrassing rather
than against zero; form a hypothesis from an empirical finding, test it honestly,
and *report it even when it loses*; and build uncertainty that communicates
*where* knowledge runs out.

The two-stage model is the clearest example of that discipline. It was a
well-motivated bet — the decomposition genuinely showed efficiency and opportunity
behave differently — and it was beaten head-to-head by a simpler single model.
Rather than bury that, the project states it plainly and keeps what the experiment
*did* produce: a validated, calibrated, asymmetric uncertainty layer that no single
model can replicate. The lasting contributions are therefore three: the
decomposition as a *diagnostic* insight about what in NFL value is predictable, the
honest benchmarking discipline that keeps every model claim grounded against strong
baselines, and the axis-aware uncertainty that makes the output usable by someone
making real decisions. The point prediction comes from the single model; the
understanding of it comes from everything built around it.
