# NFL Player Value & Fantasy Projection Lab — Full Project Reference

A complete, plain-language account of what this project contains: the data, every
model, how each one was trained and chosen, what it found, the safeguards that keep the
results honest, and the limitations that define where version 1 stops and version 2
begins. This is the working reference for rewriting the app copy and reports — it is the
"what is actually true" document, written so the numbers and design can be restated in
any voice without losing accuracy.

**How to read this.** Every model, metric, and method is explained in plain terms the
first time it appears, marked *In plain terms*. The goal is that a reader who has never
trained a model can follow the reasoning, while the numbers stay exact enough for
someone technical to trust them. Football examples are used wherever they make an idea
concrete.

---

## 1. The goal, in one paragraph

The project does two jobs with the same underlying data. The first is a **front-office
job**: measure how much value an NFL offensive player produced, predict how that value
carries into the next season, and compare value to contract cost so a team can see who
is overpaid or underpaid. The second is a **fantasy job**: project PPR fantasy points —
both season-long totals for the upcoming year and week-by-week scores during a season —
and present them as rankings a manager can act on. Around both jobs sits a research
layer (a causal study, a rookie projection model, an external benchmark) whose purpose
is to test specific football beliefs honestly rather than to ship a number.

The organizing principle throughout is **honest evaluation**: every model is graded
against a strong, hard-to-beat reference rather than against zero, results that *didn't*
work are kept on the record instead of hidden, and every prediction ships with an honest
statement of how uncertain it is.

> *In plain terms — PPR:* "Points Per Reception" is the most common fantasy scoring
> system. A player earns points for yards, touchdowns, and one point per catch. It is
> just a way of turning a real game performance into a single fantasy number.

---

## 2. Data foundation

**Source.** All on-field data comes from nflverse, a free, well-maintained public NFL
data project, pulled with its `nflreadpy` package: weekly player statistics, rosters,
schedules, depth charts, injuries, and play-by-play. Contract data comes from nflverse's
historical contracts feed, which originates from OverTheCap (a salary-cap tracking
site). Coverage is **2016 through 2025**, regular season only.

**Scope.** Four offensive skill positions: quarterback (QB), running back (RB), wide
receiver (WR), tight end (TE). Defensive players, kickers, and special teams are out of
scope.

**Grain.** "Grain" just means *what one row represents*. Two grains are used:

- **Player-season** (one row = one player in one year) for the value, salary,
  season-fantasy, and rookie work. A player traded mid-season is combined into a single
  row so a partial sample is not mistaken for a full one. A minimum-games filter (at
  least 4) keeps tiny samples out.
- **Player-week** (one row = one player in one game) for the weekly fantasy model and
  the causal study. 55,670 player-weeks have an actual fantasy result attached.

**Position-aware production.** QB production is passing + rushing value; RB/WR/TE
production is rushing + receiving value. Positions are never compared on a raw scale.
They are made comparable only after each player is judged against his own
position-and-season peers — a 250-yard receiving game means something different in a
pass-happy 2025 than it did in 2016, and a QB's numbers should never be lined up
directly against a running back's.

**What is committed vs. local.** Raw and processed data stay off of version control
(they can always be re-downloaded). The small output tables and reports are saved, so
the project can be reviewed — and the app can run — without the large raw files. The two
files that feed the Fantasy Rankings page are committed:
`2026_fantasy_football_projections.csv` (season-long) and
`weekly_fantasy_validation_predictions.csv` (weekly history).

> *In plain terms — EPA (Expected Points Added):* the building block under most of the
> value work. Before every play, based on down, distance, and field position, NFL teams
> in that situation score some average number of points on the drive. EPA measures how
> much a single play changed that expectation. A 3rd-and-8 conversion adds expected
> points (positive EPA); a sack or interception subtracts them (negative EPA). Summing a
> player's EPA captures how much his plays actually helped the team score — a far better
> measure of value than raw yards, because it credits plays by how much they mattered.

---

## 3. The pieces, at a glance

| Thread | Question it answers | Model type | Primary output |
| --- | --- | --- | --- |
| Season player value | How good was each player-season, and what carries forward? | Random Forest (point) + two-stage decomposition (uncertainty) | `value_score`, next-season projection, asymmetric intervals |
| Season-long fantasy | How many PPR points in 2026? | Elastic Net (chosen from 6 candidates) | Top-25 rankings by position |
| Weekly fantasy | How many PPR points this week? | Gradient Boosting | Weekly projection vs actual, live next-week preview |
| Salary / cap surplus | Who out-earns their contract? | Reconstructed cap hit + replacement-level / regression residual | Surplus per player-season |
| Rookie cold-start | How will a player with no NFL history do? | Hierarchical Bayesian + hurdle gate | Rookie PPR/game projection, P(plays) |
| Causal: QB injury → WR PPR | Does QB injury *cause* WR fantasy drops? | Difference-in-differences / event study | Suggestive −0.6 to −1.0 PPG effect |

The two threads that feed the **Fantasy Rankings** page are the **season-long fantasy**
projection (the ranking table) and the **weekly fantasy** model (the projection-vs-actual
breakdown). The other four power the Cap Allocation Brief, the drill-down pages, and the
Player Detail view.

---

## 4. Season player value (the analytical core)

### 4.1 The value metric and its known flaw

The headline metric is `value_score`: a player's total value (his EPA) expressed as a
**z-score within his (season, position) group**.

> *In plain terms — z-score:* a z-score restates a number as "how many standard
> deviations above or below the group average." A z-score of `0.0` is exactly average,
> `+1.0` is one standard deviation above the pack (clearly above average), `−1.0` is one
> below. Doing this *within each season and position* is what makes a 2025 tight end and
> a 2016 quarterback comparable: each is scored against his own peers, not against a raw
> total that the era and position would distort.

Its deliberate flaw: **total EPA rewards volume as much as quality.** A player who is on
the field constantly and is merely average can out-score a player used rarely who is
excellent per play. For talent evaluation that mixes two different questions — *how good
is he per play?* and *how much is he used?* — into one number. Pulling those apart is
what the rest of the value work does.

### 4.2 The decomposition (the central insight)

Value per game can be split into two parts that multiply together:
**efficiency** (value per opportunity — per dropback for a QB, per carry-or-target for a
skill player) times **opportunity** (how many chances he gets per game). The key
question is which part *repeats* from one year to the next, because a number that repeats
reflects stable skill, while a number that bounces around is mostly luck or a one-year
role.

Repeatability is measured with **lag-1 correlation**.

> *In plain terms — lag-1 correlation:* line up each player's number this year against
> the same player's number last year, and ask how tightly they track. A value near 1.0
> means "last year strongly predicts this year" (very repeatable); near 0 means "last
> year tells you almost nothing" (basically noise).

| Segment | Total value | Efficiency | Opportunity |
| --- | ---: | ---: | ---: |
| Overall | 0.42 | 0.26 | 0.76 |
| QB | 0.49 | 0.47 | 0.53 |
| RB | 0.21 | 0.22 | 0.78 |
| WR | 0.49 | 0.18 | 0.79 |
| TE | 0.50 | 0.25 | 0.77 |

**Opportunity repeats strongly (~0.76); efficiency for skill players barely repeats
(0.18–0.25). Quarterbacks are the exception — their efficiency is genuinely sticky
(0.47).** In plain terms: who gets the ball is fairly predictable; how efficient a
non-QB is with it next year is close to a coin flip. So a lot of what looks like
"predicting value" is really just predicting that roles stay similar.

### 4.3 The point model: how it was trained and chosen

The thing being predicted (the "target") is `next_value_score` — next season's value
score. Because of how the z-score is built, the target's spread is about 1.0, which
creates a trap: a model can look accurate just by guessing "average" for everyone. So
raw error is not enough; the model has to be compared to a smart simple guess.

> *In plain terms — the metrics used here:*
> - **RMSE (root mean squared error):** typical prediction miss, in the units of the
>   thing predicted, with big misses penalized extra. Lower is better.
> - **R² :** the share of the variation the model explains, from 0 (no better than
>   guessing the average) to 1 (perfect). For noisy sports data even 0.2 can be real.
> - **Skill score:** the honest headline — *how much better is the model than a strong
>   simple guess*, in percent. Beating zero is easy; beating a good baseline is the real
>   test.

> *In plain terms — what a "baseline" is and why it matters:* a baseline is the answer
> you'd get with **no machine learning at all** — the cheapest obvious guess a person
> could make on a napkin. It is the bar the model has to clear to justify its existence.
> This isn't a quirk of this project; it is the standard way forecasts are judged. The
> forecasting reference *Forecasting: Principles and Practice* (Hyndman &
> Athanasopoulos) builds its recommended accuracy measure directly on the "naive
> forecast" and frames the rule bluntly: **every method must beat naive.** The *skill
> score* is simply how far past that bar the model gets, in percent. So reporting a model
> as "+4.3% skill" is more honest than "R² = 0.23 against zero," because the comparison is
> against a guess that's actually hard to beat.

> *In plain terms — the specific baselines used here:*
> - **Persistence** ("next year = this year"): the simplest possible forecast, and the
>   canonical naive benchmark.
> - **Shrunken persistence:** persistence pulled partway toward the group average,
>   because unusually high or low seasons tend to drift back toward normal the next year
>   (regression to the mean). This is the *toughest* simple baseline — it already captures
>   most of the predictable signal.
> - **Age curve / season mean:** guesses based on typical age trajectories, or just the
>   positional average. Included so the model must beat several *different* simple ideas,
>   not a single weak one.

Training used **rolling-origin validation**.

> *In plain terms — rolling-origin validation:* train only on the past, test on the
> future, then roll forward. Predict 2022 using 2016–2021, then 2023 using 2016–2022,
> and so on. This mimics real life (you never get to peek at the season you're
> predicting) and is the honest way to test a forecasting model.

The candidates, run head-to-head:

| Method | Type | RMSE | R² | Skill vs shrunken persistence |
| --- | --- | ---: | ---: | ---: |
| Random Forest | model | 0.925 | 0.231 | **+4.3%** |
| Gradient Boosting | model | 0.958 | 0.175 | +0.9% |
| shrunken persistence | baseline | 0.966 | 0.160 | 0.0% |
| age curve | baseline | 1.052 | 0.004 | −8.9% |
| season mean | baseline | 1.054 | 0.000 | −9.1% |
| raw persistence | baseline | 1.161 | −0.213 | −20.2% |

**Random Forest was selected.**

> *In plain terms — Random Forest:* imagine asking hundreds of slightly different
> decision-tree "flowcharts" — each a chain of yes/no questions like "more than 1,200
> yards last year? younger than 27?" — and averaging their answers. Each tree on its own
> is rough and over-eager; averaging a whole "forest" of them cancels out the
> individual mistakes and produces a stable prediction. It handles messy, non-linear
> patterns well and needs little hand-tuning.

The most important honest finding: **a one-line shrunken-persistence guess is hard to
beat** — the model adds only about 4% skill. That small edge is the *true* size of the
signal, which is why the value model is described as a **tiering tool** (good for sorting
players into rough tiers) rather than an exact ranker.

What the model leans on, found with **permutation importance**:

> *In plain terms — permutation importance:* to see if a feature matters, scramble that
> one column and check how much worse the model gets. A big drop means the model really
> relied on it; no change means it was ignorable.

The model leans on current-season production (mainly total value EPA) and recent
multi-year history. Draft position and age add a little; the rest is minor.
Position-specific models (one per position) were tested and gave only small, inconsistent
gains, so the simpler **pooled model** (one model for all positions, with position as an
input) was kept.

### 4.4 The two-stage experiment — a documented "it didn't work"

The decomposition suggested a tempting idea: predict opportunity and efficiency
separately with the right tool for each, multiply them back together, and beat the single
model. It was built fully and tested fairly on the same rows:

| Method | RMSE | R² | Skill vs single model |
| --- | ---: | ---: | ---: |
| single model | 2.318 | 0.203 | — |
| two-stage | 2.417 | 0.134 | **−4.2%** |
| persistence | 2.482 | 0.087 | −7.1% |

**The single model won.** The reason is exactly the decomposition's own lesson: the
efficiency half is nearly unpredictable, so multiplying that near-random guess into the
product *adds* error the single model avoids. The idea was right about how football works
and wrong as a way to predict more accurately — and it is reported plainly rather than
hidden. (Keeping results that didn't pan out is a deliberate credibility choice.)

### 4.5 What the two-stage layer is kept for — smarter uncertainty

The two-stage version lost on accuracy but does one thing a single model cannot:
tell you *which kind* of uncertainty a player carries. Because value is
efficiency × opportunity, the uncertainty splits cleanly into "how unsure are we about
his efficiency?" and "how unsure are we about his role?" That turns into a plain label —
**efficiency-driven** or **role-driven** — so a reader knows *why* a projection is shaky.

This is delivered as a **prediction interval**.

> *In plain terms — prediction interval and coverage:* instead of a single guess, give a
> range ("most likely 180–240 PPR"). An **80% interval** should contain the real result
> about 80% of the time. **Coverage** is how often it actually does. Coverage landing
> near 80% means the honesty of the range can be trusted; an *asymmetric* interval can
> stretch further on the side where the player is harder to predict.

Validated with rolling origin:

| Segment | Coverage (target 80%) | Efficiency share of uncertainty |
| --- | ---: | ---: |
| Overall | 80.9% | 95% |
| QB | 80.0% | 98% |
| RB | 84.5% | 93% |
| TE | 80.8% | 83% |
| WR | 78.4% | 80% |

Coverage lands on target, and for skill players 80–98% of the uncertainty comes from the
efficiency axis — i.e., we usually know roughly how *much* he'll play, we just can't
pin down how *well*. So: **single model for the point estimate, two-stage layer for
honest, labeled ranges.**

---

## 5. Season-long fantasy projection (the Rankings table)

This is the model behind the **top-25 2026 rankings**. It projects 2026 full-season PPR
points from 2025 production, recent history, usage, and the EPA value features.

**Six candidates were compared:** a "same as last year" baseline, Ridge, Elastic Net,
Random Forest, Gradient Boosting, and a two-stage model (predict games played and
points-per-game separately). The winner is chosen by the lowest rolling-validation RMSE.

> *In plain terms — the linear candidates:*
> - **Ridge and Elastic Net** are "smart straight-line" models. They predict points as a
>   weighted sum of the inputs, but with a built-in penalty that stops any one input from
>   getting too much weight (this is *regularization* — it keeps the model from
>   over-trusting noise). Elastic Net can also zero out useless inputs entirely. On
>   smooth season-long totals, these disciplined linear models often beat fancier ones.

**Selected model: Elastic Net.** Performance on 2,414 rolling-validation rows:

- Rolling **MAE: 41.60** PPR points
- Rolling **RMSE: 59.09** PPR points
- **Spearman rank correlation: 0.722**
- **Top-rank hit rate: 0.615**

> *In plain terms — these last two:*
> - **MAE (mean absolute error):** the average miss, plainly (here, about 42 PPR points
>   over a full season). RMSE is its big-miss-sensitive cousin.
> - **Spearman rank correlation:** ignores exact point totals and asks "did it get the
>   *order* right?" 0.72 means the ranking is good but not perfect — strong for sorting
>   into tiers, not for splitting hairs between adjacent players.
> - **Top-rank hit rate:** of the players it called elite, the share that actually
>   finished elite — here about 62%.

505 players are projected. Each row carries the projected 2026 total, a points-per-game
and games-played split, an 80% range, a tier, and a change-vs-2025 label. The page shows
the top 25 per position.

**Honest scope:** this is a tiering projection, not a finished ranking system. It does
**not** yet include rookies (no NFL history to build on), depth-chart changes, in-season
injuries, coaching changes, or betting markets.

---

## 6. Weekly fantasy projection (the projection-vs-actual breakdown)

This model projects PPR for a player's *current* game using only information available
before kickoff — the real-life situation: late in the week, project this Sunday.

### 6.1 Model and training

**Model: a single Gradient Boosting regressor** (scikit-learn's
`HistGradientBoostingRegressor`).

> *In plain terms — Gradient Boosting:* like Random Forest, it uses many small decision
> trees, but instead of averaging independent trees it builds them **one at a time, each
> new tree fixing the leftover mistakes of the ones so far.** It is one of the most
> accurate methods for table-style data like this. The "Histogram" version buckets
> numbers into bins so it trains fast on large datasets. One pooled model is used (not
> one per position) because efficiency at the weekly level is so noisy that splitting the
> data would hurt more than specializing would help.

Settings (chosen for a stable, lightly-restrained fit, not aggressively tuned — see §10
for why heavy tuning doesn't pay here):
`max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
l2_regularization=0.1, random_state=42`.

> *In plain terms — what those knobs do:* `max_iter` is how many corrective trees to
> build; `learning_rate` is how big a step each tree takes (small = cautious, less
> overfitting); `max_leaf_nodes` / `min_samples_leaf` cap how detailed each tree gets so
> it can't memorize individual players; `l2_regularization` is an extra penalty against
> over-complex fits; `random_state` just fixes the randomness so results reproduce.

Trained with rolling-origin validation across 2020–2025. 55,670 rows, 43 features.

### 6.2 The 43 features (what the model sees), in groups

- **Recent production:** rolling PPR (last game, last-4 average and its ups-and-downs,
  last-8, season-to-date) plus last-4 averages of targets, catches, carries, and passing
  / rushing / receiving yards. This is the backbone — what he's been doing lately.
- **Player profile:** position, age, week number, career games.
- **Opponent:** how many PPR points this week's defense typically allows to the position
  — a rough "is this a good matchup?" signal.
- **Game and betting-market context:** home/away, days of rest, the point spread, the
  game's over/under total, and the *implied team total* (how many points Vegas expects
  this team to score), plus **position × market** combinations so a high-scoring game
  environment boosts a QB more than a backup tight end. Betting lines are used because
  they are the market's best guess at game flow, and game flow drives fantasy scoring.
- **Availability proxy:** whether he played recent weeks and how many in a row —
  rebuilt from whether he has a stats line, while skipping bye weeks so a bye isn't
  counted as a missed game.
- **Role / depth chart:** snap share, plus a **rebuilt depth-chart rank** (is he the RB1
  or the RB2?) computed from actual play-by-play usage, because nflverse stopped
  publishing a usable rank field around 2024.
- **Weather:** indoor flag, temperature, wind (indoor games set to a neutral 70°F / no
  wind).
- **Injury-report status:** full practice / limited / did-not-practice /
  questionable-or-worse flags.

All "recent" features are **leakage-safe** (see §11) — they only ever look at games
*before* the one being predicted.

### 6.3 Accuracy and how it is judged

The right bar is a **naive forecast** — the standard the forecasting field insists every
model must beat. Three are used: the recent-4-game average, the season-to-date average,
and the position average.

> *In plain terms:* "skill vs recent-4-avg" answers *did the model do better than simply
> averaging the player's last four games?* That rolling average is already a strong guess,
> so beating it consistently is the meaningful test — the same "must beat naive" standard
> from the forecasting literature, applied weekly. This metric choice is deliberately
> aligned with how the fantasy field itself grades projections: FantasyPros' published
> in-season accuracy method scores experts by their error against the actual points
> scored, so an error-versus-realized-points framing is the industry's own bar, not one
> invented here.

| Method | n | RMSE | MAE | Skill vs recent-4-avg |
| --- | ---: | ---: | ---: | ---: |
| Gradient Boosting (pooled) | 34,906 | 6.147 | 4.552 | +7.6% |
| Gradient Boosting per-position | 34,906 | 6.346 | 4.704 | +4.6% |
| recent-4-avg | 34,906 | 6.655 | 4.835 | 0.0% |
| season-to-date avg | 34,906 | 6.704 | 4.888 | −0.7% |
| position mean | 34,906 | 7.703 | 6.082 | −15.8% |

The edge is **steady across all six seasons** (+6.8% to +8.5% each year). A single-digit
edge is genuinely good here, because weekly fantasy is one of the least predictable
things in sports — one tipped pass or one broken tackle swings a week. Independent
accuracy research (Fantasy Football Analytics, which ranks public projection sources
each year) finds weekly projection R² in the single digits to low-twenties by position;
against that ceiling, a stable ~7–9% beat over a strong baseline, repeated across six
independent years, is a real edge. The most recent feature additions (rebuilt depth-chart rank +
weather) improved overall RMSE from 6.020 to **5.944 (+1.27%)**, every position
improving.

### 6.4 External benchmark (the one place a market line exists)

On 2020–2021, a market projection can be reverse-engineered from DraftKings salaries
(DraftKings prices players before games, so their salaries encode an implied
projection). On 11,191 shared player-weeks the model is competitive-to-slightly-ahead:
model RMSE 6.386 vs the DraftKings-implied 6.493 (**+1.65%**), winning 51–55% of
player-weeks. This is a *tough* bar because the implied projection was built using the
season's actual results. **Scope is honest:** the free data ends in 2021, so this says
nothing about beating live DraftKings, FantasyPros, or ESPN in recent years. The
benchmark code accepts any projection file dropped in later with no changes.

### 6.5 Uncertainty intervals

Weekly ranges use **conformal prediction**.

> *In plain terms — conformal intervals:* a simple, assumption-free way to build honest
> ranges. Hold out some recent games the model didn't train on, measure how big its
> misses actually were, and use those real-world miss sizes to set the width of the
> range. Because it's built from actual errors, the 80% range really does contain the
> result about 80% of the time — no bell-curve assumption required.

Overall coverage is on target (50% range → 50%, 80% range → 80%). One honest weakness:
a single global width **under-covers quarterbacks** (about 60% instead of 80%) because QB
scoring swings more wildly than the average player's. The fix used for live projections
is **per-position widths** — calibrate the range separately for QBs — which lifts QB
coverage from ~58% to ~73% without bloating the others.

### 6.6 Live (next-week) projection

The same model can project a game that hasn't happened by building each player's row two
ways: **his recent-form numbers are carried forward** from his last completed game
(already based only on past games, so nothing leaks), and **the game details** (opponent,
home/away, rest, betting lines, weather) come from the known upcoming schedule. The
actual-result column is removed entirely. The one honest shortcut: recent-form numbers
lag by one game (carried forward rather than recomputed), accepted to keep every row
complete.

### 6.7 Things that were tried and *didn't* make it (kept on the record)

- **One model per position:** lost to the pooled model everywhere.
- **Advanced tracking-data features (NGS/PFR):** an apparent gain turned out to be a
  leak (the data's missing-ness secretly revealed whether the player had played), so they
  were rejected.
- **Ensemble stacking** (blending several models): improved accuracy by only 0.07% — not
  worth running three models instead of one.
- **Quantile intervals:** wider without better coverage; conformal was kept (though this
  test is what revealed the QB coverage problem worth fixing).

---

## 7. Salary / cap surplus

This thread asks whether a player out-earns his contract compared to a freely available
replacement.

### 7.1 The cost side — a reconstructed season cap hit

The simple approach charges every player his contract's yearly *average* (APY). That's
misleading: signing bonuses are spread out and base salaries are usually backloaded, so a
star's first year actually costs the team far less against the cap than his average
suggests. The contract data has totals but **no year-by-year breakdown**, so the true cap
hit can't be looked up — it has to be estimated:

> **estimated cap hit (year k) = prorated signing bonus + backloaded base salary**

with two transparent rules: guaranteed money is spread evenly over up to five years (the
NFL's proration limit), and the rest is spread on a gently rising schedule. A nice
property: the yearly estimates **add back up to the contract's true total**, so no money
is invented. Coverage is essentially complete, and every row is flagged so a reader can
see whether it was a clean estimate or a fallback.

This is labeled an *estimate*, never presented as official cap accounting — the honest
caveat that keeps the section credible.

### 7.2 Two surplus definitions (why two different "top" players show up)

"Surplus" is reported two ways, which is why two names appear — this is not a
contradiction, just two lenses:

- **Replacement-level surplus** (the app's Cap Allocation page): value above a freely
  available backup, priced against the reconstructed cap hit. By this measure **Brock
  Purdy's 2023 season is the biggest single-season surplus** — rookie-contract QBs
  dominate because they produce like stars while costing almost nothing.
- **Salary-regression-residual surplus** (the salary-efficiency analysis): value above
  what a model would expect given a player's pay, age, experience, draft slot, and games.
  By this measure the top season is **2025 Puka Nacua** and the top team-year is 2018
  Kansas City.

> *In plain terms — "regression residual":* fit a line predicting value from salary (and
> the other factors), then measure how far above the line a player sits. That gap — the
> *residual* — is how much more he produced than his pay would predict.

Both are descriptive, not causal. A clear pattern: high-paid veteran running backs tend
to sit *below* the line, matching the well-known risk in big veteran RB contracts.

---

## 8. Rookie cold-start (Bayesian) + the hurdle gate

Rookies have no NFL history, so the rolling-history models can't touch them. Two pieces
handle the "cold start."

**Projection model: a hierarchical Bayesian regression.**

> *In plain terms — Bayesian, and "hierarchical / partial pooling":* a Bayesian model
> starts from a sensible prior belief and updates it with data, and it naturally produces
> a *range* rather than a single guess — ideal when data is thin. "Hierarchical" (or
> *partial pooling*) means the four positions share information instead of being modeled
> in total isolation: each position gets its own tendencies, but a position with few
> examples borrows strength from the others rather than over-reacting to a handful of
> players. It's the statistical version of "treat QBs and WRs as different, but let what
> you learn about one gently inform the other."

It predicts rookie points-per-game from pre-NFL facts only: draft position, age, height,
weight (college production is a planned addition). 2,265 rookies; 80% range coverage runs
about 76–88% across rookie classes. (It runs in a separate environment because its
library conflicts with the rest of the project.)

**Hurdle gate:** a first step that asks *will this rookie even play meaningfully (at
least 4 games)?* before projecting how well.

> *In plain terms — a hurdle / two-part model:* some players score zero simply because
> they never get on the field, which is a different question from how good they'd be if
> they did. A hurdle model handles the "do they clear the bar of playing at all?"
> question first, then projects production for those who clear it — so a buried backup
> isn't mistaken for a bad player.

The kept improvement is a focused **3-feature "is anyone in his way?" core** —
is there an established incumbent, did that incumbent just sign an extension, and how
productive was the prior starter — all measured from the year *before* the rookie
arrived. It nudges the marquee case correctly: **Jordan Love's chance-of-playing drops
from 0.61 to 0.51** once the model sees he was drafted behind a freshly-extended Aaron
Rodgers. Combine workout numbers and a broader team-context set were tested and
**dropped** — they overfit and never beat raw draft position (which already bakes in how
a player tested). An honest, small, targeted gain rather than an oversold one.

---

## 9. Causal study: does QB injury *cause* WR fantasy drops?

This is the one place the project asks a **causal** question — not "do they move
together?" but "does one *cause* the other?" The tool is **difference-in-differences**.

> *In plain terms — difference-in-differences (DiD) and event study:* to isolate the
> effect of an event (a QB getting hurt), compare a "treated" group (that QB's receivers)
> to a "control" group (similar receivers whose QB stayed healthy), *before vs. after*.
> If both groups were trending together beforehand, then diverge right after the event,
> the gap is the event's effect — this subtracts out league-wide trends that hit everyone.
> An "event study" just plots that gap week by week around the event. The whole thing
> only works if the two groups were genuinely tracking together before the event — the
> **parallel-trends** assumption, which is tested rather than assumed.

The first attempt defined the injury as the official "Out" ruling and found **no
effect** — and its parallel-trends test failed. The reason is timing: by the time a QB is
formally ruled Out, he's usually been playing hurt for weeks and his receivers were
already sliding, so "Out" is a *late* signal. Moving the trigger earlier — to the **first
week the starter shows up on the injury report at all** — passed the parallel-trends test
and grew the sample from 19 events to 104. On that cleaner setup:

- Event-study estimate: **−0.58 PPR per game** (p ≈ 0.04)
- Simpler before/after estimate: **−1.01 PPR per game**
- Matched-comparison estimate: **−1.48 PPR per game**

> *In plain terms — "p ≈ 0.04":* the p-value is the chance of seeing an effect this big
> if there were truly no effect. Around 0.04 means "probably real, but not airtight" —
> just under the customary 0.05 line.

**Verdict: a real but modest effect, and honestly underpowered** — a fraction of a
fantasy point to about one point per receiver per week, sitting right at the edge of
statistical significance, with a little leftover pre-trend. It's reported as *suggestive
evidence* that limited QB availability hurts receivers *before* the QB is formally out —
not as a slam-dunk finding. The value here is the discipline: a belief tested rigorously,
a flaw caught and named, the design fixed, and the modest answer stated plainly. No
players or teams were hand-picked.

---

## 10. How models are trained, chosen, and tuned (cross-cutting)

The same habits run through every thread:

- **Models are picked by how they do on unseen future seasons,** using rolling-origin
  validation — never by how well they fit the data they trained on.
- **Baselines are chosen to be hard to beat** (shrunken persistence, recent-4 average,
  the DraftKings line). A small win over these is treated as the *true* signal size and
  reported as-is, rather than dressed up against an easy target.
- **Tuning is deliberately light.** Hyperparameters are set to sensible, restrained
  values rather than exhaustively searched, because the predictability ceiling is low and
  heavy tuning mostly memorizes the test seasons (overfitting). An optional advanced
  track (automated tuning, model-explanation tools) confirms the main signals are recent
  production and history and barely moves accuracy — kept as due diligence, not shipped.
- **Simple-and-pooled keeps beating complex-and-specialized** — both the value model and
  the weekly model do better as one pooled model than as separate per-position models,
  because more training data beats narrow specialization.
- **Failures are published.** Every plausible upgrade that lost head-to-head is written
  up rather than buried — which is itself the strongest signal that the wins are real.

---

## 11. Safeguards (why the numbers can be trusted)

- **Leakage safety.** "Leakage" is the cardinal sin of forecasting: accidentally letting
  a model peek at information it wouldn't have in real life, which makes it look great in
  testing and fail in practice. Every rolling feature here is **shift-by-one safe** — it
  only ever uses games *before* the one being predicted. Automated tests deliberately
  flip a player's usage in the final week and confirm the feature still reflects only the
  earlier order.
- **Train-on-past-test-on-future only** (rolling-origin), everywhere.
- **Honest ranges** via conformal calibration, with per-position widths where one global
  width misfits (QBs).
- **A methodology-check report:** 26 automated checks, all passing — raw data kept out of
  version control, one row per player-season where intended, value scores standardized
  correctly, ranges ordered low-to-high, no future information in any feature, salary
  match rate above 90%.
- **A unit-test suite** covering the leakage logic, the metric math, the interval
  formulas, and the causal study's setup. A subset runs automatically on every code
  change.

---

## 12. Headline findings (one place)

- A smart "next year ≈ this year, nudged to average" guess is hard to beat for season
  value; the model adds about 4% — so use it to sort players into tiers, not to rank them
  exactly.
- Value splits into **role (very predictable)** and **per-play efficiency (nearly
  random)** — except QB efficiency, which is real and stable. This is the project's
  central insight.
- Modeling those two parts separately did **not** improve accuracy, but it produces
  honest, *labeled* uncertainty a single model can't.
- The weekly fantasy model beats every simple baseline by a **steady 7–9%** across six
  seasons and edges the DraftKings line on 2020–2021.
- Reconstructed cap hits make the salary analysis credible; cheap young QBs dominate
  surplus, and the running-back market overpays veterans.
- QB injury has a **modest, suggestive** negative effect on receiver scoring once the
  injury is timed to the first report — not the dramatic collapse fans assume.
- A small "is anyone in his way?" signal correctly tempers rookie-QB playing-time
  predictions (Jordan Love), where flashier features overfit and were cut.

---

## 13. Limitations (consolidated — this is where v2 starts)

**Value metric.** Built from production, so it can't fully separate a player from his
scheme, offensive line, quarterback, play-calling, or the defense's attention. Tight end
is hardest because blocking barely shows up in the data.

**Season fantasy ranking.** No rookies, depth-chart changes, in-season injuries,
coaching changes, or betting markets yet. The ordering is good for tiers, not exact ranks.

**Weekly fantasy.** No live scratch/inactive feed — a player ruled out an hour before
kickoff is still projected as playing. Opponent strength is a coarse number. Live
projections lag one game and use scheduled (not live-forecast) weather. QB ranges still
run a touch narrow even after the fix.

**External benchmark.** The only free market comparison ends in 2021; nothing checks the
model against a published projection source in recent years.

**Salary.** Cap hits are principled estimates, not official accounting. Surplus is
descriptive, not causal.

**Rookie.** College production isn't wired in yet. The model can under-cover elite
rookies (a known, one-line fix exists). The "in his way" gain is small and mostly affects
QBs.

**Causal.** Only moderately powered (~104 events), the estimate sits right at the
significance edge, and "first injury report" lumps a season-ender together with a routine
rest day.

---

## 14. Roadmap to version 2 (each item attacks a v1 limitation)

- **A live injury/inactives feed** for the weekly model — the single biggest accuracy
  win (stops projecting players who are actually out).
- **A published-projection benchmark (FantasyPros/ESPN) for 2022–2025** — modernizes the
  external check; the code already accepts it as a drop-in file.
- **Rookies in the season rankings** by feeding the Bayesian cold-start projection into
  the table.
- **College-production data** in the rookie model, plus a heavier-tailed likelihood for
  elite rookies.
- **Richer opponent modeling** beyond "points allowed to the position."
- **Real season-level cap data** to replace the reconstructed estimate.
- **An even earlier causal trigger** (first practice-report sighting) with more events
  for statistical power.

The plan is to ship v1 honestly — current models, current findings, stated limitations —
then retire the limitations one at a time so each version is measurably stronger.

---

## 15. Sources and further reading

### How the models are evaluated (the metrics standard)

The choice of metrics in this project is not arbitrary — it follows established practice
from the forecasting literature and the fantasy-accuracy field. These are the references
the evaluation approach is built on:

- **Hyndman, R.J., & Athanasopoulos, G. — *Forecasting: Principles and Practice* (3rd
  ed.).** The source for the skill-score framing and the "every method must beat naive"
  rule. It defines the Mean Absolute Scaled Error (MASE), which scales a model's error
  against the naive forecast's error: a score below 1 beats naive, above 1 loses to it.
  This is exactly why the project leads with *skill vs. a strong baseline* rather than
  raw RMSE or R² against zero. <https://otexts.com/fpp3/accuracy.html>
- **Fantasy Football Analytics — "Which Fantasy Football Projections Are Most
  Accurate?"** The source for the realistic ceiling on weekly predictability (weekly
  projection R² lands in the single digits to low-twenties by position) and for the
  principle that *consistency of accuracy across seasons* is how projection sources
  should be ranked — which is why this project stresses a steady six-season edge over a
  one-time margin.
  <https://fantasyfootballanalytics.net/2024/12/which-fantasy-football-projections-are-most-accurate.html>
- **FantasyPros — In-Season Accuracy Methodology.** The fantasy industry's own grading
  standard: experts are scored by their error (MAE) against the actual points players
  scored. This confirms that an error-versus-realized-points framing is the field's
  established bar, not a yardstick invented for this project.
  <https://www.fantasypros.com/about/faq/football-inseason-accuracy-methodology/>

In short: **skill score / MASE** (beat the naive baseline) comes from the forecasting
literature; **MAE against realized points** is the fantasy industry's own method; and the
**low R² ceiling** that makes a single-digit edge meaningful is documented by independent
accuracy researchers. The metrics were chosen to match how serious forecasters and the
fantasy field already judge projections.

### Data sources

- **nflverse** (accessed via the `nflreadpy` package) — the backbone for all on-field
  data: weekly player statistics, rosters, schedules, depth charts, injury reports,
  play-by-play, combine results, and draft picks, 2016–2025.
  <https://github.com/nflverse>
- **OverTheCap** (via nflverse's historical contracts feed) — contract terms (total
  value, guarantees, years) used to reconstruct season cap hits for the salary analysis.
  <https://overthecap.com>
- **RotoGuru** — the free DraftKings salary archive (through 2021) used to build the
  market-implied projection for the external benchmark.
  <https://www.rotoguru.net>
- **DraftKings** — the salary lines (via RotoGuru) whose implied projection serves as
  the one real-world market benchmark, on the 2020–2021 overlap.

Planned/future data sources noted in the roadmap: a historical FantasyPros or ESPN
weekly-projection archive (for a published-source benchmark in 2022–2025), and
college-production data for the rookie model.

---

## 16. File map (where each thing lives)

| Area | Source module | Report |
| --- | --- | --- |
| Value scoring & decomposition | `src/value_decomposition.py`, `src/features.py` | `report/final_project_report.md`, `report/value_decomposition.md` |
| Point model & benchmark | `src/models.py`, `src/model_benchmark.py`, `src/model_interpretation.py` | `report/model_benchmark.md`, `report/model_interpretation.md` |
| Two-stage value | `src/two_stage_value.py` | `report/two_stage_value.md` |
| Season fantasy projection | `src/fantasy_projection.py` | `report/fantasy_football_projection_summary.md` |
| Weekly fantasy projection | `src/weekly_fantasy_projection.py`, `src/pbp_features.py` | `report/weekly_fantasy_projection_summary.md`, `report/session1_pbp_weather.md` |
| Live weekly projection | `src/live_weekly_projection.py` | `report/fantasy/session7_live_projection.md` |
| External benchmark | `src/external_benchmark.py` | `report/external_benchmark.md` |
| Salary / cap hit | `src/cap_hit_reconstruction.py`, `src/salary_efficiency.py`, `src/replacement_level.py` | `report/salary/session4_cap_hit_reconstruction.md`, `report/salary_efficiency_findings.md` |
| Rookie model | `src/rookie_bayes.py`, `src/rookie_context_features.py` | `report/rookie_bayes_projection.md`, `report/rookie/session3_combine_team_context.md` |
| Causal study | `src/causal/` | `report/causal/qb_injury_session3.md` |
| Methodology checks | `src/methodology_checks.py` | `report/methodology_checks.md` |
| App | `app/streamlit_app.py`, `app/page_content.py`, `app/landing_content.py` | — |

The whole pipeline rebuilds from `python scripts/run_pipeline.py`; the app runs from
`./run_app.sh` (or `streamlit run app/streamlit_app.py`).
