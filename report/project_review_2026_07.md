# Project review — modeling, product, and a proposed talent feature

Written 2026-07-25. An honest assessment of where this project stands, with the
evidence for each claim pulled from the repo's own outputs. Three lenses:
data-science quality, user experience, and one proposed direction (peer/expert
talent signal).

---

## 0. The headline

The engineering discipline here is genuinely strong — 32 test files, a
leakage-specific test module, time-based validation, documented negative
results, quality flags on reconstructed data. That is better practice than most
student portfolio projects and it should be protected.

The problem is not rigor. It's **that the modeling has accreted rather than been
designed**, and the accretion now costs interpretability without buying accuracy.
Two numbers from the repo make the case:

1. **The best model is a linear one.** `outputs/tables/fantasy_model_comparison.csv`:

   | model | RMSE | R² | Spearman |
   |-------|------|-----|----------|
   | **Elastic Net** | **59.09** | **0.588** | 0.722 |
   | Ridge | 59.11 | 0.587 | 0.722 |
   | Random Forest | 59.61 | 0.581 | 0.711 |
   | Two-Stage HGB | 60.22 | 0.572 | 0.707 |
   | HistGradientBoosting | 61.22 | 0.558 | 0.708 |
   | current-year baseline | 67.04 | 0.469 | 0.696 |

   The gradient-boosted models — the complicated ones — are the *worst*
   non-baseline performers. A regularized linear model wins. That is a strong
   signal that the feature set is mostly linear-signal plus noise, and that
   complexity is being spent for nothing.

2. **Almost all the signal lives in ~6 features.**
   `outputs/tables/model_interpretation_feature_importance.csv` (permutation
   importance). *Note: this table is computed over `ENHANCED_FEATURES` — the
   **value** model that feeds the front-office surplus work — not
   `FANTASY_FEATURES`. The concentration pattern is the diagnosis; the exact
   ranking for the fantasy model still needs its own run (see §1B.7 step 0).*

   | feature | importance |
   |---------|-----------|
   | value_epa_total | 0.0385 |
   | value_score_last3_avg | 0.0132 |
   | value_score_last2_avg | 0.0125 |
   | position | 0.0074 |
   | value_score_prev | 0.0058 |
   | value_epa_per_game | 0.0057 |
   | draft_number | 0.0039 |
   | age | 0.0029 |
   | *…then it collapses* | |
   | games_played_last3_avg | **−0.0001** |
   | tds_per_game | **−0.0005** |
   | value_score_trend_2yr | **−0.0013** |

   A *negative* permutation importance means shuffling the feature made the
   model **better** — the model is fitting noise in it. Several features are in
   that category.

---

## 1. Data science

### 1.1 The feature set is redundant by construction

`FANTASY_FEATURES` (src/fantasy_projection.py) holds **44 features**, and they
come in near-collinear families:

- *Same quantity, three scalings:* `fantasy_points_ppr`,
  `fantasy_points_ppr_per_game`, `fantasy_points`
- *Same quantity, four lags:* `targets`, `targets_per_game`, `targets_prev`,
  `targets_last2_avg` — repeated for receptions and carries
- *Five overlapping PPR histories:* `_prev`, `_last2_avg`, `_last3_avg`,
  `_per_game_prev`, `_per_game_last2_avg`
- *Five value variants:* `value_score`, `value_epa_total`, `value_epa_per_game`,
  `value_score_prev`, `value_score_last2_avg`

Volume stats and per-game rates are related through games played, which is
itself a feature. So the matrix has multiple near-exact linear dependencies.

**Why this matters even though tree models "handle" correlation:**
- Importance is *split arbitrarily* across correlated copies, so no feature
  looks important and you can't say what the model uses. That's fatal for the
  "explain your model" question in an interview.
- Ridge/Elastic Net coefficients become uninterpretable and unstable in sign.
- Every redundant column is an extra chance to overfit a 2,414-row validation
  set.

**Recommendation:** build a deliberately small model — roughly
`value_epa_total`, one or two PPR-history lags, `position`, `age`,
`draft_number`, `games_played` — and ablate against the current 44. The
prediction: **you lose ~nothing**. If RMSE moves less than the project's own
~0.2% ablation threshold, you now have a model you can fully explain, plus a
documented experiment showing why the simple one wins. That result is *more*
impressive than the 44-feature version, not less.

### 1.2 The ablation discipline exists but was applied unevenly

`report/fantasy/injury_return_features.md` is genuinely excellent work: features
were built, tested, found not to clear the ~0.2% bar, and **excluded with a
written explanation**. That is exactly right.

But that standard was applied to *new* features only. The original 44 were never
subjected to it — they were added, not earned. Applying the existing standard
retroactively is the single highest-value modeling task available.

### 1.3 Sprawl in the codebase

`src/` is ~17,000 lines across 40 modules, with `two_stage_value.py` (1,427),
`weekly_fantasy_projection.py` (1,403), `prediction_report.py` (1,277), and
`rookie_bayes.py` (1,266) each larger than most single-purpose modules should
be. `scripts/` contains `eval_session1` … `eval_session7`, which is
session-chronology leaking into the repo structure — a reader can't tell what's
current from what was an experiment.

**Recommendation:** don't refactor for its own sake. Do one thing: make it
obvious what the *live pipeline* is versus what's archived research. A short
`ARCHITECTURE.md` naming the ~6 modules that actually produce the shipped
projections, and moving spent `eval_session*` scripts into `archive/`, would fix
most of the "where do I even look" problem for free.

### 1.4 What's genuinely good (don't lose this)

- Leakage handling is real: observation-level discipline, `shift(1)` lag
  construction, a dedicated `tests/test_features_leakage.py`.
- Time-based validation (`VALIDATION_YEARS`), not random K-fold — correct for a
  temporal problem, and something many projects get wrong.
- The external benchmark is honest: the model beats a DraftKings-implied market
  proxy by ~1.6–1.9% RMSE with a ~51–55% win rate, and the write-up says plainly
  that the benchmark only covers 2020–21. Modest, real, and honestly framed.
- Reconstructed cap hits carry `cap_hit_quality_flag`. Good instinct.

---

## 1B. Deeper audit: are the engineered features the right ones?

### 1B.1 What `value_epa` actually is

Traced through `src/features.py:55` and `src/prediction_report.py:237`:

```
value_epa_total = qb_epa                          if position == QB
                = rushing_epa + receiving_epa     otherwise
value_score     = z-score of value_epa_total within (season, position)
```

So the project's marquee engineered feature is **season-total EPA**, and
`value_score` is its within-position z-score. Two consequences worth naming:

- The top feature (`value_epa_total`) and the value model's target
  (`next_value_score`) are **the same construct at two points in time**. That's
  autoregression, not leakage — but it means "value_epa_total dominates
  importance" is close to tautological and shouldn't be read as a discovery.
- `value_score` is zero-sum inside each position-season. It measures *relative
  standing*, not level. Defensible for a "player value" framing; it is **not**
  the currency a fantasy user cares about (points).

### 1B.2 Is EPA the right basis — for a *fantasy* model?

**In its favor:** EPA is context-aware (down, distance, field position), which
makes it a genuinely better measure of football value than raw yards, and the
QB/skill split is handled correctly.

**The concern:** EPA and fantasy points reward different things. EPA credits
leverage — a 3rd-and-1 conversion is valuable; a garbage-time 40-yard
touchdown barely moves it. Fantasy scoring is the reverse. So EPA-derived
features are being used to predict a target they were not designed to track.

**But there is a real hypothesis here, and it's untested:** fantasy points are
heavily touchdown-driven, and touchdowns regress hard year over year. EPA may be
a *less noisy* signal of underlying quality, and therefore a **better predictor
of next-season PPR than past PPR is**. If that's true it's a genuinely
interesting finding and justifies the whole `value_*` family. If it's false,
those five features are dead weight in the fantasy model.

**The test is three runs:** {value_epa family} vs {PPR-history family} vs both,
same folds, same 0.2% bar. This is the single most important experiment in the
project — it either validates the central engineered feature or retires it.

### 1B.3 Data you already have and are not using

This is the biggest finding of the review. `data/raw/player_stats_2016_2025.csv`
already contains the canonical fantasy opportunity metrics — nflverse computes
them for free — and the season model uses **none** of them:

| column | refs in `src/` | in `FANTASY_FEATURES`? |
|--------|----------------|------------------------|
| `target_share` | 15 (only recomputed inside the *weekly* two-stage model) | **no** |
| `air_yards_share` | **0** | **no** |
| `wopr` | 1 | **no** |
| `racr` | 3 | no |
| `red_zone` (anything) | **0** | no |

Instead the model uses **raw counts** — `targets`, `receptions`, `carries`.

Why that's a real weakness: raw targets conflate a player's *role* with his
team's *pass volume*. Two receivers with 120 targets are not comparable if one
plays in a 700-attempt offense and the other in a 520-attempt offense.
`target_share` isolates the role — the component that actually persists year to
year, since team volume swings with scheme and game script. `wopr`
(1.5·target_share + 0.7·air_yards_share) is purpose-built as a fantasy
opportunity metric. `air_yards_share` separates a possession receiver from a
field-stretcher: same target count, very different yardage and touchdown
profile.

**Red-zone usage is entirely absent (0 references).** Touchdowns are the most
volatile component of fantasy scoring and the largest driver of year-over-year
regression; red-zone opportunity is the stable signal underneath the noise. It
is derivable from `pbp_2016_2025.parquet`, which is already downloaded.

### 1B.4 Data you have but have never connected: offseason change

There are **zero** features describing roster change — no teammate, competition,
vacated-opportunity, or new-team signals anywhere in `src/`. Yet the repo
already carries `draft_picks_2016_2025.csv`, `rosters_2016_2025.csv`, and
`depth_charts_2016_2025.csv`.

For a **season-ahead** model this is the most conspicuous gap. The first thing a
human drafter asks is "did this player's team just draft someone at his
position, and did the guy ahead of him leave?" The model cannot see either.
Constructible today:

- **Vacated opportunity** — targets/carries from players who left the roster
  (one of the most-cited inputs in professional fantasy projection).
- **Incoming competition** — draft capital the team spent at the player's
  position this offseason.
- **Team change** — did the player himself switch teams.

### 1B.5 Two smaller modeling gaps

- **No team context.** No pass rate, pass-rate-over-expected, or pace features.
  Share metrics tell you the slice; team context tells you the size of the pie.
  Both are derivable from the play-by-play file you already have.
- **Age is a bare linear term.** Position aging is not linear — running back
  production falls off a cliff around 27 while receivers peak later. A single
  linear coefficient cannot represent a cliff. A position×age interaction or a
  spline is the standard fix, and you have the data.

### 1B.6 Data you'd have to acquire

| source | value | cost |
|--------|-------|------|
| Vegas win totals / implied team totals | strong team-context prior | scrapeable, partly free |
| PFF grades | film-based talent, industry standard | paid |
| Coaching / OC changes | scheme shifts drive volume | manual or scrapeable |

### 1B.6b What must NOT change: the front-office chain

The value model and the fantasy model are **already separate**, which makes all
of the above safely additive:

| | feature list | target | downstream |
|---|---|---|---|
| **Value** (front office) | `ENHANCED_FEATURES` (`prediction_report.py`) | `next_value_score` | `salary_efficiency`, `salary_findings`, `value_decomposition`, `replacement_level`, `two_stage_value` → surplus-vs-cap |
| **Fantasy** | `FANTASY_FEATURES` (`fantasy_projection.py`) | `next_fantasy_points_ppr` | draft board, weekly + live projection |

Consequences, and they are important:

- **Every change proposed in §1B applies to `FANTASY_FEATURES` only.** The
  front-office surplus analysis reads `value_score` and is untouched by
  rewriting the fantasy feature set.
- **`value_epa` is not going anywhere.** It *is* the basis of the value model —
  `value_score` is its z-score, and the whole replacement-level/surplus story is
  built on it. The head-to-head in §1B.2 asks only whether the `value_*` family
  earns its place **as a predictor of next-season PPR in the fantasy model**. A
  "no" there retires it from one feature list; it remains the backbone of the
  front-office work either way.
- **Nothing is deleted.** Before any change, snapshot the current fantasy
  outputs (`fantasy_model_comparison.csv` and the projection tables) as a
  frozen baseline to measure against, and keep the existing `FANTASY_FEATURES`
  list in the module as `FANTASY_FEATURES_V1` so every experiment is a
  reproducible A/B rather than an overwrite.

### 1B.7 What this means for the simplification plan

§1.1 argued for cutting features. This section argues for adding a few. Those
aren't in conflict — the point is the same: **the current 44 were chosen by
accretion, not by test.** The right end state is a *small* feature set where
every member earned its place:

0. **Freeze the baseline first.** Snapshot today's fantasy outputs and preserve
   the current list as `FANTASY_FEATURES_V1`, then run permutation importance
   *on the fantasy model* (the existing table covers the value model). Nothing
   is removed until there's a measured comparison; the front-office chain is
   untouched throughout (§1B.6b).
1. Replace redundant raw counts with **share-normalized opportunity**
   (`target_share`, `air_yards_share`/`wopr`).
2. **Head-to-head the `value_epa` family against PPR history** — keep the winner,
   retire the loser.
3. Add **vacated opportunity + incoming draft capital** (the offseason signal a
   season-ahead model is blind to).
4. Add **red-zone opportunity** from play-by-play.
5. Make **age position-aware**.
6. Drop everything that fails the 0.2% bar.

The plausible outcome is a model with roughly 10–15 well-motivated features that
is *more* accurate than the current 44 and fully explainable — which is a much
stronger portfolio artifact than what exists now.

## 2. User experience

### 2.1 Navigation is better than it feels

The sidebar has only five entries — Home, Draft Board, Draft Room, Player
Detail, Methodology & Research — which is a reasonable information architecture.
**The complexity is not the nav; it's inside the pages.** `streamlit_app.py` is
**1,937 lines**, and the density of explanatory panels, embedded write-ups, and
caveats within a single page is what makes it feel heavy.

### 2.2 The QB causal study does not belong in the product

Agreed, and the repo's own write-up makes the argument: the study is described
as *"suggestive / underpowered,"* ~104 events, sitting near the 5% significance
border, with a possible pre-trend at offset −3.

That is an honest research finding. It is not a feature. A user trying to draft
a team has no action to take from it, and its presence dilutes the tool's
purpose while inviting exactly the "this is underpowered" critique on the piece
of the project a reader is least likely to need.

**Recommendation:** keep the study as a written report in `report/causal/` and
link to it from a research index (or your future portfolio landing page). Remove
it from the app's navigation. This is a *strengthening* move — the research
still exists and still demonstrates causal-inference skill, but the product
stays a draft tool.

### 2.3 The deeper product question

The app currently serves two masters: a **fantasy draft tool** (Draft Board,
Draft Room, Player Detail) and a **research portfolio** (Methodology, causal
study, benchmark pages). Those have different audiences and different success
criteria.

The cleanest fix is to separate them: the app is the tool; the research lives in
a linked write-up. Given you now own `kylelevesque.me` with a GitHub Pages root,
a research index there is the natural home — and it makes both halves better,
since the tool stops apologizing for itself and the research gets room to be
read properly.

---

## 3. The talent / peer-opinion feature

**The instinct is sound and worth pursuing.** Production stats measure *what
happened*, which conflates talent with opportunity, scheme, and teammate
quality. A player-voted or scout-derived rating is a genuinely different
information source, and the hypothesis — that peers see things box scores don't
— is testable rather than hand-wavy.

### 3.1 The problem with the NFL Top 100 specifically

- **Coverage.** 100 players a year, of whom maybe 50–60 are QB/RB/WR/TE. Against
  ~2,400 validation rows, the feature is missing for the overwhelming majority
  and effectively becomes a sparse "is elite" indicator.
- **It may be redundant.** The players most likely to make the list are the ones
  whose production already screams elite. The feature has to add signal *beyond*
  `value_epa_total`, which is a high bar.
- **Timing is actually fine** — the list is published in the summer, reflecting
  the prior season, so using it to predict the upcoming season is **not
  leakage**. That's an important point in its favor, and worth verifying per
  source before use.

### 3.2 Better-covered alternatives

| source | coverage | cost | notes |
|--------|----------|------|-------|
| **Pro Bowl / All-Pro selections** | all positions, decades of history | free (nflverse/PFR) | also peer/coach/media-voted; the natural first test |
| **Madden ratings** | *every* player, updated in-season | scrapeable | EA employs raters; a real scouted-talent proxy with no coverage gap |
| NFL Top 100 | ~100/yr | scrapeable (Wikipedia) | purely player-voted, the cleanest "peer opinion" but sparsest |
| PFF grades | comprehensive, film-based | paid | the industry standard; likely out of budget |

**Recommendation:** test **Pro Bowl/All-Pro history first** (free, deep history,
solves the coverage problem), then **Madden ratings** as the richer continuous
signal. Use the Top 100 as a third, narrower test — its value is more
rhetorical (a clean "what do players think?" story) than statistical.

### 3.3 How to do it so it's a contribution, not another feature

Frame it as the project's stated question: **does peer/expert opinion carry
predictive signal beyond production statistics?** Then:

1. Fit the simplified baseline from §1.1.
2. Add the talent feature. Measure against the same ~0.2% ablation bar.
3. Report the result either way — and **a negative result is a genuinely good
   outcome here**, because "peer reputation adds nothing once you control for
   production" is an interesting, defensible finding that directly answers the
   "analytics doesn't tell the whole story" critique with evidence.
4. Look specifically at *where* it helps: it may add nothing on average but
   matter for players with **low volume and high talent** (injury-shortened
   seasons, rookies, new starters) — exactly the cases where production stats
   are thinnest. That conditional analysis is the interesting version.

This mirrors the injury-features study that already exists, which is the
project's best piece of work.

---

## 4. Suggested order of work

1. **Simplify the feature set** and ablate against the 44 (highest value, likely
   improves or holds accuracy, transforms interpretability).
2. **Remove the causal study from the app**; keep it as a linked report.
3. **Add the talent feature** as a designed experiment on top of the simplified
   baseline.
4. **Split tool from research** — app stays the draft tool, research moves to a
   portfolio page.
5. *(Optional, later)* rebuild the front end off Streamlit, reusing the
   FastAPI + vanilla-JS pattern from the plant-ID project.
