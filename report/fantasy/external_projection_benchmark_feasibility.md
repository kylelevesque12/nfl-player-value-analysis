# External projection benchmark — feasibility spike

A short, read-only spike to answer one question: can the weekly fantasy model be
fairly benchmarked against a top external projection source like FantasyPros or
ESPN? No production code was changed; this is just a look at what's available and
what it would take.

## What's already in the repo

The project already runs an external benchmark, but the external source is **not**
FantasyPros or ESPN — it's a **DraftKings-implied** projection. `data/raw/external_projections.csv`
holds 38,262 player-weeks (2016–2021, weeks 1–18) with the source tag
`draftkings_implied_via_rotoguru`. It's built by
`scripts/build_external_projections_from_dk.py`: RotoGuru's free DK salary archive
is fuzzy-matched to nflverse `gsis_id`, then a per-(season, position) regression of
actual PPR on DK salary recovers the salary→points conversion DK is implicitly
using. The fitted value is the market-implied projection.

The evaluation lives in `src/external_benchmark.py`. It joins any projection file
matching the schema to the model's backtest predictions
(`weekly_fantasy_validation_predictions.csv`) on `(season, week, player_id)`,
keeps only player-weeks where both sides have a value, and reports RMSE, MAE, and
Spearman overall and by position. The important thing for this spike: **that
harness is source-agnostic.** It loads *every* CSV under `data/raw/` that has the
required columns, so a FantasyPros or ESPN file would be benchmarked with zero
code change.

There is even a template for exactly that: `data/raw/external_projections.example.csv`
carries two rows tagged `fantasypros_consensus` in the right schema — someone
already anticipated the FantasyPros drop-in. But it's only an example; **no real
FantasyPros or ESPN projection rows exist locally.**

## Is the current benchmark fair, and on what rows?

The DK-implied benchmark is fair in the sense that matters — it joins on stable
`gsis_id`, evaluates only player-weeks where both the model and the external source
project a value, and uses the same realized PPR target. Concretely, the model
backtest covers 2020–2025 and the DK file covers 2016–2021, so the head-to-head
sample is the **2020–2021 overlap: 11,191 matched player-weeks at a 98.3% match
rate** (high, because both sides already speak `gsis_id`). On that sample the model
posts RMSE 6.386 vs the DK-implied 6.493 — a ~1.6% edge, and a deliberately tough
one, since the DK regression is fit on the season's *actual* points and therefore
has information a live DK projection wouldn't.

Two honest limits: the DK source is a market *proxy*, not a published projection,
and its coverage stops at 2021, so the benchmark says nothing about 2022–2025.

## FantasyPros / ESPN: what acquiring it would require

Because no real FantasyPros/ESPN data is local, and per the spike's guardrails, no
scraper was built. Here's what a real benchmark against them would take.

- **Source.** FantasyPros weekly consensus projections (or ECR), or ESPN weekly
  projections.
- **Access method.** FantasyPros exposes current-week consensus on the public site,
  but a clean *historical* weekly archive sits behind their paid API / MVP tier.
  ESPN does not publish an official historical projections API; community mirrors
  exist but their stability and licensing are uncertain. Scraping live pages is
  brittle and only captures weeks going forward, not history.
- **Historical weekly availability.** Not freely. FantasyPros historical needs paid
  API access; ESPN historical isn't officially available. The free path is
  *prospective*: snapshot each week's projections as the season runs and build the
  archive over time.
- **Paid/API access.** Yes for FantasyPros historical depth.
- **Likely join keys.** `(season, week, gsis_id)`, same as today. The catch: both
  sources key on their own player IDs or names, so a name+team+position →
  `gsis_id` bridge would be needed (the repo already does this for the DK file),
  which introduces match-rate and selection risk.
- **Fairness risks.** (1) The projection must be the *pregame* value, not a
  post-game revision — otherwise it's not a fair forecast. (2) Scoring must be exact
  PPR to match the target. (3) Fuzzy name matching can drop or duplicate players and
  bias which player-weeks end up in the matched sample. (4) FantasyPros consensus
  aggregates many experts and is a genuinely strong bar; the matched universe would
  need to be confirmed as the same player-weeks the model is scored on, not a
  cherry-picked subset.

## Recommendation

**Keep DraftKings-implied as the primary benchmark now, and acquire historical
FantasyPros/ESPN projections later — don't scrape in this spike.** The reasoning:

- The matched-row evaluation machinery already exists and is source-agnostic, so
  adding FantasyPros/ESPN is a *data* problem, not a *code* problem. When a properly
  sourced file lands in `data/raw/` matching `external_projections.example.csv`,
  `external_benchmark.py` will evaluate it head-to-head with no changes.
- The honest gap in the current benchmark is *coverage* (ends 2021) more than
  *source*. A FantasyPros archive for 2022–2025 would be the higher-value
  acquisition because it both modernizes the comparison and adds a published-
  projection source.
- Until that data is acquired through a fair, stable channel (paid API or
  prospective weekly snapshots), forcing a brittle scrape would risk an unfair or
  unstable benchmark — worse than the clean DK one already in place.

No build was done in this spike beyond this note. The concrete next step, when
prioritized, is to obtain a historical FantasyPros consensus file (paid API or
accumulated weekly snapshots), map it to `gsis_id`, drop it in as
`data/raw/external_projections_fantasypros.csv`, and re-run the existing
`external_benchmark` step.
