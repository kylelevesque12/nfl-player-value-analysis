# Stage 4 — Reconstructing season cap hits from contract terms

## Why flat APY wasn't good enough

The salary track measures whether a player out-earns his contract relative to a
freely available replacement. The cost side of that comparison was
`inflated_apy` — the inflation-adjusted *average* annual value of a deal. APY is
a single number stretched flat across every year of a contract, and that is not
how the salary cap actually works. Because signing bonuses prorate and base
salaries are typically backloaded, the early years of a big extension cost far
less against the cap than the late years. Charging a star his full APY in year 1
overstates what the team actually paid, and therefore understates how much
surplus the team captured early in the deal. The honest fix is a *season-specific*
cap hit, which is what this stage builds.

## What the data does and doesn't support

`historical_contracts.csv` (nflverse / OverTheCap) is contract-level, not
cap-accounting-level. For each deal it has total `value`, `guaranteed`, `apy`,
`years`, and `year_signed` (plus inflation-adjusted versions), but **no
year-by-year base salary or signing-bonus schedule**. So a true cap hit can't be
parsed — there is no ground-truth per-year breakdown in the file. This is explicit
throughout: every reconstructed number is an *estimate from contract terms*,
flagged as such, never presented as a parsed cap figure.

## The reconstruction

For each season a player is under contract, the cap hit for that contract-year is
computed as:

> **cap hit(year k) = prorated signing bonus + backloaded base salary**

with two transparent assumptions, both documented in
`src/cap_hit_reconstruction.py`:

1. **Signing-bonus proxy.** The guaranteed money is treated as signing-bonus-like
   and prorated evenly over `min(years, 5)` — the NFL's five-year proration cap.
2. **Backloaded base.** The remaining `value − guaranteed` is spread across the
   contract years on a gently rising schedule (each year's base runs from 0.65×
   to 1.35× the average, mean 1.0).

The construction has a nice accounting property: the per-year cap hits **sum to
the contract's total value**, so no money is invented or lost. Everything is kept
in the same inflation-adjusted millions as the old `inflated_apy`, so surplus
numbers stay comparable across eras. Where contract terms are too thin to run the
curve, the code falls back to flat APY (flagged `fallback_apy`); where no contract
matches a player-season, the row is flagged `missing_contract`.

A couple of worked curves to show it behaves:

| Contract | Old (flat APY) | Reconstructed curve |
|---|---|---|
| Joe Burrow 2023 ext. (5yr, heavily guaranteed) | 73.7 every year | 61.6 → 67.7 → 73.7 (rising) |
| Brock Purdy rookie deal (4yr, ~$0.1M gtd) | 1.35 every year | 0.89 → 1.20 → 1.51 (rising) |

Fully-guaranteed rookie deals for top picks come out nearly flat, which is
correct — those contracts genuinely are.

## Coverage and fallback rates

Because the inflation-adjusted `value`/`guaranteed` columns are fully populated,
the curve runs essentially everywhere. Across all skill-position player-seasons
2016–2025 the reconstruction produces 10,513 rows, of which exactly **one** is
`missing_contract` and **zero** are `fallback_apy`. Within the surplus finding
base (8+ games played), **all 3,531 player-seasons are
`estimated_from_contract_terms`** — no fallbacks at all. The join is one row per
player-season and does not increase the finding-base row count (3,531 → 3,531).

## Before / after surplus

Replacing flat APY with the reconstructed cap hit and re-running the full
replacement-level surplus pipeline, the biggest movements are exactly where the
theory predicts — highly paid veterans in the **early years of mega-extensions**,
whose flat APY badly overstated their year-1–2 cap hits:

| Player | Season | Old salary | New cap hit | Old surplus | New surplus | Δ |
|---|---|---:|---:|---:|---:|---:|
| Jalen Hurts | 2023 | 68.3 | 54.7 | −42.8 | −30.7 | **+12.1** |
| Derek Carr | 2017 | 45.1 | 34.4 | −36.8 | −24.8 | +12.0 |
| Davante Adams | 2022 | 40.5 | 28.6 | −34.6 | −22.9 | +11.7 |
| Dak Prescott | 2024 | 70.8 | 59.3 | −68.0 | −56.7 | +11.3 |
| Joe Burrow | 2023 | 73.7 | 61.6 | −63.2 | −51.9 | +11.3 |
| Jordan Love | 2024 | 64.9 | 52.6 | −42.1 | −30.9 | +11.2 |

None of these flip to positive surplus — they were genuinely expensive — but they
look meaningfully *less* overpaid once the year-1 cap hit reflects bonus
proration. That is the credible correction a real cap analyst would expect.

## Brock Purdy

| Season | Old salary | New cap hit | Old surplus | New surplus |
|---|---:|---:|---:|---:|
| 2022 (rookie yr 1) | 1.35 | 0.89 | — | — |
| **2023 (rookie yr 2)** | **1.35** | **1.20** | **37.4** | **35.4** |
| 2024 (rookie yr 3) | 1.35 | 1.51 | 29.3 | 27.7 |
| 2025 (extension yr 1) | 57.2 | 44.7 | — | — |

Honesty note: the original roadmap *guessed* Purdy's 2023 surplus would rise once
real cap hits were used. It doesn't — it dips slightly (37.4 → 35.4). His rookie
cap hit barely changed (the deal was already tiny and near-flat), but compressing
the *top* of the QB market lowered the position-season price-per-value slope used
to dollarize his production, which trims his dollar surplus a hair. He remains the
single largest surplus player-season in the dataset either way. The number here is
what the pipeline actually produced rather than the hoped-for direction, and Purdy
is not hard-coded anywhere — the same curve runs for every contract.

## Decision

**Reconstructed cap hit replaces `inflated_apy` as the production salary
variable.** `salary_efficiency.expand_contracts_to_player_seasons` now sets
`salary_millions` to the season-specific estimate, and `replacement_level.py`
inherits it automatically. The old flat value is retained alongside as
`inflated_apy_salary` for comparison, and every player-season carries
`cap_hit_source` / `cap_hit_quality_flag` so a reader can see exactly how each
number was produced. The surplus framework no longer has to caveat that its cost
variable is an annual average rather than a cap hit — it's now a documented
season-specific estimate, which is the credibility upgrade this track needed.

The honest limitation remains stated plainly: with no year-by-year cap accounting
in the source data, this is a principled approximation, not a parsed cap hit.

## Reproduce

```
python -m scripts.eval_session4_cap_hits
```
