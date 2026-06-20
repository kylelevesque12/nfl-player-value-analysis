# Stage 9 — Detail-page component migration

## What this stage did

The detail pages had drifted apart. Some led with a title and a wall of
`st.dataframe`s, some had KPI rows, some had executive context and some didn't —
and worse, a few were still describing pre-Stage-4/5 behavior (the surplus page
called the cost variable `inflated_apy`, the causal page showed the old Out-only
null, the benchmark page led with a "beat the market" claim already scoped
down). This stage pulls every routed detail page onto one structure and brings
the copy up to the current, committed results.

Each migrated page now follows the same shape: **title + one-sentence purpose →
executive summary → KPI row → one main visual → a visible caveat callout →
detail expanders → a source footer.**

## Pages migrated

| Page | Function | Key content fix |
|---|---|---|
| Replacement-level surplus | `replacement_level_page` | Caveat corrected from `inflated_apy` to the Stage-4 reconstructed cap hit; four bare tables moved into expanders |
| Cap Allocation Brief (hero) | `front_office_executive_report` | Stale `inflated_apy` caveat replaced with the reconstructed-estimate framing |
| External benchmark | `external_benchmark_page` | Reframed from "beat the market" to the scoped framing — baseline skill (all years) primary, DK 2020-2021 secondary |
| Causal QB injury | `causal_qb_injury_page` | Rewired to the Stage 5 first-report result (104 events vs 19 Out-only, ATT ≈ −0.58 PPG, suggestive/underpowered) |
| Bayesian rookie | `rookie_bayes_page` | Added the Stage 3 incumbent-context lead and the Jordan Love 0.611 → 0.513 callout |
| Two-stage decomposition | `two_stage_weekly_page` | Standardized; negative-result framing kept; three bare tables moved into expanders |
| Methodology checks | `methodology_page` | Reorganized around trust signals (leakage-safe features, time-based validation, documented negatives, quality flags) |

## Components reused and added

Reused the existing system wherever possible: `executive_summary`,
`recommendation_callout`, `card_row` (the app's KPI-tile helper, equivalent to
`kpi_grid`), and the existing card/callout CSS.

Added three small, generic helpers to `app/components.py` (thin wrappers, no new
styling): `page_header(title, purpose)`, `caveat_callout(body, label)` (a
`recommendation_callout` of category "caveat"), `source_footer(text)`, and
`render_page_scaffold(content)` which renders the shared header + executive
summary from a config entry.

The page copy itself lives in a new **`app/page_content.py`** — a pure,
Streamlit-free config (title, purpose, summary bullets, caveat, footer per page).
This keeps the wording consistent and, importantly, unit-testable without a
Streamlit runtime, the same pattern Stage 8 used for the landing content.

## Old-style elements removed or moved

No useful tables were deleted — long/raw dataframes were moved into
`st.expander(...)` so the top of each page is a summary, not a data dump.
Examples: the surplus page's replacement-baselines, by-position, team-season, and
full top-surplus tables; the benchmark by-position/by-season tables; the causal
eligibility and coefficient tables (plus a new "Earlier Out-only analysis"
expander preserving the Stages 1-2 context); the rookie rolling-validation
table; the two-stage method/fold/per-stage tables; and the methodology full table
and report text.

## Caveats preserved (and corrected)

Every page keeps a visible caveat callout — these were strengthened, not hidden:

- **Surplus / Cap Allocation**: cap hit is a reconstructed estimate from contract
  terms, not exact NFL cap accounting; each row carries a quality flag.
- **Benchmark**: the DraftKings comparison is limited to 2020-2021 matched
  player-weeks and is a market-implied proxy fit on in-season actuals.
- **Causal**: suggestive and underpowered (~104 events, p ≈ 0.04); the
  fixed-effect parallel-trends test passes but the −3 cell-mean gap is slightly
  elevated.
- **Rookie**: combine and broad-depth features were tested and dropped; only a
  3-feature incumbent core was kept; the QB AUC gain is small.
- **Two-stage**: documented negative result, nothing in production.
- **Methodology**: includes the Stage 2 NGS/PFR availability-leakage rejection
  and the test/quality-flag disciplines.

The live-projection caveats (carry-forward player state, static schedule/weather)
already live on the Fantasy Player Board from Stage 7 and were left in place.

## Tests

`tests/test_page_content.py` (7 tests) pins: every detail page has the required
metadata; each required caveat token is present; the surplus page no longer says
`inflated_apy`; the causal page reflects first-report (19 vs 104, suggestive /
underpowered); each migrated page is wired to the config with a caveat + footer;
the old "beat the market" framing is gone; and the landing-page routing targets
are unchanged. The Stage 8 landing tests still pass (12 app tests total; 125
tests collected overall).

## Notes and what's left

- Navigation was not touched — the existing hero/drill-down radios still drive
  everything; the migration is content/layout only.
- Two unrouted legacy pages (`overview_page`, `predictions_page`, etc.) still
  contain old `inflated_apy` copy. They are not in the navigation, so they are
  effectively dead; cleaning or deleting them belongs to the Stage 12 cleanup
  pass, not here.
- **For Stage 10 (global player search):** the page scaffold and the
  `page_content` config make it straightforward to add a player-detail page in
  the same style. A search box in the sidebar plus a `player` page key in the
  config is the natural next step; nothing in this migration blocks it.
