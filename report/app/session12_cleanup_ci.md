# Stage 12 — Cleanup, CI, and report prose pass

The goal here was narrow: make the repo something a hiring manager can open
without tripping over stale claims or dead artifacts. No model logic changed.

## Dead code / artifacts

The heavy lifting was already done by an earlier cleanup pass, so this was mostly
verification:

- The modules named in the original cleanup plan — `src/advanced_modeling.py`,
  `src/position_model_comparison.py` — are already gone; their retired outputs live
  under `archive/` (which has its own README). `mlruns/` is archived, not at the
  repo root. There are no `outputs/_*.txt` scratch files.
- `git ls-files` shows **zero tracked junk** — no `.DS_Store`, `__pycache__`,
  `.pyc`, or `mlruns` committed. The `.gitignore` already allowlists exactly the
  output tables that should be in version control.

So nothing needed to be deleted or newly archived. The one genuinely dead code
path that remains is the unrouted legacy app functions (`overview_page`,
`front_office_page`, etc.) inside `app/streamlit_app.py` — they are not reachable
from `main()`. They were left in place rather than carved out of a 3,000-line file
(per the "flag, don't delete when uncertain" guardrail), but the one stale
`inflated_apy` caption they contained was corrected. Removing them cleanly is a
small, self-contained follow-up.

## The salary-data inconsistency (fixed)

The most important catch: the committed `salary_efficiency_2016_2025.csv` predated
Stage 4 — its `salary_source` was still `inflated_apy`, so the app's surplus and
player-detail pages were displaying the *old* flat-APY numbers while the code and
captions said "reconstructed cap hit." Regenerating the salary outputs
(`build_salary_efficiency_tables` + `build_salary_finding_tables` — deterministic,
no model training) brought the committed data back in line with the Stage 4 code:
`salary_source = contract_terms_curve`, and the surplus board leads with **Brock
Purdy 2023 at $35.4M**, the value documented in the Stage 4 report. This regenerated
the salary-efficiency and replacement-level output tables.

## CI

Added `.github/workflows/tests.yml`: on push and PR, set up Python 3.10, install
`requirements.txt`, byte-compile `src/`, `app/`, and `scripts/`, then run the
**data-independent test suite** (landing + detail-page content, player
search/index/assembly, and the synthetic PBP leakage checks — 24 tests). The raw
nflverse data under `data/raw/` is intentionally not committed, so the data-backed
tests can't run on a clean checkout; the workflow comment says so explicitly rather
than pretending otherwise. This keeps CI green and honest instead of red and
confusing.

## Roadmap & report prose

- **`PORTFOLIO_ROADMAP.md`** rewritten from a pre-work plan with stale checkboxes
  into a finished build log: each stage 1-12 summarized as one unified product,
  with Stage 11 (mobile/screenshots) flagged as the only remaining cosmetic item
  and the paid-data / DFS-optimizer items listed as intentional scope boundaries.
- **Stale `inflated_apy` wording corrected** in `README.md` (two spots),
  `data/README.md`, `report/salary_efficiency_findings.md`, `report/salary_efficiency_summary.md`,
  `report/final_project_report.md`, and the unrouted app caption — all now describe
  the reconstructed season cap hit with its quality flag and the "estimate, not exact
  cap accounting" caveat. The remaining `inflated_apy` mentions in the repo are in
  the Stage 4 and Stage 9 reports, where they correctly describe what was
  *replaced*.

## Consistency check

Searched the repo for `inflated_apy`, `beat DraftKings`, `beats the market`, `real
cap hit`, and `proves`. After the fixes, the only surviving matches are in correct
context: the negative-result and Stage-4/9 reports describing what changed, and
the weekly summary's explicit *negation* ("does not claim it beats DraftKings in
recent years"). No standalone overclaims remain. The causal result is consistently
described as suggestive/underpowered; the DraftKings benchmark is consistently
scoped to 2020-2021.

## Tests run

The full suite was run in chunks (the sandbox caps single commands below the time
the data-backed tests need together): **135 tests, all passing.** Breakdown — app /
pure / cap-hit / external / weekly (45), first-report + two-stage (12), rookie Bayes
+ rookie context (12), live projection (8), and causal / config / leakage / benchmark
/ replacement / decomposition / ensemble (58). CI itself runs only the 24
data-independent tests, which is the subset that can pass on a clean GitHub checkout.

## Remaining (Stage 11)

Mobile-responsive pass on the `st.columns` KPI layouts and README screenshots / a
search-to-detail GIF. Purely cosmetic — no result or data depends on it.
