# Stage 11 — Responsive pass, README visuals, presentation polish

The final cosmetic stage: make the app behave on small screens and make the
GitHub repo presentable to a hiring manager. No modeling, scoring, or data logic
changed.

## Responsive fixes

The honest starting point: Streamlit already stacks `st.columns` to full width
below its small-screen breakpoint, and Stage 9 had already moved the wide
dataframes into expanders — so the app was mostly mobile-tolerant by default. The
one spot that stayed cramped was the KPI rows, where 4-5 `st.metric` tiles sit in
a single row that's fine on desktop but tight on tablet widths.

I added a small pure helper, `app/layout.py::chunk_metrics`, that splits a metric
list into **balanced** rows of at most three tiles (4 → 2+2, 5 → 3+2, 6 → 3+3),
and routed both the app's `card_row` and the component `kpi_grid` through it. KPI
rows now wrap cleanly at tablet/phone widths instead of crushing four narrow
columns together, while desktop still reads as tidy grids. The helper is
Streamlit-free so it's unit-tested (`tests/test_layout.py`, 4 tests: short rows
stay single, wide rows wrap balanced, nothing is dropped or over-wide, order
preserved).

Everything else (landing 2×2 cards, two-column chart+table layouts, the live
board table, the player-detail sections) relies on Streamlit's native column
stacking and the existing expanders, which handle narrow viewports without code
changes.

A note on testing scope: Streamlit isn't installed in the build sandbox, so I
could not render the app at literal 375/768/1280px to eyeball it. The responsive
work is therefore code-level (balanced wrapping + leaning on Streamlit's native
breakpoints) rather than pixel-verified; a local run is the way to confirm the
final look.

## README visuals and sections

- **CI badge** added at the top, pointing at the `tests` workflow.
- **App previews** under `docs/images/` — two lightweight SVG layout renderings
  (`landing_preview.svg`, `player_detail_preview.svg`, ~5 KB each, vector so they
  stay version-control-friendly). They are faithful to the real page layouts and
  are labeled as previews, not live screenshots — the README says plainly that a
  local run produces the live, interactive version.
- **Interactive dashboard** section rewritten from the stale "v2 rebuild in
  progress / draft layer" copy into an accurate description of the finished app
  (landing page, hero pages, drill-downs, global player search → player detail).
- **Limitations / "What's done and what's left"** rewritten: the stale TODOs that
  are now complete (dashboard rebuild, causal stage 3, depth-chart rank from
  PBP, APY → reconstructed cap hit) were removed; the genuine remaining items
  (paid external projections, true OTC cap data, live screenshots / deploy) are
  listed as optional and non-blocking.

Wording was kept careful throughout, matching the earlier consistency pass:
"reduced weekly RMSE by 1.27% (6.020 → 5.944)", the DraftKings comparison scoped
to its 2020-2021 matched sample, "reconstructed cap-hit estimates" (not "real cap
hits"), and the causal result described as suggestive/underpowered.

## Deployment

Not completed this stage, and no URL was invented. The README's "what's left"
list flags a Streamlit Community Cloud deploy + live screenshots as the final
presentation step; `requirements.txt` is already sufficient for it.

## Final checks

- `tests/test_layout.py` + the CI-targeted data-independent suite: **28 passed**.
- Full suite still collects cleanly: **139 tests** (135 + 4 new), no import errors
  from moving `chunk_metrics` into `app/layout.py`.
- `git status` confirms **no `outputs/` or `data/` files were regenerated** — the
  change set is README, `app/components.py`, `app/streamlit_app.py`, the new
  `app/layout.py`, the two small SVGs under `docs/images/`, and the new test.
- Image sizes are ~5 KB and ~3.7 KB; no binaries or videos committed.

## Remaining optional work

- Capture real pixel screenshots / a search→detail GIF from a local run and swap
  them in for the SVG previews.
- Deploy to Streamlit Community Cloud and add the live URL.

Both are presentation-only and depend on a live environment; neither affects any
result.
