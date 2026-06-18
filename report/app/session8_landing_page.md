# Session 8 — App landing page

## What changed

The app used to open straight on the Cap Allocation Brief — fine for someone who
already knows the project, a confusing first screen otherwise. This session
replaces that default with a real landing page that says what the project is and
offers four obvious doors into it.

The landing page has four parts:

- **Hero** — the title ("NFL Player Value & Fantasy Projection Lab") and a
  one-sentence subtitle describing the four research threads.
- **Four findings cards** in a 2×2 grid, each with a headline, three bullet points,
  and a button that navigates straight to the relevant view.
- **A methodology strip** — five short labels summarizing how the work was done
  (leakage-safe features, time-based validation, negative results documented,
  source/quality flags on salary estimates, tests covering modeling assumptions).
- **A "How to use this app" expander** with a short orientation for each page.

The page is wired as the default: "Home (Landing)" is now the first option in the
hero navigation radio, so it's what loads on startup. The existing Cap Allocation
and Fantasy Player Board pages are unchanged and still selectable.

## Which findings are featured

Each card surfaces one session's headline result, in the project's own honest
voice — including the experiments that didn't make production:

| Card | Headline | Key points |
|---|---|---|
| Fantasy forecasting | Live weekly projections from leakage-safe role features | weekly RMSE reduced 1.27% (6.020 → 5.944) from PBP/weather; live upcoming-week scoring; per-position conformal improves QB coverage |
| Player value / cap surplus | Surplus from reconstructed cap-hit estimates | flat APY → season-specific cap hits; Purdy still top surplus; early-extension stars treated realistically |
| Rookie opportunity model | Rookie QB opportunity depends on incumbent context | combine tested-but-dropped; incumbent core kept; Jordan Love P(plays) 0.611 → 0.513 |
| QB injury causal study | First injury-report appearance matters before formal absence | Out-only found little; first-report grew events 19 → 104; ATT ≈ −0.58 PPG, suggestive/underpowered |

## What each card links to

Navigation reuses the app's existing sidebar-radio pattern — no new routing
framework. A card button stashes its target in `st.session_state["_landing_goto"]`
and reruns; `_handle_landing_nav()` (called at the top of `main()`, before the
radios are instantiated) translates that into the right radio selection. This
avoids the "can't modify a widget's state after it's created" trap.

- **Open Fantasy Player Board** → hero radio → `espn_fantasy_view`
- **Open Cap Allocation Brief** → hero radio → `front_office_executive_report`
- **Open Rookie Model** → drill-down radio → `rookie_bayes_page`
- **Open Causal Study** → drill-down radio → `causal_qb_injury_page`

A test asserts every card target string actually matches a radio option the app
uses, so a future rename can't silently break a button.

## Design notes

The pure content and navigation config (card copy, methodology labels, nav target
constants) live in a small Streamlit-free module, `app/landing_content.py`, so
they're unit-testable without a Streamlit runtime and trivial to fold into the
component system in Session 9. The render function reuses the app's native
primitives (`st.columns`, bordered `st.container`, `st.button`, `st.expander`) and
the already-injected component CSS. No bare dataframes, no recomputation — the
headline numbers are static, pulled from the prior sessions' reported results, so
the page loads instantly.

## What's left for Session 9

This session deliberately keeps the cards as simple local render code rather than
formal components. Session 9 (component migration) should:

- migrate the four cards onto a shared `player_card`/`kpi_grid`-style component so
  the landing page and detail pages share one card system;
- consider a small KPI strip at the top of the hero using `kpi_grid`;
- carry the same visual treatment into the detail pages that still use older
  layouts.

The `app/landing_content.py` split is designed to make that migration a
copy-of-config rather than a rewrite.

## Scope

Only the landing page and its navigation wiring were touched: `app/streamlit_app.py`
(landing render + radio option + dispatch + nav handler), the new
`app/landing_content.py`, and a test. No modeling logic, feature lists, salary or
causal calculations, live-projection scoring, README, or other pages were changed.
