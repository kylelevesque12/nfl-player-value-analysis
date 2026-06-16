# Session 10 — Global player search & unified player detail

## What was added

A player has always been spread across the app: their weekly projection lives on
one page, their cap surplus on another, their rookie record on a third. This
session ties them together. There's now an **always-visible sidebar search** — type
a name, get a typeahead list of `Name — POS — TEAM` labels — and selecting one
opens a single **Player Detail** page that pulls every project output for that
player into one view.

The search and the detail assembly live in a new pure module,
`app/player_search.py`, with no Streamlit and no file I/O of its own — it operates
on the DataFrames the app already caches via `load_all_data`, so it's fast and
unit-testable without a Streamlit runtime.

## The player index

`build_player_index(...)` builds one row per stable `player_id` (nflverse gsis
id) from the already-loaded outputs. For each player it resolves the most recent
name / position / team (preferring the weekly backtest, then salary, then the
rookie frame), lists the seasons they appear in, and sets a boolean flag for each
module that has data:

| Flag | Source |
|---|---|
| `has_weekly` | `weekly_fantasy_validation_predictions.csv` |
| `has_live` | `weekly_fantasy_live_projection.csv` |
| `has_surplus` | `salary_efficiency_2016_2025.csv` |
| `has_rookie` | `rookie_modeling_frame.csv` |
| `has_causal` | `causal_s3_first_report_events.csv` (treated QB, matched on `qb_id`) |

On the real data the index covers **2,721 players** with unique ids. Search is a
case-insensitive substring match on the name; the index is keyed and selected on
`player_id`, never on the name string, so two players who share a name still
resolve to distinct, stable ids.

## What the detail page shows

Built in the Session 9 shared style (header → KPI row → sections → caveats →
expanders → footer):

- **KPI row** — latest projected PPR (live week if available, else most recent
  backtest game), latest PBP depth rank, best value-over-expected season, and the
  count of first-report causal events where the player was the treated QB.
- **Weekly fantasy** — projected-vs-actual line by game, the upcoming-week live
  projection with its 80% interval when present, and a projection table in an
  expander.
- **Value & cap surplus** — per-season cap hit (reconstructed), value score, and
  efficiency tier, with the standing caveat that the cap hit is a reconstructed
  estimate, not exact NFL cap accounting.
- **Rookie model** — rookie year, draft pick, whether they played meaningfully,
  and the Bayesian projection if the player is in a validation class; a note that
  the Session 3 incumbent-context core is what sharpened the QB gate.
- **Causal study** — the player's first-injury-report events as a treated QB, with
  the "suggestive / underpowered" caveat. It explicitly does **not** force a causal
  interpretation onto non-QBs.

## Missing-data behavior

Every section degrades to a clean "not available" message instead of crashing.
`assemble_player_detail` returns `None` for any module the player has no data in,
and the page renders an `st.info(...)` for those. This is covered by tests for the
four archetypes: an active WR with weekly+live+surplus but no rookie/causal row; a
QB with rookie+causal but no live projection; a fully-missing id (every section
`None`); and the empty-index case.

## Data-join discipline

No models are recomputed in the app — everything is a filter over saved outputs.
The weekly history is collapsed to the production model method so there's one row
per (player, season, week); the surplus history is de-duplicated to one row per
(player, season). Both are pinned by tests. The one name-keyed join — the top-25
replacement-surplus board, whose saved table lacks `player_id` — is matched on the
display name and used only for a supplementary "appears in the top-25" note; the
authoritative per-season surplus comes from the `player_id`-keyed salary table.

## Navigation

Reuses the Session 8 deferred-nav pattern exactly: selecting a player sets
`_selected_player_id` and defers a hop to the new `Player Detail` hero option
(handled at the top of `main()` before the radios instantiate). The existing
hero/drill-down radios are otherwise untouched, and a "← Back to dashboard" button
returns to the landing page. No new routing framework.

## Tests

`tests/test_player_search.py` (10 tests): unique index keys + correct module
flags; case-insensitive substring search; expected display labels; stable-id
resolution on selection; missing-section handling across archetypes; no duplicate
player-week rows in the weekly summary; no duplicate player-season rows in the
surplus summary; causal section keyed on `qb_id`; nav labels still wired; and the
empty-index no-crash case. Page-content and landing tests still pass (22 app tests;
135 collected overall).

## What remains for Sessions 11/12

- **Session 11 (mobile/README):** the detail page uses `st.columns` KPI rows that
  will need the same narrow-viewport pass as the rest of the app; a recorded GIF of
  search → detail would be a natural README addition.
- **Session 12 (cleanup):** the top-25 surplus board is the last name-keyed join;
  if a `player_id` is added to that saved table during cleanup, the supplementary
  surplus note can switch to an id join. The unrouted legacy pages noted in Session
  9 also remain for removal.
