# App redesign — goal backlog

Run `/goal next` to dispatch the first unchecked item in the Queue. The
session cannot end until the test suite is green, the app-surface gate
passes, and Codex returns `VERDICT: SHIP` against the goal text.

## The decision this backlog implements

The app and the repo have been serving two different audiences out of one
set of pages, and it shows. A league-mate wants a draft board; a hiring
manager wants the causal study and the ablation write-ups. Trying to give
both of them the same navigation produced a 1,937-line `streamlit_app.py`
where the product is buried inside research tabs.

So they split:

- **The app is a fantasy football product.** It speaks plain fantasy
  language, it has no self-referential sections about its own methodology,
  and someone who has never heard of this project can draft with it.
- **The repo is the research portfolio.** `report/`, `notebooks/`, the
  modelling code in `src/`, and the pipeline that produces the tables all
  stay exactly where they are. The research is not being deleted. It is
  being *unhooked from the product's navigation* and left to live in the
  repository, where the audience that wants it already is.

One link in the app footer points at the repo. That is the entire remaining
connection between the two.

## What must survive the redesign

There is a specific way this goes wrong, and it is worth naming before any
work starts: a "simplify the interface" pass strips the honesty along with
the jargon.

Prediction intervals, tier groupings, the stable/shaky role badge, the
injury-return flag, and the caveat callouts are **product**, not research.
They are the differentiator against ESPN — the whole pitch is that this tool
tells you what it does not know. Replacing an honest range with a clean
point estimate would make the app look more polished and be worth less. The
app-surface gate checks for each of these by name and fails if one
disappears.

The same goes for accuracy claims kept in the app: any number shown must
still match what the underlying tables support, including its coverage
limits. Better to drop a claim than to round it into something cleaner than
the evidence.

## How these goals are verified

Every goal runs through three gates, in order, as a Stop hook
(`.claude/hooks/verify_goal.sh`):

1. **`pytest -q` green.** The 32 test files stay green and grow with each
   change.
2. **The app-surface gate** (`scripts/goal_check.sh` →
   `scripts/app_surface_check.py`). This project's goals are interface
   goals, so "tests pass" is not a sufficient bar — the deployed app has to
   render. The gate drives every navigation section through Streamlit's
   `AppTest` and requires a clean render, then checks that no research
   surface remains in the navigation, that the research render functions are
   gone from `app/`, that every honesty marker above is still present, and
   that no in-app navigation jump points at a page that no longer exists.
   Run it yourself any time: `.venv/bin/python scripts/app_surface_check.py`.
3. **Codex review** against the goal text, briefed on the four failure modes
   specific to this project: research deleted rather than unhooked, honesty
   stripped along with jargon, ungrounded UI claims, and dead code left
   behind by a removal.

Visual goals additionally require **rendered screenshots** before they are
called done — a page that renders without raising can still be ugly or
unusable, and `AppTest` cannot see that. Check the real app at 1280px and at
390px (phone width) and look at it.

## Queue

- [ ] **2. Split `streamlit_app.py` into one module per page.** It is 1,937
  lines, which is the reason UI work on it is risky — you cannot safely
  redesign a page you cannot find. Move each section into its own module
  under `app/` (home, draft board, draft room, player detail) with the
  shared chrome — CSS, layout, formatting helpers, data loading — factored
  out. **No behaviour change whatsoever**: this goal is pure restructuring,
  and the app-surface gate plus the existing `AppTest` tests are what pin
  that. Two known traps to fix while in here: `NAV_PLAYER` is currently
  defined twice, at `app/streamlit_app.py:187` and
  `app/landing_content.py:23`, and the `sys.modules` purge of `app.*` at the
  top of the entry script must be preserved exactly — it is what keeps
  Streamlit Cloud's hot-reload from serving new code against stale cached
  modules, it has taken the deploy down twice, and `tests/test_stale_module_reload.py`
  guards it.

- [ ] **3. Build the visual foundation.** One coherent design language
  applied everywhere instead of per-page styling: a type scale, a spacing
  scale, the color roles (including what a positive/negative/uncertain value
  looks like), and a small set of reusable components — player row, stat
  card, tier badge, range bar, section header. Replace the ad-hoc CSS in
  `inject_custom_css` / `inject_theme_css` with this. The test of success is
  that a new page can be built from the components without writing new CSS.
  Screenshots required.

- [ ] **4. Redesign Home.** This is the page a league-mate lands on with no
  context, so it has about five seconds to say what the tool is and get them
  to the board. Lead with the draft-day action, not with a description of
  the project. Screenshots required.

- [ ] **5. Redesign the Draft Board.** The core product surface: a ranked
  board someone can actually draft from. Ranges and tiers have to read
  clearly at a glance rather than hiding in a wide table — this is where the
  honest-uncertainty content either lands or fails to. Screenshots required.

- [ ] **6. Redesign Player Detail and the Draft Room shell.** Player Detail
  should answer "should I take this guy" in one screen. The Draft Room's
  *engine* is not in scope here — only its layout and readability, since the
  planner is getting functional work separately. Screenshots required.

- [ ] **7. Make it work on a phone.** People draft from their phones and
  from a laptop on a couch, and Streamlit's default layout does not survive
  a 390px viewport. Every page has to be usable narrow: no horizontal
  scrolling on the board, tap targets big enough to hit, and the Draft
  Room's pick entry reachable without pinch-zooming. Screenshots at 390px
  required.

## Not in the queue yet

Deliberately parked so the redesign stays the focus, but scoped and ready:

- **Sleeper live draft sync.** Sleeper's API is public and needs no auth:
  poll `GET /v1/draft/<draft_id>/picks` and resolve the draft from
  `GET /v1/user/<username>/drafts/nfl/2026`. The identity join was measured
  against the current board on 2026-08-07 — `gsis_id` where Sleeper has it
  (only 960 of 3,042 active skill players do), falling back to
  `src/adp.normalize_name` on name + position, matches **567 of 578 rows,
  98.1%**, with exactly one miss inside the top 150 (Kenneth Gainwell, whom
  Sleeper lists as "Kenny" — a one-line `NAME_ALIASES` entry). Poll with
  `@st.fragment(run_every="3s")`. ESPN and Yahoo are explicitly not planned:
  ESPN needs users to paste browser cookies, Yahoo needs a full OAuth app.
- **Faster manual pick entry**, which is worth more than either of those
  because it works for every platform including an in-person draft: replace
  the select-then-click-Record flow (two interactions, ~170 times a draft)
  with a type-and-Enter box with fuzzy matching.
- **Self-hosting at `nfl.kylelevesque.me`.** The infrastructure already
  exists — the DigitalOcean droplet at 157.230.226.181 runs Caddy for
  `fieldnote.kylelevesque.me`, and `ct-plant-id/deploy/Caddyfile` already
  carries a commented-out `nfl` block pointing at port 8501. Needs a
  Namecheap A record, the block uncommented, a systemd unit, and a serving
  subset of this repo. Costs nothing extra and does not sleep. Measure the
  Streamlit process's memory first: the droplet has 2GB and Fieldnote
  already uses 562MB.

## Done

<!-- Completed goals move here with the date they landed. -->

- [x] **1. Take the research out of the app.** *(2026-08-07)* Navigation went
  from five sections to four — Home, Draft Board, Draft Room, Player Detail.
  Removed the whole `Methodology & Research` section (safeguards audit,
  research-study summaries, sources, project report), the
  `Accuracy & benchmark` tab from the Draft Board, the project-overview
  panel and the "trust signals" strip from Home, and the per-section
  full-write-up expanders. That orphaned two pure-content modules
  (`app/page_content.py`, `app/section_content.py`) whose only job was
  feeding research copy and slicing `PROJECT_REFERENCE.md` into app panels,
  plus three components (`render_page_scaffold`, `executive_summary`,
  `page_header`) — all removed with their tests. `streamlit_app.py` fell
  1,937 → 1,570 lines. The research itself was untouched: the diff is
  confined to `app/` and `tests/`, every pipeline step still builds, and a
  new `test_research_stays_in_the_repository` asserts the study files still
  exist. Kept, deliberately: the plain-language "How these projections are
  built and graded" explainer under the Draft Board, and every honesty
  marker (ranges, tiers, stable/shaky, ⚕, caveat callouts). One real bug
  caught by the loop's own gates — `GITHUB_BLOB_BASE` was defined inside the
  removed block but still used by the Draft Room's report link, so the app
  raised `NameError` on that page until it was restored at module scope.
