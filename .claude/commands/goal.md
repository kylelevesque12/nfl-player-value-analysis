---
description: Set a verified goal (or "/goal next" to pull the top of GOALS.md) — the session cannot end until tests pass, the app-surface gate passes, AND Codex issues VERDICT SHIP
---
A goal loop is being started. Do these steps in order:

1. Determine the goal text:
   - If the arguments below are exactly `next` (or empty), open `GOALS.md`, take the FIRST unchecked (`- [ ]`) item in the Queue as the goal, and mark it `- [x]`. If there are none, say so and stop — do not invent a goal.
   - Otherwise, the arguments themselves are the goal.
2. Write the goal verbatim to `.claude/current-goal.md` (overwrite if it exists) and delete `.claude/goal-iterations` if present.
3. Work toward the goal autonomously. Follow `GOALS.md` and `README.md` conventions; run `.venv/bin/python -m pytest -q` as you go, and `.venv/bin/python scripts/app_surface_check.py` after any change that touches `app/`.
4. Two rules specific to this project, both of which the Stop hook enforces:
   - **Research leaves the app's navigation, not the repository.** Never delete anything under `report/`, `notebooks/`, or `src/`, and never remove a pipeline step, in service of an app goal.
   - **Do not weaken the gate to get through it.** If a check in `scripts/app_surface_check.py` is genuinely wrong, change it deliberately and say why in your summary. Loosening a check to make a failing goal pass is the one thing this loop exists to prevent.
5. If the goal is a visual one, verify it with rendered screenshots of the running app before you finish — at 1280px and at 390px. `AppTest` proves a page does not raise; it cannot tell you the page is usable.
6. When you believe the goal is complete, simply finish. The Stop hook runs the test suite, the app-surface gate, and a Codex review against the goal; any failure comes back as your next instruction. You cannot end until all gates pass or the 8-iteration safety valve trips.
7. If the goal came from `GOALS.md`, move its line to the Done section with today's date before your final finish attempt.
8. Do not edit or work around `.claude/current-goal.md`, `.claude/goal-iterations`, or the hook — the verifier owns those.

THE GOAL:

$ARGUMENTS
