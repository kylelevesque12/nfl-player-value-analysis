#!/bin/bash
# Goal-loop verifier for the NFL fantasy app, run as a Claude Code Stop hook.
#
# Blocks the session from ending while a goal is active until it verifies.
# Three gates, run in order:
#   1. pytest suite green
#   2. app-surface gate: scripts/goal_check.sh must exit 0. For this project
#      that is a UI check, not a model metric — every nav section must render
#      through Streamlit's AppTest with no exception, the research surfaces
#      must be gone from the app, and the honest-uncertainty product content
#      must still be there (see scripts/app_surface_check.py).
#   3. Codex review returning VERDICT: SHIP against the goal text.
# Any failure exits 2, which feeds the failure back to Claude as the next
# instruction — that's the loop. Safety valve after MAX_ITERATIONS.

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

GOAL_FILE=".claude/current-goal.md"
COUNT_FILE=".claude/goal-iterations"
CODEX="/Applications/Codex.app/Contents/Resources/codex"
MAX_ITERATIONS=8
PY=".venv/bin/python"

[ -f "$GOAL_FILE" ] || exit 0

n=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
if [ "$n" -ge "$MAX_ITERATIONS" ]; then
    mv "$GOAL_FILE" .claude/goal-stalled.md
    rm -f "$COUNT_FILE"
    echo "Goal loop stopped after $MAX_ITERATIONS iterations without passing." \
         "Parked in .claude/goal-stalled.md — review it manually."
    exit 0
fi

block() {
    echo $((n + 1)) > "$COUNT_FILE"
    echo "$1" >&2
    exit 2
}

# ---- gate 1: tests ----
test_out=$($PY -m pytest -q 2>&1 | tail -15)
if ! echo "$test_out" | grep -qE '^[0-9]+ passed' || echo "$test_out" | grep -qE 'failed|error'; then
    block "GOAL NOT MET (iteration $((n + 1))/$MAX_ITERATIONS): tests are not green.
Fix the failures below, then finish again.

$test_out"
fi

# ---- gate 2: app-surface / UI gate ----
if [ -x scripts/goal_check.sh ]; then
    metric_out=$(scripts/goal_check.sh 2>&1)
    if [ $? -ne 0 ]; then
        block "GOAL NOT MET (iteration $((n + 1))/$MAX_ITERATIONS): the app-surface gate failed.
A page fails to render, a research surface is still in the app, or a piece of
the product's honest-uncertainty content was stripped. Fix it below. If the
gate's own expectation is wrong, change it deliberately in
scripts/app_surface_check.py and say why — do not weaken it to get through.

$metric_out"
    fi
fi

# ---- gate 3: Codex review ----
goal=$(cat "$GOAL_FILE")
codex_out=$("$CODEX" exec --skip-git-repo-check --sandbox read-only \
    "You are the final reviewer for this project. It is a fantasy football
draft and start/sit product (Streamlit app in app/) built on a research
codebase (models in src/, write-ups in report/). The project is deliberately
splitting the two: THE APP IS A FANTASY PRODUCT FOR REAL USERS, THE REPO IS
THE RESEARCH PORTFOLIO. A league-mate should be able to draft with the app in
August with no explanation.

The developer claims this goal is complete:

---
$goal
---

Review strictly against the goal. Tests pass and the app-surface gate passed.
Check for real defects, and specifically for these four project-specific
failure modes:

1. RESEARCH DELETED RATHER THAN UNHOOKED. Research must leave the APP's
   navigation and pages, but report/, notebooks/, src/ modelling code, the
   pipeline steps and the outputs they write must remain intact in the repo.
   Removing a research page from the app must not delete the study.
2. HONESTY STRIPPED ALONG WITH THE JARGON. Prediction intervals, tier
   groupings, the stable/shaky role badge, the injury-return flag and the
   caveat callouts are PRODUCT, not research — they are the project's whole
   differentiator versus ESPN. Simplifying the interface must not turn an
   honest range into a false-precision point estimate.
3. UNGROUNDED UI CLAIMS. Any accuracy or benchmark number kept in the app
   must still match what the underlying tables actually support, including
   the stated coverage limits.
4. DEAD CODE AND BROKEN LINKS left behind by the removal — unreferenced
   render functions, imports, session-state keys, or links pointing at pages
   that no longer exist.

End with exactly one line: VERDICT: SHIP  or  VERDICT: NO-SHIP - <specific reasons>" < /dev/null 2>&1 | tail -40)

if echo "$codex_out" | grep -q "VERDICT: SHIP"; then
    rm -f "$GOAL_FILE" "$COUNT_FILE"
    echo "Goal verified: tests green, app-surface gate passed, Codex SHIP. Goal cleared."
    exit 0
fi

block "GOAL NOT MET (iteration $((n + 1))/$MAX_ITERATIONS): Codex has not approved.
Address its findings below, then finish again.

$codex_out"
