#!/bin/bash
# Per-goal gate, run by the Stop hook between pytest and Codex
# (see .claude/hooks/verify_goal.sh). Exit non-zero to block the goal.
#
# This project's goals are user-interface goals, so the gate is an app-surface
# check rather than a model metric: every nav section must render, research
# must be out of the app's navigation, and the honest-uncertainty product
# content must still be there. See scripts/app_surface_check.py.
#
# A future modelling goal should EXTEND this with its own eval-and-threshold
# so that "tests pass" can never ship a model that misses its numbers.
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/app_surface_check.py
