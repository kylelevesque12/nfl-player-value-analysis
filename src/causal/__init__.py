"""Causal analysis utilities for the NFL player value project.

Tier 2 #6 in ``PORTFOLIO_ROADMAP.md``. Submodules:

- ``treatment_identification`` builds the QB-injury treatment-event table.
- ``control_matching`` constructs same-calendar-week control panels.
- ``parallel_trends`` runs the pre-period parallel-trends check.

Each submodule is independently testable. The top-level entry point (session 2)
will live in ``did_estimator`` once the foundation here passes parallel trends.
"""
