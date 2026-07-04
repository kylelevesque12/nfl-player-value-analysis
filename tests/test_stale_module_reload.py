"""Regression test for the Streamlit Cloud stale-module outage.

Streamlit Cloud deploys code changes with a git pull plus a hot reload of the
main script only. The long-lived Python process keeps previously imported
app.* modules cached in sys.modules, so a new main script that imports a name
just added to app.landing_content raises ImportError until the app is manually
rebooted. This took the deployed app down twice. The fix purges app.* from
sys.modules at the top of the main script; this test recreates the failure
(a planted stale module missing the newer names) and asserts the app boots
anyway.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_source_contains_module_purge():
    src = (ROOT / "app" / "streamlit_app.py").read_text()
    assert 'm == "app" or m.startswith("app.")' in src
    # The purge must run before the first app.* import statement (newline-
    # anchored so a mention inside a comment does not satisfy the check).
    assert src.index('m.startswith("app.")') < src.index("\nfrom app.components import")


def test_app_boots_with_stale_landing_content_planted():
    if not (ROOT / "outputs" / "tables" / "2026_fantasy_football_projections.csv").exists():
        return  # data-dependent: skip on CI where output tables are absent

    # Simulate the cloud's stale cache: a landing_content module from "before
    # the push", missing every name the current main script imports.
    stale = types.ModuleType("app.landing_content")
    stale.LANDING_TITLE = "old"
    sys.modules["app.landing_content"] = stale
    try:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
        at.run()
        assert not at.exception, at.exception
    finally:
        sys.modules.pop("app.landing_content", None)
