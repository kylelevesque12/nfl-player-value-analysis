"""Interaction test for the Draft Room page: setup, recording picks, undo,
and reset, driven through Streamlit's AppTest so a regression in the session
state wiring is caught before it reaches the deployed app."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("streamlit")


def _has_data() -> bool:
    return (ROOT / "outputs" / "tables" / "draft_board_2026.csv").exists()


@pytest.mark.skipif(not _has_data(), reason="draft_board_2026.csv not built")
def test_draft_room_pick_tracking_and_undo():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("Draft Room").run()
    assert not at.exception
    assert list(at.session_state["dr_log"]) == []

    # Record the first pick with whatever player the board offers.
    picker = next(w for w in at.selectbox if w.options)
    picker.select(picker.options[0]).run()
    assert not at.exception

    record_button = next(b for b in at.button if b.label == "Record pick")
    record_button.click().run()
    assert not at.exception
    assert len(at.session_state["dr_log"]) == 1
    first_pick = at.session_state["dr_log"][0]

    # A second pick.
    picker2 = next(w for w in at.selectbox if w.options)
    picker2.select(picker2.options[0]).run()
    next(b for b in at.button if b.label == "Record pick").click().run()
    assert not at.exception
    assert len(at.session_state["dr_log"]) == 2

    # Undo returns to exactly the first pick.
    next(b for b in at.button if b.label == "Undo last").click().run()
    assert not at.exception
    assert at.session_state["dr_log"] == [first_pick]

    # Reset clears everything.
    next(b for b in at.button if b.label == "Reset draft").click().run()
    assert not at.exception
    assert at.session_state["dr_log"] == []
