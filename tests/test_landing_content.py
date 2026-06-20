"""Tests for the Session 8 landing-page content/config (Streamlit-free)."""

from __future__ import annotations

from pathlib import Path

from app import landing_content as lc


def test_four_cards_each_well_formed():
    cards = lc.landing_cards()
    assert len(cards) == 4
    for c in cards:
        assert {"tag", "headline", "points", "button", "target"} <= set(c)
        assert isinstance(c["points"], list) and len(c["points"]) >= 2
        assert c["headline"] and c["button"] and c["target"]


def test_card_targets_route_to_known_pages():
    targets = [c["target"] for c in lc.landing_cards()]
    assert targets == [lc.NAV_FANTASY, lc.NAV_CAP, lc.NAV_ROOKIE, lc.NAV_CAUSAL]
    # Every card target must be one of the single-nav sections.
    assert set(targets) <= set(lc.SECTIONS)


def test_sections_are_well_formed():
    # Home is first (the default landing), and the core sections are present.
    assert lc.SECTIONS[0] == lc.NAV_HOME
    for s in (lc.NAV_CAP, lc.NAV_FANTASY, lc.NAV_ROOKIE, lc.NAV_CAUSAL,
              lc.NAV_PLAYER, lc.NAV_METHOD):
        assert s in lc.SECTIONS
    # No duplicate section labels.
    assert len(lc.SECTIONS) == len(set(lc.SECTIONS))


def test_nav_targets_match_app_radio_options():
    """The app builds its sidebar from SECTIONS and routes through nav_section,
    so card targets (which are SECTIONS members) cannot silently no-op. Checked
    against the source to avoid importing Streamlit."""
    src = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text()
    assert "SECTIONS," in src  # imported from landing_content
    assert 'key="nav_section"' in src
    # The section labels that also appear verbatim in app copy/titles.
    for target in (lc.NAV_FANTASY, lc.NAV_CAP, lc.NAV_ROOKIE, lc.NAV_CAUSAL,
                   lc.NAV_METHOD):
        assert src.count(target) >= 1, f"nav target not present in app: {target!r}"


def test_methodology_strip_has_expected_labels():
    labels = lc.methodology_strip_labels()
    assert len(labels) == 5
    assert "Leakage-safe features" in labels
    assert any("Negative results" in s for s in labels)


def test_headline_findings_are_present_and_consistent():
    blob = " ".join(p for c in lc.landing_cards() for p in c["points"])
    for token in ["1.27%", "Brock Purdy", "incumbent-context", "19 to 104", "0.58"]:
        assert token in blob, f"missing headline finding: {token}"
