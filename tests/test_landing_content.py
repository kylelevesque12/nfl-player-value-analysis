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
    # Two cards go via the hero radio, two via the drill-down radio.
    hero = [t for t in targets if t in lc.HERO_TARGETS]
    detail = [t for t in targets if t not in lc.HERO_TARGETS]
    assert set(hero) == {lc.NAV_FANTASY, lc.NAV_CAP}
    assert set(detail) == {lc.NAV_ROOKIE, lc.NAV_CAUSAL}


def test_nav_targets_match_app_radio_options():
    """The card targets must equal radio option strings the app actually uses,
    or navigation silently no-ops. Checked against the source to avoid importing
    Streamlit."""
    src = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text()
    for target in (lc.NAV_FANTASY, lc.NAV_CAP, lc.NAV_ROOKIE, lc.NAV_CAUSAL, lc.NAV_DETAIL_NONE):
        assert src.count(target) >= 1, f"nav target not present in app: {target!r}"
    # Landing is wired as the default hero option.
    assert "Home (Landing)" in src


def test_methodology_strip_has_expected_labels():
    labels = lc.methodology_strip_labels()
    assert len(labels) == 5
    assert "Leakage-safe features" in labels
    assert any("Negative results" in s for s in labels)


def test_headline_findings_are_present_and_consistent():
    blob = " ".join(p for c in lc.landing_cards() for p in c["points"])
    for token in ["1.27%", "Brock Purdy", "0.611", "0.513", "19 to 104", "0.58"]:
        assert token in blob, f"missing headline finding: {token}"
