"""Tests for the landing-page / navigation content config (Streamlit-free)."""

from __future__ import annotations

from pathlib import Path

from app import landing_content as lc


def test_sections_are_well_formed():
    # Home is first (the default landing), and the core sections are present.
    assert lc.SECTIONS[0] == lc.NAV_HOME
    for s in (lc.NAV_FANTASY, lc.NAV_DRAFTROOM, lc.NAV_PLAYER, lc.NAV_METHOD):
        assert s in lc.SECTIONS
    # The Draft Board is the first product section after Home, with the Draft
    # Room beside it.
    assert lc.SECTIONS[1] == lc.NAV_FANTASY
    assert lc.SECTIONS[2] == lc.NAV_DRAFTROOM
    # Research material is consolidated: no standalone research sections in
    # the nav — those all live inside Methodology & Research now.
    for retired in ("QB Injury Study", "Project Report", "Front Office", "Rookies"):
        assert retired not in lc.SECTIONS
    # No duplicate section labels.
    assert len(lc.SECTIONS) == len(set(lc.SECTIONS))


def test_nav_targets_match_app_radio_options():
    """The app builds its sidebar from SECTIONS and routes through nav_section,
    so navigation targets cannot silently no-op. Checked against the source to
    avoid importing Streamlit."""
    src = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text()
    assert "SECTIONS," in src  # imported from landing_content
    assert 'key="nav_section"' in src
    # The section labels that also appear verbatim in app copy/titles.
    for target in (lc.NAV_FANTASY, lc.NAV_DRAFTROOM, lc.NAV_METHOD):
        assert src.count(target) >= 1, f"nav target not present in app: {target!r}"


def test_methodology_strip_has_expected_labels():
    labels = lc.methodology_strip_labels()
    assert len(labels) == 5
    assert "Leakage-safe features" in labels
    assert any("Negative results" in s for s in labels)


def test_nav_captions_cover_all_sections():
    for s in lc.SECTIONS:
        assert lc.NAV_CAPTIONS.get(s), f"nav caption missing for section: {s!r}"


def test_team_colors_cover_all_franchise_codes():
    assert len(lc.TEAM_COLORS) == 32
    for code, color in lc.TEAM_COLORS.items():
        assert color.startswith("#") and len(color) == 7, (code, color)
