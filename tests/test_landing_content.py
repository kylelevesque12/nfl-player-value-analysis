"""Tests for the landing-page / navigation content config (Streamlit-free)."""

from __future__ import annotations

from pathlib import Path

from app import landing_content as lc


def test_sections_are_well_formed():
    # Home is first (the default landing), and the core sections are present.
    assert lc.SECTIONS[0] == lc.NAV_HOME
    for s in (lc.NAV_FANTASY, lc.NAV_DRAFTROOM, lc.NAV_PLAYER):
        assert s in lc.SECTIONS
    # The Draft Board is the first product section after Home, with the Draft
    # Room beside it.
    assert lc.SECTIONS[1] == lc.NAV_FANTASY
    assert lc.SECTIONS[2] == lc.NAV_DRAFTROOM
    # No duplicate section labels.
    assert len(lc.SECTIONS) == len(set(lc.SECTIONS))


def test_nav_is_product_only():
    """The app is the fantasy product; the research is the repository.

    Every nav entry has to be something a league-mate would click during a
    draft. The research surfaces that used to live here — the methodology
    audit, the study summaries, the external benchmark, the project report —
    were removed from the app and still exist under report/ and notebooks/.
    """
    assert lc.SECTIONS == [
        lc.NAV_HOME,
        lc.NAV_FANTASY,
        lc.NAV_DRAFTROOM,
        lc.NAV_PLAYER,
    ]
    assert not hasattr(lc, "NAV_METHOD")
    banned = ("methodolog", "research", "front office", "benchmark", "report")
    for section in lc.SECTIONS:
        assert not any(word in section.lower() for word in banned), section


def test_nav_targets_match_app_radio_options():
    """The app builds its sidebar from SECTIONS and routes through nav_section,
    so navigation targets cannot silently no-op. Checked against the source to
    avoid importing Streamlit."""
    src = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text()
    assert "SECTIONS," in src  # imported from landing_content
    assert 'key="nav_section"' in src
    # The section labels that also appear verbatim in app copy/titles.
    for target in (lc.NAV_FANTASY, lc.NAV_DRAFTROOM):
        assert src.count(target) >= 1, f"nav target not present in app: {target!r}"


def test_research_stays_in_the_repository():
    """Removing the research from the app must not delete the research."""
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "report/methodology_checks.md",
        "report/causal/qb_injury_session3.md",
        "report/two_stage_weekly.md",
        "src/methodology_checks.py",
        "src/external_benchmark.py",
    ):
        assert (root / relative).exists(), f"research artifact was deleted: {relative}"


def test_nav_captions_cover_all_sections():
    for s in lc.SECTIONS:
        assert lc.NAV_CAPTIONS.get(s), f"nav caption missing for section: {s!r}"


def test_team_colors_cover_all_franchise_codes():
    assert len(lc.TEAM_COLORS) == 32
    for code, color in lc.TEAM_COLORS.items():
        assert color.startswith("#") and len(color) == 7, (code, color)
