"""Tests for the design system.

The goal these back is "a new page can be built from the components without
writing new CSS." That is only true if three things hold, and each is easy to
break by accident:

1. CSS lives in exactly one place. The moment a second module ships a
   ``<style>`` block, the override-ordering problem the design system was
   built to remove is back.
2. Components reference classes that actually exist. A typo in a class name
   renders as unstyled HTML — the page still "works", the test suite still
   passes, and nobody notices until they look at it.
3. Components reference tokens that actually exist. ``var(--nope)`` silently
   resolves to nothing, which usually reads as black text or no border.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"

from app import design  # noqa: E402


def _app_sources() -> dict[Path, str]:
    return {
        p: p.read_text()
        for p in sorted(APP.rglob("*.py"))
        if "__pycache__" not in p.parts
    }


def _stylesheet() -> str:
    return design._TOKENS + design._COMPONENTS


def test_design_module_is_the_only_source_of_css():
    offenders = [
        str(p.relative_to(ROOT))
        for p, text in _app_sources().items()
        if "<style>" in text and p.name != "design.py"
    ]
    assert not offenders, (
        "CSS must live only in app/design.py, but a <style> block appears in: "
        + ", ".join(offenders)
    )


def test_no_hardcoded_colors_outside_the_design_system():
    """Sections may pass a token value through (the team color on a tile), but
    they may not write their own hex colors into markup."""
    offenders = []
    for p, text in _app_sources().items():
        if p.name in {"design.py", "landing_content.py"}:
            continue  # the stylesheet, and the team-color lookup table
        for match in re.finditer(r'style="([^"]*)"', text):
            body = match.group(1)
            if "--team-color" in body or "left:" in body:
                continue  # a token hand-off and a computed bar position
            if re.search(r"#[0-9a-fA-F]{3,8}\b", body):
                offenders.append(f"{p.relative_to(ROOT)}: style=\"{body}\"")
    assert not offenders, "hard-coded colors in markup:\n" + "\n".join(offenders)


def _classes_in(html: str) -> set[str]:
    out: set[str] = set()
    for attr in re.findall(r'class="([^"]+)"', html):
        out.update(c for c in attr.split() if c.startswith("ds-"))
    return out


def test_every_component_class_is_defined_in_the_stylesheet():
    """Render one of every component and check each class it emits is styled."""
    css = _stylesheet()
    defined = set(re.findall(r"\.(ds-[\w-]+)", css))

    rendered = "".join(
        [
            design.stat_card_html("Label", "12.3", "PPR", "note"),
            design.stat_card_html("Label", "1", tone="positive"),
            design.stat_card_html("Label", "1", tone="negative"),
            design.stat_card_html("Label", "1", tone="uncertain"),
            design.player_tile_html(1, "A Player", "WR", "LA", "300", meta="18.7 per game"),
            design.player_row_html(1, "A Player", "WR · LA", "300", "proj PPR",
                                   badge=design.tier_badge_html(1)),
            *[design.tier_badge_html(t) for t in (1, 2, 3, 4, 9)],
            design.range_bar_html(200, 260, 340),
            design.delta_html(12.0),
            design.delta_html(-12.0),
            design.delta_html(0.0),
        ]
    )
    used = _classes_in(rendered)
    assert used, "no component classes were rendered — did the components change?"
    missing = sorted(used - defined)
    assert not missing, f"components emit classes with no CSS rule: {missing}"


def test_stylesheet_uses_only_defined_tokens():
    css = _stylesheet()
    declared = set(re.findall(r"^\s*(--[\w-]+):", css, re.MULTILINE))
    referenced = set(re.findall(r"var\((--[\w-]+)", css))
    # --team-color is supplied per-element by the tile component, with a
    # fallback in the rule itself, so it is intentionally not in :root.
    undefined = sorted(referenced - declared - {"--team-color"})
    assert not undefined, f"stylesheet references undefined tokens: {undefined}"


def test_range_bar_places_the_marker_inside_the_band():
    html = design.range_bar_html(100, 150, 200)
    pct = float(re.search(r"left:([\d.]+)%", html).group(1))
    assert pct == pytest.approx(50.0)
    # The point estimate sits where it actually falls, not at the centre.
    skewed = design.range_bar_html(100, 175, 200)
    assert float(re.search(r"left:([\d.]+)%", skewed).group(1)) == pytest.approx(75.0)


def test_range_bar_is_empty_rather_than_misleading_when_data_is_missing():
    """A missing interval must render nothing. A full-width bar would claim a
    range the model never produced."""
    assert design.range_bar_html(None, 150, 200) == ""
    assert design.range_bar_html(100, None, 200) == ""
    assert design.range_bar_html(100, 150, None) == ""
    assert design.range_bar_html(200, 150, 100) == ""  # inverted
    assert design.range_bar_html(150, 150, 150) == ""  # zero-width


def test_tier_badge_saturates_past_the_styled_range():
    assert 'ds-tier--4' in design.tier_badge_html(9)
    assert 'ds-tier--1' in design.tier_badge_html(1)
    assert design.tier_badge_html(None) == ""


def test_components_escape_interpolated_data():
    """Player names come from data files and land in raw HTML."""
    html = design.player_row_html(1, 'A <script>"&', "meta", "1")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_delta_uses_semantic_classes_not_inline_color():
    for value in (5.0, -5.0, 0.0):
        html = design.delta_html(value)
        assert "ds-delta--" in html
        assert "style=" not in html
