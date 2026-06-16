"""Tests for the Session 9 detail-page content/config (Streamlit-free).

These guard the migrated pages' structure and the required caveats, and confirm
the stale pre-Session-4/5 copy was actually replaced. They also re-check that the
landing-page navigation targets still match the app's radio options.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import page_content as pc
from app import landing_content as lc

APP_SRC = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text()
DETAIL_KEYS = ["surplus", "benchmark", "causal", "rookie", "two_stage", "methodology"]


def _blob(key: str) -> str:
    p = pc.DETAIL_PAGES[key]
    return " ".join(
        [p["title"], p["purpose"], *p["summary"], p["caveat"]["label"], p["caveat"]["body"], p["footer"]]
    ).lower()


def test_all_detail_pages_have_required_metadata():
    for key in DETAIL_KEYS:
        p = pc.DETAIL_PAGES[key]
        assert {"title", "purpose", "summary", "caveat", "footer"} <= set(p)
        assert p["title"] and p["purpose"] and p["footer"]
        assert isinstance(p["summary"], list) and len(p["summary"]) >= 2
        assert {"label", "body"} <= set(p["caveat"])
        assert p["caveat"]["body"]


def test_required_caveat_tokens_present():
    for key, tokens in pc.REQUIRED_CAVEAT_TOKENS.items():
        blob = _blob(key)
        for tok in tokens:
            assert tok.lower() in blob, f"page {key} missing required caveat token: {tok!r}"


def test_surplus_page_uses_reconstructed_cap_hit_not_inflated_apy():
    blob = _blob("surplus")
    assert "reconstructed" in blob and "not exact" in blob
    # The stale APY framing must be gone from the page config and the page body.
    assert "inflated_apy" not in blob


def test_causal_page_reflects_first_report_not_old_null():
    blob = _blob("causal")
    assert "first injury-report" in blob or "first week the established starter" in blob
    assert "19" in blob and "104" in blob          # Out-only vs first-report counts
    assert "suggestive" in blob and "underpowered" in blob


def test_migrated_pages_render_from_config_and_have_footer_caveat():
    """Each migrated page function should pull DETAIL_PAGES[...] and render a
    scaffold + caveat + footer (checked against the app source)."""
    for key in DETAIL_KEYS:
        assert f'DETAIL_PAGES["{key}"]' in APP_SRC, f"{key} page not wired to config"
    assert "render_page_scaffold(content)" in APP_SRC
    assert "caveat_callout(content[" in APP_SRC
    assert "source_footer(content[" in APP_SRC


def test_external_benchmark_overclaim_removed_from_app():
    # The old "beating the market is the qualifying bar" framing should be gone.
    assert "qualifying bar for a fantasy-projection" not in APP_SRC


def test_landing_routing_targets_unchanged():
    # Detail-page migration must not have changed the radio option strings the
    # landing-page buttons depend on.
    for target in (lc.NAV_FANTASY, lc.NAV_CAP, lc.NAV_ROOKIE, lc.NAV_CAUSAL):
        assert target in APP_SRC
    assert "Home (Landing)" in APP_SRC
