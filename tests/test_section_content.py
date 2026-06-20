"""Tests for the Streamlit-free reference slicer used by the app sections."""

from __future__ import annotations

from pathlib import Path

from app import section_content as sc

SAMPLE = """# Title

intro line

## 1. The goal

Goal body line one.
Goal body line two.

## 2. Data foundation

Data body.

## 3. The pieces

Pieces body.
"""


def test_split_reference_indexes_top_level_sections():
    secs = sc.split_reference(SAMPLE)
    assert set(secs) == {1, 2, 3}
    assert secs[1]["title"] == "The goal"
    assert "Goal body line one." in secs[1]["body"]
    # Body stops before the next heading.
    assert "Data body." not in secs[1]["body"]


def test_reference_markdown_orders_and_skips_missing():
    md = sc.reference_markdown(SAMPLE, [3, 1, 99])
    # Requested order preserved, missing section skipped.
    assert md.index("Pieces body.") < md.index("Goal body line one.")
    assert "99" not in md


def test_section_map_keys_present():
    for key in ("home", "value", "fantasy", "rookie", "causal", "methodology"):
        assert key in sc.SECTION_REFERENCE_MAP
        assert sc.SECTION_REFERENCE_MAP[key]


def test_empty_text_is_safe():
    assert sc.split_reference("") == {}
    assert sc.reference_markdown("", [1, 2]) == ""


def test_real_reference_slices_cleanly_if_present():
    ref = Path(__file__).resolve().parents[1] / "PROJECT_REFERENCE.md"
    if not ref.exists():
        return
    secs = sc.split_reference(ref.read_text(encoding="utf-8"))
    # The reference should expose the sections the app maps to.
    for nums in sc.SECTION_REFERENCE_MAP.values():
        for n in nums:
            assert n in secs, f"reference section {n} missing"
    # No session jargon should leak through the sliced app content.
    joined = " ".join(s["body"] for s in secs.values()).lower()
    assert "session 1" not in joined and "session 2" not in joined
