"""Tests for the Session 11 responsive layout helper (Streamlit-free)."""

from __future__ import annotations

from app.layout import chunk_metrics


def _metrics(n):
    return [(f"L{i}", "v", None) for i in range(n)]


def test_short_rows_stay_in_one_row():
    for n in (0, 1, 2, 3):
        rows = chunk_metrics(_metrics(n), max_per_row=3)
        assert len(rows) == (0 if n == 0 else 1)


def test_wide_rows_wrap_balanced():
    assert [len(r) for r in chunk_metrics(_metrics(4), 3)] == [2, 2]
    assert [len(r) for r in chunk_metrics(_metrics(5), 3)] == [3, 2]
    assert [len(r) for r in chunk_metrics(_metrics(6), 3)] == [3, 3]


def test_no_row_exceeds_max_and_nothing_lost():
    for n in range(0, 12):
        rows = chunk_metrics(_metrics(n), max_per_row=3)
        assert sum(len(r) for r in rows) == n          # nothing dropped
        assert all(len(r) <= 3 for r in rows)          # cap respected
        # balanced: row sizes differ by at most 1
        sizes = [len(r) for r in rows]
        assert not sizes or (max(sizes) - min(sizes) <= 1)


def test_order_preserved():
    flat = [m for row in chunk_metrics(_metrics(5), 3) for m in row]
    assert flat == _metrics(5)
