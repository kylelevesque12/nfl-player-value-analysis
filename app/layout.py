"""Pure (Streamlit-free) layout helpers for the dashboard.

Kept separate from ``components.py`` (which imports Streamlit) so the responsive
chunking logic can be unit-tested without a Streamlit runtime.
"""

from __future__ import annotations

import math


def chunk_metrics(metrics: list, max_per_row: int = 3) -> list[list]:
    """Split a metric list into balanced rows of at most ``max_per_row`` tiles.

    Balancing keeps rows even — 4 metrics at max 3 become 2 + 2 rather than
    3 + 1 — which reads better on tablet/phone widths where a single 4-wide KPI
    row gets cramped. (Streamlit already stacks columns to full width below its
    small-screen breakpoint; this keeps the mid-width range tidy too.)
    """
    metrics = list(metrics)
    n = len(metrics)
    if n == 0:
        return []
    if n <= max_per_row:
        return [metrics]
    n_rows = math.ceil(n / max_per_row)
    base, extra = divmod(n, n_rows)
    rows, i = [], 0
    for r in range(n_rows):
        size = base + (1 if r < extra else 0)
        rows.append(metrics[i:i + size])
        i += size
    return rows
