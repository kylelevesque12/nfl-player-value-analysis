"""Pure (Streamlit-free) helpers for slicing PROJECT_REFERENCE.md into the
per-section write-ups the app embeds.

The reference document is the single clean, plain-language source of truth (no
behind-the-scenes session jargon), so the app pulls its "full write-up" panels
from it rather than from the raw report/ files. Kept Streamlit-free so the
slicing logic can be unit-tested without a runtime.
"""

from __future__ import annotations

import re

_HEADING = re.compile(r"^##\s+(\d+)\.\s+(.*)$", re.MULTILINE)

# Which reference sections back each app section.
SECTION_REFERENCE_MAP: dict[str, list[int]] = {
    "home": [1, 3, 12],
    "value": [4, 7],
    "fantasy": [5, 6],
    "rookie": [8],
    "causal": [9],
    "methodology": [10, 11, 13, 14],
}


def split_reference(text: str) -> dict[int, dict[str, str]]:
    """Split the reference markdown into {number: {"title", "body"}} by its
    top-level '## N. Title' headings."""
    if not text:
        return {}
    matches = list(_HEADING.finditer(text))
    out: dict[int, dict[str, str]] = {}
    for i, m in enumerate(matches):
        num = int(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[num] = {"title": title, "body": text[start:end].strip()}
    return out


def reference_markdown(
    text: str, numbers: list[int], include_heading: bool = True
) -> str:
    """Return the concatenated markdown for the given reference section numbers,
    in the order requested. Missing sections are skipped."""
    sections = split_reference(text)
    parts: list[str] = []
    for n in numbers:
        sec = sections.get(n)
        if not sec:
            continue
        if include_heading:
            parts.append(f"## {n}. {sec['title']}\n\n{sec['body']}")
        else:
            parts.append(sec["body"])
    return "\n\n".join(parts)


def section_reference_markdown(text: str, key: str, include_heading: bool = True) -> str:
    """Convenience: full write-up markdown for a named app section."""
    return reference_markdown(
        text, SECTION_REFERENCE_MAP.get(key, []), include_heading=include_heading
    )
