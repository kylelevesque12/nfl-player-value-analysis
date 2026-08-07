"""App-surface gate for the goal loop.

This project's goals are user-interface goals, so "the tests pass" is not a
sufficient bar — the deployed app has to actually render, and the redesign has
to keep the promises the project makes to its users. This script is the
mechanical part of that bar. The Stop hook runs it between pytest and the
Codex review (see .claude/hooks/verify_goal.sh).

It checks four things, in order of how badly each one would hurt if it broke:

1. EVERY NAV SECTION RENDERS. Each section is driven through Streamlit's
   AppTest and must produce no exception. A page that throws is the single
   worst outcome of a refactor and the easiest to miss, because Streamlit
   renders the rest of the app around the failure.

2. THE APP IS A FANTASY PRODUCT, NOT A RESEARCH DASHBOARD. No navigation
   entry may be a research surface, and the research-only render functions
   must be gone from app/. Note what this does NOT check: report/,
   notebooks/, and src/ are untouched by design — the research leaves the
   app's navigation, it does not leave the repository.

3. THE HONEST-UNCERTAINTY CONTENT SURVIVED. Calibrated ranges, tiers, the
   role stable/shaky badge and the injury-return flag are the product's
   differentiator against ESPN, and they are the thing most likely to be
   swept away by a "simplify the interface" pass. Stripping them would turn
   an honest range into a false-precision point estimate, so the gate treats
   their absence as a failure just like a crash.

4. NO DANGLING NAV TARGETS. Anything the app tries to navigate to must be a
   section that still exists, which is the specific way removing a page
   tends to break the pages that linked to it.

Run it directly for a readable report:

    .venv/bin/python scripts/app_surface_check.py

Exit code 0 means every check passed; 1 means at least one failed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
APP_ENTRY = APP_DIR / "app.py"
if not APP_ENTRY.exists():  # current entry point; the split may rename it
    APP_ENTRY = APP_DIR / "streamlit_app.py"

# Data the app needs in order to render at all. Missing data is a hard
# failure rather than a skip: a gate that quietly passes when it could not
# actually look at anything is worse than no gate.
REQUIRED_DATA = [
    ROOT / "outputs" / "tables" / "draft_board_2026.csv",
    ROOT / "outputs" / "tables" / "2026_fantasy_football_projections.csv",
]

# A navigation entry whose name matches any of these is a research surface
# and belongs in the repo, not in the product's nav bar.
RESEARCH_NAV_PATTERNS = [
    r"methodolog",
    r"research",
    r"front office",
    r"benchmark",
    r"salary",
    r"causal",
    r"report",
]

# Render functions that exist only to show research inside the app. The goal
# is for these to be gone from app/ entirely.
RESEARCH_RENDER_FUNCTIONS = [
    "methodology_research_section",
    "methodology_page",
    "external_benchmark_page",
    "reports_page",
    "_project_report_tab",
    "_research_studies_tab",
    "_study_card",
    "_full_writeup_expander",
]

# Product content that must survive the redesign. Each entry is (label, list
# of source tokens, at least one of which must still appear in app/).
HONESTY_MARKERS = [
    ("prediction intervals (floors and ceilings)",
     ["prediction_interval_low", "prediction_interval_high"]),
    ("tier grouping", ["tier"]),
    ("role stable/shaky badge", ["stable", "shaky"]),
    ("injury-return flag", ["injury_return", "⚕"]),
    ("caveat callouts", ["caveat_callout"]),
]


def _fail(msg: str) -> str:
    return f"FAIL  {msg}"


def _ok(msg: str) -> str:
    return f"ok    {msg}"


def _app_sources() -> dict[Path, str]:
    """Every Python source file under app/, excluding caches and worktrees."""
    out: dict[Path, str] = {}
    for path in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        out[path] = path.read_text(encoding="utf-8")
    return out


def check_data_present() -> list[str]:
    problems = []
    for path in REQUIRED_DATA:
        if not path.exists():
            problems.append(
                _fail(
                    f"{path.relative_to(ROOT)} is missing — the gate cannot "
                    "render the app. Run `python scripts/run_pipeline.py` first."
                )
            )
    return problems or [_ok("required data tables are present")]


def check_sections_render() -> list[str]:
    """Drive every nav section through AppTest and require a clean render."""
    from streamlit.testing.v1 import AppTest

    results: list[str] = []
    at = AppTest.from_file(str(APP_ENTRY), default_timeout=180)
    at.run()
    if at.exception:
        return [_fail(f"the app raised on first load: {at.exception[0].value}")]

    if not at.sidebar.radio:
        return [_fail("no navigation radio found in the sidebar")]

    sections = list(at.sidebar.radio[0].options)
    results.append(_ok(f"navigation has {len(sections)} sections: {', '.join(sections)}"))

    for section in sections:
        run = AppTest.from_file(str(APP_ENTRY), default_timeout=180)
        run.run()
        run.sidebar.radio[0].set_value(section).run()
        if run.exception:
            results.append(_fail(f"section '{section}' raised: {run.exception[0].value}"))
        else:
            results.append(_ok(f"section '{section}' renders"))

    return results


def check_no_research_in_nav() -> list[str]:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_ENTRY), default_timeout=180)
    at.run()
    if at.exception or not at.sidebar.radio:
        return [_fail("could not read navigation to check for research sections")]

    results = []
    for section in at.sidebar.radio[0].options:
        for pattern in RESEARCH_NAV_PATTERNS:
            if re.search(pattern, section, re.IGNORECASE):
                results.append(
                    _fail(
                        f"navigation entry '{section}' is a research surface "
                        f"(matched /{pattern}/) — the app is the product, the "
                        "repo is the portfolio"
                    )
                )
                break
    return results or [_ok("no research surfaces in the navigation")]


def check_research_renderers_removed() -> list[str]:
    sources = _app_sources()
    results = []
    for name in RESEARCH_RENDER_FUNCTIONS:
        hits = [
            str(path.relative_to(ROOT))
            for path, text in sources.items()
            if re.search(rf"\b{re.escape(name)}\b", text)
        ]
        if hits:
            results.append(
                _fail(f"research renderer '{name}' still referenced in {', '.join(hits)}")
            )
    return results or [_ok("research render functions are gone from app/")]


def check_honesty_markers() -> list[str]:
    blob = "\n".join(_app_sources().values()).lower()
    results = []
    for label, tokens in HONESTY_MARKERS:
        if not any(token.lower() in blob for token in tokens):
            results.append(
                _fail(
                    f"{label} no longer appears anywhere in app/ — this is "
                    "product content, not research; a simplification pass must "
                    "not remove it"
                )
            )
    return results or [_ok("honest-uncertainty content is still present")]


def check_nav_targets_resolve() -> list[str]:
    """Every target handed to the nav-jump helper must be a real section.

    Call sites pass either a string literal or a NAV_* constant, so both are
    collected and the constants are resolved through the module that defines
    them. A NAV_* name the resolver cannot find is reported rather than
    ignored — silently skipping unknown names is how this check would rot
    into always passing.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_ENTRY), default_timeout=180)
    at.run()
    if at.exception or not at.sidebar.radio:
        return [_fail("could not read navigation to check jump targets")]
    sections = set(at.sidebar.radio[0].options)

    # Resolve NAV_* constants to their values wherever app/ defines them.
    constants: dict[str, str] = {}
    for text in _app_sources().values():
        for name, value in re.findall(
            r"^(NAV_[A-Z_]+)\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE
        ):
            constants[name] = value

    blob = "\n".join(_app_sources().values())
    raw = set(re.findall(r"_go_to\(\s*([^)]+?)\s*\)", blob))

    results: list[str] = []
    targets: set[str] = set()
    for token in raw:
        literal = re.fullmatch(r"[\"']([^\"']+)[\"']", token)
        if literal:
            targets.add(literal.group(1))
        elif token in constants:
            targets.add(constants[token])
        elif token.startswith("NAV_"):
            results.append(_fail(f"_go_to({token}) references an undefined nav constant"))
        # anything else is a runtime-computed target this static check cannot
        # follow; the render check above still exercises the real pages.

    dangling = sorted(t for t in targets if t not in sections)
    if dangling:
        results.append(
            _fail(
                "navigation jumps point at sections that no longer exist: "
                + ", ".join(dangling)
            )
        )
    if not targets and not results:
        return [_fail("found no _go_to targets at all — has the helper been renamed?")]
    return results or [_ok(f"all {len(targets)} in-app navigation targets resolve")]


CHECKS = [
    ("data", check_data_present),
    ("render", check_sections_render),
    ("no research in nav", check_no_research_in_nav),
    ("research renderers removed", check_research_renderers_removed),
    ("honesty preserved", check_honesty_markers),
    ("nav targets resolve", check_nav_targets_resolve),
]


def main() -> int:
    if not APP_ENTRY.exists():
        print(_fail(f"no app entry point found at {APP_DIR.relative_to(ROOT)}/"))
        return 1

    failed = False
    for name, check in CHECKS:
        print(f"\n[{name}]")
        try:
            lines = check()
        except Exception as exc:  # a check that crashes is a failure, not a pass
            lines = [_fail(f"check crashed: {type(exc).__name__}: {exc}")]
        for line in lines:
            print("  " + line)
            if line.startswith("FAIL"):
                failed = True
        # Data is a precondition: without it nothing below can be trusted.
        if name == "data" and failed:
            break

    print("\n" + ("APP SURFACE GATE: FAIL" if failed else "APP SURFACE GATE: PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
