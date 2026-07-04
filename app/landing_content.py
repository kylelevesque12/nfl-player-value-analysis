"""Pure (Streamlit-free) content + navigation config for the landing page.

Kept separate from ``streamlit_app.py`` so the copy, methodology labels, and
navigation targets can be unit-tested without a Streamlit runtime.
"""

from __future__ import annotations

LANDING_TITLE = "NFL Player Value & Fantasy Forecasting"
LANDING_SUBTITLE = (
    "2026 draft rankings, weekly PPR projections with honest uncertainty, and "
    "contract-value analytics, built on ten years of NFL data and graded "
    "against the baselines every forecast must beat."
)

# Navigation targets, must match the sidebar radio option strings exactly.
# A single sidebar selects one section. Product sections (the tables a fantasy
# or front-office user acts on) come first; the research and methodology
# material is consolidated under one section at the end.
NAV_HOME = "Home"
NAV_CAP = "Player Value & Cap"
NAV_VALUE = NAV_CAP  # alias
NAV_FANTASY = "Draft Board"
NAV_ROOKIE = "Rookies"
NAV_PLAYER = "Player Detail"
NAV_METHOD = "Methodology & Research"

# The full ordered list of sidebar sections (the app's only navigation control).
# The QB injury study, the decomposition experiments, and the project report
# now live inside Methodology & Research rather than as their own sections.
SECTIONS = [
    NAV_HOME,
    NAV_FANTASY,
    NAV_CAP,
    NAV_ROOKIE,
    NAV_PLAYER,
    NAV_METHOD,
]


def landing_cards() -> list[dict]:
    """Config for the four section cards on the Home page."""
    return [
        {
            "tag": "Fantasy",
            "headline": "Season rankings and weekly PPR projections you can act on",
            "points": [
                "Beats the naive forecasting baselines by a steady 7–9% in every season, 2020–2025.",
                "Play-by-play role and weather features cut weekly error a further 1.27%.",
                "Every projection ships with a calibrated range, honest about uncertainty.",
            ],
            "button": "Open the Draft Board",
            "target": NAV_FANTASY,
        },
        {
            "tag": "Front office",
            "headline": "Who out-earns their contract, priced properly",
            "points": [
                "Season cap hits reconstructed from contract terms replace the misleading flat APY.",
                "Brock Purdy's 2023 season is the largest single-season cap surplus of 2016–2025.",
                "The veteran running back market consistently over-pays for production.",
            ],
            "button": "Open Player Value & Cap",
            "target": NAV_CAP,
        },
        {
            "tag": "Rookies",
            "headline": "Projections for players with no NFL history",
            "points": [
                "A hierarchical Bayesian model projects rookies from draft capital and profile.",
                "An incumbent-context signal lowers playing-time odds for blocked rookies.",
                "Combine features were tested and dropped; they never beat draft position.",
            ],
            "button": "Open Rookie Model",
            "target": NAV_ROOKIE,
        },
        {
            "tag": "Research",
            "headline": "The methodology, checked and kept on the record",
            "points": [
                "Re-timing the QB injury study grew its sample from 19 to 104 events and surfaced a suggestive −0.58 PPG effect.",
                "26 automated methodology checks pass, covering leakage safety and calibration.",
                "Ideas that lost head-to-head are documented, not hidden.",
            ],
            "button": "Open Methodology & Research",
            "target": NAV_METHOD,
        },
    ]


def methodology_strip_labels() -> list[str]:
    return [
        "Leakage-safe features",
        "Time-based validation",
        "Negative results documented",
        "Source/quality flags for salary estimates",
        "Tests covering key modeling assumptions",
    ]
