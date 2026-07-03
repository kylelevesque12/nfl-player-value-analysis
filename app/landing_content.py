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
    """Config for the four findings cards."""
    return [
        {
            "tag": "Fantasy forecasting",
            "headline": "Live weekly projections from leakage-safe player role features",
            "points": [
                "PBP role + weather features reduced weekly fantasy RMSE by 1.27% (6.020 → 5.944).",
                "A live projection frame now scores upcoming weeks without outcome data.",
                "Per-position conformal intervals improve QB coverage.",
            ],
            "button": "Open the Draft Board",
            "target": NAV_FANTASY,
        },
        {
            "tag": "Player value / cap surplus",
            "headline": "Surplus value using reconstructed cap-hit estimates",
            "points": [
                "Replaced flat APY with season-specific reconstructed cap-hit estimates.",
                "Brock Purdy remains the top surplus player-season.",
                "Early-extension stars are treated more realistically than under flat APY.",
            ],
            "button": "Open Player Value & Cap",
            "target": NAV_CAP,
        },
        {
            "tag": "Rookie opportunity model",
            "headline": "Rookie QB opportunity depends on incumbent context",
            "points": [
                "Combine features were tested but not kept.",
                "A focused incumbent-context core improved the QB rookie case.",
                "Rookies blocked by an established starter are projected to play less.",
            ],
            "button": "Open Rookie Model",
            "target": NAV_ROOKIE,
        },
        {
            "tag": "QB injury causal study",
            "headline": "First injury-report appearance matters before formal absence",
            "points": [
                "An Out-only treatment found little signal.",
                "First-report treatment expanded events from 19 to 104.",
                "Post-period ATT was about −0.58 PPG, suggestive and underpowered.",
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
