"""Pure (Streamlit-free) content + navigation config for the landing page.

Kept separate from ``streamlit_app.py`` so the card copy, methodology labels, and
navigation targets can be unit-tested without a Streamlit runtime, and so they're
trivial to migrate into the component system in Session 9.
"""

from __future__ import annotations

LANDING_TITLE = "NFL Player Value & Fantasy Projection Lab"
LANDING_SUBTITLE = (
    "A portfolio research app combining weekly fantasy forecasting, "
    "contract-adjusted surplus value, rookie opportunity modeling, and causal "
    "analysis of quarterback injury reports."
)

# Navigation targets — must match the sidebar radio option strings exactly.
# A single sidebar selects one section; each section pairs a short explanation
# with its tool and results.
NAV_HOME = "Home"
NAV_CAP = "Player Value & Cap"
NAV_VALUE = NAV_CAP  # alias
NAV_FANTASY = "Fantasy Rankings"
NAV_ROOKIE = "Rookies"
NAV_CAUSAL = "QB Injury Study"
NAV_PLAYER = "Player Detail"
NAV_METHOD = "Methodology & Sources"

# The full ordered list of sidebar sections (the app's only navigation control).
SECTIONS = [
    NAV_HOME,
    NAV_CAP,
    NAV_FANTASY,
    NAV_ROOKIE,
    NAV_CAUSAL,
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
            "button": "Open Fantasy Rankings",
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
                "Jordan Love P(plays) moved from 0.611 to 0.513.",
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
                "Post-period ATT was about −0.58 PPG — suggestive and underpowered.",
            ],
            "button": "Open Causal Study",
            "target": NAV_CAUSAL,
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
