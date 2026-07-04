"""Pure (Streamlit-free) content + navigation config for the landing page.

Kept separate from ``streamlit_app.py`` so the copy, methodology labels, and
navigation targets can be unit-tested without a Streamlit runtime.
"""

from __future__ import annotations

LANDING_TITLE = "NFL Player Value & Fantasy Forecasting"
LANDING_SUBTITLE = (
    "PPR draft rankings and weekly projections with honest floors and "
    "ceilings, plus the front-office view of who is actually worth the money."
)

# Navigation targets, must match the sidebar radio option strings exactly.
# A single sidebar selects one section. The fantasy cluster comes first
# (Draft Board, Rookies, Player Detail), then the front-office view, then the
# methodology material consolidated at the end.
NAV_HOME = "Home"
NAV_CAP = "Front Office"
NAV_VALUE = NAV_CAP  # alias
NAV_FANTASY = "Draft Board"
NAV_ROOKIE = "Rookies"
NAV_PLAYER = "Player Detail"
NAV_METHOD = "Methodology & Research"

# The full ordered list of sidebar sections (the app's only navigation control).
# The QB injury study, the decomposition experiments, and the project report
# live inside Methodology & Research rather than as their own sections.
SECTIONS = [
    NAV_HOME,
    NAV_FANTASY,
    NAV_ROOKIE,
    NAV_PLAYER,
    NAV_CAP,
    NAV_METHOD,
]

# One-line caption under each nav item, so the sidebar itself distinguishes
# the fantasy pages from the front-office view and the methodology material.
NAV_CAPTIONS = {
    NAV_HOME: "Start here",
    NAV_FANTASY: "2026 PPR rankings & accuracy",
    NAV_ROOKIE: "First-year player projections",
    NAV_PLAYER: "Everything on one player",
    NAV_CAP: "Cap value & contract surplus",
    NAV_METHOD: "How it all works",
}

# Team primary colors for player tiles (nflverse team codes). Fallback is the
# brand navy for any code not listed here.
TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00",
    "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LA": "#003594", "LAC": "#0080C6", "LV": "#000000", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#002244",
    "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#0C2340", "WAS": "#5A1414",
}
DEFAULT_TEAM_COLOR = "#0d2b45"


def landing_cards() -> list[dict]:
    """Config for the four takeaway cards on the Home page. Written in plain
    fantasy language: the technical depth lives in the repo, the app translates
    it into something a league-mate can read."""
    return [
        {
            "tag": "Draft prep",
            "headline": "2026 rankings built to beat the averages",
            "points": [
                "Out-projects a player's own recent form by 7–9% in every season since 2020.",
                "Every player gets a floor and a ceiling, not just one number.",
                "Held its own against DraftKings' market prices where the data allows a comparison.",
            ],
            "button": "Open the Draft Board",
            "target": NAV_FANTASY,
        },
        {
            "tag": "Front office",
            "headline": "Who's actually worth the money",
            "points": [
                "Brock Purdy's 2023 season produced about $35M more than a replacement QB, at almost no cap cost.",
                "Cheap young quarterbacks are where the surplus lives; veteran RB deals rarely pay it back.",
                "Cap hits rebuilt season by season from contract terms, not misleading yearly averages.",
            ],
            "button": "Open Front Office",
            "target": NAV_CAP,
        },
        {
            "tag": "Rookies",
            "headline": "Rookie fliers, handicapped honestly",
            "points": [
                "No NFL stats needed: projections built from draft slot, age, and physical profile.",
                "Knows when a rookie is buried behind an established starter, and says so.",
                "Ranges that admit how uncertain rookie seasons really are.",
            ],
            "button": "Open Rookie Projections",
            "target": NAV_ROOKIE,
        },
        {
            "tag": "Under the hood",
            "headline": "Why you can trust these numbers",
            "points": [
                "Every model must beat a hard baseline before it ships.",
                "26 automated checks guard against hindsight leaking into projections.",
                "The ideas that failed are documented, not buried.",
            ],
            "button": "How it works",
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
