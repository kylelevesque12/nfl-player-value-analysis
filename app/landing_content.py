"""Pure (Streamlit-free) content + navigation config for the landing page.

Kept separate from ``streamlit_app.py`` so the copy, methodology labels, and
navigation targets can be unit-tested without a Streamlit runtime.
"""

from __future__ import annotations

LANDING_TITLE = "NFL Player Value & Fantasy Forecasting"
LANDING_SUBTITLE = (
    "2026 PPR rankings with honest floors, ceilings, and tiers, plus a Draft "
    "Room that plans your whole draft, not just your next pick."
)

# Navigation targets, must match the sidebar radio option strings exactly.
# A single sidebar selects one section. Every section is user-facing fantasy
# product; the front-office surplus study, the rookie methodology, the QB
# injury study, the decomposition experiments, and the project report all live
# as research summaries inside Methodology & Research (full depth in the repo).
NAV_HOME = "Home"
NAV_FANTASY = "Draft Board"
NAV_DRAFTROOM = "Draft Room"
NAV_PLAYER = "Player Detail"
NAV_METHOD = "Methodology & Research"

# The full ordered list of sidebar sections (the app's only navigation control).
SECTIONS = [
    NAV_HOME,
    NAV_FANTASY,
    NAV_DRAFTROOM,
    NAV_PLAYER,
    NAV_METHOD,
]

# One-line caption under each nav item.
NAV_CAPTIONS = {
    NAV_HOME: "Start here",
    NAV_FANTASY: "2026 rankings, tiers & accuracy",
    NAV_DRAFTROOM: "Plan your whole draft",
    NAV_PLAYER: "Everything on one player",
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


def methodology_strip_labels() -> list[str]:
    return [
        "Leakage-safe features",
        "Time-based validation",
        "Negative results documented",
        "Source/quality flags for salary estimates",
        "Tests covering key modeling assumptions",
    ]
