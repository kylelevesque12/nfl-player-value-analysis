"""Streamlit entry point for the NFL fantasy app.

This file does four things and nothing else: fix up the import path, purge
stale app modules, set the page up, and route the sidebar selection to one of
the section modules in ``app/sections/``. Every page's content lives in its
own module so it can be found and changed without reading this one.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `from app.components import ...` work regardless of how Streamlit is
# invoked. Streamlit's cwd is the script's directory (app/), not the project
# root, so we prepend the project root to sys.path here.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Purge our own app.* modules so every script run imports them fresh from disk.
# Streamlit Cloud updates code with a git pull + hot reload of THIS file only:
# the long-lived Python process keeps the old app.* modules cached in
# sys.modules, so a new main script importing a name added to app.landing_content
# in the same push raises ImportError until someone manually reboots the app.
# (This took the deployed app down twice.) Re-importing these small pure-config
# modules costs well under a millisecond per run; the heavy third-party imports
# are untouched.
#
# This purge MUST stay above every `from app...` import below, and must keep
# matching the whole app package (not a hand-listed set of modules), or the
# outage comes back the next time a section module gains a new name.
for _mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
    del sys.modules[_mod]

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="NFL Player Value & Fantasy Forecasting",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#157A6E",
    "#C8553D",
    "#3D6B99",
    "#B08900",
    "#7A5C99",
    "#4C956C",
    "#D17A22",
]

from app.data_access import load_all_data, show_missing_data_warning  # noqa: E402
from app.landing_content import (  # noqa: E402
    NAV_CAPTIONS,
    NAV_DRAFTROOM,
    NAV_FANTASY,
    NAV_PLAYER,
    SECTIONS,
)
from app.navigation import (  # noqa: E402
    handle_landing_nav,
    player_index_from_data,
    render_player_search,
)
from app.sections.draft_board import fantasy_section  # noqa: E402
from app.sections.draft_room import draft_room_section  # noqa: E402
from app.sections.home import landing_page  # noqa: E402
from app.sections.player_detail import player_detail_page  # noqa: E402
from app.theme import inject_all_css  # noqa: E402

inject_all_css()


def main() -> None:
    data = load_all_data()
    handle_landing_nav()
    missing = [
        name
        for name, df in data.items()
        if df.empty
        and name
        in {
            "salary",
            "methodology",
            "fantasy",
            "weekly_fantasy",
            "two_stage_projection",
        }
    ]
    show_missing_data_warning(missing)

    player_index = player_index_from_data(data)

    st.sidebar.markdown(
        """
        <div class="side-brand">🏈&nbsp;NFL Player Value
        <span class="sub">&amp; Fantasy Forecasting</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    section = st.sidebar.radio(
        "Section",
        SECTIONS,
        key="nav_section",
        label_visibility="collapsed",
        captions=[NAV_CAPTIONS.get(s, "") for s in SECTIONS],
    )
    render_player_search(player_index)
    st.sidebar.divider()
    st.sidebar.caption(
        "Projections cover the 2016-2025 seasons and the 2026 outlook. "
        "How the models are built, how they are graded, and the full research "
        "write-ups: "
        "[GitHub](https://github.com/kylelevesque12/nfl-player-value-analysis)."
    )

    if section == NAV_FANTASY:
        fantasy_section(data)
    elif section == NAV_DRAFTROOM:
        draft_room_section(data)
    elif section == NAV_PLAYER:
        player_detail_page(data, player_index)
    else:
        landing_page(data)


if __name__ == "__main__":
    main()
