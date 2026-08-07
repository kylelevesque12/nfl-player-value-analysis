"""Sidebar navigation and the deferred section-jump used across the app.

Streamlit will not let a widget's key be reassigned after the widget has been
instantiated, so a button that wants to change section stashes its target and
reruns; ``handle_landing_nav`` applies it at the top of the next run, before
the sidebar radio is built.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import player_search as ps
from app.landing_content import NAV_PLAYER


def go_to(target: str) -> None:
    """Defer navigation to the next run (radio keys can't be set after the
    widgets are instantiated). Handled by handle_landing_nav at top of main."""
    st.session_state["_landing_goto"] = target
    st.rerun()


def handle_landing_nav() -> None:
    goto = st.session_state.pop("_landing_goto", None)
    if not goto:
        return
    # Single-section navigation: every target is one sidebar section.
    st.session_state["nav_section"] = goto



def render_player_search(index: pd.DataFrame) -> None:
    """Always-visible sidebar player search. Selecting a player navigates to the
    unified Player Detail view (reuses the deferred-nav pattern)."""
    st.sidebar.divider()
    st.sidebar.markdown("### Player search")
    if index is None or index.empty:
        st.sidebar.caption("Player index unavailable.")
        return
    label_map = {row["player_id"]: ps.display_label(row) for _, row in index.iterrows()}
    options = [""] + index["player_id"].tolist()
    choice = st.sidebar.selectbox(
        "Type a player name",
        options,
        format_func=lambda p: "type to search…" if p == "" else label_map.get(p, p),
        key="player_search_select",
    )
    if choice and choice != st.session_state.get("_selected_player_id"):
        st.session_state["_selected_player_id"] = choice
        go_to(NAV_PLAYER)


def player_index_from_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return ps.build_player_index(
        data.get("weekly_fantasy"),
        data.get("weekly_fantasy_live"),
        data.get("salary"),
        data.get("rookie_modeling_frame"),
        data.get("causal_s3_events"),
    )

