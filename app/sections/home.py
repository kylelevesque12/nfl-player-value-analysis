"""Home: the page a league-mate lands on with no context."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import fantasy_content as fc
from app.landing_content import (
    LANDING_TITLE,
    LANDING_SUBTITLE,
    NAV_FANTASY,
    TEAM_COLORS,
    DEFAULT_TEAM_COLOR,
)
from app.navigation import go_to


def _top_projected_strip(data: dict[str, pd.DataFrame]) -> None:
    """ESPN-style strip: the top five projected players for 2026 as team-color
    tiles, with position quick-links into the Draft Board."""
    fantasy = data.get("fantasy", pd.DataFrame())
    if fantasy.empty or "predicted_2026_fantasy_points_ppr" not in fantasy.columns:
        return

    st.markdown("### Top projected players, 2026")
    top = fantasy.sort_values(
        "predicted_2026_fantasy_points_ppr", ascending=False
    ).head(5)
    team_col = "primary_team_2025" if "primary_team_2025" in top.columns else "team"
    cols = st.columns(len(top))
    for col, (rank, (_, row)) in zip(cols, enumerate(top.iterrows(), start=1)):
        team = str(row.get(team_col, "") or "")
        color = TEAM_COLORS.get(team, DEFAULT_TEAM_COLOR)
        ppg = row.get("predicted_2026_ppr_per_game")
        ppg_txt = f"{ppg:.1f} per game" if pd.notna(ppg) else ""
        with col:
            st.markdown(
                f"""
                <div class="player-tile" style="--team-color:{color}">
                    <div class="rank">#{rank} · {row['position']}</div>
                    <div class="name">{row['player_display_name']}</div>
                    <div class="meta">{team} · {ppg_txt}</div>
                    <div class="points">{row['predicted_2026_fantasy_points_ppr']:.0f}
                    <span>proj PPR</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("Full rankings by position, with floors and ceilings:")
    pos_cols = st.columns(4)
    for col, pos in zip(pos_cols, ["QB", "RB", "WR", "TE"]):
        with col:
            if st.button(f"{pos} rankings", key=f"home_pos_{pos}", width="stretch"):
                st.session_state["rank_pos"] = pos
                go_to(NAV_FANTASY)


def _player_content_modules(data: dict[str, pd.DataFrame]) -> None:
    """Home player-content modules: draft-day values, projected risers, the
    regression watch, and (once the current class has been scored) rookie
    fliers."""
    fantasy = data.get("fantasy", pd.DataFrame())
    two_stage = data.get("two_stage_projection", pd.DataFrame())
    board = data.get("draft_board", pd.DataFrame())

    values = fc.draft_values_frame(board)
    risers = fc.risers_frame(fantasy)
    watch = fc.regression_watch_frame(fantasy, two_stage)
    rookies = fc.rookie_fliers_frame(fantasy)
    if values.empty and risers.empty and watch.empty and rookies.empty:
        return

    col_values, col_risers, col_watch = st.columns(3)
    with col_values:
        st.markdown("### Draft-day values")
        st.caption(
            "The market drafts these players well after where the model "
            "ranks them. Edge is the gap in overall rank."
        )
        if values.empty:
            st.info("Values need the ADP snapshot and the overall board.")
        else:
            show = pd.DataFrame(
                {
                    "Player": values["player_display_name"],
                    "Pos": values["position"],
                    "ADP": values.get("adp_formatted", ""),
                    "Edge": values["edge_vs_adp"].round(0).astype(int),
                }
            )
            st.dataframe(show, width="stretch", hide_index=True)

    with col_risers:
        st.markdown("### Projected risers")
        st.caption(
            "Projected to beat last season's total by the most. Several are "
            "returns from injury-shortened seasons."
        )
        if risers.empty:
            st.info("Riser data unavailable.")
        else:
            show = pd.DataFrame(
                {
                    "Player": risers["player_display_name"],
                    "Pos": risers["position"],
                    "Proj": risers[fc.PROJ_COL].round(0).astype(int),
                    "vs '25": risers[fc.DELTA_COL].round(0).astype(int),
                }
            )
            st.dataframe(show, width="stretch", hide_index=True)

    with col_watch:
        st.markdown("### Regression watch")
        st.caption(
            "Big seasons that leaned on per-play efficiency, which barely "
            "repeats for RB/WR/TE. Role-driven players are safer."
        )
        if watch.empty:
            st.info("Regression-watch data unavailable.")
        else:
            show = pd.DataFrame(
                {
                    "Player": watch["player_display_name"],
                    "Pos": watch["position"],
                    "Proj": watch[fc.PROJ_COL].round(0).astype(int),
                    "vs '25": watch[fc.DELTA_COL].round(0).astype(int),
                }
            )
            st.dataframe(show, width="stretch", hide_index=True)

    st.caption(
        "Why no quarterbacks on the regression watch: QB efficiency is the "
        "one kind that genuinely repeats, so the fade-the-fluke logic does "
        "not apply at that position."
    )

    if not rookies.empty:
        st.divider()
        st.markdown("### Rookie fliers")
        st.caption(
            "Top projected rookies from the current draft class — no NFL "
            "stats yet, so these come from a Bayesian model on draft "
            "capital, age, physical profile, and incumbent context. "
            "P(plays) is the modeled chance he wins a meaningful role in 2026."
        )
        show = pd.DataFrame(
            {
                "Player": rookies["player_display_name"],
                "Pos": rookies["position"],
                "Team": rookies["team"],
                "Pick": rookies["draft_number"].astype("Int64"),
                "P(plays)": rookies["p_plays"].apply(
                    lambda v: f"{v:.0%}" if pd.notna(v) else "—"
                ),
                "Proj PPR": rookies[fc.PROJ_COL].round(0).astype(int),
            }
        )
        st.dataframe(show, width="stretch", hide_index=True)


def landing_page(data: dict[str, pd.DataFrame]) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{LANDING_TITLE}</h1>
            <p>{LANDING_SUBTITLE}</p>
            <span class="pill">2026 draft prep</span>
            <span class="pill">Honest ranges</span>
            <span class="pill">QB · RB · WR · TE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _top_projected_strip(data)

    st.divider()
    _player_content_modules(data)

    st.divider()
    with st.expander("How to use this app"):
        st.markdown(
            "The **Draft Board** has the 2026 rankings with tiers, floors, and "
            "ceilings. The **Draft Room** plans your whole draft, not just your "
            "next pick, and tracks picks as they happen. **Player Detail** "
            "assembles everything on one player.\n\n"
            "Projections are ranges, not promises. Where two players sit in the "
            "same tier, the model cannot confidently separate them — take the "
            "one you prefer."
        )

