"""Draft Room: league setup, live pick tracking, and the whole-draft plan."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app import draft_planner
from app import fantasy_content as fc
from app.data_access import GITHUB_BLOB_BASE
from app.design import caveat_callout, section_header


def _draft_room_player_label(row: pd.Series, team_col: str | None) -> str:
    team = ""
    if team_col and pd.notna(row.get(team_col)):
        team = f", {row[team_col]}"
    return f"{row['player_display_name']} ({row['position']}{team})"


def _draft_room_planner(board: pd.DataFrame) -> None:
    """League setup, live pick tracking, and the whole-draft plan."""
    st.session_state.setdefault("dr_log", [])
    log: list[str] = st.session_state["dr_log"]

    with st.expander("League setup", expanded=not log):
        cols = st.columns(3)
        teams = cols[0].number_input("Teams", min_value=2, max_value=20, value=12, key="dr_teams")
        my_slot = cols[1].number_input(
            "Your pick slot", min_value=1, max_value=int(teams), value=1, key="dr_slot"
        )
        rounds = cols[2].number_input("Rounds", min_value=1, max_value=25, value=15, key="dr_rounds")
        st.caption(
            "Standard roster assumed: QB, 2 RB, 2 WR, TE, one RB/WR/TE flex, plus bench."
        )
    teams = int(st.session_state["dr_teams"])
    my_slot = min(int(st.session_state["dr_slot"]), teams)
    rounds = int(st.session_state["dr_rounds"])

    total_picks = teams * rounds
    current_pick = len(log) + 1
    draft_over = current_pick > total_picks

    drafted_ids = set(log)
    planner_board = draft_planner.build_board(board, drafted_ids=drafted_ids)
    team_col = "primary_team_2025" if "primary_team_2025" in board.columns else None

    st.subheader("Draft complete" if draft_over else f"Pick #{current_pick}")
    if not draft_over:
        on_the_clock = draft_planner.is_my_pick(current_pick, teams, my_slot)
        if on_the_clock:
            st.success("**You're on the clock.**")
        else:
            st.info(f"Team {draft_planner.slot_on_the_clock(current_pick, teams)} is on the clock.")

    undrafted = board[~board["player_id"].astype(str).isin(drafted_ids)].sort_values(
        "overall_rank"
    ).copy()
    undrafted["_label"] = undrafted.apply(_draft_room_player_label, axis=1, team_col=team_col)
    options = undrafted["player_id"].astype(str).tolist()
    label_map = dict(zip(undrafted["player_id"].astype(str), undrafted["_label"]))

    pick_cols = st.columns([3, 1, 1])
    with pick_cols[0]:
        selected = st.selectbox(
            f"Who was taken with pick #{current_pick}?" if not draft_over else "Draft complete",
            options,
            format_func=lambda pid: label_map.get(pid, pid),
            index=None,
            placeholder="Search a player…",
            disabled=draft_over or not options,
        )
    with pick_cols[1]:
        if st.button("Record pick", width="stretch", disabled=draft_over or selected is None):
            log.append(selected)
            st.rerun()
    with pick_cols[2]:
        if st.button("Undo last", width="stretch", disabled=not log):
            log.pop()
            st.rerun()
    if st.button("Reset draft"):
        st.session_state["dr_log"] = []
        st.rerun()

    if not draft_over:
        plan = draft_planner.plan_whole_draft(
            planner_board, current_pick=current_pick, teams=teams, my_slot=my_slot, rounds=rounds
        )
        st.markdown(draft_planner.recommendation_text(plan, on_the_clock))

        if not plan.cost_of_waiting.empty:
            st.markdown("**Cost of waiting one round at each position**")
            cow = plan.cost_of_waiting.rename(
                columns={
                    "position": "Position",
                    "best_now": "Best available now",
                    "best_at_next_pick": "Best at your next pick",
                    "points_lost_if_you_wait": "Points lost if you wait",
                }
            )
            st.dataframe(cow.round(1), width="stretch", hide_index=True)

        if plan.steps:
            st.markdown("**Your plan from here**")
            plan_df = pd.DataFrame(
                [
                    {
                        "Pick #": s.pick_number,
                        "Position": s.position,
                        "Player": s.player_name,
                        "Value (VORP)": round(s.value, 1),
                    }
                    for s in plan.steps
                ]
            )
            st.dataframe(plan_df, width="stretch", hide_index=True)

        caveat_callout(
            "The plan assumes opponents draft the best remaining player by "
            "market ADP; real drafters have their own needs and biases. "
            "Adjacent players are also often within the projections' own "
            "uncertainty, so use the plan to see where the value is, not as "
            "an exact script. A planned Monte Carlo upgrade replaces this "
            "single assumption with a distribution over many simulated "
            "drafts.",
            "Deterministic v1",
        )

    my_ids = [pid for i, pid in enumerate(log) if draft_planner.is_my_pick(i + 1, teams, my_slot)]
    if my_ids:
        st.markdown("### Your roster so far")
        mine = board[board["player_id"].astype(str).isin(my_ids)]
        show = pd.DataFrame(
            {
                "Player": mine["player_display_name"],
                "Pos": mine["position"],
                "Proj PPR": mine["predicted_2026_fantasy_points_ppr"].round(0).astype(int),
                "VORP": mine["vorp"].round(0).astype(int),
            }
        )
        st.dataframe(show, width="stretch", hide_index=True)

    with st.expander("How the plan is built"):
        st.markdown(
            "The planner walks the rest of the draft forward, one shared "
            "timeline: at every pick that isn't yours, it assumes the best "
            "remaining player by market ADP gets taken, and at every one of "
            "your picks it searches all the positions you could draft. "
            "Because your own picks and the simulated opponent picks draw "
            "from the same pool, nothing is ever double-counted — taking a "
            "player now genuinely removes him from every later pick's "
            "shelf. The plan searches every sequence up to your team's "
            "starting roster (position minimums plus the flex slot) and "
            f"returns the highest-value sequence. Full write-up: "
            f"[report/draft_room_planner.md]({GITHUB_BLOB_BASE}/report/draft_room_planner.md)."
        )


def _draft_room_scarcity_reference(data: dict[str, pd.DataFrame]) -> None:
    """Supporting reference: the positional scarcity chart the planner is
    built on, for a quick look during a live draft."""
    fantasy = data.get("fantasy", pd.DataFrame())
    scarcity = fc.scarcity_frame(fantasy)
    if scarcity.empty:
        return

    st.subheader("Positional scarcity: where the cliffs are")
    st.caption(
        "Each line is one position's projected points by positional rank. A "
        "steep line falls off fast, so waiting a round costs real points."
    )
    fig = px.line(
        scarcity,
        x="positional_rank",
        y="projected_points",
        color="position",
        hover_data={"player": True, "positional_rank": True, "projected_points": ":.0f"},
        labels={
            "positional_rank": "Positional rank",
            "projected_points": "Projected 2026 PPR",
            "position": "",
        },
        color_discrete_map={
            "QB": "#C8553D", "RB": "#157A6E", "WR": "#3D6B99", "TE": "#B08900",
        },
    )
    fig.update_layout(height=380, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    dropoffs = fc.starter_window_dropoffs(scarcity)
    if not dropoffs.empty:
        with st.expander("The cost of drafting the 12th-best instead of the best"):
            show = dropoffs.rename(
                columns={
                    "position": "Position",
                    "top_projection": "Best (proj PPR)",
                    "rank12_projection": "12th-best (proj PPR)",
                    "dropoff": "Points lost",
                }
            )
            st.dataframe(show.round(0), width="stretch", hide_index=True)


def draft_room_section(data: dict[str, pd.DataFrame]) -> None:
    """Plan the whole draft: league setup, live pick tracking, and a
    dynamic-program plan for every remaining pick, with the positional
    scarcity picture kept below as reference."""
    section_header(
        "Draft prep",
        "Draft Room",
        "Plan the whole draft, not just the next pick.",
    )

    board = data.get("draft_board", pd.DataFrame())
    if board.empty:
        st.info(
            "The overall board is missing. Run "
            "`python scripts/run_pipeline.py --steps draft_board` "
            "(fetch `python scripts/fetch_adp.py --year 2026` first for the "
            "market-comparison columns; the board still builds without it)."
        )
        return

    _draft_room_planner(board)
    st.divider()
    _draft_room_scarcity_reference(data)

