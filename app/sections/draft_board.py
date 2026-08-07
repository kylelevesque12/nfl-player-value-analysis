"""Draft Board: the 2026 rankings, overall and by position."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import fantasy_content as fc
from app.design import section_header
from app.formatting import card_row, col_or_na


def _overall_board_view(data: dict[str, pd.DataFrame]) -> None:
    """Cross-position draft board: VORP ranks, auction values, ADP edge."""
    board = data.get("draft_board", pd.DataFrame())
    if board.empty:
        st.info(
            "The overall board is missing. Run "
            "`python scripts/run_pipeline.py --steps draft_board` "
            "(fetch `python scripts/fetch_adp.py --year 2026` first for the "
            "market-comparison columns; the board still builds without it)."
        )
        return

    top = board.head(75).copy()
    edge = pd.to_numeric(top["edge_vs_adp"], errors="coerce")
    is_rookie = col_or_na(top, "is_rookie_projection").fillna(False).astype(bool)
    player_label = top["player_display_name"] + is_rookie.map({True: " (R)", False: ""})
    show = pd.DataFrame(
        {
            "Rank": top["overall_rank"].astype(int),
            "Player": player_label,
            "Pos": top["position"],
            "Team": col_or_na(top, "primary_team_2025"),
            "Bye": pd.to_numeric(col_or_na(top, "bye"), errors="coerce")
            .fillna(0).astype(int).replace(0, pd.NA),
            "Proj PPR": top["predicted_2026_fantasy_points_ppr"].round(0).astype(int),
            "VORP": top["vorp"].round(0).astype(int),
            "$": top["auction_value"].astype(int),
            "ADP": col_or_na(top, "adp_formatted"),
            "Edge": edge.round(0),
        }
    )
    st.subheader("Overall draft board: top 75 by value over replacement")
    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
        height=740,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Player": st.column_config.TextColumn(
                "Player",
                width="medium",
                help="(R) marks a 2026 rookie: no NFL stats yet, projected "
                "from draft capital, age, and profile via a Bayesian model.",
            ),
            "VORP": st.column_config.NumberColumn(
                "VORP",
                width="small",
                help="Value over replacement player: projected points above "
                "the best freely available player at the position once every "
                "starting lineup in a 12-team league is filled. This is what "
                "makes players comparable across positions.",
            ),
            "$": st.column_config.NumberColumn(
                "$",
                width="small",
                format="$%d",
                help="Auction value for a 12-team, $200-budget league: a $1 "
                "floor everywhere, with the league's discretionary budget "
                "split in proportion to positive VORP.",
            ),
            "ADP": st.column_config.TextColumn(
                "ADP",
                width="small",
                help="Average draft position (round.pick) across real 12-team "
                "PPR drafts on Fantasy Football Calculator.",
            ),
            "Edge": st.column_config.NumberColumn(
                "Edge",
                width="small",
                format="%+d",
                help="ADP overall rank minus the model's overall rank. "
                "Positive: the market lets you draft him later than the model "
                "ranks him (a value). Negative: the market takes him earlier "
                "(a fade). Blank: not drafted in the ADP sample.",
            ),
        },
    )
    meta_drafts = board.get("adp_total_drafts", pd.Series(dtype="float64")).dropna()
    meta_end = board.get("adp_window_end", pd.Series(dtype="object")).dropna()
    if not meta_drafts.empty and not meta_end.empty:
        st.caption(
            f"ADP snapshot: {int(meta_drafts.iloc[0]):,} real 12-team PPR "
            f"drafts through {meta_end.iloc[0]}. "
            "2026 rookies are not on the board yet — they join once the "
            "rookie class is scored (see the roadmap)."
        )
    st.download_button(
        "Download the full overall board",
        board.to_csv(index=False),
        file_name="draft_board_2026.csv",
        mime="text/csv",
    )


def espn_fantasy_view(data: dict[str, pd.DataFrame]) -> None:
    """Fantasy rankings: top-25 2026 season projections per position, plus
    week-by-week projection-vs-actual for completed games. Table-first."""
    fantasy = data["fantasy"]
    weekly = data["weekly_fantasy"]

    if fantasy.empty:
        st.info(
            "Fantasy projections are missing. Run "
            "`python scripts/run_pipeline.py --steps fantasy`."
        )
        return

    position = st.radio(
        "Position", ["Overall", "QB", "RB", "WR", "TE"], horizontal=True, key="rank_pos"
    )
    if position == "Overall":
        _overall_board_view(data)
        return

    pos = (
        fantasy[fantasy["position"].eq(position)]
        .sort_values("predicted_2026_fantasy_points_ppr", ascending=False)
        .head(25)
        .reset_index(drop=True)
        .copy()
    )
    pos.insert(0, "Rank", range(1, len(pos) + 1))
    team_col = "primary_team_2025" if "primary_team_2025" in pos.columns else "team"

    low = pos["prediction_interval_low"].round(0)
    high = pos["prediction_interval_high"].round(0)
    profile_short = (
        pos.get("fantasy_projection_tier", pd.Series([""] * len(pos)))
        .astype(str)
        .str.replace(" Fantasy Profile", "", regex=False)
        .str.replace(" Profile", "", regex=False)
    )
    tiers = fc.assign_tiers(pos)
    badges = fc.stability_labels(data.get("two_stage_projection", pd.DataFrame()))
    role = (
        pos.merge(badges, on="player_id", how="left")["role_badge"].fillna("")
        if not badges.empty
        else pd.Series([""] * len(pos))
    )
    is_rookie = pos.get("is_rookie_projection", pd.Series(False, index=pos.index)).fillna(False)
    # Injury-shortened prior season: a non-rookie who played <= 8 games in
    # 2025. The projection rests on that small sample, so the honest read is
    # the wide 80% range, not the point estimate (see the injury-return note).
    gp_2025 = pd.to_numeric(pos.get("games_played_2025"), errors="coerce")
    injury_short = (gp_2025 <= 8) & (~is_rookie.astype(bool)) & gp_2025.notna()
    player_label = (
        pos["player_display_name"]
        + is_rookie.map({True: " (R)", False: ""})
        + injury_short.map({True: " ⚕", False: ""})
    )
    ranking = pd.DataFrame({
        "Rank": pos["Rank"].astype(int),
        "Tier": tiers.reindex(pos.index).astype(int),
        "Player": player_label,
        "Team": pos.get(team_col, ""),
        "Proj PPR": pos["predicted_2026_fantasy_points_ppr"].round(1),
        "PPR/G": pos["predicted_2026_ppr_per_game"].round(1),
        "GP": pos["predicted_2026_games_played"].round(0).astype(int),
        "80% range": [f"{lo:.0f}–{hi:.0f}" for lo, hi in zip(low, high)],
        "Role": role,
        "Profile": profile_short,
    })
    if "projection_change_from_2025" in pos.columns:
        ranking["Δ vs '25"] = pos["projection_change_from_2025"].round(1)

    proj_max = float(pos["predicted_2026_fantasy_points_ppr"].max() or 1.0)
    st.subheader(f"Top 25 {position}s: 2026 projected PPR")
    st.dataframe(
        ranking,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Player": st.column_config.TextColumn(
                "Player",
                width="medium",
                help="(R) marks a 2026 rookie: no NFL stats yet, projected "
                "from draft capital, age, and profile via a Bayesian model "
                "rather than the veteran production model. ⚕ marks a player "
                "who missed most of 2025 to injury (8 or fewer games): his "
                "point projection rests on a small sample, so read the 80% "
                "range, which is wide on purpose. The model does not try to "
                "guess his exact bounce-back — those seasons are genuinely "
                "high-variance.",
            ),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Proj PPR": st.column_config.ProgressColumn(
                "Proj PPR",
                help="Projected 2026 PPR points",
                format="%.1f",
                min_value=0.0,
                max_value=proj_max,
            ),
            "PPR/G": st.column_config.NumberColumn("PPR/G", width="small", format="%.1f"),
            "GP": st.column_config.NumberColumn("GP", width="small", format="%d"),
            "80% range": st.column_config.TextColumn(
                "80% range",
                width="small",
                help="80% prediction interval: the actual total should land in "
                "this range about 8 times in 10. Season totals are genuinely "
                "hard to predict, so honest ranges are wide.",
            ),
            "Tier": st.column_config.NumberColumn(
                "Tier",
                width="small",
                help="Players in the same tier are close enough that the model "
                "cannot confidently order them — take the tier, not the exact "
                "rank. A new tier starts where the drop in projected points is "
                "large relative to the projections' own uncertainty.",
            ),
            "Role": st.column_config.TextColumn(
                "Role",
                width="small",
                help="Stable = the projection rests on role (targets, snaps, "
                "carries), which repeats strongly year to year. Shaky = it "
                "leans on per-play efficiency, which barely repeats for "
                "RB/WR/TE. QBs get no badge: QB efficiency genuinely repeats, "
                "so the label would mislead there.",
            ),
            "Profile": st.column_config.TextColumn("Profile", width="medium"),
            "Δ vs '25": st.column_config.NumberColumn(
                "Δ vs '25",
                width="small",
                format="%+.1f",
                help="Projected 2026 total minus the player's actual 2025 total. "
                "Negative for most top players by design: career-best seasons "
                "tend to regress toward the mean, so the model projects below "
                "last year's peak.",
            ),
        },
    )
    st.download_button(
        f"Download {position} rankings",
        ranking.to_csv(index=False),
        file_name=f"fantasy_rankings_2026_{position}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Weekly projection vs actual")

    options = pos["player_id"].tolist()
    labels = dict(zip(pos["player_id"], pos["player_display_name"]))
    sel = st.selectbox(
        "Player", options, format_func=lambda p: labels.get(p, p), key="rank_weekly_player"
    )

    wk = weekly.copy()
    if "method" in wk.columns:
        wk = wk[wk["method"].eq("hist_gradient_boosting")]
    wk = wk[wk["player_id"].astype(str).eq(str(sel))].copy()
    if wk.empty:
        st.info("No completed-game projections on record for this player.")
        return

    latest = int(pd.to_numeric(wk["season"], errors="coerce").max())
    wk = wk[wk["season"].eq(latest)].sort_values("week")
    proj = wk["prediction"].to_numpy()
    actual = wk["target_fantasy_points_ppr"].to_numpy()

    weekly_table = pd.DataFrame({
        "Week": wk["week"].astype(int),
        "Opp": wk.get("opponent_team", ""),
        "Projected": wk["prediction"].round(1),
        "Actual": wk["target_fantasy_points_ppr"].round(1),
        "Error": (wk["target_fantasy_points_ppr"] - wk["prediction"]).round(1),
    })
    card_row([
        ("Games", f"{len(wk)}", None),
        ("Avg projected", f"{proj.mean():.1f}", None),
        ("Avg actual", f"{actual.mean():.1f}", None),
        ("RMSE", f"{float(np.sqrt(np.mean((actual - proj) ** 2))):.1f}", None),
    ])
    st.caption(f"{labels.get(sel, sel)}, {latest} season, projected vs actual PPR by game.")
    st.dataframe(weekly_table, width="stretch", hide_index=True)



def fantasy_section(data: dict[str, pd.DataFrame]) -> None:
    section_header(
        "Fantasy",
        "Draft Board",
        "2026 PPR projections by position, with tiers and honest ranges.",
    )
    espn_fantasy_view(data)
    st.divider()
    with st.expander("How these projections are built and graded"):
        st.markdown(
            "Two models feed this section. Season-long 2026 totals come from an "
            "Elastic Net (a disciplined linear model chosen from six candidates by "
            "lowest validation error). Weekly scores come from a gradient-boosting "
            "model (many small decision trees, each correcting the last) using only "
            "information known before kickoff (recent form, opponent, betting lines, "
            "weather, injury status). Accuracy is judged the way forecasters do: "
            "against a strong naive baseline (a player's recent-game average), which the "
            "weekly model beats by a steady 7–9% across six seasons, and against a "
            "DraftKings-implied market line on 2020–2021, where it is competitive to "
            "slightly ahead.\n\n"
            "**Tier** groups players the model cannot confidently separate: a new "
            "tier starts only where the drop in projected points is large relative "
            "to the projections' own uncertainty, so within a tier you should take "
            "the player you prefer, not the higher row. **Role** comes from the "
            "season value model's decomposition: production built on role "
            "(targets, snaps, carries) repeats strongly year to year, while "
            "per-play efficiency barely repeats for RB/WR/TE — so *Stable* marks "
            "role-driven projections and *Shaky* marks efficiency-driven ones. "
            "QBs get no Role badge because QB efficiency is the documented "
            "exception that does repeat."
        )

