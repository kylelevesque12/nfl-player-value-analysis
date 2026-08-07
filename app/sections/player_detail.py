"""Player Detail: every saved output the project has for one player."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app import player_search as ps
from app.components import caveat_callout, source_footer
from app.formatting import card_row, kpi_or_dash
from app.landing_content import NAV_HOME
from app.navigation import go_to


def player_detail_page(data: dict[str, pd.DataFrame], index: pd.DataFrame) -> None:
    pid = st.session_state.get("_selected_player_id")
    if not pid:
        st.title("Player Detail")
        st.info("Search for a player in the sidebar to open a detail view.")
        return

    detail = ps.assemble_player_detail(
        pid,
        weekly=data.get("weekly_fantasy"),
        live=data.get("weekly_fantasy_live"),
        salary=data.get("salary"),
        top_surplus=data.get("replacement_top_surplus"),
        rookie=data.get("rookie_modeling_frame"),
        rookie_pred=data.get("rookie_bayes_validation_predictions"),
        causal=data.get("causal_s3_events"),
    )
    meta = index[index["player_id"].astype(str) == str(pid)] if not index.empty else index
    meta_row = meta.iloc[0] if meta is not None and not meta.empty else None
    name = detail["player_name"]
    pos = meta_row["position"] if meta_row is not None else None
    team = meta_row["team"] if meta_row is not None else None
    seasons = meta_row["seasons"] if meta_row is not None else "—"

    st.title(str(name))
    bits = " · ".join([str(x) for x in [pos, team, f"seasons {seasons}"] if x and pd.notna(x)])
    st.caption(f"Unified player view, every project output available for this player. {bits}")
    if st.button("← Back to dashboard"):
        st.session_state["_selected_player_id"] = None
        go_to(NAV_HOME)

    wk = detail["weekly_history"]
    live = detail["live"]
    sal = detail["surplus_history"]
    rk = detail["rookie"]
    rk_pred = detail["rookie_pred"]
    cz = detail["causal"]

    # ---- KPI row ----
    latest_proj = None
    if live is not None and "projected_points" in live.columns:
        latest_proj = float(live["projected_points"].iloc[0])
    elif wk is not None and "prediction" in wk.columns and not wk["prediction"].dropna().empty:
        latest_proj = float(wk["prediction"].dropna().iloc[-1])
    depth_rank = None
    if wk is not None and "pbp_depth_chart_rank_last1" in wk.columns:
        dr = wk["pbp_depth_chart_rank_last1"].dropna()
        depth_rank = float(dr.iloc[-1]) if not dr.empty else None
    best_surplus = None
    if sal is not None and "value_above_expected_salary" in sal.columns:
        v = sal["value_above_expected_salary"].dropna()
        best_surplus = float(v.max()) if not v.empty else None
    n_causal = 0 if cz is None else len(cz)

    card_row(
        [
            ("Latest projected PPR", kpi_or_dash(latest_proj),
             "Live week if available, else most recent backtest game."),
            ("Latest depth rank (PBP)", kpi_or_dash(depth_rank, "{:.0f}"),
             "1 = top of the position group."),
            ("Best value-over-expected ($M)", kpi_or_dash(best_surplus),
             "Peak surplus season on record." if best_surplus is not None else "No salary record."),
            ("Causal QB events", f"{n_causal}",
             "First-injury-report events where this player was the treated QB."),
        ]
    )

    # ---- Weekly fantasy ----
    st.subheader("Weekly fantasy")
    if wk is None:
        st.info("No weekly fantasy projection history for this player.")
    else:
        plot = wk.copy()
        plot["game"] = plot["season"].astype(int).astype(str) + "-W" + plot["week"].astype(int).astype(str).str.zfill(2)
        ycols = [c for c in ["prediction", "target_fantasy_points_ppr"] if c in plot.columns]
        fig = px.line(
            plot, x="game", y=ycols,
            labels={"value": "PPR points", "game": "Game", "variable": ""},
            title="Projected vs actual PPR by game (production HGB)",
        )
        fig.update_layout(height=380, xaxis=dict(showticklabels=False))
        st.plotly_chart(fig, width="stretch")
        if live is not None:
            lrow = live.iloc[0]
            opp = lrow.get("opponent_team", "")
            lo = lrow.get("interval_low_80"); hi = lrow.get("interval_high_80")
            st.markdown(
                f"**Upcoming-week projection:** {lrow.get('projected_points', float('nan')):.1f} PPR "
                f"vs {opp}, 80% band [{lo:.1f}, {hi:.1f}]." if pd.notna(lo) else
                f"**Upcoming-week projection:** {lrow.get('projected_points', float('nan')):.1f} PPR vs {opp}."
            )
        with st.expander("Weekly projection table"):
            cols = [c for c in ["season", "week", "team", "opponent_team", "prediction",
                                "interval_low_80", "interval_high_80", "target_fantasy_points_ppr"]
                    if c in wk.columns]
            st.dataframe(wk[cols].tail(40), width="stretch", hide_index=True)

    # ---- Value / surplus ----
    st.subheader("Value & cap surplus")
    if sal is None:
        st.info("No salary / value record for this player.")
    else:
        cols = [c for c in ["season", "team", "games_played", "value_score", "salary_millions",
                            "value_above_expected_salary", "salary_efficiency_tier", "salary_source"]
                if c in sal.columns]
        show = sal[cols].rename(columns={"salary_millions": "cap_hit_$M"})
        st.dataframe(show, width="stretch", hide_index=True)
        caveat_callout(
            "Cap cost is a season-specific cap hit reconstructed from contract terms "
            "(prorated signing bonus + backloaded base), an estimate, not exact NFL "
            "cap accounting. See the salary_source column.",
            "Reconstructed estimate",
        )
        if detail["top_surplus"] is not None:
            st.caption("This player appears in the top-25 replacement-level surplus board.")

    # ---- Rookie model ----
    st.subheader("Rookie model")
    if rk is None:
        st.info("This player is not in the rookie modeling frame.")
    else:
        r = rk.iloc[0]
        ry = int(r["rookie_year"]) if "rookie_year" in r and pd.notna(r["rookie_year"]) else None
        dn = r.get("draft_number"); played = r.get("played_meaningfully")
        st.markdown(
            f"- Rookie year: **{ry or '—'}**  ·  draft pick: "
            f"**{int(dn) if pd.notna(dn) else '—'}**  ·  played meaningfully (>=4 games): "
            f"**{'Yes' if pd.notna(played) and int(played) == 1 else 'No'}**"
        )
        if rk_pred is not None and "predicted_ppr_per_game_mean" in rk_pred.columns:
            pr = rk_pred.iloc[0]
            st.markdown(
                f"- Bayesian projection: **{pr['predicted_ppr_per_game_mean']:.1f} PPR/game** "
                "(validation class)."
            )
        st.caption(
            "A 3-feature incumbent-context core sharpens the hurdle gate "
            "(combine and broad-depth features were tested and dropped). The gain is "
            "concentrated in the blocked-QB cell."
        )

    # ---- Causal study ----
    st.subheader("Causal study (QB injury)")
    if cz is None:
        st.info("This player is not a treated QB in the first-injury-report causal panel.")
    else:
        cols = [c for c in ["season", "team", "event_week", "first_injury_status",
                            "games_started_before_event"] if c in cz.columns]
        st.dataframe(cz[cols], width="stretch", hide_index=True)
        caveat_callout(
            "The first-report causal effect (ATT ~= -0.58 PPG) is suggestive and "
            "underpowered; appearing here means the player was a treated QB, not that "
            "a specific effect is attributed to him.",
            "Suggestive / underpowered",
        )

    with st.expander("ID diagnostics"):
        st.write({"player_id (gsis)": str(pid), "display_name": str(name)})

    source_footer(
        "Assembled from saved outputs only (weekly backtest, live projection, salary/"
        "value, rookie frame, causal events), no models are recomputed in the app."
    )

