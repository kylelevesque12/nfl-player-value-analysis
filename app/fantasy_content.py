"""Pure (Streamlit-free) builders for the fantasy player-content modules.

These turn the committed output tables into the frames the app renders on Home,
the Draft Board, and the Draft Room: projection tiers, the positional scarcity
curve, projected risers, the regression watch, and the stable/shaky role badge.
Kept Streamlit-free so every rule here can be unit-tested directly.

The stability signal comes from the season value model's decomposition
(`two_stage_2026_projection.csv`): `efficiency_variance_share` is the share of a
player's projection uncertainty that comes from per-play efficiency rather than
role. Efficiency barely repeats year to year for non-QBs (lag-1 correlation
0.18-0.25) while role repeats strongly (~0.76), so a projection that leans on
efficiency is genuinely shakier than one resting on role.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PROJ_COL = "predicted_2026_fantasy_points_ppr"
DELTA_COL = "projection_change_from_2025"
LOW_COL = "prediction_interval_low"
HIGH_COL = "prediction_interval_high"

# efficiency_variance_share cutoffs for the role badge. Roughly the observed
# quartiles of the 505-player table: above the top quartile the projection
# leans on efficiency (shaky), below the bottom quartile it rests on role
# (stable). The middle half gets no badge rather than a manufactured label.
SHAKY_SHARE = 0.75
STABLE_SHARE = 0.45

# The efficiency-doesn't-repeat logic applies to skill players only. QB
# efficiency is the project's documented exception (lag-1 correlation 0.47 vs
# 0.18-0.25 for RB/WR/TE), so labeling a QB "shaky" because his projection
# leans on efficiency would contradict the research this signal comes from.
EFFICIENCY_SIGNAL_POSITIONS = {"RB", "WR", "TE"}

# A new tier starts when the drop from the tier's top player exceeds this
# fraction of the position's typical 80%-interval halfwidth. Scaling to the
# interval keeps the rule honest: tiers are only as fine as the projections'
# own uncertainty allows.
TIER_GAP_FRACTION = 0.30

# Floor for "fantasy-relevant" rows in the Home modules, so deep-roster players
# with tiny projections don't dominate percentage-flavored lists.
MIN_RELEVANT_PROJ = 120.0


def _require(df: pd.DataFrame, cols: list[str]) -> bool:
    return df is not None and not df.empty and all(c in df.columns for c in cols)


def assign_tiers(pos_df: pd.DataFrame) -> pd.Series:
    """Tier numbers (1 = best) for one position's rows.

    Players sort by projected points; a new tier opens when the drop from the
    current tier's top player exceeds ``TIER_GAP_FRACTION`` of the position's
    median interval halfwidth. Within a tier, players are statistically close
    enough that agonizing over the order is not supported by the model.
    """
    if not _require(pos_df, [PROJ_COL]):
        return pd.Series(dtype="int64")
    df = pos_df.sort_values(PROJ_COL, ascending=False)
    proj = df[PROJ_COL].to_numpy(dtype="float64")

    if _require(pos_df, [LOW_COL, HIGH_COL]):
        halfwidth = float(np.nanmedian((df[HIGH_COL] - df[LOW_COL]) / 2.0))
    else:
        halfwidth = float(np.nanstd(proj))
    threshold = max(TIER_GAP_FRACTION * halfwidth, 1.0)

    tiers = np.ones(len(proj), dtype="int64")
    anchor = proj[0] if len(proj) else 0.0
    for i in range(1, len(proj)):
        if anchor - proj[i] > threshold:
            tiers[i] = tiers[i - 1] + 1
            anchor = proj[i]
        else:
            tiers[i] = tiers[i - 1]
    return pd.Series(tiers, index=df.index)


def stability_labels(two_stage: pd.DataFrame) -> pd.DataFrame:
    """player_id -> role badge: 'Stable' (role-driven), 'Shaky'
    (efficiency-driven), or '' for the unremarkable middle.

    QBs never receive a badge: their efficiency genuinely repeats, so the
    stable/shaky reading does not apply to them (see
    ``EFFICIENCY_SIGNAL_POSITIONS``)."""
    if not _require(two_stage, ["player_id", "efficiency_variance_share"]):
        return pd.DataFrame(columns=["player_id", "role_badge"])
    share = pd.to_numeric(two_stage["efficiency_variance_share"], errors="coerce")
    eligible = (
        two_stage["position"].isin(EFFICIENCY_SIGNAL_POSITIONS)
        if "position" in two_stage.columns
        else pd.Series(True, index=two_stage.index)
    )
    badge = pd.Series("", index=two_stage.index, dtype="object")
    badge[eligible & (share >= SHAKY_SHARE)] = "Shaky"
    badge[eligible & (share <= STABLE_SHARE)] = "Stable"
    return pd.DataFrame(
        {"player_id": two_stage["player_id"], "role_badge": badge}
    )


def scarcity_frame(fantasy: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Positional scarcity curve: projected points by positional rank."""
    if not _require(fantasy, ["position", PROJ_COL]):
        return pd.DataFrame()
    rows = []
    for pos, grp in fantasy.groupby("position"):
        top = grp.sort_values(PROJ_COL, ascending=False).head(top_n)
        rows.append(
            pd.DataFrame(
                {
                    "position": pos,
                    "positional_rank": range(1, len(top) + 1),
                    "projected_points": top[PROJ_COL].to_numpy(),
                    "player": top["player_display_name"].to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def starter_window_dropoffs(
    scarcity: pd.DataFrame, window: int = 12
) -> pd.DataFrame:
    """Points lost between positional rank 1 and rank ``window`` per position —
    the plain-number version of 'how steep is this position's cliff'."""
    if not _require(scarcity, ["position", "positional_rank", "projected_points"]):
        return pd.DataFrame()
    rows = []
    for pos, grp in scarcity.groupby("position"):
        grp = grp.sort_values("positional_rank")
        top = float(grp["projected_points"].iloc[0])
        edge_rows = grp[grp["positional_rank"] == window]
        if edge_rows.empty:
            continue
        edge = float(edge_rows["projected_points"].iloc[0])
        rows.append(
            {"position": pos, "top_projection": top,
             "rank12_projection": edge, "dropoff": top - edge}
        )
    return (
        pd.DataFrame(rows).sort_values("dropoff", ascending=False).reset_index(drop=True)
        if rows else pd.DataFrame()
    )


def risers_frame(fantasy: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Players projected to beat their 2025 total by the most, among
    fantasy-relevant projections."""
    needed = ["player_display_name", "position", PROJ_COL, DELTA_COL]
    if not _require(fantasy, needed):
        return pd.DataFrame()
    df = fantasy[fantasy[PROJ_COL] >= MIN_RELEVANT_PROJ].copy()
    df = df[pd.to_numeric(df[DELTA_COL], errors="coerce") > 0]
    df = df.sort_values(DELTA_COL, ascending=False).head(top_n)
    team_col = "primary_team_2025" if "primary_team_2025" in df.columns else "team"
    out = df[["player_id", "player_display_name", "position", PROJ_COL, DELTA_COL]].copy()
    out["team"] = df.get(team_col, "")
    return out.reset_index(drop=True)


def draft_values_frame(
    board: pd.DataFrame,
    top_n: int = 8,
    max_rank: int = 120,
    min_edge: float = 10.0,
) -> pd.DataFrame:
    """Draft-day values: players the market drafts meaningfully later than the
    model ranks them (edge_vs_adp = ADP overall rank − model overall rank)."""
    needed = ["player_display_name", "position", "overall_rank", "edge_vs_adp"]
    if not _require(board, needed):
        return pd.DataFrame()
    df = board[pd.to_numeric(board["edge_vs_adp"], errors="coerce").notna()].copy()
    df = df[(df["overall_rank"] <= max_rank) & (df["edge_vs_adp"] >= min_edge)]
    df = df.sort_values("edge_vs_adp", ascending=False).head(top_n)
    keep = [
        "player_display_name", "position", "overall_rank",
        "adp_formatted", "edge_vs_adp",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def regression_watch_frame(
    fantasy: pd.DataFrame, two_stage: pd.DataFrame, top_n: int = 8
) -> pd.DataFrame:
    """Projected decliners whose production profile is efficiency-driven.

    Two conditions, both required: the model projects a lower 2026 total than
    the player's 2025 actual (negative delta), and the decomposition attributes
    most of the projection's uncertainty to per-play efficiency
    (``efficiency_variance_share`` at or above the shaky cutoff) — the
    combination that says "the big season leaned on the part that doesn't
    repeat."
    """
    needed = ["player_id", "player_display_name", "position", PROJ_COL, DELTA_COL]
    if not _require(fantasy, needed) or not _require(
        two_stage, ["player_id", "efficiency_variance_share"]
    ):
        return pd.DataFrame()
    merged = fantasy.merge(
        two_stage[["player_id", "efficiency_variance_share"]],
        on="player_id",
        how="inner",
    )
    # Skill players only: QB efficiency repeats, so the "leaned on the part
    # that doesn't repeat" argument does not apply at that position.
    merged = merged[merged["position"].isin(EFFICIENCY_SIGNAL_POSITIONS)]
    merged = merged[merged[PROJ_COL] >= MIN_RELEVANT_PROJ]
    merged = merged[pd.to_numeric(merged[DELTA_COL], errors="coerce") < 0]
    merged = merged[merged["efficiency_variance_share"] >= SHAKY_SHARE]
    merged = merged.sort_values(DELTA_COL).head(top_n)
    team_col = "primary_team_2025" if "primary_team_2025" in merged.columns else "team"
    out = merged[
        ["player_id", "player_display_name", "position", PROJ_COL, DELTA_COL,
         "efficiency_variance_share"]
    ].copy()
    out["team"] = merged.get(team_col, "")
    return out.reset_index(drop=True)


def rookie_fliers_frame(fantasy: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Top projected rookies, from the current draft class's Bayesian hurdle
    projection folded into the season table (``is_rookie_projection``).

    Empty until the rookie class has been scored (a separate .venv-bayes
    step; see PORTFOLIO_ROADMAP.md) and merged in — this frame is simply
    empty on a veterans-only table, same graceful-degradation pattern as the
    other Home modules.
    """
    needed = ["player_id", "player_display_name", "position", PROJ_COL, "is_rookie_projection"]
    if not _require(fantasy, needed):
        return pd.DataFrame()
    rookies = fantasy[fantasy["is_rookie_projection"] == True]  # noqa: E712
    if rookies.empty:
        return pd.DataFrame()
    rookies = rookies.sort_values(PROJ_COL, ascending=False).head(top_n)
    team_col = "primary_team_2025" if "primary_team_2025" in rookies.columns else "team"
    out = rookies[["player_id", "player_display_name", "position", PROJ_COL]].copy()
    out["team"] = rookies.get(team_col, "")
    out["draft_number"] = rookies.get("draft_number", pd.NA)
    out["p_plays"] = rookies.get("predicted_p_plays_meaningfully", pd.NA)
    return out.reset_index(drop=True)
