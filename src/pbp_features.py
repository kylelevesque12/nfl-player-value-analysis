"""Feature engineering from nflverse play-by-play data.

The nflverse depth-chart file dropped its numeric ``list_rank`` field around
the 2024 season, so the "RB1 vs RB2" distinction — which is one of the
highest-signal weekly fantasy features — disappeared. This module rebuilds
it from play-by-play. For every player-week, I count their involvement on
offensive plays (pass attempts for QBs, rushes + targets for RBs, targets
for WR/TE), rank players within each (team, season, week, position), and
then expose two rolling-history versions of that rank as leakage-safe
features:

- ``pbp_depth_chart_rank_last1``: their rank in the previous game they played
- ``pbp_depth_chart_rank_last4_avg``: rolling mean of their rank over the
  previous four games they played

Both columns are ``groupby(player_id).shift(1)``-safe — the current week's
play-by-play never enters the feature for the current week. A small synthetic
test in ``tests/`` pins this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


def _load_pbp(project_root: Path | None = None) -> pd.DataFrame | None:
    """Find and load the PBP parquet if it exists; return None otherwise."""
    if project_root is None:
        from src.load_data import find_project_root

        project_root = find_project_root()
    candidates = sorted((project_root / "data" / "raw").glob("pbp_*.parquet"))
    if not candidates:
        return None
    # Use the widest-year file by name (e.g. pbp_2016_2025.parquet). Only the
    # columns used by the depth-chart ranker are read — the full PBP file has
    # ~380 columns and reading all of them costs >2 GB of RAM for no benefit.
    needed = [
        "play_type",
        "posteam",
        "season",
        "week",
        "passer_player_id",
        "rusher_player_id",
        "receiver_player_id",
    ]
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(candidates[-1]).schema.names)
    cols = [c for c in needed if c in available]
    return pd.read_parquet(candidates[-1], columns=cols)


def _aggregate_player_week_opportunities(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (player_id, team, season, week) emit opportunity totals.

    Opportunities follow the natural per-position metric:
        QB: pass attempts (PBP rows where the player is ``passer_player_id``)
        RB: rushes + targets (rusher_player_id + receiver_player_id)
        WR/TE: targets (receiver_player_id)

    The PBP file is offensive-plays-only after the filter; one row per play.
    """
    plays = pbp[
        pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()
    ]

    def _count(role_col: str) -> pd.DataFrame:
        sub = plays[plays[role_col].notna()]
        return (
            sub.groupby([role_col, "posteam", "season", "week"], as_index=False)
            .size()
            .rename(
                columns={
                    role_col: "player_id",
                    "posteam": "team",
                    "size": "_count",
                }
            )
        )

    pass_attempts = _count("passer_player_id").rename(
        columns={"_count": "pbp_pass_attempts"}
    )
    rush_attempts = _count("rusher_player_id").rename(
        columns={"_count": "pbp_rush_attempts"}
    )
    targets = _count("receiver_player_id").rename(
        columns={"_count": "pbp_targets"}
    )

    usage = pass_attempts.merge(
        rush_attempts, on=["player_id", "team", "season", "week"], how="outer"
    ).merge(
        targets, on=["player_id", "team", "season", "week"], how="outer"
    )
    for col in ["pbp_pass_attempts", "pbp_rush_attempts", "pbp_targets"]:
        if col not in usage.columns:
            usage[col] = 0
    usage = usage.fillna(
        {"pbp_pass_attempts": 0, "pbp_rush_attempts": 0, "pbp_targets": 0}
    )
    return usage


def _attach_position_and_opportunity_metric(
    usage: pd.DataFrame, position_lookup: pd.DataFrame
) -> pd.DataFrame:
    """Join position via ``position_lookup`` (player_id, season, position) and
    compute the position-specific opportunity total used for ranking."""
    enriched = usage.merge(
        position_lookup, on=["player_id", "season"], how="inner"
    )
    enriched = enriched[enriched["position"].isin(SKILL_POSITIONS)]
    opp = np.where(
        enriched["position"].eq("QB"),
        enriched["pbp_pass_attempts"],
        np.where(
            enriched["position"].eq("RB"),
            enriched["pbp_rush_attempts"] + enriched["pbp_targets"],
            enriched["pbp_targets"],
        ),
    )
    enriched = enriched.assign(opportunities=opp)
    return enriched


def build_pbp_depth_chart_rank(
    pbp: pd.DataFrame,
    position_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Per (player_id, season, week) emit a leakage-safe depth-chart rank.

    Returned columns: ``player_id, season, week, pbp_depth_chart_rank_last1,
    pbp_depth_chart_rank_last4_avg, pbp_targets_last4_avg, pbp_touches_last4_avg``.

    The rolling features use ``groupby(player_id).shift(1)`` so the current
    game's play-by-play never contributes to the current game's features.
    """
    usage = _aggregate_player_week_opportunities(pbp)
    usage = _attach_position_and_opportunity_metric(usage, position_lookup)

    # Current-week rank: rank players within (team, season, week, position) by
    # opportunities, with rank 1 going to the most involved player. This is the
    # raw quantity; we DO NOT expose it as a feature directly — we only expose
    # shifted/rolling versions below.
    usage = usage.sort_values(
        ["team", "season", "week", "position", "opportunities"],
        ascending=[True, True, True, True, False],
    )
    usage["_current_week_rank"] = (
        usage.groupby(["team", "season", "week", "position"])["opportunities"]
        .rank(method="dense", ascending=False)
    )

    # Shift(1) safe rolling features. Sort by player then walk forward.
    usage = usage.sort_values(["player_id", "season", "week"]).reset_index(
        drop=True
    )
    grp = usage.groupby("player_id", group_keys=False)
    usage["pbp_depth_chart_rank_last1"] = grp["_current_week_rank"].shift(1)
    usage["pbp_depth_chart_rank_last4_avg"] = grp["_current_week_rank"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    usage["pbp_targets_last4_avg"] = grp["pbp_targets"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    # Touches = rushes + targets for skill-position players; pass attempts for QBs.
    touches = np.where(
        usage["position"].eq("QB"),
        usage["pbp_pass_attempts"],
        usage["pbp_rush_attempts"] + usage["pbp_targets"],
    )
    # Assign in place so the existing ``grp`` (bound to this same object) sees
    # the new column — ``usage.assign`` would return a fresh frame and leave
    # ``grp`` pointing at the pre-_touches version.
    usage["_touches"] = touches
    usage["pbp_touches_last4_avg"] = grp["_touches"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )

    return usage[
        [
            "player_id",
            "season",
            "week",
            "pbp_depth_chart_rank_last1",
            "pbp_depth_chart_rank_last4_avg",
            "pbp_targets_last4_avg",
            "pbp_touches_last4_avg",
        ]
    ].reset_index(drop=True)


def attach_pbp_features(
    weekly_modeling: pd.DataFrame,
    pbp: pd.DataFrame | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Wire PBP-derived depth-chart-rank features into the weekly modeling frame.

    If the PBP parquet is absent on disk, returns the input unchanged so the
    pipeline still runs (with the documented loss of accuracy).
    """
    if pbp is None:
        pbp = _load_pbp(project_root)
        if pbp is None:
            return weekly_modeling

    # Build a position lookup directly from the modeling frame so we use the
    # same position tags downstream consumers see.
    position_lookup = (
        weekly_modeling[["player_id", "season", "position"]]
        .dropna(subset=["player_id", "season", "position"])
        .drop_duplicates(subset=["player_id", "season"])
        .copy()
    )
    position_lookup["season"] = pd.to_numeric(
        position_lookup["season"], errors="coerce"
    ).astype("Int64")

    pbp = pbp.copy()
    pbp["season"] = pd.to_numeric(pbp.get("season"), errors="coerce").astype("Int64")
    pbp["week"] = pd.to_numeric(pbp.get("week"), errors="coerce").astype("Int64")
    pbp = pbp.dropna(subset=["season", "week", "posteam"])
    pbp["season"] = pbp["season"].astype(int)
    pbp["week"] = pbp["week"].astype(int)

    rank_table = build_pbp_depth_chart_rank(pbp, position_lookup)
    return weekly_modeling.merge(
        rank_table, on=["player_id", "season", "week"], how="left"
    )
