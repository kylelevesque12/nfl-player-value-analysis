"""Share-normalized opportunity features for the season fantasy model.

Why these exist
---------------
The season model currently uses raw counts — ``targets``, ``receptions``,
``carries``. A raw count conflates two different things: how big a role the
player has, and how many plays his offense runs. Two receivers with 120 targets
are not comparable if one plays in a 700-attempt offense and the other in a
520-attempt offense.

Share metrics separate them. ``target_share`` is the fraction of his team's
targets a player commands — the *role* signal, and the component that actually
persists year over year, because team volume swings with scheme, game script and
coaching. ``wopr`` (Weighted Opportunity Rating) is the standard fantasy
composite of volume and downfield role, and ``air_yards_share`` distinguishes a
possession receiver from a field-stretcher who sees the same number of targets.

How they are computed (and why not the easy way)
------------------------------------------------
nflverse ships weekly ``target_share`` / ``air_yards_share`` / ``wopr`` columns,
but a *season* feature must not be the mean of those weekly ratios: averaging
ratios weights a 2-target game the same as a 12-target game, which biases the
estimate toward a player's quiet weeks. Instead we aggregate numerator and
denominator separately —

    season target_share = player season targets / team season targets

— which is the share he actually commanded over the year.

Team totals are summed over the skill-position frame. Targets go essentially
only to RB/WR/TE, so this is a very close approximation of true team targets
(trick-play targets to linemen are negligible and not in the frame).

Traded players
--------------
Shares are computed at the player-season-**team** level and then combined to one
player-season row with a games-weighted mean, so a player who splits a year
between two teams gets the share he held while on each, weighted by how long he
was there — not a figure distorted by the larger team's volume.

Quarterbacks
------------
Receiving shares are meaningless for QBs, so they are left as NaN rather than
filled with 0 (a 0 would read as "had a role and lost it"). The model pipeline
median-imputes, and ``position`` is a feature, so the model can separate them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The columns the share denominators are built from, and the resulting features.
_SHARE_SPECS = [
    # (feature, player numerator, team denominator source)
    ("target_share", "targets", "targets"),
    ("air_yards_share", "receiving_air_yards", "receiving_air_yards"),
    ("carry_share", "carries", "carries"),
]

# Weighted Opportunity Rating — the standard fantasy composite.
_WOPR_TARGET_WEIGHT = 1.5
_WOPR_AIR_YARDS_WEIGHT = 0.7

# Receiving-share features are not meaningful for quarterbacks.
_RECEIVING_SHARES = ("target_share", "air_yards_share", "wopr")


def _safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise share with divide-by-zero mapped to NaN, not inf or 0."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return num / den


def add_team_opportunity_shares(player_season_team: pd.DataFrame) -> pd.DataFrame:
    """Attach share features to a player-season-**team** frame.

    Expects one row per (season, team, player) with season-summed counting
    stats — i.e. the output of ``aggregate_weekly_to_player_season``.
    """
    df = player_season_team.copy()
    if not {"season", "team"}.issubset(df.columns):
        raise ValueError("share features need 'season' and 'team' columns")

    for feature, player_col, team_col in _SHARE_SPECS:
        if player_col not in df.columns:
            continue
        team_total = df.groupby(["season", "team"])[team_col].transform("sum")
        df[feature] = _safe_share(df[player_col], team_total)

    if {"target_share", "air_yards_share"}.issubset(df.columns):
        df["wopr"] = (
            _WOPR_TARGET_WEIGHT * df["target_share"].fillna(0)
            + _WOPR_AIR_YARDS_WEIGHT * df["air_yards_share"].fillna(0)
        )
        # Keep NaN where the player had no receiving role at all, so "no data"
        # stays distinct from "genuine zero share".
        no_receiving = df["target_share"].isna() & df["air_yards_share"].isna()
        df.loc[no_receiving, "wopr"] = np.nan

    if "position" in df.columns:
        is_qb = df["position"].eq("QB")
        for col in _RECEIVING_SHARES:
            if col in df.columns:
                df.loc[is_qb, col] = np.nan

    return df


def collapse_shares_to_player_season(
    player_season_team: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse per-team shares to one row per (season, player_id).

    Uses a games-weighted mean so a traded player's share reflects where he
    actually played. Falls back to an unweighted mean if games are unavailable.
    """
    df = player_season_team
    share_cols = [c for c in ("target_share", "air_yards_share", "carry_share", "wopr")
                  if c in df.columns]
    if not share_cols:
        return pd.DataFrame(columns=["season", "player_id"])

    weights = (
        pd.to_numeric(df["games_played"], errors="coerce").fillna(0)
        if "games_played" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    work = df[["season", "player_id"] + share_cols].copy()
    work["_w"] = weights.where(weights > 0, np.nan)

    out = {}
    for col in share_cols:
        w = work["_w"].where(work[col].notna())
        weighted = (work[col] * w).groupby([work["season"], work["player_id"]]).sum(min_count=1)
        total_w = w.groupby([work["season"], work["player_id"]]).sum(min_count=1)
        out[col] = weighted / total_w.replace(0, np.nan)

    collapsed = pd.DataFrame(out).reset_index()
    return collapsed


def attach_opportunity_features(
    player_season: pd.DataFrame,
    player_season_team: pd.DataFrame,
) -> pd.DataFrame:
    """Merge share features (plus one-season lags) onto a player-season frame.

    ``player_season`` is the collapsed modeling frame; ``player_season_team`` is
    the pre-collapse frame the shares are derived from.
    """
    shares = collapse_shares_to_player_season(
        add_team_opportunity_shares(player_season_team)
    )
    merged = player_season.merge(shares, on=["season", "player_id"], how="left")

    # Prior-season shares: what the player's role was last year. Sorted and
    # shifted per player, so no future information enters.
    merged = merged.sort_values(["player_id", "season"])
    grouped = merged.groupby("player_id", group_keys=False)
    for col in ("target_share", "wopr"):
        if col in merged.columns:
            merged[f"{col}_prev"] = grouped[col].shift(1)

    return merged
