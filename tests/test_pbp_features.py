"""Leakage-safety tests for the PBP depth-chart rank features.

The depth-chart rank is derived from play-by-play usage, but the features
exposed to the model must only ever use PRIOR weeks — the current week's
play-by-play cannot enter the current week's feature row. These synthetic
tests pin that contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pbp_features import build_pbp_depth_chart_rank


def _synthetic_pbp() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two WRs on one team over three weeks.

    WR_A is the clear WR1 in weeks 1-2 (more targets); in week 3 the usage
    flips and WR_B leads. A leakage-free week-3 feature must still reflect the
    OLD ordering (A ahead of B), because it can only see weeks 1-2.
    """
    rows = []

    def add(week, receiver, n):
        for _ in range(n):
            rows.append(
                {
                    "play_type": "pass",
                    "posteam": "AAA",
                    "season": 2020,
                    "week": week,
                    "passer_player_id": "QB",
                    "rusher_player_id": np.nan,
                    "receiver_player_id": receiver,
                }
            )

    add(1, "WR_A", 10); add(1, "WR_B", 4)
    add(2, "WR_A", 9); add(2, "WR_B", 3)
    add(3, "WR_A", 2); add(3, "WR_B", 12)  # usage flips in week 3
    pbp = pd.DataFrame(rows)

    position_lookup = pd.DataFrame(
        {
            "player_id": ["WR_A", "WR_B"],
            "season": [2020, 2020],
            "position": ["WR", "WR"],
        }
    )
    return pbp, position_lookup


def test_rank_last1_uses_only_prior_week():
    pbp, lookup = _synthetic_pbp()
    out = build_pbp_depth_chart_rank(pbp, lookup)

    def rank(pid, week):
        m = out[(out["player_id"] == pid) & (out["week"] == week)]
        return m["pbp_depth_chart_rank_last1"].iloc[0]

    # Week 1 has no prior week -> feature is NaN (no leakage from week 1 itself).
    assert pd.isna(rank("WR_A", 1))
    assert pd.isna(rank("WR_B", 1))

    # Week 2's last1 reflects week 1, where A (rank 1) led B (rank 2).
    assert rank("WR_A", 2) == 1
    assert rank("WR_B", 2) == 2

    # Week 3's last1 reflects week 2 (A still ahead) — NOT week 3's flipped
    # usage. If the current week leaked in, A would be rank 2 here.
    assert rank("WR_A", 3) == 1
    assert rank("WR_B", 3) == 2


def test_no_current_week_leakage_in_rolling_avg():
    pbp, lookup = _synthetic_pbp()
    out = build_pbp_depth_chart_rank(pbp, lookup)
    # The rolling-4 average is built on shift(1) data, so the most recent row
    # for each player must never equal a value computed from its own week.
    a3 = out[(out["player_id"] == "WR_A") & (out["week"] == 3)]
    # A averaged rank 1 over weeks 1-2; the leak-free rolling avg stays 1.0.
    assert a3["pbp_depth_chart_rank_last4_avg"].iloc[0] == 1.0
