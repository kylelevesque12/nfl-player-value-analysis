"""Tests for the Session 3 rookie pre-season context features.

These pin the two things that matter most: (1) leakage discipline — every
non-combine feature is built from the season BEFORE the rookie year and the
rookie is excluded from his own position's veteran tallies; and (2) the join is
one-row-in/one-row-out with no rookie duplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.load_data import find_project_root
from src import rookie_bayes as rb
from src import rookie_context_features as rc


# ---------------------------------------------------------------------------
# Synthetic scenario: a rookie QB on AAA in 2022 behind a veteran starter.
# ---------------------------------------------------------------------------
def _synthetic():
    rosters = pd.DataFrame(
        {
            "gsis_id": ["rook", "vet", "wr_rook", "wr_vet"],
            "full_name": ["Rookie QB", "Vet QB", "Rookie WR", "Vet WR"],
            "position": ["QB", "QB", "WR", "WR"],
            "team": ["AAA", "AAA", "AAA", "AAA"],
            "season": [2022, 2022, 2022, 2022],
            "rookie_year": [2022, 2018, 2022, 2019],
            "entry_year": [2022, 2018, 2022, 2019],
            "draft_number": [40, 20, 60, 120],
            "birth_date": ["2000-01-01", "1995-01-01", "2000-02-02", "1996-02-02"],
            "height": ["6-4", "6-3", "6-0", "6-1"],
            "weight": [220, 225, 195, 200],
            "college": ["X", "Y", "Z", "W"],
            "draft_club": ["AAA", "AAA", "AAA", "AAA"],
        }
    )

    def wk(pid, season, pos, pts, n, team="AAA"):
        return pd.DataFrame({
            "player_id": [pid] * n, "season": [season] * n,
            "week": list(range(1, n + 1)), "season_type": ["REG"] * n,
            "position": [pos] * n, "recent_team": [team] * n,
            "fantasy_points_ppr": [pts] * n, "attempts": [30 if pos == "QB" else 0] * n,
            "carries": [2] * n,
        })

    player_stats = pd.concat([
        # Veteran QB: the 2021 (prior-year) AAA starter, 16 games @ 20 ppr.
        wk("vet", 2021, "QB", 20.0, 16),
        # Veteran QB also plays the rookie year — DIFFERENT value (must be ignored).
        wk("vet", 2022, "QB", 5.0, 16),
        # Veteran WR: prior-year production (2021) @ 14 ppr.
        wk("wr_vet", 2021, "WR", 14.0, 15),
        # Rookie WR plays in 2022 (own rookie year — must never feed his features).
        wk("wr_rook", 2022, "WR", 25.0, 12),
    ], ignore_index=True)
    return rosters, player_stats


def test_team_context_uses_prior_season_only():
    rosters, player_stats = _synthetic()
    frame = rb.build_rookie_modeling_frame(rosters, player_stats, attach_context=True)
    rook = frame[frame["player_id"].eq("rook")].iloc[0]
    # prior_qb_pprpg must be the 2021 starter value (20.0), NOT the vet's 2022
    # rookie-year value (5.0). This is the core leakage guarantee.
    assert rook["prior_qb_pprpg"] == 20.0
    assert rook["established_incumbent"] == 1.0  # 20 ppr >= 8 threshold


def test_veteran_depth_excludes_the_rookie_himself():
    rosters, player_stats = _synthetic()
    frame = rb.build_rookie_modeling_frame(rosters, player_stats, attach_context=True)
    wr = frame[frame["player_id"].eq("wr_rook")].iloc[0]
    # AAA has two WRs in 2022 (wr_rook + wr_vet) but only wr_vet is a veteran
    # (entry < 2022) and the rookie excludes himself -> exactly 1 veteran.
    assert wr["pos_vet_count"] == 1.0
    # The max veteran prior-year ppr is the 2021 vet WR value (14.0), not the
    # rookie's own 2022 production (25.0).
    assert wr["pos_vet_max_pprpg"] == 14.0


def test_combine_join_does_not_duplicate_rookies():
    root = find_project_root()
    if not (root / "data" / "raw").glob("combine_*.csv"):
        return
    ros = pd.read_csv(root / "data/raw/rosters_2016_2025.csv", low_memory=False)
    ps = pd.read_csv(root / "data/raw/player_stats_2016_2025.csv", low_memory=False)
    frame = rb.build_rookie_modeling_frame(ros, ps)
    assert frame["player_id"].is_unique
    # combine lookup itself is one row per player.
    combine = rc.load_combine_features(root, ros)
    if combine is not None:
        assert combine["player_id"].is_unique


def test_no_rookie_year_outcome_columns_in_feature_set():
    forbidden = {
        "season_ppr_total", "season_ppr_per_game", "games_played",
        "played_meaningfully", "rookie_year_ppr_per_game_full",
    }
    model_features = set(rb.FEATURE_COLUMNS) | set(rb.OPTIONAL_FEATURES)
    assert model_features.isdisjoint(forbidden), (
        f"rookie-year outcomes leaked into features: {model_features & forbidden}"
    )
    # The Session-3 context features must not be outcome columns either.
    assert set(rb.CONTEXT_FEATURES).isdisjoint(forbidden)


def test_training_path_runs_with_context_features():
    """PyMC isn't installed here, so this is the logistic surrogate smoke test
    that the standardized context features feed a fittable stage-1 model."""
    root = find_project_root()
    ros = pd.read_csv(root / "data/raw/rosters_2016_2025.csv", low_memory=False)
    ps = pd.read_csv(root / "data/raw/player_stats_2016_2025.csv", low_memory=False)
    frame = rb.build_rookie_modeling_frame(ros, ps)
    train = frame[frame["rookie_year"] < 2023]
    test = frame[frame["rookie_year"] >= 2023]
    train_z, test_z, _ = rb.standardize_features(train, test)
    zcols = [c for c in train_z.columns if c.endswith("_z")]
    assert any("incumbent" in c or "prior_qb" in c for c in zcols)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_z[zcols].fillna(0.0), train_z["played_meaningfully"].astype(int))
    proba = clf.predict_proba(test_z[zcols].fillna(0.0))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
