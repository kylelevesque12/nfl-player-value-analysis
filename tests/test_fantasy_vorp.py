"""Tests for ADP matching and the VORP draft board (Streamlit-free)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.adp import match_adp_to_projections, normalize_name
from src.fantasy_vorp import (
    PROJ_COL,
    auction_values,
    build_draft_board,
    compute_replacement_points,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Name normalization + matching
# ---------------------------------------------------------------------------
def test_normalize_name_strips_suffixes_and_punctuation():
    assert normalize_name("Marvin Harrison Jr.") == "marvin harrison"
    # Apostrophe styles must collide onto the same key across sources.
    assert normalize_name("Ja'Marr Chase") == normalize_name("JaMarr Chase")
    assert normalize_name("Brian Thomas Jr") == "brian thomas"
    assert normalize_name("Hollywood Brown") == "marquise brown"  # alias


def test_adp_matching_by_name_and_position_with_fallback():
    fantasy = pd.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "player_display_name": ["Ja'Marr Chase", "Travis Etienne Jr.", "Sam Smith"],
            "position": ["WR", "RB", "TE"],
            PROJ_COL: [280.0, 180.0, 90.0],
        }
    )
    adp = pd.DataFrame(
        {
            "adp_name": ["JaMarr Chase", "Travis Etienne", "Rookie Man"],
            "position": ["WR", "RB", "RB"],
            "adp": [3.7, 15.7, 30.0],
            "adp_formatted": ["1.04", "2.04", "3.06"],
            "adp_overall_rank": [1, 2, 3],
        }
    )
    merged, diag = match_adp_to_projections(fantasy, adp)
    assert merged.loc[merged.player_id == "a", "adp"].iloc[0] == 3.7
    assert merged.loc[merged.player_id == "b", "adp"].iloc[0] == 15.7
    assert pd.isna(merged.loc[merged.player_id == "c", "adp"]).all()
    assert diag["adp_matched"] == 2 and diag["adp_players"] == 3
    # The unmatched ADP player is inside the top 100 and must be reported.
    assert diag["top100_unmatched"][0]["adp_name"] == "Rookie Man"


# ---------------------------------------------------------------------------
# Replacement level + VORP
# ---------------------------------------------------------------------------
def _tiny_league() -> pd.DataFrame:
    """2-team league fixture: roster QB, 1 RB, 1 WR; 1 flex (RB/WR)."""
    rows = []
    for pos, points in [
        ("QB", [30, 28, 26, 24]),
        ("RB", [25, 20, 15, 10, 5]),
        ("WR", [22, 18, 14, 12, 8]),
        ("TE", [12, 9, 6]),
    ]:
        for i, p in enumerate(points):
            rows.append(
                {
                    "player_id": f"{pos}{i}",
                    "player_display_name": f"{pos} Player {i}",
                    "position": pos,
                    PROJ_COL: float(p),
                }
            )
    return pd.DataFrame(rows)


def test_replacement_points_fill_lineups_including_flex():
    fantasy = _tiny_league()
    replacement = compute_replacement_points(
        fantasy,
        teams=2,
        roster={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        flex_slots=1,
    )
    # Fixed starters: 2 QB, 2 RB, 2 WR, 2 TE. Flex pool remainders:
    # RB [15,10,5], WR [14,12,8], TE [6]. Two flex picks: RB 15, then WR 14.
    # Replacement = best player left at each position.
    assert replacement["QB"] == 26.0
    assert replacement["RB"] == 10.0
    assert replacement["WR"] == 12.0
    assert replacement["TE"] == 6.0


def test_vorp_ordering_is_cross_position():
    fantasy = _tiny_league()
    board, diag = build_draft_board(fantasy, adp=None, teams=2)
    # VORP must equal projection minus the position's replacement level.
    qb0 = board[board.player_id == "QB0"].iloc[0]
    assert qb0["vorp"] == pytest.approx(30.0 - diag["replacement_points"]["QB"])
    # Board is sorted by VORP with a 1..n overall rank.
    assert list(board["overall_rank"]) == list(range(1, len(board) + 1))
    assert board["vorp"].is_monotonic_decreasing


def test_auction_values_respect_budget_and_floor():
    fantasy = _tiny_league()
    board, _ = build_draft_board(fantasy, adp=None, teams=2)
    values = auction_values(board, teams=2, budget=100, roster_spots=5)
    assert (values >= 1).all()
    # League budget 200; every player carries the $1 floor and the
    # discretionary pool (200 - 10) is split over positive-VORP players.
    # Their total should equal discretionary + their floors, up to integer
    # rounding (at most $0.5 per player).
    positive = board["vorp"] > 0
    expected = (200 - 10) + positive.sum()
    assert values[positive].sum() == pytest.approx(expected, abs=len(values))


# ---------------------------------------------------------------------------
# Real-data smoke checks
# ---------------------------------------------------------------------------
def test_real_board_builds_and_is_sane_if_data_present():
    fantasy_path = ROOT / "outputs" / "tables" / "2026_fantasy_football_projections.csv"
    board_path = ROOT / "outputs" / "tables" / "draft_board_2026.csv"
    if not fantasy_path.exists():
        return
    fantasy = pd.read_csv(fantasy_path)
    board, diag = build_draft_board(fantasy, adp=None)
    # Replacement levels: QB well above the skill positions in a 1-QB league.
    rep = diag["replacement_points"]
    assert rep["QB"] > rep["RB"] and rep["QB"] > rep["WR"] and rep["QB"] > rep["TE"]
    # A 1-QB league's top 10 should not be QB-heavy.
    assert (board.head(10)["position"] == "QB").sum() <= 2

    if board_path.exists():
        saved = pd.read_csv(board_path)
        assert {"overall_rank", "vorp", "auction_value", "edge_vs_adp"} <= set(saved.columns)
        assert saved["overall_rank"].iloc[0] == 1
