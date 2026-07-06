"""Tests for the whole-draft planner (pure, Streamlit-free).

The hand-worked toy scenario below mirrors the RB-scarcity intuition the
planner exists to capture: a running back pool with a steep early cliff
(40 -> 10 points) next to a smoother quarterback pool. Taking the running
back first should beat taking the quarterback first, and by exactly the
margin a full, honest simulation produces — worked out by hand in the
module's dev notes and reproduced here so a regression cannot silently
change the numbers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app import draft_planner as dp

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Snake draft math
# ---------------------------------------------------------------------------
def test_snake_pick_numbers_two_team_four_round():
    assert dp.my_pick_numbers(teams=2, my_slot=1, rounds=4) == [1, 4, 5, 8]
    assert dp.my_pick_numbers(teams=2, my_slot=2, rounds=4) == [2, 3, 6, 7]


def test_slot_on_the_clock_matches_snake_reversal():
    # 12-team league, round 2 reverses: pick 13 is slot 12, pick 24 is slot 1.
    assert dp.slot_on_the_clock(1, teams=12) == 1
    assert dp.slot_on_the_clock(12, teams=12) == 12
    assert dp.slot_on_the_clock(13, teams=12) == 12
    assert dp.slot_on_the_clock(24, teams=12) == 1


def test_is_my_pick_consistent_with_pick_numbers():
    picks = set(dp.my_pick_numbers(teams=6, my_slot=3, rounds=5))
    for overall in range(1, 6 * 5 + 1):
        assert dp.is_my_pick(overall, teams=6, my_slot=3) == (overall in picks)


# ---------------------------------------------------------------------------
# Toy scenario: RB cliff vs. smooth QB pool
# ---------------------------------------------------------------------------
def _toy_players() -> pd.DataFrame:
    rows = [
        ("RB_A", "RB", 40.0, 1),
        ("QB_A", "QB", 30.0, 2),
        ("RB_B", "RB", 10.0, 3),
        ("QB_B", "QB", 20.0, 4),
        ("RB_C", "RB", 9.0, 5),
        ("QB_C", "QB", 10.0, 6),
        ("RB_D", "RB", 8.0, 7),
        ("QB_D", "QB", 5.0, 8),
    ]
    return pd.DataFrame(
        [
            {"player_id": pid, "player_display_name": pid, "position": pos, "vorp": val, "adp": adp}
            for pid, pos, val, adp in rows
        ]
    )


def test_toy_board_pools_and_adp_order():
    board = dp.build_board(_toy_players())
    assert [p["player_id"] for p in board.pools["RB"]] == ["RB_A", "RB_B", "RB_C", "RB_D"]
    assert [p["player_id"] for p in board.pools["QB"]] == ["QB_A", "QB_B", "QB_C", "QB_D"]
    assert [p["player_id"] for p in board.adp_order[:3]] == ["RB_A", "QB_A", "RB_B"]


def test_plan_prefers_the_steep_position_first():
    """Two-team, two-pick horizon (QB1, RB1, no flex): taking the scarce RB
    first should beat taking QB first, and the totals should match the exact
    hand-calculation (see module docstring for the worked reasoning)."""
    board = dp.build_board(_toy_players())
    plan = dp.plan_whole_draft(
        board, current_pick=1, teams=2, my_slot=1, rounds=4,
        roster={"QB": 1, "RB": 1}, flex_slots=0,
    )
    assert [s.position for s in plan.steps] == ["RB", "QB"]
    assert [s.player_id for s in plan.steps] == ["RB_A", "QB_B"]
    assert plan.total_value == pytest.approx(60.0)


def test_plan_beats_the_naive_alternative_by_hand_calculated_margin():
    """The RB-first plan (60) must beat what a QB-first draft would have
    produced (39): taking QB first lets opponents snipe both RB_A and RB_B
    before your next pick, which is exactly the scarcity trap the planner
    exists to avoid."""
    board = dp.build_board(_toy_players())
    plan = dp.plan_whole_draft(
        board, current_pick=1, teams=2, my_slot=1, rounds=4,
        roster={"QB": 1, "RB": 1}, flex_slots=0,
    )
    assert plan.total_value == pytest.approx(60.0)
    assert plan.total_value > 39.0  # the QB-first alternative, worked by hand


def test_cost_of_waiting_table_matches_hand_calculation():
    """Teams=2, my_slot=1: my picks are [1, 4]; the only two picks strictly
    between them (2 and 3) belong to the opponent, so exactly 2 players are
    removed by ADP (RB_A, QB_A) — not 3. An earlier version of this table
    looped `next_pick - current_pick` times instead of checking whose pick
    each slot actually was, which over-counted by exactly one opponent pick
    (it would have popped RB_B too, understating what's left at RB)."""
    board = dp.build_board(_toy_players())
    table = dp.cost_of_waiting_table(
        board, current_pick=1, my_future_picks=[1, 4], roster={"QB": 1, "RB": 1},
        teams=2, my_slot=1,
    )
    by_pos = table.set_index("position")
    assert by_pos.loc["RB", "best_now"] == pytest.approx(40.0)
    assert by_pos.loc["RB", "best_at_next_pick"] == pytest.approx(10.0)
    assert by_pos.loc["RB", "points_lost_if_you_wait"] == pytest.approx(30.0)
    assert by_pos.loc["QB", "points_lost_if_you_wait"] == pytest.approx(10.0)
    # Sorted with the steepest cost first.
    assert table.iloc[0]["position"] == "RB"


def test_opponent_only_shelf_excludes_the_users_own_picks():
    """The pick slots that belong to the user (both endpoints, and any of
    the user's own picks that fall strictly inside the range) must never be
    counted as an opponent removal."""
    board = dp.build_board(_toy_players())
    # Range [1, 4): picks 1 (mine), 2 (opp), 3 (opp). Only 2 opponent pops.
    removed = dp.opponent_only_shelf(board, current_pick=1, next_pick=4, teams=2, my_slot=1)
    assert sum(removed.values()) == 2


def test_plan_respects_roster_caps_and_flex():
    """QB1, RB1, WR1 + 1 flex: exactly 4 stages, no position exceeding its
    cap (RB/WR each boundable at 1 + flex = 2, but only one flex slot total
    exists across RB/WR/TE combined)."""
    rows = []
    for pos, values in [("QB", [30, 20]), ("RB", [25, 20, 15]), ("WR", [24, 19, 14])]:
        for i, v in enumerate(values):
            rows.append({"player_id": f"{pos}{i}", "player_display_name": f"{pos}{i}",
                         "position": pos, "vorp": float(v)})
    players = pd.DataFrame(rows)
    board = dp.build_board(players)
    plan = dp.plan_whole_draft(
        board, current_pick=1, teams=2, my_slot=1, rounds=4,
        roster={"QB": 1, "RB": 1, "WR": 1}, flex_slots=1,
    )
    counts: dict[str, int] = {}
    for step in plan.steps:
        counts[step.position] = counts.get(step.position, 0) + 1
    assert counts.get("QB", 0) <= 1
    # RB + WR combined can use at most (1+1) base + 1 flex = 3 slots.
    assert counts.get("RB", 0) + counts.get("WR", 0) <= 3


def test_empty_board_and_finished_draft_are_safe():
    empty = dp.build_board(pd.DataFrame(columns=["player_id", "player_display_name", "position", "vorp"]))
    plan = dp.plan_whole_draft(empty, current_pick=1, teams=2, my_slot=1, rounds=1)
    assert plan.steps == []
    assert dp.recommendation_text(plan, on_the_clock=True).startswith("No further")


# ---------------------------------------------------------------------------
# Real-data smoke test
# ---------------------------------------------------------------------------
def test_real_board_plans_quickly_and_sanely_if_data_present():
    board_path = ROOT / "outputs" / "tables" / "draft_board_2026.csv"
    if not board_path.exists():
        return
    import time

    board_df = pd.read_csv(board_path)
    board = dp.build_board(board_df)
    start = time.perf_counter()
    plan = dp.plan_whole_draft(board, current_pick=25, teams=12, my_slot=1, rounds=15)
    elapsed = time.perf_counter() - start

    assert elapsed < 15.0, f"planner took {elapsed:.1f}s, too slow for interactive use"
    assert len(plan.steps) == 7  # QB1 + RB2 + WR2 + TE1 + 1 flex
    counts: dict[str, int] = {}
    for step in plan.steps:
        counts[step.position] = counts.get(step.position, 0) + 1
    assert counts.get("QB", 0) <= 1
    assert not plan.cost_of_waiting.empty
