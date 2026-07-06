"""The Draft Room's whole-draft planner (pure, Streamlit-free).

The idea, from the roadmap: a draft pick's value is not the player in front of
you, it is the best full plan for every pick you have left. This module
answers "what should I do right now, given the whole rest of my draft" rather
than just "who is best available."

How it works:

1. **Snake math.** Given the number of teams and your draft slot, compute
   which overall pick numbers are yours for the rest of the draft.

2. **One consistent timeline.** Between two of your picks, only opponents
   draft (a no-trade snake draft has no other structure). Opponents are
   modeled as drafting the best remaining player by average draft position
   (ADP), the same simplifying assumption a static "value based drafting"
   calculator makes. The planner walks the draft forward pick by pick, one
   shared timeline of removed players: at every opponent pick it removes the
   ADP-best player still on the board, and at every one of your picks it
   branches over the positions you could draft, each branch removing the
   best remaining player at that position. Whichever specific player a
   branch removes is gone for every pick that follows in that branch — your
   own picks and simulated opponent picks draw from the same, single pool,
   so nothing is ever double-counted.

3. **Search over your remaining picks.** The plan searches every sequence of
   position choices for your picks up to your team's starting roster (roster
   minimums plus the flex slot; bench rounds beyond that aren't planned in
   per-position detail) and returns the sequence with the highest total
   value, subject to the roster's position minimums and shared flex slot.

The one real approximation is the opponent model itself: real opponents do
not draft purely by ADP, they have their own team needs and biases. A later
Monte Carlo layer (sampling opponent picks around ADP many times and
reporting how often each opening move wins) replaces this determinism with a
distribution instead of removing the assumption; see the roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

VALUE_COL = "vorp"
NAME_COL = "player_display_name"

# Kept in sync by hand with src/fantasy_vorp.py's STANDARD_ROSTER / FLEX_SLOTS
# (the overall board's replacement-level lineup and the planner's target
# roster must describe the same league shape).
DEFAULT_ROSTER = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
DEFAULT_FLEX_SLOTS = 1
FLEX_POSITIONS = ("RB", "WR", "TE")

# ADP for a player with no market data: treated as drafted very late, so an
# unranked (e.g. rookie) player never displaces a market-known player in the
# simulated opponent queue, but can still fill in once ranked players are gone.
UNRANKED_ADP_PENALTY = 1000.0


# ---------------------------------------------------------------------------
# Snake draft math
# ---------------------------------------------------------------------------
def slot_on_the_clock(overall_pick: int, teams: int) -> int:
    """1-indexed team slot whose turn it is at ``overall_pick`` in a snake
    draft (round 1 goes 1..teams, round 2 reverses, and so on)."""
    if overall_pick < 1 or teams < 1:
        raise ValueError("overall_pick and teams must be positive")
    round_number = (overall_pick - 1) // teams + 1
    pick_in_round = (overall_pick - 1) % teams + 1
    return pick_in_round if round_number % 2 == 1 else teams - pick_in_round + 1


def my_pick_numbers(teams: int, my_slot: int, rounds: int) -> list[int]:
    """All overall pick numbers belonging to ``my_slot`` across ``rounds``."""
    return [
        p for p in range(1, teams * rounds + 1) if slot_on_the_clock(p, teams) == my_slot
    ]


def is_my_pick(overall_pick: int, teams: int, my_slot: int) -> bool:
    return slot_on_the_clock(overall_pick, teams) == my_slot


# ---------------------------------------------------------------------------
# Board setup
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DraftBoard:
    """Undrafted players, indexed for the planner.

    ``pools`` holds each position's undrafted players sorted best-to-worst by
    value; ``adp_order`` holds all undrafted players sorted by ADP ascending
    (missing ADP pushed to the back). Both exclude every id in ``drafted``
    at construction time; the planner's own search removes further players
    from a shared in-memory timeline as it explores (see module docstring).
    """

    pools: dict[str, list[dict]]
    adp_order: list[dict]


def build_board(
    players: pd.DataFrame, drafted_ids: set[str] | frozenset[str] = frozenset()
) -> DraftBoard:
    """Build the undrafted-player indexes from a projection/VORP table.

    ``players`` needs ``player_id``, ``player_display_name``, ``position``,
    and a value column (``vorp`` by default; anything monotonic in "how good"
    works, e.g. plain projected points for a board without VORP computed).
    An ``adp`` column is optional; without one, every player is treated as
    equally (unrankedly) late, which still yields a valid, if less market-
    aware, plan.
    """
    drafted = {str(d) for d in drafted_ids}
    available = players[~players["player_id"].astype(str).isin(drafted)].copy()
    value_col = VALUE_COL if VALUE_COL in available.columns else "predicted_2026_fantasy_points_ppr"

    pools: dict[str, list[dict]] = {}
    for pos, grp in available.groupby("position"):
        ordered = grp.sort_values(value_col, ascending=False)
        pools[pos] = [
            {
                "player_id": str(r.player_id),
                "name": getattr(r, NAME_COL),
                "value": float(getattr(r, value_col)),
            }
            for r in ordered.itertuples(index=False)
        ]

    adp = available.copy()
    if "adp" in adp.columns:
        adp["_adp_sort"] = pd.to_numeric(adp["adp"], errors="coerce")
        max_known = adp["_adp_sort"].max()
        fill = (max_known if pd.notna(max_known) else 0.0) + UNRANKED_ADP_PENALTY
        adp["_adp_sort"] = adp["_adp_sort"].fillna(fill)
    else:
        adp["_adp_sort"] = 0.0
    adp = adp.sort_values("_adp_sort")
    adp_order = [
        {"player_id": str(r.player_id), "position": r.position}
        for r in adp.itertuples(index=False)
    ]
    return DraftBoard(pools=pools, adp_order=adp_order)


# ---------------------------------------------------------------------------
# Cost of waiting (for display): opponent-only depletion, one position at a
# time. This does not have the double-counting issue the full plan search
# has to avoid, because it only ever asks "if I draft something ELSE this
# turn, what happens to position P by my next pick" — position P is never
# also the thing hypothetically taken this turn.
# ---------------------------------------------------------------------------
def opponent_only_shelf(
    board: DraftBoard, current_pick: int, next_pick: int, teams: int, my_slot: int
) -> dict[str, int]:
    """Count of each position opponents will remove from ``current_pick``
    (inclusive) up to ``next_pick`` (exclusive), assuming they draft the best
    remaining player by ADP at every pick that is not yours.

    Every overall pick in the range is checked against the snake order, not
    assumed to be an opponent's: if ``current_pick`` is your own pick right
    now, it is skipped here (this table asks "what happens to position P if I
    draft something else this turn," so your own pick removes nothing from
    P's shelf, whatever it is)."""
    removed: dict[str, int] = {}
    idx = 0
    for overall in range(current_pick, next_pick):
        if is_my_pick(overall, teams, my_slot):
            continue
        while idx < len(board.adp_order):
            entry = board.adp_order[idx]
            idx += 1
            removed[entry["position"]] = removed.get(entry["position"], 0) + 1
            break
    return removed


def cost_of_waiting_table(
    board: DraftBoard,
    current_pick: int,
    my_future_picks: list[int],
    roster: dict[str, int],
    teams: int,
    my_slot: int,
) -> pd.DataFrame:
    """Best available now vs. best available at your next pick, per position."""
    if not my_future_picks:
        return pd.DataFrame()
    next_pick = my_future_picks[1] if len(my_future_picks) > 1 else None
    removed_by_next = (
        opponent_only_shelf(board, current_pick, next_pick, teams, my_slot)
        if next_pick
        else {}
    )
    rows = []
    for pos in roster:
        pool = board.pools.get(pos, [])
        if not pool:
            continue
        best_now = pool[0]["value"]
        idx_next = removed_by_next.get(pos, 0)
        best_next = pool[idx_next]["value"] if idx_next < len(pool) else None
        rows.append(
            {
                "position": pos,
                "best_now": best_now,
                "best_at_next_pick": best_next,
                "points_lost_if_you_wait": (
                    best_now - best_next if best_next is not None else None
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        "points_lost_if_you_wait", ascending=False, na_position="last"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The whole-draft plan: exact forward simulation over one shared timeline
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanStep:
    stage: int
    pick_number: int
    position: str
    player_id: str | None
    player_name: str | None
    value: float


@dataclass(frozen=True)
class DraftPlan:
    steps: list[PlanStep]
    total_value: float
    cost_of_waiting: pd.DataFrame


def _feasible_positions(
    counts: dict[str, int], roster: dict[str, int], flex_slots: int
) -> list[str]:
    """Positions still draftable given roster minimums and the shared flex."""
    extra_used = sum(
        max(counts.get(p, 0) - roster.get(p, 0), 0) for p in FLEX_POSITIONS
    )
    out = [pos for pos, cap in roster.items() if counts.get(pos, 0) < cap]
    if extra_used < flex_slots:
        for pos in FLEX_POSITIONS:
            if pos in roster and pos not in out:
                out.append(pos)
    return out


def _next_available(pool: list[dict], removed: frozenset[str]) -> dict | None:
    for player in pool:
        if player["player_id"] not in removed:
            return player
    return None


def _simulate_from(
    stage_index: int,
    timeline_pick: int,
    future_picks: list[int],
    pools: dict[str, list[dict]],
    adp_order: list[dict],
    adp_pos: int,
    removed: frozenset[str],
    counts: dict[str, int],
    roster: dict[str, int],
    flex_slots: int,
) -> tuple[float, list[PlanStep]]:
    """Walk the shared timeline forward: consume opponent picks up to the next
    stage's pick number (advancing the ADP pointer through already-removed
    players), then branch over feasible positions for that pick."""
    if stage_index >= len(future_picks):
        return 0.0, []

    target_pick = future_picks[stage_index]
    ptr = adp_pos
    ids = removed
    for _ in range(max(target_pick - timeline_pick, 0)):
        while ptr < len(adp_order) and adp_order[ptr]["player_id"] in ids:
            ptr += 1
        if ptr < len(adp_order):
            ids = ids | {adp_order[ptr]["player_id"]}
            ptr += 1

    best_total, best_steps = float("-inf"), None
    for pos in _feasible_positions(counts, roster, flex_slots):
        player = _next_available(pools.get(pos, []), ids)
        if player is None:
            continue
        new_ids = ids | {player["player_id"]}
        new_counts = dict(counts)
        new_counts[pos] = new_counts.get(pos, 0) + 1
        rest_value, rest_steps = _simulate_from(
            stage_index + 1, target_pick + 1, future_picks, pools, adp_order,
            ptr, new_ids, new_counts, roster, flex_slots,
        )
        total = player["value"] + rest_value
        if total > best_total:
            step = PlanStep(
                stage=stage_index + 1, pick_number=target_pick, position=pos,
                player_id=player["player_id"], player_name=player["name"],
                value=player["value"],
            )
            best_total, best_steps = total, [step] + rest_steps

    if best_steps is None:
        return 0.0, []
    return best_total, best_steps


def plan_whole_draft(
    board: DraftBoard,
    current_pick: int,
    teams: int,
    my_slot: int,
    rounds: int,
    roster: dict[str, int] | None = None,
    flex_slots: int = DEFAULT_FLEX_SLOTS,
) -> DraftPlan:
    """The recommended position at every remaining starter-relevant pick.

    The plan horizon is the team's total starter slots (roster minimums plus
    the flex slots); picks beyond that are bench rounds and are not planned
    in per-position detail. Returns an empty plan if the draft (for this
    team) is already over.
    """
    roster = DEFAULT_ROSTER if roster is None else roster
    horizon = sum(roster.values()) + flex_slots

    future_picks = [p for p in my_pick_numbers(teams, my_slot, rounds) if p >= current_pick]
    future_picks = future_picks[:horizon]
    if not future_picks:
        return DraftPlan(steps=[], total_value=0.0, cost_of_waiting=pd.DataFrame())

    total, steps = _simulate_from(
        0, current_pick, future_picks, board.pools, board.adp_order,
        0, frozenset(), {}, roster, flex_slots,
    )
    cost_df = cost_of_waiting_table(
        board, current_pick, future_picks, roster, teams, my_slot
    )
    return DraftPlan(steps=steps, total_value=total, cost_of_waiting=cost_df)


def recommendation_text(plan: DraftPlan, on_the_clock: bool) -> str:
    if not plan.steps:
        return "No further starter picks to plan for this team."
    first = plan.steps[0]
    lead = "Take" if on_the_clock else "When you're on the clock, take"
    return (
        f"{lead} **{first.position}** — the plan's top choice is "
        f"**{first.player_name}**."
    )
