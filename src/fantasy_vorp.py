"""Fantasy VORP: value over replacement player, and the overall draft board.

The problem VORP solves: projected points are not comparable across positions.
The 12th quarterback and the 12th running back both start in a 12-team league,
but the freely available player behind them differs enormously — QB13 scores
nearly as much as QB12, while the next running back up is far worse. A player's
draft value is therefore his projection minus the replacement level at his
position: the points a team could get from the best player nobody starts.

Replacement level is computed by actually filling the league's starting
lineups, not by a hand-picked rank. For a 12-team league with a standard
roster (QB, 2 RB, 2 WR, TE, one RB/WR/TE flex): take the top 12 QBs, top 24
RBs, top 24 WRs, and top 12 TEs as fixed starters, then fill the 12 flex
slots with the best remaining skill players. The replacement level at each
position is the projection of the best player left on the board after all
starters are gone.

Auction values follow from VORP: a 12-team, $200-budget league has $2,400 to
spend and must pay at least $1 for each of its 180 roster spots. The
remaining discretionary budget is split across positive-VORP players in
proportion to their VORP.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.adp import load_adp_snapshot, match_adp_to_projections
from src.load_data import ensure_project_dirs, find_project_root

PROJ_COL = "predicted_2026_fantasy_points_ppr"
LOW_COL = "prediction_interval_low"
HIGH_COL = "prediction_interval_high"

DEFAULT_TEAMS = 12
STANDARD_ROSTER = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLEX_SLOTS = 1
FLEX_POSITIONS = ("RB", "WR", "TE")

AUCTION_BUDGET_PER_TEAM = 200
ROSTER_SPOTS_PER_TEAM = 15


def compute_replacement_points(
    fantasy: pd.DataFrame,
    teams: int = DEFAULT_TEAMS,
    roster: dict[str, int] | None = None,
    flex_slots: int = FLEX_SLOTS,
) -> dict[str, float]:
    """Replacement-level projected points per position, by filling lineups.

    Returns {position: points of the best player left after every starting
    slot in the league, including flex, is filled}.
    """
    roster = STANDARD_ROSTER if roster is None else roster
    pools = {
        pos: grp.sort_values(PROJ_COL, ascending=False)[PROJ_COL].to_list()
        for pos, grp in fantasy.groupby("position")
    }

    # Fixed starters come off the top of each position pool.
    taken = {pos: teams * count for pos, count in roster.items()}

    # Flex slots go to the best remaining skill players, one at a time.
    for _ in range(teams * flex_slots):
        best_pos, best_points = None, float("-inf")
        for pos in FLEX_POSITIONS:
            pool = pools.get(pos, [])
            idx = taken.get(pos, 0)
            if idx < len(pool) and pool[idx] > best_points:
                best_pos, best_points = pos, pool[idx]
        if best_pos is None:
            break
        taken[best_pos] = taken.get(best_pos, 0) + 1

    replacement = {}
    for pos in set(roster) | set(FLEX_POSITIONS):
        pool = pools.get(pos, [])
        idx = min(taken.get(pos, 0), max(len(pool) - 1, 0))
        replacement[pos] = float(pool[idx]) if pool else 0.0
    return replacement


def auction_values(
    board: pd.DataFrame,
    teams: int = DEFAULT_TEAMS,
    budget: int = AUCTION_BUDGET_PER_TEAM,
    roster_spots: int = ROSTER_SPOTS_PER_TEAM,
) -> pd.Series:
    """Dollar values: $1 floor everywhere, discretionary budget split in
    proportion to positive VORP."""
    league_budget = teams * budget
    discretionary = league_budget - teams * roster_spots  # $1 per roster spot
    positive = board["vorp"].clip(lower=0.0)
    total = float(positive.sum())
    share = positive / total if total > 0 else 0.0
    return (1.0 + share * discretionary).round(0).astype(int)


# Columns that only ever come from the ADP merge. When no ADP snapshot is
# available the board is still fully usable (VORP, auction values, and ranks
# all stand on their own), but these columns must still exist, filled with
# NaN, so every consumer sees one stable schema instead of having to
# special-case "column missing" versus "column present but blank."
ADP_DERIVED_COLUMNS = [
    "adp", "adp_formatted", "adp_overall_rank", "bye",
    "adp_total_drafts", "adp_window_end",
]


def build_draft_board(
    fantasy: pd.DataFrame,
    adp: pd.DataFrame | None = None,
    teams: int = DEFAULT_TEAMS,
) -> tuple[pd.DataFrame, dict]:
    """The overall (cross-position) draft board, with ADP merged when given.

    This is a legitimate base board even without ADP: value over replacement,
    auction values, and overall rank all come from the season projections
    alone. ADP only adds the market-comparison columns on top.

    edge_vs_adp = ADP overall rank − VORP overall rank: positive means the
    market lets you draft the player later than the model ranks him (a value),
    negative means the market takes him earlier (a fade).
    """
    replacement = compute_replacement_points(fantasy, teams=teams)

    board = fantasy.copy()
    board["replacement_points"] = board["position"].map(replacement)
    board["vorp"] = board[PROJ_COL] - board["replacement_points"]
    if LOW_COL in board.columns and HIGH_COL in board.columns:
        board["vorp_low"] = board[LOW_COL] - board["replacement_points"]
        board["vorp_high"] = board[HIGH_COL] - board["replacement_points"]

    board = board.sort_values("vorp", ascending=False).reset_index(drop=True)
    board["overall_rank"] = range(1, len(board) + 1)
    board["auction_value"] = auction_values(board, teams=teams)

    diagnostics: dict = {"replacement_points": replacement}
    if adp is not None and not adp.empty:
        board, adp_diag = match_adp_to_projections(board, adp)
        diagnostics.update(adp_diag)
        board["edge_vs_adp"] = board["adp_overall_rank"] - board["overall_rank"]
    else:
        board["edge_vs_adp"] = pd.NA

    for col in ADP_DERIVED_COLUMNS:
        if col not in board.columns:
            board[col] = pd.NA

    return board, diagnostics


def build_draft_board_outputs(
    project_root: str | Path | None = None,
    year: int = 2026,
    save_outputs: bool = True,
) -> dict:
    """Build and save the overall draft board table + a plain-language report."""
    root = find_project_root() if project_root is None else Path(project_root).resolve()
    dirs = ensure_project_dirs(root)

    fantasy = pd.read_csv(
        dirs["tables"] / f"{year}_fantasy_football_projections.csv"
    )
    adp = load_adp_snapshot(root, year)
    board, diagnostics = build_draft_board(fantasy, adp)

    keep = [
        "overall_rank", "player_id", "player_display_name", "position",
        "primary_team_2025", PROJ_COL, LOW_COL, HIGH_COL,
        "replacement_points", "vorp", "vorp_low", "vorp_high",
        "auction_value", "adp", "adp_formatted", "adp_overall_rank",
        "edge_vs_adp", "bye", "adp_total_drafts", "adp_window_end",
    ]
    out = board[[c for c in keep if c in board.columns]]

    if save_outputs:
        out.to_csv(
            dirs["tables"] / f"draft_board_{year}.csv",
            index=False,
            float_format=config.CSV_FLOAT_FORMAT,
        )
        _write_report(root, board, diagnostics, year)

    return {"board": out, "diagnostics": diagnostics}


def _write_report(root: Path, board: pd.DataFrame, diagnostics: dict, year: int) -> None:
    replacement = diagnostics["replacement_points"]
    top10 = board.head(10)
    lines = [
        "# The overall draft board: VORP and auction values",
        "",
        "This report documents how the cross-position draft board is built.",
        "",
        "## The question",
        "",
        "Projected points are not comparable across positions: the 12th-best",
        "quarterback and the 12th-best running back both start in a 12-team",
        "league, but the freely available player behind each differs",
        "enormously. The board therefore ranks players by VORP (value over",
        "replacement player): projected points minus the replacement level at",
        "the position.",
        "",
        "## How replacement level is computed",
        "",
        "The league's starting lineups are actually filled: 12 QB, 24 RB, 24",
        "WR, and 12 TE as fixed starters, then 12 flex slots go to the best",
        "remaining skill players one at a time. The replacement level at each",
        "position is the projection of the best player left after every",
        "starting slot is gone. The computed levels:",
        "",
        "| Position | Replacement-level projected PPR |",
        "| --- | ---: |",
    ]
    for pos in ("QB", "RB", "WR", "TE"):
        if pos in replacement:
            lines.append(f"| {pos} | {replacement[pos]:.1f} |")
    lines += [
        "",
        "## The top of the board",
        "",
        "| Overall | Player | Pos | VORP | Auction $ | ADP |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in top10.iterrows():
        adp_txt = row.get("adp_formatted", "")
        adp_txt = adp_txt if isinstance(adp_txt, str) else "—"
        lines.append(
            f"| {int(row['overall_rank'])} | {row['player_display_name']} | "
            f"{row['position']} | {row['vorp']:.0f} | "
            f"${int(row['auction_value'])} | {adp_txt} |"
        )
    if "adp_match_rate" in diagnostics:
        unmatched = diagnostics.get("top100_unmatched", [])
        lines += [
            "",
            "## ADP match diagnostics",
            "",
            f"{diagnostics['adp_matched']} of {diagnostics['adp_players']} ADP",
            f"players matched to the projection table",
            f"({diagnostics['adp_match_rate']:.1%}). However, one gap needs to",
            "be stated plainly: the unmatched players inside the top 100 picks",
            "are 2026 rookies, who are not in the season projection table yet.",
            "Until the rookie class is scored (a planned item on the roadmap),",
            "the board is honest about veterans but silent on rookies:",
            "",
        ]
        for row in unmatched:
            lines.append(
                f"- {row['adp_formatted']} {row['position']} {row['adp_name']}"
            )
    lines += [
        "",
        "## Limitations",
        "",
        "- Replacement level uses a standard 12-team lineup (QB, 2 RB, 2 WR,",
        "  TE, one flex). Other league shapes shift the levels; the code takes",
        "  the league shape as a parameter.",
        "- Auction values split the league's discretionary budget in",
        "  proportion to positive VORP. That is a defensible convention, not",
        "  market truth.",
        "- VORP inherits the projections' uncertainty. The board carries",
        "  vorp_low / vorp_high from the 80% projection intervals, and",
        "  adjacent players are often statistically indistinguishable.",
        "",
    ]
    report_path = root / "report" / "draft_board_vorp.md"
    report_path.write_text("\n".join(lines))


if __name__ == "__main__":
    result = build_draft_board_outputs()
    print(result["board"].head(15).to_string(index=False))
    print(result["diagnostics"]["replacement_points"])
