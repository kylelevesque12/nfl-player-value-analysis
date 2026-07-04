"""Fetch the current ADP snapshot from Fantasy Football Calculator.

Usage:
    python scripts/fetch_adp.py --year 2026
    python scripts/fetch_adp.py --year 2026 --teams 12 --scoring ppr

Writes data/external/adp_{year}_ppr.csv (committed: it is a small snapshot
and the deployed app must not call external APIs). Prints the snapshot
metadata and the match diagnostics against the season projection table so a
broken join is visible immediately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.adp import fetch_ffc_adp, match_adp_to_projections, save_adp_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument("--scoring", default="ppr", choices=["ppr", "half-ppr", "standard"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    adp = fetch_ffc_adp(args.year, scoring=args.scoring, teams=args.teams)
    path = save_adp_snapshot(adp, root, args.year)
    print(
        f"saved {len(adp)} players to {path.relative_to(root)} "
        f"({int(adp['adp_total_drafts'].iloc[0]):,} drafts through "
        f"{adp['adp_window_end'].iloc[0]})"
    )

    fantasy_path = root / "outputs" / "tables" / "2026_fantasy_football_projections.csv"
    if fantasy_path.exists():
        fantasy = pd.read_csv(fantasy_path)
        _, diag = match_adp_to_projections(fantasy, adp)
        print(
            f"projection match: {diag['adp_matched']}/{diag['adp_players']} "
            f"ADP players ({diag['adp_match_rate']:.1%})"
        )
        if diag["top100_unmatched"]:
            print("unmatched inside the top 100 picks (fix via NAME_ALIASES or rookies):")
            for row in diag["top100_unmatched"]:
                print(f"  {row['adp_formatted']:>6}  {row['position']:<3} {row['adp_name']}")
        else:
            print("no unmatched players inside the top 100 picks")


if __name__ == "__main__":
    main()
