"""Fetch the current draft class's bio data for the rookie cold-start model.

Unlike the historical rookie training data (built from ``rosters_2016_2025.csv``,
which carries birth_date/height/weight once a player has an NFL roster entry),
a brand-new draft class's roster snapshot is missing that bio data for weeks
after the draft — nflverse's roster feed hasn't been enriched yet. This script
pulls three sources:

    * ``load_draft_picks()`` — draft slot, team, college, and age (already
      computed at draft time, so no birth_date is needed).
    * ``load_combine()`` — height and weight from pre-draft testing, joined
      on normalized player name.
    * ``load_rosters()`` — the identity anchor. ``load_draft_picks()`` also
      has a ``gsis_id`` column, but it is NOT the same identifier scheme used
      everywhere else in this project (verified against the real 2026 class:
      every value disagreed with the roster-sourced gsis_id on the players
      where both were populated). Only the roster-sourced gsis_id is trusted;
      see ``src/rookie_class.py`` for the full explanation and the match-rate
      report.

Usage:
    python scripts/fetch_rookie_class.py --year 2026

Writes ``data/raw/rookie_class_<year>.csv`` (small, committed — a few hundred
rows, refreshed by re-running this script; not the multi-year historical
files, which stay untouched).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rookie_class import build_rookie_class_frame, SKILL_POSITIONS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    try:
        import nflreadpy as nfl
    except ImportError:
        sys.stderr.write("nflreadpy is not installed. Install it with:\n    pip install nflreadpy\n")
        return 2

    root = Path(__file__).resolve().parents[1]

    print(f"Fetching {args.year} draft picks...")
    draft_picks = nfl.load_draft_picks().to_pandas()
    draft_picks = draft_picks[
        (draft_picks["season"] == args.year)
        & (draft_picks["position"].isin(SKILL_POSITIONS))
    ].copy()
    print(f"  {len(draft_picks)} skill-position picks")

    print(f"Fetching {args.year} combine data...")
    combine = nfl.load_combine().to_pandas()
    combine = combine[combine["season"] == args.year].copy()
    print(f"  {len(combine)} combine participants (all positions)")

    print(f"Fetching {args.year} rosters (identity anchor)...")
    rosters = nfl.load_rosters([args.year]).to_pandas()
    print(f"  {len(rosters)} roster rows")

    frame, diagnostics = build_rookie_class_frame(
        draft_picks, combine, rosters, year=args.year
    )

    out_path = root / "data" / "raw" / f"rookie_class_{args.year}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    print(f"Wrote {len(frame)} rows to {out_path.relative_to(root)}")
    print(
        f"Combine match: {diagnostics['combine_matched']}/{diagnostics['total_picks']} "
        f"({diagnostics['combine_match_rate']:.1%}) — the rest get a neutral "
        "height/weight prior (z-score 0) from the model, same fallback used "
        "for any missing feature."
    )
    print(
        f"gsis_id coverage: {diagnostics['gsis_id_coverage']}/{diagnostics['total_picks']} "
        f"({diagnostics['gsis_id_rate']:.1%}) — rows missing a gsis_id cannot "
        "be scored (no stable player_id to attach a projection to) and are "
        "dropped; see the printed list below if any were."
    )
    if diagnostics["missing_gsis_id_names"]:
        print("  missing gsis_id, dropped: " + ", ".join(diagnostics["missing_gsis_id_names"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
