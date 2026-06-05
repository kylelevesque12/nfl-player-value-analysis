"""Fetch college production scores for rookie projections (stub).

The Bayesian rookie model (``src/rookie_bayes.py``) accepts an optional
per-player ``college_score`` feature. The intended source is the
CollegeFootballData API (https://collegefootballdata.com/), which is free
but requires registering for an API key.

This file is a STUB. The full implementation should:

1. Pull career-level production for each rookie's draft year minus 1 (last
   college season). For receivers/RBs/QBs the proxies are:
   - WR/TE: career receiving yards per game, target share, breakout age,
     yards after catch rate
   - RB: career rushing yards per game, yards per carry, receiving usage
   - QB: career passing efficiency, career rushing yards, completion %
2. Adjust for conference strength (Power 5 vs G5 multiplier).
3. Standardize into a single ``college_score`` per (position, draft_year).
4. Write ``data/raw/college_production.csv`` with columns
   ``player_id,college_score``.

API reference: https://api.collegefootballdata.com/
Recommended Python client: ``cfbd-py`` (https://pypi.org/project/cfbd-py/)
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch CFBData college production for rookies (STUB)."
    )
    parser.add_argument("--api-key", help="CFBData API key.", required=False)
    args = parser.parse_args()

    sys.stderr.write(
        "\nThis script is a stub. See the module docstring for the\n"
        "implementation plan and PORTFOLIO_ROADMAP.md Tier 2 #4 for context.\n\n"
        "Until college production is wired in, the Bayesian rookie model\n"
        "uses draft capital and physical features only — which still produces\n"
        "a non-trivial cold-start projection.\n\n"
    )
    if args.api_key:
        sys.stderr.write(f"(API key supplied but unused: {args.api_key[:6]}...)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
