"""Fetch historical DraftKings salaries from RotoGuru.

RotoGuru1 (http://rotoguru1.com) maintains free historical DK / FD / Yahoo
DFS salary CSVs. Their free DK archive covers 2014-2021 inclusive. 2022 and
later return empty pages (a paid source like Stokastic or FantasyData is
needed for newer seasons; see PORTFOLIO_ROADMAP.md Tier 1 item #1 for options).

What this script does:
  * Loops over (year, week) pairs and downloads the semicolon-separated DK
    salary block from each page.
  * Caches per-week files under ``data/raw/rotoguru_cache/`` so reruns skip
    pages that already succeeded.
  * Concatenates all cached weeks into ``data/raw/dk_salaries_<start>_<end>.csv``.

The fetcher is intentionally polite: a small delay between requests, HTTP
errors are logged but do not abort the run.

Usage::

    python scripts/fetch_rotoguru_salaries.py --years 2014-2021

The output CSV is later transformed into the schema expected by
``src/external_benchmark.py`` via ``scripts/build_external_projections_from_dk.py``.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests


ROTOGURU_URL = (
    "http://rotoguru1.com/cgi-bin/fyday.pl?week={week}&year={year}&game=dk&scsv=1"
)
HEADER_LINE = "Week;Year;GID;Name;Pos;Team;h/a;Oppt;DK points;DK salary"
DEFAULT_YEARS = "2014-2021"
DEFAULT_WEEKS = list(range(1, 19))  # weeks 1-18 cover the regular season
REQUEST_DELAY_SECONDS = 0.4
USER_AGENT = (
    "nfl-player-value-analysis/portfolio (educational; "
    "contact via repository)"
)


def _parse_years(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch historical DK salaries from RotoGuru."
    )
    parser.add_argument(
        "--years",
        default=DEFAULT_YEARS,
        help=(
            'Year range or comma list. Default "2014-2021". RotoGuru free DK '
            "archive only covers through 2021."
        ),
    )
    parser.add_argument(
        "--weeks",
        default="",
        help='Comma list of weeks to fetch. Default: 1-18.',
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch even weeks already cached.",
    )
    return parser.parse_args()


def _find_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extract_scsv_block(html: str) -> str | None:
    """Pull the semicolon-separated data block out of the RotoGuru HTML."""
    if HEADER_LINE not in html:
        return None
    # The block starts at the header line and ends at the </pre> closing tag.
    start = html.index(HEADER_LINE)
    end_match = re.search(r"</pre>", html[start:])
    if end_match is None:
        return None
    block = html[start : start + end_match.start()]
    return block.strip()


def _parse_scsv_block(block: str) -> pd.DataFrame:
    """Parse the SCSV block into a DataFrame."""
    lines = [line for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return pd.DataFrame()
    header_cols = [c.strip() for c in lines[0].split(";")]
    rows: list[list[str]] = []
    for line in lines[1:]:
        parts = line.split(";")
        if len(parts) != len(header_cols):
            continue  # malformed row
        rows.append([p.strip() for p in parts])
    df = pd.DataFrame(rows, columns=header_cols)
    # Friendly column names that match downstream conventions.
    return df.rename(
        columns={
            "Week": "week",
            "Year": "year",
            "GID": "game_id_rotoguru",
            "Name": "name_lastfirst",
            "Pos": "position",
            "Team": "team_rotoguru",
            "h/a": "home_away",
            "Oppt": "opponent_rotoguru",
            "DK points": "dk_points_scored",
            "DK salary": "dk_salary",
        }
    )


def fetch_week(
    year: int,
    week: int,
    cache_dir: Path,
    refresh: bool,
    session: requests.Session,
) -> pd.DataFrame:
    cache_path = cache_dir / f"rotoguru_dk_{year}_w{week:02d}.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, dtype=str)

    url = ROTOGURU_URL.format(week=week, year=year)
    try:
        response = session.get(url, timeout=20)
    except requests.RequestException as exc:
        sys.stderr.write(f"  year={year} week={week} request failed: {exc}\n")
        return pd.DataFrame()

    if response.status_code != 200:
        sys.stderr.write(
            f"  year={year} week={week} HTTP {response.status_code}\n"
        )
        return pd.DataFrame()

    block = _extract_scsv_block(response.text)
    if block is None:
        sys.stderr.write(
            f"  year={year} week={week} no SCSV block found in response\n"
        )
        return pd.DataFrame()

    df = _parse_scsv_block(block)
    if df.empty:
        sys.stderr.write(f"  year={year} week={week} parsed empty\n")
        return df

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def main() -> int:
    args = parse_args()
    years = _parse_years(args.years)
    weeks = (
        [int(w.strip()) for w in args.weeks.split(",") if w.strip()]
        if args.weeks
        else DEFAULT_WEEKS
    )

    project_root = _find_project_root()
    raw_dir = project_root / "data" / "raw"
    cache_dir = raw_dir / "rotoguru_cache"
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rows: list[pd.DataFrame] = []
    fetched = 0
    skipped = 0
    empty = 0
    for year in years:
        for week in weeks:
            cache_path = cache_dir / f"rotoguru_dk_{year}_w{week:02d}.csv"
            already_cached = cache_path.exists() and not args.refresh
            if already_cached:
                df = pd.read_csv(cache_path, dtype=str)
                skipped += 1
            else:
                df = fetch_week(year, week, cache_dir, args.refresh, session)
                fetched += 1
                time.sleep(REQUEST_DELAY_SECONDS)
            if df.empty:
                empty += 1
                continue
            rows.append(df)
            print(
                f"  year={year} week={week} rows={len(df):,} "
                f"{'cached' if already_cached else 'fetched'}"
            )

    if not rows:
        sys.stderr.write("No RotoGuru data could be loaded.\n")
        return 1

    combined = pd.concat(rows, ignore_index=True)
    out_path = raw_dir / f"dk_salaries_{years[0]}_{years[-1]}.csv"
    combined.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path.relative_to(project_root)} "
        f"with {len(combined):,} rows "
        f"(fetched={fetched}, cached={skipped}, empty={empty})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
