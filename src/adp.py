"""Average draft position (ADP) from Fantasy Football Calculator's public API.

ADP is the market price of every player: across thousands of real mock and
money drafts, the average pick at which each player is taken. The Draft Room
needs it to predict what other drafters will do, and the Draft Board uses it
to show where the model disagrees with the market.

Source: https://fantasyfootballcalculator.com/api/v1/adp/{scoring} — a free,
public, keyless API. The fetch is a small snapshot (a few hundred rows), so
the result is committed to ``data/external/`` and the deployed app never
calls the network. Matching to the projection table is by normalized name +
position (never by team: ADP reflects current rosters while the projection
table carries prior-season teams), with the match rate reported so a silent
join failure cannot slip through.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{scoring}?teams={teams}&year={year}"
SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

# Tokens dropped during name normalization (suffixes that one source carries
# and the other does not).
_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}

# Known cross-source name differences (FFC name -> nflverse display name),
# applied after normalization. Extend as diagnostics surface new ones.
NAME_ALIASES = {
    "hollywood brown": "marquise brown",
    "cam ward": "cameron ward",
}


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation and generational suffixes, collapse spaces.

    Apostrophes and periods are deleted rather than replaced with spaces:
    sources disagree on writing "Ja'Marr" vs "JaMarr", and both must land on
    the same key ("jamarr"). Hyphens and other separators become spaces.
    """
    text = str(name).lower().replace("'", "").replace(".", "")
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in _SUFFIX_TOKENS]
    normalized = " ".join(tokens)
    return NAME_ALIASES.get(normalized, normalized)


def fetch_ffc_adp(year: int, scoring: str = "ppr", teams: int = 12) -> pd.DataFrame:
    """Fetch one ADP snapshot from Fantasy Football Calculator.

    Returns one row per player with the draft-market columns plus the
    snapshot metadata (total drafts, window end date) repeated on every row,
    so the saved CSV is self-describing about its freshness.
    """
    url = FFC_URL.format(scoring=scoring, teams=teams, year=year)
    # FFC returns 403 to the default Python-urllib user agent; identify as a
    # normal client instead.
    request = urllib.request.Request(
        url, headers={"User-Agent": "nfl-player-value-analysis/1.0 (portfolio project)"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "Success":
        raise RuntimeError(f"FFC ADP API returned status={payload.get('status')!r}")

    players = pd.DataFrame(payload["players"])
    players = players[players["position"].isin(SKILL_POSITIONS)].copy()
    players = players.rename(columns={"name": "adp_name", "team": "adp_team"})
    keep = [
        "adp_name", "position", "adp_team", "adp", "adp_formatted",
        "times_drafted", "high", "low", "stdev", "bye",
    ]
    players = players[[c for c in keep if c in players.columns]]
    players["adp_overall_rank"] = players["adp"].rank(method="first").astype(int)

    meta = payload.get("meta", {})
    players["adp_total_drafts"] = meta.get("total_drafts")
    players["adp_window_end"] = meta.get("end_date")
    players["adp_scoring"] = meta.get("type", scoring.upper())
    return players.sort_values("adp").reset_index(drop=True)


def match_adp_to_projections(
    fantasy: pd.DataFrame, adp: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Left-join ADP onto the projection table by normalized name + position.

    A name-only fallback catches position-label disagreements, but only when
    the name is unique on both sides. Returns the merged frame plus
    diagnostics: overall match rate on the ADP side, and the unmatched ADP
    players inside the top 100 picks (the ones a draft would actually miss).
    """
    fan = fantasy.copy()
    fan["_key_name"] = fan["player_display_name"].map(normalize_name)
    mkt = adp.copy()
    mkt["_key_name"] = mkt["adp_name"].map(normalize_name)

    merged = fan.merge(
        mkt, left_on=["_key_name", "position"], right_on=["_key_name", "position"],
        how="left",
    )

    # Name-only fallback for rows the strict key missed, when unambiguous.
    missed = merged["adp"].isna()
    fan_unique = ~fan["_key_name"].duplicated(keep=False)
    mkt_unique = ~mkt["_key_name"].duplicated(keep=False)
    fallback = fan.loc[missed[missed].index]
    if not fallback.empty:
        candidates = mkt[mkt_unique].set_index("_key_name")
        for idx, row in fallback.iterrows():
            key = row["_key_name"]
            if key in candidates.index and fan_unique.loc[idx]:
                hit = candidates.loc[key]
                for col in mkt.columns:
                    if col not in ("_key_name", "position"):
                        merged.loc[idx, col] = hit[col]

    matched_keys = set(merged.loc[merged["adp"].notna(), "_key_name"])
    unmatched = mkt[~mkt["_key_name"].isin(matched_keys)]
    top100_unmatched = unmatched[unmatched["adp_overall_rank"] <= 100]
    diagnostics = {
        "adp_players": int(len(mkt)),
        "adp_matched": int(len(mkt) - len(unmatched)),
        "adp_match_rate": float((len(mkt) - len(unmatched)) / max(len(mkt), 1)),
        "top100_unmatched": top100_unmatched[
            ["adp_name", "position", "adp_formatted"]
        ].to_dict("records"),
    }
    return merged.drop(columns=["_key_name"]), diagnostics


def save_adp_snapshot(adp: pd.DataFrame, project_root: Path, year: int) -> Path:
    out_dir = Path(project_root) / "data" / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"adp_{year}_ppr.csv"
    adp.to_csv(path, index=False)
    return path


def load_adp_snapshot(project_root: Path, year: int) -> pd.DataFrame:
    path = Path(project_root) / "data" / "external" / f"adp_{year}_ppr.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
