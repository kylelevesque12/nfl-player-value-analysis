"""Global player search index + unified player-detail assembly (Session 10).

All functions here are pure: they take the already-loaded output DataFrames (the
same ones `load_all_data` caches) and return plain frames/dicts. No file I/O, no
model training, no Streamlit — so the search index and the detail assembly are
unit-testable without a Streamlit runtime, and the app stays fast because nothing
is recomputed.

The index is keyed by the stable nflverse `player_id` (gsis id). A player is
included if they appear in any project output (weekly backtest, live projection,
salary/value, rookie model, or as a treated QB in the causal panel), and carries
boolean flags for which modules have data so the detail page can show clean
"not available" states instead of crashing.
"""

from __future__ import annotations

import pandas as pd

ID = "player_id"
NAME = "player_display_name"
POS = "position"
TEAM = "team"


def _latest_meta(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    """One row per player_id carrying the most recent name/position/team."""
    if df is None or df.empty or ID not in df.columns:
        return pd.DataFrame(columns=[ID, NAME, POS, TEAM])
    keep = [c for c in [ID, NAME, POS, TEAM, *sort_cols] if c in df.columns]
    d = df[keep].dropna(subset=[ID]).copy()
    present_sort = [c for c in sort_cols if c in d.columns]
    if present_sort:
        d = d.sort_values(present_sort)
    d = d.drop_duplicates(ID, keep="last")
    for col in (NAME, POS, TEAM):
        if col not in d.columns:
            d[col] = pd.NA
    return d[[ID, NAME, POS, TEAM]]


def _ids(df: pd.DataFrame, col: str = ID) -> set:
    if df is None or df.empty or col not in df.columns:
        return set()
    return set(df[col].dropna().astype(str))


def _seasons_by_player(*frames: pd.DataFrame) -> dict:
    out: dict[str, set] = {}
    for df in frames:
        if df is None or df.empty or ID not in df.columns or "season" not in df.columns:
            continue
        sub = df[[ID, "season"]].dropna()
        for pid, season in zip(sub[ID].astype(str), pd.to_numeric(sub["season"], errors="coerce")):
            if pd.notna(season):
                out.setdefault(pid, set()).add(int(season))
    return out


def build_player_index(
    weekly: pd.DataFrame | None = None,
    live: pd.DataFrame | None = None,
    salary: pd.DataFrame | None = None,
    rookie: pd.DataFrame | None = None,
    causal: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per player_id with metadata + per-module availability flags.

    Columns: player_id, player_display_name, position, team, seasons (str),
    has_weekly, has_live, has_surplus, has_rookie, has_causal.
    """
    # Metadata: prefer the most recent weekly record, then salary, then rookie.
    meta_sources = [
        _latest_meta(weekly, ["season", "week"]),
        _latest_meta(salary, ["season"]),
        _latest_meta(rookie, ["rookie_year"]),
    ]
    non_empty = [m for m in meta_sources if not m.empty]
    meta = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame(
        columns=[ID, NAME, POS, TEAM]
    )
    if meta.empty:
        return pd.DataFrame(
            columns=[ID, NAME, POS, TEAM, "seasons", "has_weekly", "has_live",
                     "has_surplus", "has_rookie", "has_causal"]
        )
    # First source wins (weekly is most authoritative for current team).
    meta = meta.drop_duplicates(ID, keep="first")

    weekly_ids = _ids(weekly)
    live_ids = _ids(live)
    salary_ids = _ids(salary)
    rookie_ids = _ids(rookie)
    # Causal: a player is "involved" if they are a treated QB (qb_id).
    causal_ids = _ids(causal, "qb_id")

    all_ids = weekly_ids | live_ids | salary_ids | rookie_ids | causal_ids
    # Any id present in a source but missing metadata gets a placeholder row.
    known = set(meta[ID].astype(str))
    missing = all_ids - known
    if missing:
        extra = pd.DataFrame({ID: sorted(missing)})
        extra[NAME] = extra[ID]
        extra[POS] = pd.NA
        extra[TEAM] = pd.NA
        meta = pd.concat([meta, extra], ignore_index=True)
    meta = meta[meta[ID].astype(str).isin(all_ids)].copy()

    seasons = _seasons_by_player(weekly, salary, rookie)
    meta["seasons"] = meta[ID].astype(str).map(
        lambda p: ", ".join(str(s) for s in sorted(seasons.get(p, set()))) or "—"
    )
    meta["has_weekly"] = meta[ID].astype(str).isin(weekly_ids)
    meta["has_live"] = meta[ID].astype(str).isin(live_ids)
    meta["has_surplus"] = meta[ID].astype(str).isin(salary_ids)
    meta["has_rookie"] = meta[ID].astype(str).isin(rookie_ids)
    meta["has_causal"] = meta[ID].astype(str).isin(causal_ids)

    meta = meta.drop_duplicates(ID, keep="first").reset_index(drop=True)
    assert meta[ID].is_unique, "player index must have unique player_id keys"
    return meta.sort_values(NAME, na_position="last").reset_index(drop=True)


def display_label(row: pd.Series | dict) -> str:
    """e.g. 'Justin Jefferson · WR · MIN'."""
    name = row.get(NAME) or row.get(ID) or "Unknown"
    pos = row.get(POS)
    team = row.get(TEAM)
    parts = [str(name)]
    if pos is not None and pd.notna(pos):
        parts.append(str(pos))
    if team is not None and pd.notna(team):
        parts.append(str(team))
    return " · ".join(parts)


def search_players(index: pd.DataFrame, query: str, limit: int = 30) -> pd.DataFrame:
    """Case-insensitive substring match on player name; returns matched index
    rows with a ``label`` column, capped at ``limit``."""
    if index is None or index.empty or not query or not query.strip():
        return index.head(0).assign(label=pd.Series(dtype=str)) if index is not None else pd.DataFrame()
    q = query.strip().lower()
    mask = index[NAME].astype(str).str.lower().str.contains(q, na=False, regex=False)
    hits = index[mask].copy().head(limit)
    hits["label"] = hits.apply(display_label, axis=1)
    return hits


# ---------------------------------------------------------------------------
# Per-player detail assembly
# ---------------------------------------------------------------------------
def _filter(df: pd.DataFrame | None, player_id: str, col: str = ID) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame()
    return df[df[col].astype(str) == str(player_id)].copy()


def assemble_player_detail(
    player_id: str,
    *,
    weekly: pd.DataFrame | None = None,
    live: pd.DataFrame | None = None,
    salary: pd.DataFrame | None = None,
    top_surplus: pd.DataFrame | None = None,
    rookie: pd.DataFrame | None = None,
    rookie_pred: pd.DataFrame | None = None,
    causal: pd.DataFrame | None = None,
    weekly_method: str = "hist_gradient_boosting",
) -> dict:
    """Filter every source to one player. Each section is None/empty when the
    player has no data there, so the page can render clean unavailable states."""
    wk = _filter(weekly, player_id)
    if not wk.empty and "method" in wk.columns:
        wk = wk[wk["method"].eq(weekly_method)].copy()
    wk = wk.sort_values([c for c in ["season", "week"] if c in wk.columns]) if not wk.empty else wk

    sal = _filter(salary, player_id)
    if not sal.empty and "season" in sal.columns:
        sal = sal.sort_values("season")
        # Surplus history must be one row per (player, season).
        sal = sal.drop_duplicates([ID, "season"], keep="last")

    live_f = _filter(live, player_id)
    rk = _filter(rookie, player_id)
    rk_pred = _filter(rookie_pred, player_id)
    cz = _filter(causal, player_id, col="qb_id")

    def _or_none(df: pd.DataFrame) -> pd.DataFrame | None:
        return df if df is not None and not df.empty else None

    detail = {
        "player_id": str(player_id),
        "weekly_history": _or_none(wk),
        "live": _or_none(live_f),
        "surplus_history": _or_none(sal),
        "rookie": _or_none(rk),
        "rookie_pred": _or_none(rk_pred),
        "causal": _or_none(cz),
    }

    # Top-surplus headline is name-keyed (the saved table lacks player_id); match
    # by the player's display name where available.
    name = None
    for src in (wk, sal, rk):
        if src is not None and not src.empty and NAME in src.columns:
            name = src[NAME].iloc[-1]
            break
    detail["player_name"] = name or str(player_id)
    if top_surplus is not None and not top_surplus.empty and name and NAME in top_surplus.columns:
        ts = top_surplus[top_surplus[NAME].astype(str) == str(name)].copy()
        detail["top_surplus"] = ts if not ts.empty else None
    else:
        detail["top_surplus"] = None
    return detail
