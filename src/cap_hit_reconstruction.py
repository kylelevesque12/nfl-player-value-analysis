"""Reconstruct season-level cap hits from historical contract terms.

WHY THIS EXISTS. The salary track originally used ``inflated_apy`` — the
inflation-adjusted *average* annual value of a contract — as the cost variable.
APY is flat across a deal, but real cap hits are not: signing-bonus proration and
backloaded base salaries make the early years of a contract cost far less against
the cap than the late years, and rookie deals are cheap throughout. Charging a
star the same APY in year 1 as in year 5 understates how much surplus a team
captures early in a deal (the Brock Purdy problem), so the front-office story
needs a season-specific cap hit.

WHAT THE DATA SUPPORTS. ``historical_contracts.csv`` (nflverse / OverTheCap) has
contract-level fields only — total ``value``, ``guaranteed``, ``apy``, ``years``,
``year_signed`` (and inflation-adjusted versions) — but **no year-by-year base
salary or signing-bonus schedule**. We therefore cannot parse true cap hits. What
we *can* do is reconstruct a realistic cap-hit curve from the contract terms:

    cap_hit(year k) = prorated signing bonus + backloaded base salary

with two transparent, documented assumptions:

  1. The guaranteed money is treated as a signing-bonus-like proxy, prorated
     evenly over min(years, 5) — the NFL's max proration window.
  2. The remaining (value - guaranteed) base pool is distributed across the
     contract years on a gently rising (backloaded) schedule, the typical real
     contract shape.

By construction the reconstructed cap hits over a contract sum to its total
value, so no money is invented or lost. This is an *estimate from contract
terms*, not a parsed cap hit, and every row is flagged as such. Where contract
terms are too thin we fall back to flat APY (clearly flagged), and where no
contract matches we mark the row missing.

All amounts are kept in the same inflation-adjusted millions as the prior
``inflated_apy`` variable, so surplus numbers stay comparable across eras.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BACKLOAD = 0.35          # base salary rises from (1-BACKLOAD)*mean to (1+BACKLOAD)*mean
MAX_PRORATION_YEARS = 5  # NFL caps signing-bonus proration at 5 years

# Quality flags (confidence in the estimate).
QUALITY_TERMS = "estimated_from_contract_terms"
QUALITY_FALLBACK = "fallback_apy"
QUALITY_MISSING = "missing_contract"
# Source (method used).
SOURCE_CURVE = "contract_terms_curve"
SOURCE_FALLBACK = "apy_flat_fallback"
SOURCE_MISSING = "missing"


def load_contracts(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def normalize_contracts(contracts: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns and derive the contract's active-season window.

    Uses inflation-adjusted amounts (``inflated_value``/``inflated_guaranteed``/
    ``inflated_apy``) when present so the reconstructed cap hits are in the same
    units as the legacy variable; falls back to nominal otherwise.
    """
    c = _num(
        contracts,
        ["year_signed", "years", "value", "apy", "guaranteed",
         "inflated_value", "inflated_apy", "inflated_guaranteed"],
    )
    c = c[c["year_signed"].notna() & c["years"].notna()
          & c["year_signed"].ge(1990) & c["years"].gt(0)].copy()

    c["value_m"] = c["inflated_value"] if "inflated_value" in c.columns else c.get("value")
    c["apy_m"] = c["inflated_apy"] if "inflated_apy" in c.columns else c.get("apy")
    c["guaranteed_m"] = (
        c["inflated_guaranteed"] if "inflated_guaranteed" in c.columns else c.get("guaranteed")
    )
    # Fall back to nominal where the inflation-adjusted figure is missing.
    if "value" in c.columns:
        c["value_m"] = c["value_m"].fillna(c["value"])
    if "apy" in c.columns:
        c["apy_m"] = c["apy_m"].fillna(c["apy"])
    if "guaranteed" in c.columns:
        c["guaranteed_m"] = c["guaranteed_m"].fillna(c["guaranteed"])

    c["years_int"] = np.ceil(c["years"]).clip(lower=1).astype(int)
    c["contract_start_season"] = c["year_signed"].astype(int)
    c["contract_end_season"] = c["contract_start_season"] + c["years_int"] - 1
    return c


def season_cap_hit_curve(
    value_m: float,
    guaranteed_m: float,
    years_int: int,
    contract_year: int,
    backload: float = BACKLOAD,
    max_proration: int = MAX_PRORATION_YEARS,
) -> float:
    """Estimated cap hit (millions) for one year of a contract.

    ``contract_year`` is 1-indexed within the deal. Cap hit =
    prorated-signing-bonus + backloaded-base; the per-year values sum to
    ``value_m`` across the whole contract.
    """
    Y = max(int(years_int), 1)
    k = min(max(int(contract_year), 1), Y)
    value_m = float(value_m) if np.isfinite(value_m) else 0.0
    g = float(guaranteed_m) if np.isfinite(guaranteed_m) else 0.0
    sb = min(max(g, 0.0), max(value_m, 0.0))         # signing-bonus proxy
    P = min(Y, max_proration)
    bonus_k = (sb / P) if k <= P else 0.0
    base_pool = max(value_m - sb, 0.0)
    if Y == 1:
        weight = 1.0
    else:
        weight = (1.0 - backload) + (k - 1) / (Y - 1) * (2.0 * backload)  # mean 1
    base_k = base_pool * (weight / Y)
    return bonus_k + base_k


def add_estimated_cap_hit(
    expanded: pd.DataFrame,
    season_col: str = "season",
    start_col: str = "contract_start_season",
    years_col: str = "years_int",
    value_col: str = "value_m",
    guaranteed_col: str = "guaranteed_m",
    apy_col: str = "apy_m",
) -> pd.DataFrame:
    """Given player-season rows that already carry contract terms, add
    ``estimated_cap_hit`` plus ``cap_hit_source`` and ``cap_hit_quality_flag``.

    Curve estimate where value + years are usable; flat-APY fallback otherwise.
    """
    df = expanded.copy()
    contract_year = (df[season_col] - df[start_col] + 1).astype("Int64")

    value = pd.to_numeric(df.get(value_col), errors="coerce")
    guaranteed = pd.to_numeric(df.get(guaranteed_col), errors="coerce").fillna(0.0)
    years = pd.to_numeric(df.get(years_col), errors="coerce")
    apy = pd.to_numeric(df.get(apy_col), errors="coerce")

    can_curve = value.gt(0) & years.gt(0) & contract_year.notna()
    est = np.full(len(df), np.nan)
    for i in np.where(can_curve.to_numpy())[0]:
        est[i] = season_cap_hit_curve(
            value.iloc[i], guaranteed.iloc[i], int(years.iloc[i]), int(contract_year.iloc[i])
        )

    source = np.where(can_curve, SOURCE_CURVE, SOURCE_MISSING).astype(object)
    quality = np.where(can_curve, QUALITY_TERMS, QUALITY_MISSING).astype(object)

    # Fallback to flat APY where the curve isn't usable OR produced a
    # non-positive cap hit (e.g. a fully-guaranteed year past the proration
    # window), but APY is available.
    with np.errstate(invalid="ignore"):
        curve_ok = est > 0
    fallback = (~curve_ok) & apy.gt(0).to_numpy()
    est = np.where(fallback, apy.to_numpy(), est)
    source = np.where(fallback, SOURCE_FALLBACK, source)
    quality = np.where(fallback, QUALITY_FALLBACK, quality)

    df["estimated_cap_hit"] = est
    df["cap_hit_source"] = source
    df["cap_hit_quality_flag"] = quality
    return df


def reconstruct_cap_hits(
    contracts: pd.DataFrame,
    seasons: list[int] | range,
    rosters: pd.DataFrame | None = None,
    positions: set[str] | None = None,
) -> pd.DataFrame:
    """One row per (gsis_id, season) of reconstructed cap hits.

    For each season, the player's active contract (most recently signed whose
    window covers the season) determines the contract-year index and the curve.
    """
    c = normalize_contracts(contracts)
    if "gsis_id" not in c.columns:
        raise ValueError("contracts must carry gsis_id to key cap hits")
    c = c[c["gsis_id"].notna()].copy()
    if positions is not None and "position" in c.columns:
        c = c[c["position"].isin(positions)].copy()

    records = []
    for season in seasons:
        active = c[(c["contract_start_season"] <= season)
                   & (c["contract_end_season"] >= season)].copy()
        if active.empty:
            continue
        active["season"] = int(season)
        # Most recently signed contract wins; APY breaks ties (dominant deal).
        active = active.sort_values(
            ["gsis_id", "year_signed", "apy_m"], ascending=[True, False, False]
        ).drop_duplicates(["gsis_id", "season"], keep="first")
        records.append(active)

    if not records:
        return pd.DataFrame(
            columns=["gsis_id", "season", "estimated_cap_hit",
                     "cap_hit_source", "cap_hit_quality_flag"]
        )

    out = pd.concat(records, ignore_index=True)
    out = add_estimated_cap_hit(out)

    keep = ["gsis_id", "season", "player", "position", "team",
            "contract_start_season", "years_int", "value_m", "apy_m",
            "guaranteed_m", "inflated_apy", "estimated_cap_hit",
            "cap_hit_source", "cap_hit_quality_flag"]
    keep = [k for k in keep if k in out.columns]
    out = out[keep].rename(columns={"player": "player_name"})
    assert not out.duplicated(["gsis_id", "season"]).any(), (
        "cap-hit table must be one row per player-season"
    )
    return out.reset_index(drop=True)


def attach_cap_hits(
    player_season_frame: pd.DataFrame,
    cap_hits: pd.DataFrame,
    id_col: str = "player_id",
    season_col: str = "season",
) -> pd.DataFrame:
    """Left-join reconstructed cap hits onto a player-season frame.

    Joins on (id_col == gsis_id, season). Asserts the join does not add rows and
    reports the unmatched rate via a ``cap_hit_quality_flag`` of MISSING for
    rows with no contract.
    """
    before = len(player_season_frame)
    ch = cap_hits.rename(columns={"gsis_id": id_col})
    cols = [id_col, season_col, "estimated_cap_hit", "cap_hit_source",
            "cap_hit_quality_flag"]
    cols = [c for c in cols if c in ch.columns]
    merged = player_season_frame.merge(ch[cols], on=[id_col, season_col], how="left")
    assert len(merged) == before, (
        f"cap-hit attach changed row count: {before} -> {len(merged)}"
    )
    merged["cap_hit_quality_flag"] = merged["cap_hit_quality_flag"].fillna(QUALITY_MISSING)
    merged["cap_hit_source"] = merged["cap_hit_source"].fillna(SOURCE_MISSING)
    return merged
