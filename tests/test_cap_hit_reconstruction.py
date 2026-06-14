"""Tests for Session 4 cap-hit reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.load_data import find_project_root
from src import cap_hit_reconstruction as ch
from src import salary_efficiency as se


def _contract(gsis, value, guaranteed, years, year_signed, apy=None, pos="QB"):
    apy = apy if apy is not None else value / years
    return {
        "player": gsis, "position": pos, "team": "AAA",
        "year_signed": year_signed, "years": years,
        "value": value, "apy": apy, "guaranteed": guaranteed,
        "inflated_value": value, "inflated_apy": apy,
        "inflated_guaranteed": guaranteed, "gsis_id": gsis,
    }


# ---------------------------------------------------------------------------
# Curve math
# ---------------------------------------------------------------------------
def test_curve_sums_to_total_value_and_is_backloaded():
    value, guaranteed, years = 100.0, 20.0, 5
    hits = [ch.season_cap_hit_curve(value, guaranteed, years, k) for k in range(1, years + 1)]
    # No money invented or lost: the per-year cap hits sum to total value.
    assert sum(hits) == pytest.approx(value, abs=1e-6)
    # Backloaded: early years cheaper than late years.
    assert hits[0] < hits[-1]
    # Year 1 is below the flat APY (value/years); the last year is above.
    apy = value / years
    assert hits[0] < apy < hits[-1]


def test_rookie_early_year_below_apy():
    # A backloaded rookie-style deal: most money is base, lightly guaranteed.
    hit_y1 = ch.season_cap_hit_curve(value_m=8.0, guaranteed_m=0.4, years_int=4, contract_year=1)
    assert hit_y1 < 8.0 / 4  # below flat APY


# ---------------------------------------------------------------------------
# reconstruct_cap_hits table
# ---------------------------------------------------------------------------
def test_reconstruct_one_row_per_player_season():
    contracts = pd.DataFrame([
        _contract("p1", 100.0, 20.0, 5, 2020),
        _contract("p2", 40.0, 8.0, 4, 2021),
        # An overlapping re-signing for p1 — most recent should win, still one row.
        _contract("p1", 200.0, 80.0, 4, 2022),
    ])
    out = ch.reconstruct_cap_hits(contracts, range(2020, 2026))
    assert not out.duplicated(["gsis_id", "season"]).any()
    assert (out["cap_hit_quality_flag"] == ch.QUALITY_TERMS).all()


def test_fallback_apy_is_flagged_when_value_missing():
    c = _contract("p3", np.nan, np.nan, 3, 2021, apy=5.0)
    c["inflated_value"] = np.nan
    contracts = pd.DataFrame([c])
    out = ch.reconstruct_cap_hits(contracts, [2021, 2022, 2023])
    assert (out["cap_hit_quality_flag"] == ch.QUALITY_FALLBACK).all()
    assert (out["estimated_cap_hit"] == 5.0).all()


def test_attach_does_not_add_rows_and_flags_missing():
    contracts = pd.DataFrame([_contract("p1", 100.0, 20.0, 5, 2020)])
    cap_hits = ch.reconstruct_cap_hits(contracts, range(2020, 2024))
    frame = pd.DataFrame({
        "player_id": ["p1", "p1", "ghost"],
        "season": [2021, 2022, 2021],
        "x": [1, 2, 3],
    })
    out = ch.attach_cap_hits(frame, cap_hits)
    assert len(out) == len(frame)            # no row increase
    # The unmatched 'ghost' row is flagged missing, not dropped or crashed.
    ghost = out[out["player_id"].eq("ghost")].iloc[0]
    assert ghost["cap_hit_quality_flag"] == ch.QUALITY_MISSING


def test_missing_contracts_handled_without_crashing():
    # Contracts with junk year_signed get filtered; reconstruct must not crash.
    contracts = pd.DataFrame([
        _contract("p1", 100.0, 20.0, 5, 0),       # junk year
        _contract("p2", 50.0, 10.0, 3, 2022),
    ])
    out = ch.reconstruct_cap_hits(contracts, range(2020, 2025))
    assert "p1" not in set(out["gsis_id"])  # junk contract excluded
    assert "p2" in set(out["gsis_id"])


# ---------------------------------------------------------------------------
# End-to-end pipeline still runs and uses the reconstructed cap hit
# ---------------------------------------------------------------------------
def test_salary_pipeline_runs_with_reconstructed_cap_hit():
    root = find_project_root()
    value_scores = pd.read_csv(root / "data/processed/player_value_scores_2016_2025.csv")
    contracts = se.load_contracts(root)
    merged = se.merge_value_and_salary(value_scores, contracts)
    # The production salary variable is now the reconstructed cap hit.
    assert "estimated_cap_hit" in merged.columns
    assert "cap_hit_quality_flag" in merged.columns
    assert "inflated_apy_salary" in merged.columns  # legacy retained for comparison
    assert (merged["salary_source"] == ch.SOURCE_CURVE).any()
    # Players without a matched contract carry NaN salary; matched rows are > 0.
    matched = merged["salary_millions"].notna()
    assert matched.any()
    assert merged.loc[matched, "salary_millions"].gt(0).all()
    # Real rookie deals should land below their flat inflated-APY in early years
    # at least on average (backloading).
    rookies = merged[merged["years_exp"].le(2)] if "years_exp" in merged else merged
    assert (rookies["salary_millions"] <= rookies["inflated_apy_salary"] + 1e-9).mean() > 0.5
