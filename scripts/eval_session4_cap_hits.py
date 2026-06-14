"""Session 4 before/after: reconstructed cap hit vs flat inflated-APY.

Runs the real salary -> replacement-surplus pipeline twice on the same value
scores and contracts:

  OLD = salary_millions := inflated_apy (flat across the deal)
  NEW = salary_millions := reconstructed season cap hit (prorated bonus +
        backloaded base, from contract terms)

Then compares dollar surplus over replacement for the headline cuts: top surplus
players, QB leaders, rookie-contract players, and Brock Purdy 2023.
"""

from __future__ import annotations

import warnings
import pandas as pd

from src.load_data import find_project_root
from src import salary_efficiency as se
from src.salary_findings import prepare_salary_finding_base
from src.replacement_level import compute_replacement_level_surplus

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def _surplus_frame(value_scores, contracts, *, mode):
    merged = se.merge_value_and_salary(value_scores, contracts)
    if mode == "old":
        # Restore the legacy flat inflated-APY salary.
        merged["salary_millions"] = pd.to_numeric(
            merged["inflated_apy_salary"], errors="coerce"
        )
        merged = merged[merged["salary_millions"] > 0].copy()
    merged = se.add_efficiency_metrics(merged)
    base = prepare_salary_finding_base(merged, min_games=8)
    enriched, _, _ = compute_replacement_level_surplus(base)
    return enriched


def main():
    root = find_project_root()
    value_scores = pd.read_csv(root / "data/processed/player_value_scores_2016_2025.csv")
    contracts = se.load_contracts(root)

    new = _surplus_frame(value_scores, contracts, mode="new")
    old = _surplus_frame(value_scores, contracts, mode="old")

    # Coverage / fallback rates on the NEW table.
    print("=== cap_hit quality flags (NEW, player-seasons in finding base) ===")
    print(new["cap_hit_quality_flag"].value_counts(dropna=False).to_dict())
    print(f"rows  old={len(old)}  new={len(new)}")

    keys = ["player_display_name", "season", "position"]
    m = old[keys + ["salary_millions", "dollar_surplus_millions"]].rename(
        columns={"salary_millions": "old_salary", "dollar_surplus_millions": "old_surplus"}
    ).merge(
        new[keys + ["salary_millions", "dollar_surplus_millions", "value_score"]].rename(
            columns={"salary_millions": "new_cap_hit", "dollar_surplus_millions": "new_surplus"}
        ),
        on=keys, how="inner",
    )
    m["surplus_change"] = m["new_surplus"] - m["old_surplus"]

    def show(df, cols=None):
        cols = cols or ["player_display_name", "season", "position", "value_score",
                        "old_salary", "new_cap_hit", "old_surplus", "new_surplus", "surplus_change"]
        print(df[cols].round(2).to_string(index=False))

    print("\n=== Top 12 surplus players (by NEW surplus) ===")
    show(m.sort_values("new_surplus", ascending=False).head(12))

    print("\n=== QB surplus leaders (by NEW surplus) ===")
    show(m[m.position == "QB"].sort_values("new_surplus", ascending=False).head(10))

    print("\n=== Biggest surplus INCREASES (cheap young deals) ===")
    show(m.sort_values("surplus_change", ascending=False).head(10))

    print("\n=== Brock Purdy ===")
    show(m[m.player_display_name.str.contains("Purdy", na=False)].sort_values("season"))


if __name__ == "__main__":
    main()
