"""Session 6 eval: ensemble stacking + quantile intervals vs Session 1 baseline.

Builds the exact Session 1 modeling frame, runs the four-arm point backtest and
the interval backtest, and prints the benchmark + interval-quality tables.
"""

from __future__ import annotations

import warnings
import pandas as pd

from src.load_data import find_project_root, load_csv
from src import weekly_fantasy_projection as wk
from src import weekly_ensemble_experiment as ex

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def main():
    root = find_project_root()
    frame = wk.build_modeling_frame(
        load_csv("data/raw/player_stats_2016_2025.csv", root, low_memory=False),
        load_csv("data/raw/schedules_2016_2025.csv", root, low_memory=False),
        load_csv("data/raw/rosters_2016_2025.csv", root, low_memory=False),
        project_root=root,
    )
    feature_cols = wk._available(frame, wk.WEEKLY_FANTASY_FEATURES)
    leaky = [c for c in feature_cols if c.startswith(("ngs_", "pfr_"))]
    print(f"rows={len(frame):,}  features={len(feature_cols)}  NGS/PFR leak check={leaky or 'NONE'}")

    print("\n=== POINT MODELS (rolling 2023-2025) ===")
    preds = ex.run_point_backtest(frame, feature_cols)
    pm = ex.point_metrics(preds)
    base = pm[pm["arm"] == "pooled_hgb"]["rmse"].iloc[0]
    pm["rmse_vs_pooled_%"] = (base - pm["rmse"]) / base * 100
    print(pm[["arm", "rmse", "mae", "rmse_vs_pooled_%", "rmse_QB", "rmse_RB", "rmse_WR", "rmse_TE"]].round(4).to_string(index=False))

    print("\n=== INTERVALS: quantile-GB vs conformal ===")
    intervals = ex.run_interval_backtest(frame, feature_cols)
    iq = ex.interval_quality(intervals)
    print(iq.round(4).to_string(index=False))

    print("\n=== INTERVAL coverage by position (80% level) ===")
    eighty = intervals[intervals["level"] == "80"]
    rows = []
    for (method, pos), g in eighty.groupby(["method", "position"]):
        cov = ((g["actual"] >= g["lo"]) & (g["actual"] <= g["hi"])).mean()
        rows.append({"method": method, "position": pos, "coverage": round(float(cov), 3),
                     "mean_width": round(float((g["hi"] - g["lo"]).mean()), 2)})
    print(pd.DataFrame(rows).sort_values(["position", "method"]).to_string(index=False))


if __name__ == "__main__":
    main()
