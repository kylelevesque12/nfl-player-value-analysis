"""Session 7 eval: per-position vs global conformal coverage, + write the live
projection table for the Streamlit board.

Part 1 — interval comparison. Train on seasons < 2025, calibrate on the last 20%
of training rows, then on the 2025 hold-out measure empirical coverage and width
by position for the GLOBAL halfwidth (production) vs PER-POSITION halfwidths
(Session 7). The point of interest is QB coverage, which Session 6 showed the
global interval badly misses.

Part 2 — write ``outputs/tables/weekly_fantasy_live_projection.csv`` so the
Fantasy Player Board can show a live "next week" table.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from src.load_data import find_project_root, load_csv
from src import weekly_fantasy_projection as wk
from src import live_weekly_projection as lp

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
TARGET = "target_fantasy_points_ppr"


def _coverage_table(test, point, halfwidths, label):
    rows = []
    for method in ["global", "per_position"]:
        for pos in ["QB", "RB", "WR", "TE"]:
            mask = test["position"].eq(pos).to_numpy()
            if method == "global":
                hw = halfwidths["global"][label]
            else:
                hw = halfwidths["by_position"][pos][label]
            lo = np.clip(point[mask] - hw, 0, None)
            hi = point[mask] + hw
            y = test.loc[mask, TARGET].to_numpy()
            rows.append({"method": method, "position": pos,
                         "empirical_coverage": round(float(((y >= lo) & (y <= hi)).mean()), 3),
                         "mean_width": round(float((hi - lo).mean()), 2), "n": int(mask.sum())})
    return pd.DataFrame(rows)


def main():
    root = find_project_root()
    ps = load_csv("data/raw/player_stats_2016_2025.csv", root, low_memory=False)
    sch = load_csv("data/raw/schedules_2016_2025.csv", root, low_memory=False)
    ros = load_csv("data/raw/rosters_2016_2025.csv", root, low_memory=False)

    modeling = wk.build_modeling_frame(ps, sch, ros, project_root=root)
    fc = wk._available(modeling, wk.WEEKLY_FANTASY_FEATURES)

    train_all = modeling[modeling["season"] < 2025].dropna(subset=[TARGET]).sort_values(["season", "week"])
    test = modeling[modeling["season"] == 2025].dropna(subset=[TARGET]).copy()
    cal_size = max(int(round(0.2 * len(train_all))), 1)
    train_fit, cal = train_all.iloc[:-cal_size], train_all.iloc[-cal_size:]

    model = wk._make_main_model(fc)
    model.fit(train_fit[fc], train_fit[TARGET])
    hw = lp.compute_position_conformal_halfwidths(cal, fc, model)
    point = np.clip(model.predict(test[fc]), 0, None)

    print("=== 80% interval coverage by position: global vs per-position (2025 hold-out) ===")
    tbl80 = _coverage_table(test, point, hw, "80")
    print(tbl80.pivot(index="position", columns="method", values=["empirical_coverage", "mean_width"]).to_string())
    print("\n=== 50% interval coverage by position ===")
    tbl50 = _coverage_table(test, point, hw, "50")
    print(tbl50.pivot(index="position", columns="method", values=["empirical_coverage", "mean_width"]).to_string())
    print("\nhalfwidths 80:", {p: round(hw['by_position'][p]['80'], 2) for p in ['QB', 'RB', 'WR', 'TE']},
          "| global 80:", round(hw['global']['80'], 2))

    # Part 2: write the live projection table (as of the last week with a next
    # REG week in the static data).
    live = lp.build_live_projection_frame(ps, sch, ros, as_of=(2025, 17), project_root=root)
    proj, _ = lp.score_live_projection_frame(live, ps, sch, ros, project_root=root)
    out_path = root / "outputs" / "tables" / "weekly_fantasy_live_projection.csv"
    proj.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nWrote {len(proj)} live projections (week {int(proj['week'].iloc[0])}) -> {out_path.name}")


if __name__ == "__main__":
    main()
