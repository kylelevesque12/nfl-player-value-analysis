"""Session 1 before/after eval: PBP depth-chart rank + weather features.

Builds the weekly modeling frame ONCE (so both arms see identical rows and
folds), then runs the rolling-origin backtest twice:

  baseline  = WEEKLY_FANTASY_FEATURES minus the new session-1 columns
  enhanced  = full WEEKLY_FANTASY_FEATURES (PBP rank + weather added)

Reports overall and by-position RMSE for the pooled HGB model, plus the
absolute and relative improvement. This isolates the feature contribution
cleanly: same frame, same splits, same model, only the feature list changes.
"""

from __future__ import annotations

import pandas as pd

from src.load_data import find_project_root, load_csv
from src import weekly_fantasy_projection as wk

NEW_FEATURES = [
    "pbp_depth_chart_rank_last1",
    "pbp_depth_chart_rank_last4_avg",
    "pbp_targets_last4_avg",
    "pbp_touches_last4_avg",
    "is_indoor",
    "game_temp",
    "game_wind",
]


def _overall_rmse(preds: pd.DataFrame) -> float:
    m = preds[preds["method"] == "hist_gradient_boosting"]
    return wk._rmse(m["target_fantasy_points_ppr"], m["prediction"])


def _by_pos_rmse(preds: pd.DataFrame) -> pd.Series:
    m = preds[preds["method"] == "hist_gradient_boosting"]
    return m.groupby("position").apply(
        lambda g: wk._rmse(g["target_fantasy_points_ppr"], g["prediction"])
    )


def main() -> None:
    root = find_project_root()
    player_stats = load_csv("data/raw/player_stats_2016_2025.csv", root, low_memory=False)
    schedules = load_csv("data/raw/schedules_2016_2025.csv", root, low_memory=False)
    rosters = load_csv("data/raw/rosters_2016_2025.csv", root, low_memory=False)

    frame = wk.build_modeling_frame(player_stats, schedules, rosters, project_root=root)

    full = wk._available(frame, wk.WEEKLY_FANTASY_FEATURES)
    baseline_cols = [c for c in full if c not in NEW_FEATURES]
    added = [c for c in full if c in NEW_FEATURES]

    print(f"rows={len(frame):,}  baseline_features={len(baseline_cols)}  "
          f"enhanced_features={len(full)}  added={added}")
    # Coverage of the new columns (how often the join actually landed).
    print("\nnew-feature non-null coverage:")
    for c in added:
        print(f"  {c:32s} {frame[c].notna().mean():.3f}")

    import gc

    val_years = [2023, 2024, 2025]
    base_preds, _ = wk.collect_rolling_predictions(frame, baseline_cols, val_years)
    gc.collect()
    enh_preds, _ = wk.collect_rolling_predictions(frame, full, val_years)
    gc.collect()

    b_all, e_all = _overall_rmse(base_preds), _overall_rmse(enh_preds)
    print("\n=== Overall pooled-HGB RMSE ===")
    print(f"  baseline : {b_all:.4f}")
    print(f"  enhanced : {e_all:.4f}")
    print(f"  abs delta: {b_all - e_all:+.4f}   rel: {(b_all - e_all)/b_all*100:+.2f}%")

    b_pos, e_pos = _by_pos_rmse(base_preds), _by_pos_rmse(enh_preds)
    print("\n=== By-position RMSE ===")
    print(f"  {'pos':4s} {'baseline':>9s} {'enhanced':>9s} {'rel %':>8s}")
    for pos in ["QB", "RB", "WR", "TE"]:
        if pos in b_pos.index and pos in e_pos.index:
            b, e = b_pos[pos], e_pos[pos]
            print(f"  {pos:4s} {b:9.4f} {e:9.4f} {(b-e)/b*100:+8.2f}")


if __name__ == "__main__":
    main()
