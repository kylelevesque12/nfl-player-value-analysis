"""Session 2 diagnostic: is the NGS/PFR gain from the VALUES or from COVERAGE?

Lean version: a single pooled HGB per arm (train on seasons < 2025, validate on
2025), no per-position models, so all arms run in well under a minute.

Arms:
  baseline
  + NGS values            (16 lag/roll cols)
  + NGS values shuffled   (same NaN pattern, scrambled values)
  + NGS coverage flags    (3 binary: tracked receiver/rusher/passer last week)
  + PFR values
  + PFR values shuffled
  + NGS + PFR values
  + NGS + PFR coverage flags
"""

from __future__ import annotations

import warnings
import numpy as np

from src.load_data import find_project_root, load_csv
from src import weekly_fantasy_projection as wk
from src.external_player_features import (
    NGS_BASE_COLS,
    PFR_BASE_COLS,
    NGS_COVERAGE_FLAGS,
    attach_external_weekly_features,
    attach_ngs_coverage_flags,
)

warnings.filterwarnings("ignore")
VAL_YEAR = 2025


def _suffix(bases):
    out = []
    for c in bases:
        out += [f"{c}_lag1", f"{c}_roll3"]
    return out


def _fit_rmse(frame, cols):
    train = frame[frame["season"] < VAL_YEAR].dropna(subset=["target_fantasy_points_ppr"])
    valid = frame[frame["season"] == VAL_YEAR].dropna(subset=["target_fantasy_points_ppr"])
    pipe = wk._make_main_model(cols)
    pipe.fit(train[cols], train["target_fantasy_points_ppr"])
    pred = pipe.predict(valid[cols]).clip(min=0)
    return wk._rmse(valid["target_fantasy_points_ppr"], pred), valid, pred


def _by_pos(valid, pred):
    import pandas as pd
    d = valid[["position"]].copy()
    d["e2"] = (valid["target_fantasy_points_ppr"].to_numpy() - pred) ** 2
    return {p: float(np.sqrt(g["e2"].mean())) for p, g in d.groupby("position")}


def _shuffle_keep_nan(frame, cols, seed=0):
    out = frame.copy()
    rng = np.random.RandomState(seed)
    for c in cols:
        mask = out[c].notna().values
        vals = out.loc[mask, c].values.copy()
        rng.shuffle(vals)
        out.loc[mask, c] = vals
    return out


def main():
    root = find_project_root()
    # build_modeling_frame already attaches the production NGS coverage flags.
    f = wk.build_modeling_frame(
        load_csv("data/raw/player_stats_2016_2025.csv", root, low_memory=False),
        load_csv("data/raw/schedules_2016_2025.csv", root, low_memory=False),
        load_csv("data/raw/rosters_2016_2025.csv", root, low_memory=False),
        project_root=root,
    )
    # Attach the leak-free coverage flags and the NGS/PFR *value* columns
    # (both only used by this diagnostic; neither is in the production model).
    f = attach_ngs_coverage_flags(f, project_root=root)
    f = attach_external_weekly_features(f, project_root=root)
    ngs = [c for c in _suffix(NGS_BASE_COLS) if c in f.columns]
    pfr = [c for c in _suffix(PFR_BASE_COLS) if c in f.columns]
    ngs_flags = list(NGS_COVERAGE_FLAGS)
    base = [
        c for c in wk._available(f, wk.WEEKLY_FANTASY_FEATURES)
        if c not in set(ngs + pfr + ngs_flags)
    ]
    # A PFR coverage flag, for completeness (shown to add little).
    f["pfr_rec_tracked_lag1"] = f["pfr_receiving_drop_pct_lag1"].notna().astype(float)
    pfr_flags = ["pfr_rec_tracked_lag1"]

    b, vb, pb = _fit_rmse(f, base)
    bpos = _by_pos(vb, pb)
    print(f"\n=== Pooled-HGB RMSE (train<{VAL_YEAR}, validate {VAL_YEAR}) ===")
    print(f"  {'arm':26s} {'RMSE':>8s} {'vs base':>9s}")
    print(f"  {'baseline':26s} {b:8.4f} {'+0.00%':>9s}")

    def show(name, frame, cols):
        r, v, p = _fit_rmse(frame, cols)
        print(f"  {name:26s} {r:8.4f} {(b-r)/b*100:+8.2f}%")
        return r

    show("+NGS values", f, base + ngs)
    show("+NGS shuffled", _shuffle_keep_nan(f, ngs), base + ngs)
    show("+NGS flags only", f, base + ngs_flags)
    show("+PFR values", f, base + pfr)
    show("+PFR shuffled", _shuffle_keep_nan(f, pfr), base + pfr)
    show("+NGS+PFR values", f, base + ngs + pfr)
    show("+NGS+PFR flags only", f, base + ngs_flags + pfr_flags)

    # by-position for the two finalists
    print("\n=== By-position RMSE (baseline vs flags-only) ===")
    _, vf, pf = _fit_rmse(f, base + ngs_flags + pfr_flags)
    fpos = _by_pos(vf, pf)
    for pos in ["QB", "RB", "WR", "TE"]:
        print(f"  {pos}: {bpos.get(pos, float('nan')):.3f} -> {fpos.get(pos, float('nan')):.3f}")


if __name__ == "__main__":
    main()
