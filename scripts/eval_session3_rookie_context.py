"""Session 3 benchmark: do combine + team-context features improve the rookie
hurdle's stage-1 P(plays meaningfully)?

The production hurdle is a PyMC hierarchical logistic (stage 1) + Normal
(stage 2). PyMC is heavy and not always installed, and these features target
stage 1 specifically (whether a rookie plays), so for fast feature comparison we
use a frequentist logistic-regression surrogate on the SAME stage-1 target and
the SAME rolling-by-rookie-year validation. Features that help the surrogate are
then wired into the production model's FEATURE_COLUMNS.

Four arms, identical frame and folds:
  baseline      = current model features (draft_log, age_at_draft, height, weight)
  +combine      = baseline + combine athletic metrics + bmi
  +team_context = baseline + prior-season team/incumbent/depth features
  +both         = baseline + combine + team_context

Metrics: AUC, log loss, Brier, on pooled out-of-fold rookies. Plus the Jordan
Love case study (his P(plays) baseline vs +team_context).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from src.load_data import find_project_root
from src.rookie_bayes import (
    build_rookie_modeling_frame,
    FEATURE_COLUMNS,
    CONTEXT_FEATURES,
    DEFAULT_VALIDATION_YEARS,
)
from src.rookie_context_features import COMBINE_FEATURES, TEAM_CONTEXT_FEATURES

warnings.filterwarnings("ignore")
TARGET = "played_meaningfully"


def _fit_predict(train, test, cols):
    """Z-score on train stats, mean-impute NaN (=0 after centering), logistic."""
    mu = train[cols].mean()
    sd = train[cols].std().replace(0, 1.0)
    Xtr = ((train[cols] - mu) / sd).fillna(0.0).to_numpy()
    Xte = ((test[cols] - mu) / sd).fillna(0.0).to_numpy()
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, train[TARGET].astype(int).to_numpy())
    return clf.predict_proba(Xte)[:, 1], clf


def _rolling_oof(frame, cols, years):
    rows = []
    for y in years:
        tr = frame[frame["rookie_year"] < y]
        te = frame[frame["rookie_year"] == y]
        if tr[TARGET].nunique() < 2 or te.empty:
            continue
        p, _ = _fit_predict(tr, te, cols)
        rows.append(pd.DataFrame({"y": te[TARGET].astype(int).to_numpy(), "p": p,
                                  "position": te["position"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def _metrics(oof):
    return (roc_auc_score(oof["y"], oof["p"]),
            log_loss(oof["y"], oof["p"], labels=[0, 1]),
            brier_score_loss(oof["y"], oof["p"]))


def main():
    root = find_project_root()
    ros = pd.read_csv(root / "data/raw/rosters_2016_2025.csv", low_memory=False)
    ps = pd.read_csv(root / "data/raw/player_stats_2016_2025.csv", low_memory=False)
    # build_rookie_modeling_frame attaches the context features itself.
    frame = build_rookie_modeling_frame(ros, ps, project_root=root)

    base = [c for c in FEATURE_COLUMNS if c in frame.columns]
    combine = [c for c in COMBINE_FEATURES if c in frame.columns]
    team = [c for c in TEAM_CONTEXT_FEATURES if c in frame.columns]
    core = [c for c in CONTEXT_FEATURES if c in frame.columns]  # the kept set
    arms = {
        "baseline": base,
        "+combine": base + combine,
        "+team_context(7)": base + team,
        "+both": base + combine + team,
        "+incumbent_core(3)": base + core,  # the set wired into production
    }
    years = DEFAULT_VALIDATION_YEARS

    print(f"rookies={len(frame)}  play-rate={frame[TARGET].mean():.3f}  folds={years}")
    print(f"kept (CONTEXT_FEATURES) = {core}")
    print(f"\n=== Stage-1 P(plays) surrogate — pooled out-of-fold (all positions) ===")
    print(f"  {'arm':20s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s}")
    for name, cols in arms.items():
        oof = _rolling_oof(frame, cols, years)
        auc, ll, br = _metrics(oof)
        print(f"  {name:20s} {auc:7.4f} {ll:9.4f} {br:8.4f}")

    # QB-only view: this is the cell the incumbent features actually target,
    # and it mirrors the production model's per-position (hierarchical) slopes.
    qb = frame[frame["position"].eq("QB")]
    print(f"\n=== QB-only out-of-fold AUC (n={len(qb)}) ===")
    for name in ["baseline", "+team_context(7)", "+incumbent_core(3)"]:
        oof = _rolling_oof(qb, arms[name], years)
        print(f"  {name:20s} AUC={roc_auc_score(oof['y'], oof['p']):.4f}")

    # Coverage by position and draft-year bucket.
    print("\n=== Combine coverage by position (non-null forty) ===")
    print(frame.groupby("position")["forty"].apply(lambda s: round(s.notna().mean(), 2)).to_dict())
    print("=== Combine coverage by draft-year ===")
    print(frame.groupby("rookie_year")["forty"].apply(lambda s: round(s.notna().mean(), 2)).to_dict())

    # ---- Case studies: blocked first-round QBs ----
    print("\n=== Case study — P(plays meaningfully), QB-only model ===")
    for who, yr in [("Jordan Love", 2020), ("Patrick Mahomes", 2017)]:
        row = qb[qb["player_display_name"].str.contains(who, na=False)]
        if row.empty:
            continue
        tr = qb[qb["rookie_year"] < yr]
        b, _ = _fit_predict(tr, row, base)
        c, _ = _fit_predict(tr, row, base + core)
        print(f"  {who} ({yr}): baseline={b[0]:.3f} -> +incumbent_core={c[0]:.3f}"
              f"  (actual played={int(row[TARGET].iloc[0])})")


if __name__ == "__main__":
    main()
