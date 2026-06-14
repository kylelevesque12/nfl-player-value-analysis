"""Session 6 experiment: ensemble stacking + quantile intervals for the weekly
fantasy model.

This is an OFF-PRODUCTION experiment harness. It reuses the exact Session 1
leakage-safe modeling frame, feature list, folds, and target, and compares four
point-prediction arms plus a quantile-interval method against the production
conformal intervals. Production code is only changed if something here wins;
otherwise the result is documented as negative and nothing ships.

Arms:
  A pooled HGB           — the Session 1 production point model
  B position-specific HGB — one HGB per position, pooled fallback
  C linear (Ridge)        — same features/preprocessing, different error profile
  D stacked ensemble      — Ridge meta-model over OOF predictions of A, B, C

Stacking leakage control: the meta-model trains only on OUT-OF-FOLD base
predictions. For each validation year, base learners produce predictions on the
last ``n_oof_seasons`` training seasons using models fit on strictly earlier
seasons; the meta-model is fit on those, then the base learners are refit on the
full training window and the meta-model maps their validation predictions to the
ensemble. No base model ever scores a row it trained on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.models import make_model_pipeline
from src.weekly_fantasy_projection import (
    WEEKLY_FANTASY_HGB_PARAMS,
    SKILL_POSITIONS,
    _available,
    _conformal_halfwidth,
)

TARGET = "target_fantasy_points_ppr"
DEFAULT_FOLDS = [2023, 2024, 2025]
DEFAULT_N_OOF = 3

# HGB params used by the experiment. Defaults to the exact Session 1 production
# config; ``set_hgb_params`` lets the eval substitute a lighter config (same
# across ALL arms) when compute is constrained — the decision rules rest on the
# RELATIVE arm comparison, which a uniform lighter config preserves.
_HGB_PARAMS = dict(WEEKLY_FANTASY_HGB_PARAMS)


def set_hgb_params(**overrides):
    _HGB_PARAMS.update(overrides)


def _hgb(feature_cols):
    return make_model_pipeline(
        feature_cols, HistGradientBoostingRegressor(**_HGB_PARAMS)
    )


def _ridge(feature_cols):
    return make_model_pipeline(feature_cols, Ridge(alpha=10.0))


def _fit_pooled(train, feature_cols):
    m = _hgb(feature_cols)
    m.fit(train[feature_cols], train[TARGET])
    return m


def _fit_linear(train, feature_cols):
    m = _ridge(feature_cols)
    m.fit(train[feature_cols], train[TARGET])
    return m


def _fit_per_position(train, feature_cols):
    models = {}
    for pos in SKILL_POSITIONS:
        sub = train[train["position"].eq(pos)]
        if len(sub) >= 50:  # enough data for a position-specific model
            m = _hgb(feature_cols)
            m.fit(sub[feature_cols], sub[TARGET])
            models[pos] = m
    return models


def _predict_per_position(models, pooled, df, feature_cols):
    """Per-position predictions with a documented pooled fallback for any
    position lacking its own trained model."""
    pred = pooled.predict(df[feature_cols])  # fallback baseline
    for pos, m in models.items():
        mask = df["position"].eq(pos).to_numpy()
        if mask.any():
            pred[mask] = m.predict(df.loc[mask, feature_cols])
    return np.clip(pred, 0, None)


def _base_predictions(train, predict_df, feature_cols):
    """Fit all three base learners on ``train`` and score ``predict_df``."""
    pooled = _fit_pooled(train, feature_cols)
    linear = _fit_linear(train, feature_cols)
    per_pos = _fit_per_position(train, feature_cols)
    return pd.DataFrame({
        "pooled": np.clip(pooled.predict(predict_df[feature_cols]), 0, None),
        "per_position": _predict_per_position(per_pos, pooled, predict_df, feature_cols),
        "linear": np.clip(linear.predict(predict_df[feature_cols]), 0, None),
    }, index=predict_df.index)


def run_point_backtest(
    frame: pd.DataFrame,
    feature_cols: list[str] | None = None,
    folds: list[int] = DEFAULT_FOLDS,
    n_oof_seasons: int = DEFAULT_N_OOF,
) -> pd.DataFrame:
    """Rolling-origin backtest of all four arms. Returns long predictions with
    columns: season, position, actual, arm, prediction."""
    if feature_cols is None:
        feature_cols = _available(frame, _session1_feature_cols(frame))
    records = []

    for year in folds:
        train_all = frame[frame["season"] < year].dropna(subset=[TARGET]).copy()
        valid = frame[frame["season"] == year].dropna(subset=[TARGET]).copy()
        if train_all.empty or valid.empty:
            continue
        train_seasons = sorted(train_all["season"].unique())
        oof_seasons = train_seasons[-n_oof_seasons:]

        # --- OOF base predictions on the last n_oof training seasons ---
        oof_frames = []
        for s in oof_seasons:
            inner_train = train_all[train_all["season"] < s]
            oof_rows = train_all[train_all["season"] == s]
            if inner_train.empty or oof_rows.empty:
                continue
            bp = _base_predictions(inner_train, oof_rows, feature_cols)
            bp[TARGET] = oof_rows[TARGET].to_numpy()
            oof_frames.append(bp)
        if not oof_frames:
            continue
        meta_train = pd.concat(oof_frames, ignore_index=True)
        meta = Ridge(alpha=1.0, positive=True)  # constrained: non-negative weights
        meta.fit(meta_train[["pooled", "per_position", "linear"]], meta_train[TARGET])

        # --- Refit base learners on full train; score validation ---
        base_valid = _base_predictions(train_all, valid, feature_cols)
        ensemble = np.clip(
            meta.predict(base_valid[["pooled", "per_position", "linear"]]), 0, None
        )

        for arm, col in [("pooled_hgb", "pooled"),
                         ("per_position_hgb", "per_position"),
                         ("linear_ridge", "linear")]:
            records.append(pd.DataFrame({
                "season": valid["season"].to_numpy(), "position": valid["position"].to_numpy(),
                "actual": valid[TARGET].to_numpy(), "arm": arm,
                "prediction": base_valid[col].to_numpy(),
            }))
        records.append(pd.DataFrame({
            "season": valid["season"].to_numpy(), "position": valid["position"].to_numpy(),
            "actual": valid[TARGET].to_numpy(), "arm": "stacked_ensemble",
            "prediction": ensemble,
        }))

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Quantile intervals vs conformal
# ---------------------------------------------------------------------------
# Nominal coverage -> (lower q, upper q) for the quantile models.
QUANTILE_LEVELS = {"50": (0.25, 0.75), "80": (0.10, 0.90)}
CONFORMAL_COVERAGE = {"50": 0.50, "80": 0.80}


def _fit_quantile(train, feature_cols, q):
    m = make_model_pipeline(
        feature_cols,
        HistGradientBoostingRegressor(loss="quantile", quantile=q, **_HGB_PARAMS),
    )
    m.fit(train[feature_cols], train[TARGET])
    return m


def run_interval_backtest(
    frame: pd.DataFrame,
    feature_cols: list[str] | None = None,
    folds: list[int] = DEFAULT_FOLDS,
) -> pd.DataFrame:
    """For each fold, produce quantile-GB and conformal intervals on validation
    rows. Returns long rows with columns: season, position, actual, level,
    method, lo, hi."""
    if feature_cols is None:
        feature_cols = _available(frame, _session1_feature_cols(frame))
    records = []
    for year in folds:
        train_all = frame[frame["season"] < year].dropna(subset=[TARGET]).copy()
        valid = frame[frame["season"] == year].dropna(subset=[TARGET]).copy()
        if train_all.empty or valid.empty:
            continue
        # Calibration split for conformal: last 20% of train by time.
        train_all = train_all.sort_values(["season", "week"]).reset_index(drop=True)
        cal_size = max(int(round(0.2 * len(train_all))), 1)
        train_fit = train_all.iloc[: len(train_all) - cal_size]
        train_cal = train_all.iloc[len(train_all) - cal_size :]

        pooled = _fit_pooled(train_fit, feature_cols)
        point = np.clip(pooled.predict(valid[feature_cols]), 0, None)
        cal_resid = train_cal[TARGET].to_numpy() - pooled.predict(train_cal[feature_cols])

        # Quantile models (fit on the same train_fit).
        qmods = {q: _fit_quantile(train_fit, feature_cols, q)
                 for q in sorted({q for pair in QUANTILE_LEVELS.values() for q in pair})}
        qpred = {q: m.predict(valid[feature_cols]) for q, m in qmods.items()}

        for level, (lq, uq) in QUANTILE_LEVELS.items():
            hw = _conformal_halfwidth(cal_resid, CONFORMAL_COVERAGE[level])
            # conformal (symmetric)
            records.append(pd.DataFrame({
                "season": valid["season"].to_numpy(), "position": valid["position"].to_numpy(),
                "actual": valid[TARGET].to_numpy(), "level": level, "method": "conformal",
                "lo": np.clip(point - hw, 0, None), "hi": point + hw,
            }))
            # quantile (asymmetric, enforce lo<=hi)
            lo = np.clip(qpred[lq], 0, None)
            hi = np.clip(qpred[uq], 0, None)
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            records.append(pd.DataFrame({
                "season": valid["season"].to_numpy(), "position": valid["position"].to_numpy(),
                "actual": valid[TARGET].to_numpy(), "level": level, "method": "quantile",
                "lo": lo, "hi": hi,
            }))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def interval_quality(intervals: pd.DataFrame) -> pd.DataFrame:
    """Empirical coverage and mean width by (method, level)."""
    rows = []
    for (method, level), g in intervals.groupby(["method", "level"]):
        covered = ((g["actual"] >= g["lo"]) & (g["actual"] <= g["hi"])).mean()
        rows.append({"method": method, "level": level, "n": len(g),
                     "target_coverage": int(level) / 100,
                     "empirical_coverage": float(covered),
                     "mean_width": float((g["hi"] - g["lo"]).mean())})
    return pd.DataFrame(rows).sort_values(["level", "method"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(p)) ** 2)))


def _mae(a, p):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(p))))


def point_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, g in preds.groupby("arm"):
        row = {"arm": arm, "rmse": _rmse(g["actual"], g["prediction"]),
               "mae": _mae(g["actual"], g["prediction"]), "n": len(g)}
        for pos in ["QB", "RB", "WR", "TE"]:
            gp = g[g["position"].eq(pos)]
            row[f"rmse_{pos}"] = _rmse(gp["actual"], gp["prediction"]) if len(gp) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def _session1_feature_cols(frame):
    from src.weekly_fantasy_projection import WEEKLY_FANTASY_FEATURES
    return WEEKLY_FANTASY_FEATURES
