"""Two-stage value model: opportunity x efficiency.

Motivation
----------
The value decomposition stage established a structural fact about this data:
the two components of player value behave very differently year over year.

- **Opportunity** (usage per game) is highly persistent (lag-1 correlation
  ~0.76 for skill positions). It is governed by role, depth chart, and scheme.
- **Efficiency** (value EPA per opportunity) is much noisier (~0.06-0.25 for
  skill positions once small samples are excluded), and is dominated by
  regression to the mean. Quarterbacks are the exception (~0.47).

Because ``value_epa_total = efficiency_per_opportunity * opportunity_per_game *
games_played``, forcing a single model to predict the blended total wastes
capacity learning a product whose two factors want different functional forms
and different amounts of shrinkage. A two-stage model predicts each factor with
the right tool and recombines them, which is both more accurate in principle and
more *legible*: it can say "role should hold, but expect efficiency regression"
and propagate asymmetric uncertainty into the final interval.

Scope of this module
--------------------
This file currently implements **Stage 1: the opportunity model** end to end
(target construction, leakage-safe history features, rolling-origin validation,
a strong persistence baseline, skill score, and a per-position report). The
efficiency stage and the final value reconstruction are scaffolded with clear
``NotImplementedError`` placeholders and a documented plan so the second stage
can be added without reshaping stage one.

Dependency note
---------------
Data preparation, target/feature construction, and the baseline are pandas/numpy
only and are unit-tested without the modeling stack. scikit-learn is imported
lazily inside the model-training functions so the rest of the module loads and
tests cleanly in minimal environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src import config
from src.value_decomposition import (
    MIN_EFFICIENCY_OPPORTUNITY,
    DEFAULT_MIN_EFFICIENCY_OPPORTUNITY,
    add_opportunity_and_efficiency,
    add_talent_rate_features,
)

# Stage-1 target: next season's opportunity per game.
OPPORTUNITY_TARGET = "next_opportunity_per_game"
OPPORTUNITY_CURRENT = "opportunity_per_game"

# Leakage-safe opportunity history features (all are lagged/rolling on prior
# seasons only, built by ``add_opportunity_history_features``).
OPPORTUNITY_HISTORY_FEATURES = [
    "opportunity_per_game_prev",
    "opportunity_per_game_last2_avg",
    "opportunity_per_game_last3_avg",
    "opportunity_per_game_trend_2yr",
    "games_played_prev",
    "games_played_last2_avg",
    "prior_qualifying_seasons",
]
OPPORTUNITY_CONTEXT_FEATURES = ["position", "age", "years_exp", "draft_number"]
OPPORTUNITY_FEATURES = OPPORTUNITY_CONTEXT_FEATURES + OPPORTUNITY_HISTORY_FEATURES

DEFAULT_VALIDATION_YEARS = list(config.ROLLING_VALIDATION_YEARS)


# ---------------------------------------------------------------------------
# Metrics (duplicated lightweight, no sklearn dependency)
# ---------------------------------------------------------------------------
def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, "float64") - np.asarray(y_pred, "float64"))))


def _r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def add_opportunity_history_features(player_season: pd.DataFrame) -> pd.DataFrame:
    """Add lagged/rolling opportunity history features without future leakage.

    Every feature uses ``shift(1)`` (or rolling windows on shifted values) so a
    given season row only ever sees strictly earlier seasons for that player.
    """
    out = player_season.sort_values(["player_id", "season"]).copy()
    grouped = out.groupby("player_id", group_keys=False)

    out["prior_qualifying_seasons"] = grouped.cumcount()

    for col in ["opportunity_per_game", "games_played"]:
        if col not in out.columns:
            continue
        out[f"{col}_prev"] = grouped[col].shift(1)
        out[f"{col}_last2_avg"] = grouped[col].apply(
            lambda s: s.shift(1).rolling(2, min_periods=1).mean()
        )
        out[f"{col}_last3_avg"] = grouped[col].apply(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

    if {"opportunity_per_game_prev", "opportunity_per_game_last2_avg"}.issubset(out.columns):
        out["opportunity_per_game_trend_2yr"] = (
            out["opportunity_per_game_prev"] - out["opportunity_per_game_last2_avg"]
        )

    return out


def add_next_season_opportunity_target(player_season: pd.DataFrame) -> pd.DataFrame:
    """Attach next-season opportunity-per-game target (consecutive seasons only)."""
    out = player_season.sort_values(["player_id", "season"]).copy()
    grouped = out.groupby("player_id")
    out["next_season"] = grouped["season"].shift(-1)
    out[OPPORTUNITY_TARGET] = grouped[OPPORTUNITY_CURRENT].shift(-1)

    has_next = out["next_season"].eq(out["season"] + 1)
    out.loc[~has_next, OPPORTUNITY_TARGET] = np.nan
    out["has_next_season"] = has_next.astype(int)
    return out


def build_opportunity_modeling_frame(
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load player seasons, decompose, add opportunity history + target.

    Returns the modeling frame (rows with a known next-season target) and the
    list of available opportunity features.
    """
    if project_root is None:
        project_root = _find_project_root()
    project_root = Path(project_root)

    processed_dir = project_root / "data" / "processed"
    value_scores_path = processed_dir / "player_value_scores_2016_2025.csv"
    skill_seasons_path = processed_dir / "skill_player_seasons_2016_2025.csv"
    if value_scores_path.exists():
        player_season = pd.read_csv(value_scores_path)
    elif skill_seasons_path.exists():
        player_season = pd.read_csv(skill_seasons_path)
    else:
        raise FileNotFoundError("No processed player-season file in " + str(processed_dir))

    player_season = add_opportunity_and_efficiency(player_season)
    player_season = add_talent_rate_features(player_season)
    player_season = add_opportunity_history_features(player_season)
    player_season = add_next_season_opportunity_target(player_season)

    feature_cols = [c for c in OPPORTUNITY_FEATURES if c in player_season.columns]
    modeling_df = player_season.dropna(subset=[OPPORTUNITY_TARGET]).copy()
    return modeling_df, feature_cols


# ---------------------------------------------------------------------------
# Baseline + models for the opportunity stage
# ---------------------------------------------------------------------------
def predict_opportunity_persistence(
    train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> np.ndarray:
    """Baseline: next-season opportunity = current-season opportunity per game."""
    current = pd.to_numeric(valid_df[OPPORTUNITY_CURRENT], errors="coerce")
    fill = float(pd.to_numeric(train_df[OPPORTUNITY_CURRENT], errors="coerce").mean())
    return current.fillna(fill).to_numpy()


def _make_opportunity_regressor(name: str):
    """Lazy sklearn import so the module loads without the modeling stack."""
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            max_features=0.5,
            min_samples_leaf=15,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=3,
            learning_rate=0.05,
            max_iter=400,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=config.RANDOM_STATE,
        )
    raise ValueError(f"Unknown opportunity model: {name}")


def _make_opportunity_pipeline(feature_cols: list[str], model_name: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical = [c for c in feature_cols if c == "position"]
    numeric = [c for c in feature_cols if c not in categorical]
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", pre), ("model", _make_opportunity_regressor(model_name))])


OPPORTUNITY_MODELS = ["random_forest", "gradient_boosting"]


def collect_opportunity_rolling_predictions(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    validation_years: list[int] | None = None,
) -> pd.DataFrame:
    """Rolling-origin out-of-sample predictions for baseline + models."""
    if validation_years is None:
        validation_years = DEFAULT_VALIDATION_YEARS

    frames: list[pd.DataFrame] = []
    for year in validation_years:
        train_df = modeling_df[modeling_df["season"].lt(year)].copy()
        valid_df = modeling_df[modeling_df["season"].eq(year)].copy()
        if train_df.empty or valid_df.empty:
            continue

        base = valid_df[["player_id", "position", "season", OPPORTUNITY_TARGET]].copy()

        persist = base.copy()
        persist["method"] = "persistence"
        persist["method_type"] = "baseline"
        persist["prediction"] = predict_opportunity_persistence(train_df, valid_df)
        frames.append(persist)

        for name in OPPORTUNITY_MODELS:
            pipe = _make_opportunity_pipeline(feature_cols, name)
            pipe.fit(train_df[feature_cols], train_df[OPPORTUNITY_TARGET])
            out = base.copy()
            out["method"] = name
            out["method_type"] = "model"
            out["prediction"] = pipe.predict(valid_df[feature_cols])
            frames.append(out)

    if not frames:
        return pd.DataFrame()
    preds = pd.concat(frames, ignore_index=True)
    preds["residual"] = preds[OPPORTUNITY_TARGET] - preds["prediction"]
    preds["abs_residual"] = preds["residual"].abs()
    return preds


def summarize_opportunity_methods(predictions: pd.DataFrame) -> pd.DataFrame:
    """Overall metrics per method with skill score vs the persistence baseline."""
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (method, mtype), grp in predictions.groupby(["method", "method_type"]):
        rows.append(
            {
                "method": method,
                "method_type": mtype,
                "n": int(len(grp)),
                "rmse": _rmse(grp[OPPORTUNITY_TARGET], grp["prediction"]),
                "mae": _mae(grp[OPPORTUNITY_TARGET], grp["prediction"]),
                "r2": _r2(grp[OPPORTUNITY_TARGET], grp["prediction"]),
            }
        )
    summary = pd.DataFrame(rows)
    ref = summary.loc[summary["method"] == "persistence", "rmse"]
    ref_rmse = float(ref.iloc[0]) if len(ref) else float("nan")
    summary["skill_vs_persistence"] = 1.0 - summary["rmse"] / ref_rmse
    return summary.sort_values("rmse").reset_index(drop=True)


def summarize_opportunity_by_position(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (position, method, mtype), grp in predictions.groupby(
        ["position", "method", "method_type"]
    ):
        rows.append(
            {
                "position": position,
                "method": method,
                "method_type": mtype,
                "n": int(len(grp)),
                "rmse": _rmse(grp[OPPORTUNITY_TARGET], grp["prediction"]),
                "r2": _r2(grp[OPPORTUNITY_TARGET], grp["prediction"]),
            }
        )
    by_pos = pd.DataFrame(rows)
    ref = (
        by_pos[by_pos["method"] == "persistence"][["position", "rmse"]]
        .rename(columns={"rmse": "ref_rmse"})
    )
    by_pos = by_pos.merge(ref, on="position", how="left")
    by_pos["skill_vs_persistence"] = 1.0 - by_pos["rmse"] / by_pos["ref_rmse"]
    return by_pos.drop(columns=["ref_rmse"]).sort_values(["position", "rmse"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 2: efficiency model
# ---------------------------------------------------------------------------
# Stage-2 target: next season's efficiency per opportunity (qualified rows only).
EFFICIENCY_TARGET = "next_efficiency_per_opportunity"
EFFICIENCY_CURRENT = "efficiency_per_opportunity"

# Efficiency rate features (how production is earned) + lagged efficiency history.
EFFICIENCY_RATE_FEATURES = [
    "catch_rate",
    "yards_per_target",
    "yards_per_reception",
    "adot",
    "yac_per_reception",
    "racr",
    "yards_per_carry",
    "completion_pct",
    "yards_per_attempt",
    "passing_adot",
    "pacr",
]
EFFICIENCY_HISTORY_FEATURES = [
    "efficiency_per_opportunity_prev",
    "efficiency_per_opportunity_last2_avg",
    "opportunity_per_game_prev",  # volume context informs efficiency reliability
    "prior_qualifying_seasons",
]
EFFICIENCY_CONTEXT_FEATURES = ["position", "age", "years_exp"]
EFFICIENCY_FEATURES = (
    EFFICIENCY_CONTEXT_FEATURES + EFFICIENCY_RATE_FEATURES + EFFICIENCY_HISTORY_FEATURES
)


def add_efficiency_history_features(player_season: pd.DataFrame) -> pd.DataFrame:
    """Add lagged/rolling efficiency history features (no future leakage)."""
    out = player_season.sort_values(["player_id", "season"]).copy()
    grouped = out.groupby("player_id", group_keys=False)
    if "prior_qualifying_seasons" not in out.columns:
        out["prior_qualifying_seasons"] = grouped.cumcount()
    if EFFICIENCY_CURRENT in out.columns:
        out["efficiency_per_opportunity_prev"] = grouped[EFFICIENCY_CURRENT].shift(1)
        out["efficiency_per_opportunity_last2_avg"] = grouped[EFFICIENCY_CURRENT].apply(
            lambda s: s.shift(1).rolling(2, min_periods=1).mean()
        )
    return out


def add_next_season_efficiency_target(player_season: pd.DataFrame) -> pd.DataFrame:
    """Attach next-season efficiency target on consecutive, qualified seasons.

    The target is only defined when BOTH the current and next season clear the
    efficiency-opportunity floor, because efficiency on tiny samples is noise on
    either end of the pair.
    """
    out = player_season.sort_values(["player_id", "season"]).copy()
    grouped = out.groupby("player_id")
    if "next_season" not in out.columns:
        out["next_season"] = grouped["season"].shift(-1)
    out[EFFICIENCY_TARGET] = grouped[EFFICIENCY_CURRENT].shift(-1)
    next_qualified = grouped["efficiency_qualified"].shift(-1)
    next_qualified = next_qualified.astype("boolean").fillna(False).astype(bool)
    consecutive = out["next_season"].eq(out["season"] + 1)
    keep = consecutive & out["efficiency_qualified"].fillna(False) & next_qualified
    out.loc[~keep, EFFICIENCY_TARGET] = np.nan
    out["efficiency_pair_qualified"] = keep.astype(int)
    return out


def build_efficiency_modeling_frame(
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load player seasons, decompose, add efficiency history + qualified target."""
    if project_root is None:
        project_root = _find_project_root()
    project_root = Path(project_root)

    processed_dir = project_root / "data" / "processed"
    value_scores_path = processed_dir / "player_value_scores_2016_2025.csv"
    skill_seasons_path = processed_dir / "skill_player_seasons_2016_2025.csv"
    if value_scores_path.exists():
        player_season = pd.read_csv(value_scores_path)
    elif skill_seasons_path.exists():
        player_season = pd.read_csv(skill_seasons_path)
    else:
        raise FileNotFoundError("No processed player-season file in " + str(processed_dir))

    player_season = add_opportunity_and_efficiency(player_season)
    player_season = add_talent_rate_features(player_season)
    player_season = add_efficiency_history_features(player_season)
    player_season = add_next_season_efficiency_target(player_season)

    feature_cols = [c for c in EFFICIENCY_FEATURES if c in player_season.columns]
    modeling_df = player_season.dropna(subset=[EFFICIENCY_TARGET]).copy()
    return modeling_df, feature_cols


def predict_efficiency_shrink_to_mean(
    train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> np.ndarray:
    """Baseline: predict the training positional mean efficiency.

    Because skill-position efficiency barely autocorrelates, predicting the
    position's mean is a genuinely strong baseline — this is the "regression to
    the mean" null the model must beat.
    """
    pos_means = train_df.groupby("position")[EFFICIENCY_TARGET].mean()
    overall = float(train_df[EFFICIENCY_TARGET].mean())
    return valid_df["position"].map(pos_means).fillna(overall).to_numpy()


def predict_efficiency_shrunken_persistence(
    train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> np.ndarray:
    """Baseline: per-position least-squares shrink of current efficiency.

    Fits next = a + b * current within each position on the training fold, which
    is the optimal linear shrinkage of persistence toward the mean. For QBs
    (high autocorrelation) b will be large; for skill positions it will be small.
    """
    preds = np.empty(len(valid_df), dtype="float64")
    overall_mean = float(train_df[EFFICIENCY_TARGET].mean())
    valid_pos = valid_df["position"].to_numpy()
    valid_cur = pd.to_numeric(valid_df[EFFICIENCY_CURRENT], errors="coerce").to_numpy()
    for pos in np.unique(valid_pos):
        tr = train_df[train_df["position"].eq(pos)]
        mask = valid_pos == pos
        x = pd.to_numeric(tr[EFFICIENCY_CURRENT], errors="coerce")
        y = pd.to_numeric(tr[EFFICIENCY_TARGET], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() >= 10 and float(x[ok].std()) > 1e-9:
            b = float(np.cov(x[ok], y[ok])[0, 1] / np.var(x[ok]))
            a = float(y[ok].mean() - b * x[ok].mean())
            pos_pred = a + b * np.nan_to_num(valid_cur[mask], nan=float(x[ok].mean()))
        else:
            pos_pred = np.full(int(mask.sum()), float(y[ok].mean()) if ok.any() else overall_mean)
        preds[mask] = pos_pred
    return preds


EFFICIENCY_BASELINES: dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = {
    "shrink_to_mean": predict_efficiency_shrink_to_mean,
    "shrunken_persistence": predict_efficiency_shrunken_persistence,
}


def collect_efficiency_rolling_predictions(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    validation_years: list[int] | None = None,
) -> pd.DataFrame:
    """Rolling-origin out-of-sample predictions for efficiency baselines + models."""
    if validation_years is None:
        validation_years = DEFAULT_VALIDATION_YEARS

    frames: list[pd.DataFrame] = []
    for year in validation_years:
        train_df = modeling_df[modeling_df["season"].lt(year)].copy()
        valid_df = modeling_df[modeling_df["season"].eq(year)].copy()
        if train_df.empty or valid_df.empty:
            continue
        base = valid_df[["player_id", "position", "season", EFFICIENCY_TARGET]].copy()

        for name, fn in EFFICIENCY_BASELINES.items():
            out = base.copy()
            out["method"] = name
            out["method_type"] = "baseline"
            out["prediction"] = fn(train_df, valid_df)
            frames.append(out)

        for name in OPPORTUNITY_MODELS:  # same RF/HGB regressors, efficiency target
            pipe = _make_opportunity_pipeline(feature_cols, name)
            pipe.fit(train_df[feature_cols], train_df[EFFICIENCY_TARGET])
            out = base.copy()
            out["method"] = name
            out["method_type"] = "model"
            out["prediction"] = pipe.predict(valid_df[feature_cols])
            frames.append(out)

    if not frames:
        return pd.DataFrame()
    preds = pd.concat(frames, ignore_index=True)
    preds["residual"] = preds[EFFICIENCY_TARGET] - preds["prediction"]
    preds["abs_residual"] = preds["residual"].abs()
    return preds


def summarize_efficiency_methods(predictions: pd.DataFrame) -> pd.DataFrame:
    """Overall efficiency metrics with skill score vs the shrink-to-mean baseline."""
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (method, mtype), grp in predictions.groupby(["method", "method_type"]):
        rows.append(
            {
                "method": method,
                "method_type": mtype,
                "n": int(len(grp)),
                "rmse": _rmse(grp[EFFICIENCY_TARGET], grp["prediction"]),
                "mae": _mae(grp[EFFICIENCY_TARGET], grp["prediction"]),
                "r2": _r2(grp[EFFICIENCY_TARGET], grp["prediction"]),
            }
        )
    summary = pd.DataFrame(rows)
    ref = summary.loc[summary["method"] == "shrink_to_mean", "rmse"]
    ref_rmse = float(ref.iloc[0]) if len(ref) else float("nan")
    summary["skill_vs_shrink_to_mean"] = 1.0 - summary["rmse"] / ref_rmse
    return summary.sort_values("rmse").reset_index(drop=True)


def summarize_efficiency_by_position(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (position, method, mtype), grp in predictions.groupby(
        ["position", "method", "method_type"]
    ):
        rows.append(
            {
                "position": position,
                "method": method,
                "method_type": mtype,
                "n": int(len(grp)),
                "rmse": _rmse(grp[EFFICIENCY_TARGET], grp["prediction"]),
                "r2": _r2(grp[EFFICIENCY_TARGET], grp["prediction"]),
            }
        )
    by_pos = pd.DataFrame(rows)
    ref = (
        by_pos[by_pos["method"] == "shrink_to_mean"][["position", "rmse"]]
        .rename(columns={"rmse": "ref_rmse"})
    )
    by_pos = by_pos.merge(ref, on="position", how="left")
    by_pos["skill_vs_shrink_to_mean"] = 1.0 - by_pos["rmse"] / by_pos["ref_rmse"]
    return by_pos.drop(columns=["ref_rmse"]).sort_values(["position", "rmse"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reconstruction: combine the two stages into a value projection
# ---------------------------------------------------------------------------
def standardize_to_value_score(
    predicted_value_per_game: pd.Series,
    positions: pd.Series,
    train_group_stats: pd.DataFrame,
) -> pd.Series:
    """Standardize a predicted per-game value using frozen training group stats.

    The existing ``value_score`` is a within season-position z-score, but a
    future season's cross-section is unknown at prediction time. We therefore
    freeze the per-position mean/std of ``value_epa_per_game`` from the training
    seasons and apply them, which keeps the projected score on a comparable
    scale without peeking at the season being predicted.
    """
    means = positions.map(train_group_stats["mean"])
    stds = positions.map(train_group_stats["std"]).replace(0, np.nan)
    return (predicted_value_per_game.to_numpy() - means.to_numpy()) / stds.to_numpy()


def reconstruct_value_predictions(
    opportunity_pred: pd.Series,
    efficiency_pred: pd.Series,
) -> pd.Series:
    """Combine the two stages: per-game value = efficiency x opportunity-per-game.

    Uses the verified identity ``value_epa_per_game = efficiency_per_opportunity
    * opportunity_per_game``. (Total-season value would additionally multiply by
    games played; per-game is the cleaner, availability-neutral target and is
    what the project standardizes into ``value_score_per_game``.)
    """
    return pd.Series(
        np.asarray(efficiency_pred, dtype="float64")
        * np.asarray(opportunity_pred, dtype="float64")
    )


# ---------------------------------------------------------------------------
# Asymmetric uncertainty: propagate per-stage error into a value interval
# ---------------------------------------------------------------------------
def propagate_product_interval(
    efficiency_pred: np.ndarray,
    opportunity_pred: np.ndarray,
    sigma_efficiency: np.ndarray,
    sigma_opportunity: np.ndarray,
    z: float = config.PREDICTION_INTERVAL_MULTIPLIER,
) -> dict[str, np.ndarray]:
    """Propagate two independent stage errors through value = efficiency × opportunity.

    For independent errors on the two factors, the variance of the product is
    exactly::

        Var(E·O) = O² σ_E² + E² σ_O² + σ_E² σ_O²

    The first term is the uncertainty contributed by the **efficiency** axis, the
    second by the **opportunity** axis, and the third is the (usually small)
    interaction. This is what makes the band *legible*: we can report what share
    of a player's value uncertainty comes from "how good per play" vs "how much
    they play". Because skill-position efficiency is nearly unpredictable while
    opportunity is highly predictable, the efficiency term typically dominates —
    the interval is wide along the axis the model genuinely cannot pin down.

    Returns a dict of arrays: predicted value, total sigma, interval bounds,
    interval width, and the variance contributed by each axis plus the
    efficiency variance share.
    """
    e = np.asarray(efficiency_pred, dtype="float64")
    o = np.asarray(opportunity_pred, dtype="float64")
    se = np.asarray(sigma_efficiency, dtype="float64")
    so = np.asarray(sigma_opportunity, dtype="float64")

    value = e * o
    var_from_efficiency = np.square(o) * np.square(se)
    var_from_opportunity = np.square(e) * np.square(so)
    var_interaction = np.square(se) * np.square(so)
    total_var = var_from_efficiency + var_from_opportunity + var_interaction
    sigma = np.sqrt(total_var)

    with np.errstate(invalid="ignore", divide="ignore"):
        efficiency_share = np.where(total_var > 0, var_from_efficiency / total_var, np.nan)

    return {
        "value_pred": value,
        "sigma": sigma,
        "interval_low": value - z * sigma,
        "interval_high": value + z * sigma,
        "interval_width": 2.0 * z * sigma,
        "var_from_efficiency": var_from_efficiency,
        "var_from_opportunity": var_from_opportunity,
        "var_interaction": var_interaction,
        "efficiency_variance_share": efficiency_share,
    }


def _per_position_residual_sigma(
    residual_df: pd.DataFrame, residual_col: str
) -> tuple[dict[str, float], float]:
    """Per-position residual std (and an overall fallback) from a calibration set."""
    overall = float(residual_df[residual_col].std(ddof=1)) if len(residual_df) > 1 else 0.0
    per_pos = residual_df.groupby("position")[residual_col].std(ddof=1).to_dict()
    per_pos = {k: (v if np.isfinite(v) else overall) for k, v in per_pos.items()}
    return per_pos, overall


def combined_rolling_validation(
    project_root: Path,
    validation_years: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the recombined two-stage value against a single-model baseline.

    For each validation season the two stages are trained on earlier seasons,
    their predictions are multiplied to reconstruct per-game value, and the
    result is standardized with frozen training group stats into a
    ``value_score_per_game``-comparable number. This is compared, on the same
    rows, against:

    - ``persistence``: next per-game value = current per-game value.
    - ``single_model``: one model predicting ``value_score_per_game`` directly
      from the same opportunity+efficiency feature union (the architecture the
      two-stage design is meant to beat).

    Only efficiency-qualified consecutive pairs are scored, because the
    efficiency stage is only defined there. Returns (per-row predictions,
    summary).
    """
    if validation_years is None:
        validation_years = DEFAULT_VALIDATION_YEARS

    # Build a single frame carrying everything both stages need.
    value_scores_path = project_root / "data" / "processed" / "player_value_scores_2016_2025.csv"
    ps = pd.read_csv(value_scores_path)
    ps = add_opportunity_and_efficiency(ps)
    ps = add_talent_rate_features(ps)
    ps = add_opportunity_history_features(ps)
    ps = add_efficiency_history_features(ps)
    ps = add_next_season_opportunity_target(ps)
    ps = add_next_season_efficiency_target(ps)

    # Per-game value target and its lag for the persistence baseline.
    ps = ps.sort_values(["player_id", "season"])
    grp = ps.groupby("player_id")
    if "value_epa_per_game" not in ps.columns:
        ps["value_epa_per_game"] = ps["efficiency_per_opportunity"] * ps["opportunity_per_game"]
    ps["next_value_per_game"] = grp["value_epa_per_game"].shift(-1)
    ps.loc[ps["next_season"].ne(ps["season"] + 1), "next_value_per_game"] = np.nan

    opp_feats = [c for c in OPPORTUNITY_FEATURES if c in ps.columns]
    eff_feats = [c for c in EFFICIENCY_FEATURES if c in ps.columns]
    single_feats = sorted(set(opp_feats) | set(eff_feats))

    # Scoring rows: qualified efficiency pairs with a known per-game value target.
    scored_mask = ps[EFFICIENCY_TARGET].notna() & ps["next_value_per_game"].notna()

    frames: list[pd.DataFrame] = []
    for year in validation_years:
        train = ps[ps["season"].lt(year)]
        valid = ps[ps["season"].eq(year) & scored_mask]
        if valid.empty:
            continue

        # Frozen training group stats for standardization.
        group_stats = (
            train.dropna(subset=["value_epa_per_game"])
            .groupby("position")["value_epa_per_game"]
            .agg(["mean", "std"])
        )

        base = valid[["player_id", "position", "season"]].copy()
        base["actual_value_per_game"] = valid["next_value_per_game"].to_numpy()

        # --- two-stage reconstruction ---
        opp_train = train.dropna(subset=[OPPORTUNITY_TARGET])
        eff_train = train.dropna(subset=[EFFICIENCY_TARGET])
        opp_pipe = _make_opportunity_pipeline(opp_feats, "gradient_boosting")
        opp_pipe.fit(opp_train[opp_feats], opp_train[OPPORTUNITY_TARGET])
        eff_pipe = _make_opportunity_pipeline(eff_feats, "gradient_boosting")
        eff_pipe.fit(eff_train[eff_feats], eff_train[EFFICIENCY_TARGET])

        opp_hat = opp_pipe.predict(valid[opp_feats])
        eff_hat = eff_pipe.predict(valid[eff_feats])
        two_stage_pg = reconstruct_value_predictions(pd.Series(opp_hat), pd.Series(eff_hat))

        rec = base.copy()
        rec["method"] = "two_stage"
        rec["prediction_value_per_game"] = two_stage_pg.to_numpy()
        frames.append(rec)

        # --- single-model baseline (predict per-game value directly) ---
        single_train = train.dropna(subset=["next_value_per_game"])
        single_pipe = _make_opportunity_pipeline(single_feats, "gradient_boosting")
        single_pipe.fit(single_train[single_feats], single_train["next_value_per_game"])
        sm = base.copy()
        sm["method"] = "single_model"
        sm["prediction_value_per_game"] = single_pipe.predict(valid[single_feats])
        frames.append(sm)

        # --- persistence baseline ---
        pb = base.copy()
        pb["method"] = "persistence"
        pb["prediction_value_per_game"] = valid["value_epa_per_game"].to_numpy()
        frames.append(pb)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    preds = pd.concat(frames, ignore_index=True)
    preds["residual"] = preds["actual_value_per_game"] - preds["prediction_value_per_game"]
    preds["abs_residual"] = preds["residual"].abs()

    rows: list[dict[str, Any]] = []
    for method, g in preds.groupby("method"):
        rows.append(
            {
                "method": method,
                "n": int(len(g)),
                "rmse": _rmse(g["actual_value_per_game"], g["prediction_value_per_game"]),
                "mae": _mae(g["actual_value_per_game"], g["prediction_value_per_game"]),
                "r2": _r2(g["actual_value_per_game"], g["prediction_value_per_game"]),
            }
        )
    summary = pd.DataFrame(rows)
    ref = summary.loc[summary["method"] == "persistence", "rmse"]
    ref_rmse = float(ref.iloc[0]) if len(ref) else float("nan")
    summary["skill_vs_persistence"] = 1.0 - summary["rmse"] / ref_rmse
    sm_ref = summary.loc[summary["method"] == "single_model", "rmse"]
    sm_rmse = float(sm_ref.iloc[0]) if len(sm_ref) else float("nan")
    summary["skill_vs_single_model"] = 1.0 - summary["rmse"] / sm_rmse
    summary = summary.sort_values("rmse").reset_index(drop=True)
    return preds, summary


def interval_rolling_validation(
    project_root: Path,
    validation_years: list[int] | None = None,
    target_coverage: float = config.PREDICTION_INTERVAL_TARGET_COVERAGE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate asymmetric two-stage value intervals with rolling origin.

    For each validation season the training years are split into a proper
    training set and a calibration set (the most recent prior season). Both
    stages are fit on the proper-train set; their residuals on the calibration
    set give per-position error sigmas for each axis. Those sigmas are propagated
    through the product to build a value prediction interval for the validation
    rows, and empirical coverage is checked against the target.

    Returns (per-row predictions+intervals, coverage summary, variance-share
    summary).
    """
    if validation_years is None:
        validation_years = DEFAULT_VALIDATION_YEARS
    z = config.PREDICTION_INTERVAL_MULTIPLIER

    value_scores_path = project_root / "data" / "processed" / "player_value_scores_2016_2025.csv"
    ps = pd.read_csv(value_scores_path)
    ps = add_opportunity_and_efficiency(ps)
    ps = add_talent_rate_features(ps)
    ps = add_opportunity_history_features(ps)
    ps = add_efficiency_history_features(ps)
    ps = add_next_season_opportunity_target(ps)
    ps = add_next_season_efficiency_target(ps)
    ps = ps.sort_values(["player_id", "season"])
    grp = ps.groupby("player_id")
    if "value_epa_per_game" not in ps.columns:
        ps["value_epa_per_game"] = ps["efficiency_per_opportunity"] * ps["opportunity_per_game"]
    ps["next_value_per_game"] = grp["value_epa_per_game"].shift(-1)
    ps.loc[ps["next_season"].ne(ps["season"] + 1), "next_value_per_game"] = np.nan

    opp_feats = [c for c in OPPORTUNITY_FEATURES if c in ps.columns]
    eff_feats = [c for c in EFFICIENCY_FEATURES if c in ps.columns]
    scored_mask = ps[EFFICIENCY_TARGET].notna() & ps["next_value_per_game"].notna()

    fold_frames: list[pd.DataFrame] = []
    for year in validation_years:
        train_all = ps[ps["season"].lt(year)]
        valid = ps[ps["season"].eq(year) & scored_mask]
        if valid.empty or train_all.empty:
            continue

        calib_season = int(train_all["season"].max())
        proper = train_all[train_all["season"].lt(calib_season)]
        calib = train_all[train_all["season"].eq(calib_season)]
        if proper.empty or len(calib) < 30:
            # Fall back to a random split if there's too little history.
            calib = train_all.sample(frac=0.25, random_state=config.RANDOM_STATE)
            proper = train_all.drop(calib.index)
            if proper.empty:
                proper = train_all

        # Fit each stage on the proper-train set only.
        opp_train = proper.dropna(subset=[OPPORTUNITY_TARGET])
        eff_train = proper.dropna(subset=[EFFICIENCY_TARGET])
        opp_pipe = _make_opportunity_pipeline(opp_feats, "gradient_boosting")
        opp_pipe.fit(opp_train[opp_feats], opp_train[OPPORTUNITY_TARGET])
        eff_pipe = _make_opportunity_pipeline(eff_feats, "gradient_boosting")
        eff_pipe.fit(eff_train[eff_feats], eff_train[EFFICIENCY_TARGET])

        # Per-position residual sigma for each stage, from the calibration set.
        calib_opp = calib.dropna(subset=[OPPORTUNITY_TARGET]).copy()
        calib_opp["resid"] = (
            calib_opp[OPPORTUNITY_TARGET].to_numpy() - opp_pipe.predict(calib_opp[opp_feats])
        )
        opp_sigma_pos, opp_sigma_all = _per_position_residual_sigma(calib_opp, "resid")

        calib_eff = calib.dropna(subset=[EFFICIENCY_TARGET]).copy()
        calib_eff["resid"] = (
            calib_eff[EFFICIENCY_TARGET].to_numpy() - eff_pipe.predict(calib_eff[eff_feats])
        )
        eff_sigma_pos, eff_sigma_all = _per_position_residual_sigma(calib_eff, "resid")

        # Predict both stages on the validation rows and build the interval.
        opp_hat = opp_pipe.predict(valid[opp_feats])
        eff_hat = eff_pipe.predict(valid[eff_feats])
        pos = valid["position"].astype(str)
        so = pos.map(opp_sigma_pos).fillna(opp_sigma_all).to_numpy()
        se = pos.map(eff_sigma_pos).fillna(eff_sigma_all).to_numpy()

        prop = propagate_product_interval(eff_hat, opp_hat, se, so, z=z)
        out = valid[["player_id", "position", "season"]].copy()
        out["actual_value_per_game"] = valid["next_value_per_game"].to_numpy()
        for key in (
            "value_pred",
            "sigma",
            "interval_low",
            "interval_high",
            "interval_width",
            "var_from_efficiency",
            "var_from_opportunity",
            "efficiency_variance_share",
        ):
            out[key] = prop[key]
        out["covered"] = out["actual_value_per_game"].between(
            out["interval_low"], out["interval_high"]
        )
        fold_frames.append(out)

    if not fold_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    predictions = pd.concat(fold_frames, ignore_index=True)

    def _coverage(group_df: pd.DataFrame) -> dict[str, float]:
        return {
            "n": int(len(group_df)),
            "coverage": float(group_df["covered"].mean()),
            "mean_width": float(group_df["interval_width"].mean()),
            "mean_efficiency_variance_share": float(
                group_df["efficiency_variance_share"].mean()
            ),
        }

    overall = {"segment": "overall", "segment_value": "all", **_coverage(predictions)}
    rows = [overall]
    for pos, g in predictions.groupby("position"):
        rows.append({"segment": "position", "segment_value": pos, **_coverage(g)})
    coverage_summary = pd.DataFrame(rows)
    coverage_summary["target_coverage"] = target_coverage
    coverage_summary["coverage_gap"] = coverage_summary["coverage"] - target_coverage

    # Variance-share table: how much of value uncertainty is from each axis.
    var_rows = []
    for seg_value, g in [("all", predictions)] + list(predictions.groupby("position")):
        total_e = float(g["var_from_efficiency"].sum())
        total_o = float(g["var_from_opportunity"].sum())
        denom = total_e + total_o
        var_rows.append(
            {
                "segment_value": "all" if isinstance(seg_value, str) and seg_value == "all" else seg_value,
                "efficiency_variance_share": total_e / denom if denom > 0 else np.nan,
                "opportunity_variance_share": total_o / denom if denom > 0 else np.nan,
            }
        )
    variance_share = pd.DataFrame(var_rows)

    return predictions, coverage_summary, variance_share


# ---------------------------------------------------------------------------
# Report + orchestration
# ---------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.1f}%"


def _metric_table(summary: pd.DataFrame, skill_col: str, skill_label: str) -> list[str]:
    lines = [
        f"| Method | Type | RMSE | MAE | R² | {skill_label} |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in summary.iterrows():
        mtype = r.get("method_type", "")
        lines.append(
            f"| {r['method']} | {mtype} | {r['rmse']:.3f} | {r['mae']:.3f} | "
            f"{r['r2']:.3f} | {_fmt_pct(r[skill_col])} |"
        )
    return lines


def build_two_stage_report_markdown(
    opp_summary: pd.DataFrame,
    opp_by_pos: pd.DataFrame,
    eff_summary: pd.DataFrame,
    eff_by_pos: pd.DataFrame,
    combined_summary: pd.DataFrame,
    interval_coverage: pd.DataFrame | None = None,
    variance_share: pd.DataFrame | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# Two-Stage Value Model: Opportunity × Efficiency")
    lines.append("")
    lines.append(
        "Player value factors exactly as `value_epa_per_game = "
        "efficiency_per_opportunity × opportunity_per_game`. The decomposition "
        "analysis showed the two factors behave very differently year over year, "
        "so this model predicts each separately and recombines them rather than "
        "predicting blended value with one model. All metrics use rolling-origin "
        "validation."
    )
    lines.append("")
    lines.append("## Stage 1 — Opportunity (usage per game)")
    lines.append("")
    lines.append(
        "Skill score is RMSE reduction versus a persistence baseline (next "
        "opportunity = current opportunity per game)."
    )
    lines.append("")
    lines.extend(_metric_table(opp_summary, "skill_vs_persistence", "Skill vs persistence"))
    lines.append("")
    if not opp_by_pos.empty:
        lines.append("Models by position:")
        lines.append("")
        lines.append("| Position | Method | RMSE | R² | Skill vs persistence |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for _, r in opp_by_pos[opp_by_pos["method_type"] == "model"].iterrows():
            lines.append(
                f"| {r['position']} | {r['method']} | {r['rmse']:.3f} | {r['r2']:.3f} | "
                f"{_fmt_pct(r['skill_vs_persistence'])} |"
            )
        lines.append("")
    lines.append("## Stage 2 — Efficiency (value EPA per opportunity)")
    lines.append("")
    lines.append(
        "Computed on efficiency-qualified seasons only (a position-specific "
        "minimum opportunity load), because efficiency on tiny samples is noise. "
        "Skill score is RMSE reduction versus shrink-to-mean (predict the "
        "positional mean), which is a strong null when efficiency barely "
        "autocorrelates."
    )
    lines.append("")
    lines.extend(_metric_table(eff_summary, "skill_vs_shrink_to_mean", "Skill vs shrink-to-mean"))
    lines.append("")
    if not eff_by_pos.empty:
        lines.append("By position (all methods, to show where efficiency is learnable):")
        lines.append("")
        lines.append("| Position | Method | RMSE | R² | Skill vs shrink-to-mean |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for _, r in eff_by_pos.iterrows():
            lines.append(
                f"| {r['position']} | {r['method']} | {r['rmse']:.3f} | {r['r2']:.3f} | "
                f"{_fmt_pct(r['skill_vs_shrink_to_mean'])} |"
            )
        lines.append("")
        lines.append(
            "The expected pattern: quarterback efficiency is genuinely learnable "
            "(meaningful skill over the positional mean), while RB/WR/TE "
            "efficiency is close to pure regression to the mean — which is itself "
            "the key front-office insight."
        )
        lines.append("")
    if not combined_summary.empty:
        lines.append("## Recombined value vs single-model baseline")
        lines.append("")
        lines.append(
            "The two stages multiply to a per-game value projection, scored on "
            "efficiency-qualified pairs against a single model that predicts "
            "per-game value directly from the same feature union, and against "
            "persistence."
        )
        lines.append("")
        lines.append(
            "| Method | RMSE | MAE | R² | Skill vs persistence | Skill vs single model |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for _, r in combined_summary.iterrows():
            lines.append(
                f"| {r['method']} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['r2']:.3f} | "
                f"{_fmt_pct(r['skill_vs_persistence'])} | {_fmt_pct(r['skill_vs_single_model'])} |"
            )
        lines.append("")
    if interval_coverage is not None and not interval_coverage.empty:
        lines.append("## Asymmetric prediction intervals")
        lines.append("")
        lines.append(
            "Each stage's calibration-set residuals give a per-position error "
            "sigma, and these are propagated through the product "
            "`value = efficiency × opportunity` via "
            "`Var(E·O) = O²σ_E² + E²σ_O² + σ_E²σ_O²`. The first term is the "
            "uncertainty from the efficiency axis, the second from opportunity. "
            "This makes the band *legible*: the table below reports empirical "
            "coverage against the "
            f"{_fmt_pct(float(interval_coverage['target_coverage'].iloc[0]))} target "
            "and the share of value uncertainty coming from each axis — something "
            "a single blended model cannot decompose."
        )
        lines.append("")
        lines.append(
            "| Segment | Coverage | Target | Gap | Mean width | Efficiency variance share |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for _, r in interval_coverage.iterrows():
            lines.append(
                f"| {r['segment_value']} | {_fmt_pct(r['coverage'])} | "
                f"{_fmt_pct(r['target_coverage'])} | {_fmt_pct(r['coverage_gap'])} | "
                f"{r['mean_width']:.2f} | {_fmt_pct(r['mean_efficiency_variance_share'])} |"
            )
        lines.append("")
        if variance_share is not None and not variance_share.empty:
            lines.append(
                "Variance attribution (share of total value uncertainty by axis):"
            )
            lines.append("")
            lines.append("| Segment | Efficiency share | Opportunity share |")
            lines.append("| --- | ---: | ---: |")
            for _, r in variance_share.iterrows():
                lines.append(
                    f"| {r['segment_value']} | {_fmt_pct(r['efficiency_variance_share'])} | "
                    f"{_fmt_pct(r['opportunity_variance_share'])} |"
                )
            lines.append("")
            lines.append(
                "For wide receivers and tight ends almost all value uncertainty "
                "comes from the efficiency axis (the model cannot pin down "
                "per-target quality), while for quarterbacks and running backs the "
                "opportunity axis carries more of it. The interval is therefore "
                "wide along exactly the axis the model genuinely cannot predict — "
                "the practical payoff of modeling the two factors separately."
            )
            lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Separating the axes lets the model speak the language a front office "
        "uses — \"role should hold, expect efficiency regression\" — and "
        "concentrate confidence where the signal actually is (opportunity, and "
        "QB efficiency) while shrinking hard where it is not (skill-position "
        "efficiency). Whether the recombined value beats a single blended model "
        "on raw RMSE is reported above honestly; even when the gain is modest, "
        "the decomposition's value is in interpretability and calibrated, "
        "axis-aware uncertainty."
    )
    lines.append("")
    return "\n".join(lines)


def _find_project_root() -> Path:
    current = Path.cwd()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "processed").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root from " + str(current))


def build_two_stage_value_outputs(
    project_root: Path | None = None,
    save_outputs: bool = True,
    validation_years: list[int] | None = None,
    include_combined: bool = True,
) -> dict[str, Any]:
    """Build both stages, the recombined validation, tables, and report.

    Stage 1 (opportunity) and Stage 2 (efficiency) each train RF/HGB models, so
    they require scikit-learn. ``include_combined`` additionally runs the
    recombined-vs-single-model comparison (also model-based). Set it to False to
    skip the heaviest step.
    """
    if project_root is None:
        project_root = _find_project_root()
    project_root = Path(project_root)

    # Stage 1: opportunity
    opp_df, opp_feats = build_opportunity_modeling_frame(project_root)
    opp_preds = collect_opportunity_rolling_predictions(opp_df, opp_feats, validation_years)
    opp_summary = summarize_opportunity_methods(opp_preds)
    opp_by_pos = summarize_opportunity_by_position(opp_preds)

    # Stage 2: efficiency
    eff_df, eff_feats = build_efficiency_modeling_frame(project_root)
    eff_preds = collect_efficiency_rolling_predictions(eff_df, eff_feats, validation_years)
    eff_summary = summarize_efficiency_methods(eff_preds)
    eff_by_pos = summarize_efficiency_by_position(eff_preds)

    # Recombination vs single-model baseline
    if include_combined:
        combined_preds, combined_summary = combined_rolling_validation(
            project_root, validation_years
        )
        interval_preds, interval_coverage, variance_share = interval_rolling_validation(
            project_root, validation_years
        )
    else:
        combined_preds, combined_summary = pd.DataFrame(), pd.DataFrame()
        interval_preds, interval_coverage, variance_share = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    report_md = build_two_stage_report_markdown(
        opp_summary,
        opp_by_pos,
        eff_summary,
        eff_by_pos,
        combined_summary,
        interval_coverage,
        variance_share,
    )

    outputs = {
        "opportunity_predictions": opp_preds,
        "opportunity_summary": opp_summary,
        "opportunity_by_position": opp_by_pos,
        "efficiency_predictions": eff_preds,
        "efficiency_summary": eff_summary,
        "efficiency_by_position": eff_by_pos,
        "combined_predictions": combined_preds,
        "combined_summary": combined_summary,
        "interval_predictions": interval_preds,
        "interval_coverage": interval_coverage,
        "variance_share": variance_share,
        "report_markdown": report_md,
        "opportunity_features": opp_feats,
        "efficiency_features": eff_feats,
    }

    if save_outputs:
        tables_dir = project_root / "outputs" / "tables"
        report_dir = project_root / "report"
        tables_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        opp_summary.to_csv(tables_dir / "two_stage_opportunity_summary.csv", index=False)
        opp_by_pos.to_csv(tables_dir / "two_stage_opportunity_by_position.csv", index=False)
        eff_summary.to_csv(tables_dir / "two_stage_efficiency_summary.csv", index=False)
        eff_by_pos.to_csv(tables_dir / "two_stage_efficiency_by_position.csv", index=False)
        if not combined_summary.empty:
            combined_summary.to_csv(
                tables_dir / "two_stage_combined_summary.csv", index=False
            )
        if not interval_coverage.empty:
            interval_coverage.to_csv(
                tables_dir / "two_stage_interval_coverage.csv", index=False
            )
            variance_share.to_csv(
                tables_dir / "two_stage_variance_share.csv", index=False
            )
        (report_dir / "two_stage_value.md").write_text(report_md)

    return outputs


if __name__ == "__main__":
    result = build_two_stage_value_outputs()
    print("Opportunity:\n", result["opportunity_summary"].to_string(index=False))
    print("\nEfficiency:\n", result["efficiency_summary"].to_string(index=False))
    if not result["combined_summary"].empty:
        print("\nCombined:\n", result["combined_summary"].to_string(index=False))
