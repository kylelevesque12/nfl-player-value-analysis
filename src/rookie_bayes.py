"""Hierarchical Bayesian rookie-season PPR projections.

This is the methodology piece from Tier 2 #4 of ``PORTFOLIO_ROADMAP.md``. The
problem it solves is one the existing stack literally cannot: projecting a
*rookie's* fantasy production before they've played an NFL snap. The HGB
weekly model needs rolling history; rookies don't have any. Public projectors
solve this with cold-start models built on draft capital, age, athletic
profile, and college production translation.

The model
---------

Hierarchical Normal regression on PPR-per-game, partial-pooled across the four
skill positions::

    ppg[i] ~ Normal(mu[i], sigma[pos[i]])

    mu[i] = alpha[pos[i]]
          + beta_draft[pos[i]] * draft_log_z[i]
          + beta_age[pos[i]]   * age_z[i]
          + beta_height[pos[i]]* height_z[i]
          + beta_weight[pos[i]]* weight_z[i]
          + beta_college[pos[i]] * college_score[i]   # optional, see below

    alpha[pos]       ~ Normal(alpha_mu, alpha_tau)
    beta_draft[pos]  ~ Normal(beta_draft_mu, beta_draft_tau)
    beta_age[pos]    ~ Normal(beta_age_mu, beta_age_tau)
    beta_height[pos] ~ Normal(0, 1)
    beta_weight[pos] ~ Normal(0, 1)
    beta_college[pos]~ Normal(0, 1)

    alpha_mu, beta_*_mu ~ Normal(0, 5)
    alpha_tau, beta_*_tau, sigma[pos] ~ HalfNormal(2)

Partial pooling on the intercept and the high-information slopes (draft, age)
lets positions with small samples (TE) borrow strength from positions with big
samples (WR), while still allowing position-specific effects.

PyMC isolation
--------------

PyMC's ABI is incompatible with the rest of the project's numpy/pandas pins.
This module imports PyMC **inside** ``fit_rookie_model``; the rest of the
module — modeling-frame construction, validation orchestration, summary
formatting — runs in the project's normal venv. To run the model itself, set
up a separate venv from ``requirements-bayes.txt``:

    python -m venv .venv-bayes
    source .venv-bayes/bin/activate
    pip install -r requirements-bayes.txt
    python -c "from src.rookie_bayes import build_rookie_bayes_outputs; \\
               build_rookie_bayes_outputs()"

The data-prep helpers below are exercised by ``tests/test_rookie_bayes.py``
in the main env so the architecture stays under test even when PyMC is not
importable.

College stats
-------------

The ``college_score`` column is optional. If ``data/raw/college_production.csv``
exists with columns ``player_id,college_score`` it gets joined in; otherwise
the corresponding term is dropped. The intended source is the CollegeFootballData
API (free, requires registration) — see ``scripts/fetch_college_production.py``
for the acquisition stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.load_data import ensure_project_dirs, find_project_root


SKILL_POSITIONS = list(config.SKILL_POSITIONS)
CSV_FLOAT_FORMAT = config.CSV_FLOAT_FORMAT

# Rookies are typically projected when their rookie season is "next" — the
# current model produces 2026 rookie projections from training data on rookies
# whose rookie_year was <= 2025. Validation holds out 2020-2024 rookies; 2025
# rookies have observed targets too thanks to your data ending at 2025.
DEFAULT_VALIDATION_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
MIN_GAMES_FOR_TARGET = 4  # exclude rookies who barely played from training
MIN_DRAFT_PICK_FOR_TRAINING = 1  # include all drafted players
UNDRAFTED_PICK_NUMBER = 300.0  # impute for undrafted free agents


# ---------------------------------------------------------------------------
# Modeling-frame construction (no PyMC dependency)
# ---------------------------------------------------------------------------
def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _safe_age_at_draft(birth_date: pd.Series, draft_year: pd.Series) -> pd.Series:
    """Approximate age at the April draft. Good to within a few months."""
    birth = pd.to_datetime(birth_date, errors="coerce")
    draft_day = pd.to_datetime(
        draft_year.astype("string").str.cat(["-04-15"] * len(draft_year)),
        errors="coerce",
    )
    return ((draft_day - birth).dt.days / 365.25).astype("float64")


def _height_to_inches(height: pd.Series) -> pd.Series:
    """Convert nflverse height strings like "6-2" or "74" to total inches."""
    s = height.astype("string").fillna("")

    def convert(value: str) -> float:
        v = value.strip()
        if not v:
            return float("nan")
        if "-" in v:
            try:
                feet, inches = v.split("-", 1)
                return float(feet) * 12.0 + float(inches)
            except (TypeError, ValueError):
                return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    return s.map(convert)


def _build_rookie_player_season_targets(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Collapse weekly stats into per-player-season PPR totals and games."""
    df = player_stats.copy()
    df = df[df["season_type"].eq("REG")]
    df = _to_numeric(df, ["season", "week", "fantasy_points_ppr"])
    df = df.dropna(subset=["season", "player_id", "fantasy_points_ppr"])
    df["season"] = df["season"].astype(int)
    df["played"] = df["fantasy_points_ppr"].notna().astype(int)
    agg = df.groupby(["player_id", "season"], as_index=False).agg(
        season_ppr_total=("fantasy_points_ppr", "sum"),
        games_played=("played", "sum"),
    )
    agg["season_ppr_per_game"] = agg["season_ppr_total"] / agg["games_played"].clip(lower=1)
    return agg


def _load_college_score(project_root: Path) -> pd.DataFrame:
    """Optional college-production score keyed by ``player_id``.

    Expected CSV columns: ``player_id``, ``college_score``. If absent the
    feature is dropped from the model.
    """
    path = project_root / "data" / "raw" / "college_production.csv"
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "college_score"])
    return pd.read_csv(path)


def build_rookie_modeling_frame(
    rosters: pd.DataFrame,
    player_stats: pd.DataFrame,
    project_root: Path | None = None,
    attach_context: bool = True,
) -> pd.DataFrame:
    """Build the per-rookie modeling frame.

    Each row is a player's rookie season, with pre-draft features and the
    observed rookie-season PPR-per-game target.

    When ``attach_context`` is True (default), pre-season incumbent / team
    context features are attached (see ``src/rookie_context_features``). Only a
    small, theoretically-grounded subset (``CONTEXT_FEATURES``) is fed to the
    model; the rest are attached for inspection but left out of the feature set.
    All context features are strictly pre-rookie-season — Session 3 documents the
    leakage discipline and the combine/broad-depth features that were tested and
    dropped.
    """
    root = (
        find_project_root() if project_root is None else Path(project_root).resolve()
    )

    df = rosters.copy()
    df = df[df["position"].isin(SKILL_POSITIONS)]
    df = _to_numeric(df, ["season", "draft_number", "rookie_year", "entry_year", "weight"])

    # Restrict each player to their rookie-year row.
    df = df.dropna(subset=["rookie_year", "season"]).copy()
    df["rookie_year"] = df["rookie_year"].astype(int)
    df["season"] = df["season"].astype(int)
    df = df[df["season"].eq(df["rookie_year"])].copy()
    # Some players appear twice in their rookie season due to trades — keep
    # the first.
    df = df.drop_duplicates(subset=["gsis_id"]).copy()
    df = df.rename(columns={"gsis_id": "player_id"})

    df["draft_number"] = df["draft_number"].fillna(UNDRAFTED_PICK_NUMBER)
    df["age_at_draft"] = _safe_age_at_draft(df["birth_date"], df["rookie_year"])
    if "age_at_draft_hint" in df.columns:
        # A brand-new draft class has no birth_date in nflverse's roster feed
        # yet (it's populated weeks after the draft). draft_picks carries a
        # draft-day age directly, computed the same way (age at the ~April
        # draft) — fill the gap with it rather than leaving the feature NaN.
        # A no-op for historical rows, which always have a real birth_date.
        df["age_at_draft"] = df["age_at_draft"].fillna(
            pd.to_numeric(df["age_at_draft_hint"], errors="coerce")
        )
    df["height_inches"] = _height_to_inches(df["height"])
    df["draft_log"] = np.log(df["draft_number"].clip(lower=1.0))

    targets = _build_rookie_player_season_targets(player_stats)
    df = df.merge(
        targets,
        left_on=["player_id", "rookie_year"],
        right_on=["player_id", "season"],
        how="left",
        suffixes=("", "_target"),
    ).drop(columns=["season_target"], errors="ignore")

    college = _load_college_score(root)
    if not college.empty:
        df = df.merge(college[["player_id", "college_score"]], on="player_id", how="left")

    # Hurdle-model targets.
    #   played_meaningfully = 1 if the rookie had >= MIN_GAMES_FOR_TARGET games
    #     in their rookie year, 0 otherwise (covers Jordan Love 2020 — no
    #     appearances at all behind Rodgers, currently NaN -> dropped from
    #     training entirely).
    #   rookie_year_ppr_per_game_full = season_ppr_per_game if played, 0 if not.
    #     This is what we actually want to project: expected rookie-year PPR/game
    #     averaged over the cases where the player played and the cases where
    #     they didn't.
    games = df.get("games_played")
    df["played_meaningfully"] = (
        (games.fillna(0) >= MIN_GAMES_FOR_TARGET).astype(int)
        if games is not None
        else 0
    )
    df["rookie_year_ppr_per_game_full"] = df.get(
        "season_ppr_per_game", pd.Series(dtype="float64")
    ).fillna(0.0)

    keep = [
        "player_id",
        "player_display_name" if "player_display_name" in df.columns else "full_name",
        "position",
        "rookie_year",
        "draft_number",
        "draft_log",
        "age_at_draft",
        "height_inches",
        "weight",
        "college",
        "draft_club",
        "season_ppr_total",
        "games_played",
        "season_ppr_per_game",
        "played_meaningfully",
        "rookie_year_ppr_per_game_full",
    ]
    if "college_score" in df.columns:
        keep.insert(keep.index("college") + 1, "college_score")
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    if "full_name" in out.columns and "player_display_name" not in out.columns:
        out = out.rename(columns={"full_name": "player_display_name"})
    out = out.reset_index(drop=True)

    if attach_context:
        from src.rookie_context_features import attach_rookie_context_features

        out = attach_rookie_context_features(
            out, rosters, player_stats, project_root=root
        )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature standardization (computed within a training fold to avoid leakage)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = ["draft_log", "age_at_draft", "height_inches", "weight"]
# Session 3 incumbent-context features. A focused, theoretically-grounded set:
# whether the rookie's position has an established returning starter, whether
# that incumbent recently signed a meaningful extension, and the prior-year
# starting QB's production. These chiefly sharpen the QB "blocked behind a
# starter" cell (Jordan Love, Mahomes). Combine and the broader depth features
# were tested and dropped — see report/rookie/session3_combine_team_context.md.
# They are OPTIONAL: absent if build_rookie_modeling_frame(attach_context=False).
CONTEXT_FEATURES = [
    "established_incumbent",
    "incumbent_recent_extension",
    "prior_qb_pprpg",
]
OPTIONAL_FEATURES = ["college_score"] + CONTEXT_FEATURES


def standardize_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    """Z-score features using **training-fold** statistics only."""
    train_z = train.copy()
    test_z = test.copy()
    stats: dict[str, tuple[float, float]] = {}
    for col in FEATURE_COLUMNS + OPTIONAL_FEATURES:
        if col not in train.columns:
            continue
        mu = float(train[col].mean(skipna=True))
        sd = float(train[col].std(skipna=True))
        if sd <= 0 or not np.isfinite(sd):
            continue
        train_z[f"{col}_z"] = (train[col] - mu) / sd
        test_z[f"{col}_z"] = (test[col] - mu) / sd
        stats[col] = (mu, sd)
    return train_z, test_z, stats


# ---------------------------------------------------------------------------
# PyMC model (imports inside the function — see module docstring)
# ---------------------------------------------------------------------------
def fit_rookie_model(
    train_df: pd.DataFrame,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> Any:
    """Fit the hierarchical Normal model. Returns an arviz ``InferenceData``.

    Imports PyMC at call time so the rest of the project keeps working in
    envs where PyMC is not installable.
    """
    import pymc as pm  # noqa: PLC0415

    train = train_df.dropna(subset=["season_ppr_per_game"]).copy()
    train = train[train["games_played"].ge(MIN_GAMES_FOR_TARGET)].copy()
    if train.empty:
        raise ValueError("No usable training rookies after filtering.")

    pos_idx, pos_levels = pd.factorize(train["position"], sort=True)
    n_positions = len(pos_levels)
    feature_z_cols = [c for c in train.columns if c.endswith("_z")]
    if not feature_z_cols:
        raise ValueError(
            "No standardized feature columns found. Call standardize_features first."
        )

    feature_matrix = train[feature_z_cols].fillna(0.0).to_numpy()
    n_features = feature_matrix.shape[1]
    y = train["season_ppr_per_game"].to_numpy()

    with pm.Model() as model:
        # Hyperpriors on the position-level intercept and slope means.
        alpha_mu = pm.Normal("alpha_mu", mu=8.0, sigma=5.0)
        alpha_tau = pm.HalfNormal("alpha_tau", sigma=3.0)
        # Non-centered parameterization for alpha: prevents the funnel
        # geometry that causes divergences when alpha_tau is small.
        alpha_offset = pm.Normal("alpha_offset", mu=0.0, sigma=1.0, shape=n_positions)
        alpha = pm.Deterministic("alpha", alpha_mu + alpha_tau * alpha_offset)

        beta_mu = pm.Normal("beta_mu", mu=0.0, sigma=2.0, shape=n_features)
        beta_tau = pm.HalfNormal("beta_tau", sigma=1.5, shape=n_features)
        # Non-centered for beta too. Shape (positions, features).
        beta_offset = pm.Normal(
            "beta_offset", mu=0.0, sigma=1.0, shape=(n_positions, n_features)
        )
        beta = pm.Deterministic("beta", beta_mu + beta_tau * beta_offset)

        sigma = pm.HalfNormal("sigma", sigma=3.0, shape=n_positions)

        mu = alpha[pos_idx] + pm.math.sum(beta[pos_idx, :] * feature_matrix, axis=1)
        pm.Normal(
            "y",
            mu=mu,
            sigma=sigma[pos_idx],
            observed=y,
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
        )
        idata.attrs["position_levels"] = list(pos_levels)
        idata.attrs["feature_columns"] = feature_z_cols

    return idata


def predict_rookies(idata: Any, test_df: pd.DataFrame) -> pd.DataFrame:
    """Posterior-predictive mean and 50/80 intervals per test rookie."""
    pos_levels = list(idata.attrs["position_levels"])
    feature_cols = list(idata.attrs["feature_columns"])

    alpha = idata.posterior["alpha"].stack(sample=("chain", "draw")).values  # (positions, samples)
    beta = idata.posterior["beta"].stack(sample=("chain", "draw")).values  # (positions, features, samples)
    sigma = idata.posterior["sigma"].stack(sample=("chain", "draw")).values  # (positions, samples)

    pos_to_idx = {p: i for i, p in enumerate(pos_levels)}
    rows = test_df.copy()
    feature_matrix = rows[feature_cols].fillna(0.0).to_numpy()
    pos_idx = rows["position"].map(pos_to_idx).fillna(0).astype(int).to_numpy()

    n_samples = alpha.shape[1]
    n_rows = len(rows)
    pred_samples = np.zeros((n_rows, n_samples))
    for i in range(n_rows):
        p = pos_idx[i]
        # mu_i = alpha[p] + beta[p, :] @ feature_matrix[i]
        mu_i = alpha[p] + beta[p, :, :].T @ feature_matrix[i]
        pred_samples[i] = np.random.normal(mu_i, sigma[p])

    rows["predicted_ppr_per_game_mean"] = pred_samples.mean(axis=1)
    rows["predicted_ppr_per_game_p10"] = np.percentile(pred_samples, 10, axis=1)
    rows["predicted_ppr_per_game_p25"] = np.percentile(pred_samples, 25, axis=1)
    rows["predicted_ppr_per_game_p75"] = np.percentile(pred_samples, 75, axis=1)
    rows["predicted_ppr_per_game_p90"] = np.percentile(pred_samples, 90, axis=1)
    return rows


# ---------------------------------------------------------------------------
# Hurdle model: stage 1 (logistic) for P(plays meaningfully)
# ---------------------------------------------------------------------------
def fit_played_meaningfully_model(
    train_df: pd.DataFrame,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> Any:
    """Stage 1 of the hurdle: hierarchical logistic regression on whether the
    rookie played a meaningful number of games (default >= 4) in their
    rookie season.

    Trains on **every** rookie in the frame — both players who played and
    players who didn't. The target is binary.
    """
    import pymc as pm  # noqa: PLC0415

    train = train_df.dropna(subset=["played_meaningfully"]).copy()
    if train.empty:
        raise ValueError("No rookies with a played_meaningfully label.")

    pos_idx, pos_levels = pd.factorize(train["position"], sort=True)
    n_positions = len(pos_levels)
    feature_z_cols = [c for c in train.columns if c.endswith("_z")]
    if not feature_z_cols:
        raise ValueError(
            "No standardized feature columns found. Call standardize_features first."
        )

    feature_matrix = train[feature_z_cols].fillna(0.0).to_numpy()
    n_features = feature_matrix.shape[1]
    y = train["played_meaningfully"].astype(int).to_numpy()

    with pm.Model() as model:
        # Hierarchical structure mirrors stage 2 — partial pooling on
        # intercept and slopes across positions.
        alpha_mu = pm.Normal("alpha_mu", mu=0.0, sigma=2.0)
        alpha_tau = pm.HalfNormal("alpha_tau", sigma=2.0)
        alpha_offset = pm.Normal("alpha_offset", mu=0.0, sigma=1.0, shape=n_positions)
        alpha = pm.Deterministic("alpha", alpha_mu + alpha_tau * alpha_offset)

        beta_mu = pm.Normal("beta_mu", mu=0.0, sigma=1.0, shape=n_features)
        beta_tau = pm.HalfNormal("beta_tau", sigma=1.0, shape=n_features)
        beta_offset = pm.Normal(
            "beta_offset", mu=0.0, sigma=1.0, shape=(n_positions, n_features)
        )
        beta = pm.Deterministic("beta", beta_mu + beta_tau * beta_offset)

        logit_p = alpha[pos_idx] + pm.math.sum(
            beta[pos_idx, :] * feature_matrix, axis=1
        )
        pm.Bernoulli("y", logit_p=logit_p, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
        )
        idata.attrs["position_levels"] = list(pos_levels)
        idata.attrs["feature_columns"] = feature_z_cols
    return idata


def _stack_posterior(idata: Any, name: str) -> np.ndarray:
    return idata.posterior[name].stack(sample=("chain", "draw")).values


def predict_hurdle(
    stage1_idata: Any,
    stage2_idata: Any,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combined hurdle projection: P(plays) * E[PPR/game | plays].

    Returns a DataFrame with the joint posterior mean and 50/80 percentile
    intervals over the combined product, plus diagnostic columns showing
    each stage's contribution separately.
    """
    pos_levels = list(stage1_idata.attrs["position_levels"])
    feature_cols = list(stage1_idata.attrs["feature_columns"])

    alpha1 = _stack_posterior(stage1_idata, "alpha")
    beta1 = _stack_posterior(stage1_idata, "beta")
    alpha2 = _stack_posterior(stage2_idata, "alpha")
    beta2 = _stack_posterior(stage2_idata, "beta")
    sigma2 = _stack_posterior(stage2_idata, "sigma")

    pos_to_idx = {p: i for i, p in enumerate(pos_levels)}
    rows = test_df.copy()
    feature_matrix = rows[feature_cols].fillna(0.0).to_numpy()
    pos_idx = rows["position"].map(pos_to_idx).fillna(0).astype(int).to_numpy()

    n_samples = alpha1.shape[1]
    n_rows = len(rows)

    p_plays_samples = np.zeros((n_rows, n_samples))
    ppr_if_plays_samples = np.zeros((n_rows, n_samples))

    for i in range(n_rows):
        p = pos_idx[i]
        logit1 = alpha1[p] + beta1[p, :, :].T @ feature_matrix[i]
        p_plays_samples[i] = 1.0 / (1.0 + np.exp(-logit1))
        mu2 = alpha2[p] + beta2[p, :, :].T @ feature_matrix[i]
        ppr_if_plays_samples[i] = np.random.normal(mu2, sigma2[p])

    # Expected rookie-year PPR/game averaged over both "didn't play" (yields 0)
    # and "did play" (yields the stage-2 sample). This is the calibrated
    # answer to "what should we expect from this rookie in year 1?"
    combined_samples = p_plays_samples * ppr_if_plays_samples.clip(min=0.0)

    rows["predicted_p_plays_meaningfully"] = p_plays_samples.mean(axis=1)
    rows["predicted_ppr_per_game_if_plays_mean"] = ppr_if_plays_samples.mean(axis=1)
    rows["predicted_rookie_year_ppr_per_game_mean"] = combined_samples.mean(axis=1)
    rows["predicted_rookie_year_ppr_per_game_p10"] = np.percentile(
        combined_samples, 10, axis=1
    )
    rows["predicted_rookie_year_ppr_per_game_p25"] = np.percentile(
        combined_samples, 25, axis=1
    )
    rows["predicted_rookie_year_ppr_per_game_p75"] = np.percentile(
        combined_samples, 75, axis=1
    )
    rows["predicted_rookie_year_ppr_per_game_p90"] = np.percentile(
        combined_samples, 90, axis=1
    )
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def rolling_rookie_validation(
    modeling_df: pd.DataFrame,
    validation_years: list[int] | None = None,
    **fit_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin validation across rookie classes.

    For each validation year, fit on all rookies whose rookie_year is strictly
    earlier, predict on that year's rookies, and report metrics.
    """
    if validation_years is None:
        validation_years = DEFAULT_VALIDATION_YEARS

    all_preds: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for year in validation_years:
        train = modeling_df[modeling_df["rookie_year"].lt(year)].copy()
        test = modeling_df[modeling_df["rookie_year"].eq(year)].copy()
        if train.empty or test.empty:
            continue
        train_z, test_z, _stats = standardize_features(train, test)
        idata = fit_rookie_model(train_z, **fit_kwargs)
        preds = predict_rookies(idata, test_z)
        preds["validation_year"] = int(year)
        all_preds.append(preds)

        eval_df = preds.dropna(subset=["season_ppr_per_game"])
        if eval_df.empty:
            continue
        y = eval_df["season_ppr_per_game"].to_numpy()
        p = eval_df["predicted_ppr_per_game_mean"].to_numpy()
        metric_rows.append(
            {
                "validation_year": int(year),
                "n_rookies": int(len(eval_df)),
                "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                "mae": float(np.mean(np.abs(y - p))),
                "bias": float(np.mean(p - y)),
                "interval_50_coverage": float(
                    eval_df["season_ppr_per_game"]
                    .between(
                        eval_df["predicted_ppr_per_game_p25"],
                        eval_df["predicted_ppr_per_game_p75"],
                    )
                    .mean()
                ),
                "interval_80_coverage": float(
                    eval_df["season_ppr_per_game"]
                    .between(
                        eval_df["predicted_ppr_per_game_p10"],
                        eval_df["predicted_ppr_per_game_p90"],
                    )
                    .mean()
                ),
            }
        )

    preds_df = (
        pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    )
    metrics_df = pd.DataFrame(metric_rows)
    return preds_df, metrics_df


def rolling_hurdle_validation(
    modeling_df: pd.DataFrame,
    validation_years: list[int] | None = None,
    **fit_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling validation for the two-stage hurdle model.

    Evaluates the combined projection (P(plays) * E[PPR|plays]) against the
    *full* rookie-year PPR/game target (which is 0 for rookies who didn't
    play, not NaN). This is the version that gives Jordan Love a meaningful
    target instead of silently dropping him.
    """
    if validation_years is None:
        validation_years = DEFAULT_VALIDATION_YEARS

    all_preds: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for year in validation_years:
        train = modeling_df[modeling_df["rookie_year"].lt(year)].copy()
        test = modeling_df[modeling_df["rookie_year"].eq(year)].copy()
        if train.empty or test.empty:
            continue
        train_z, test_z, _stats = standardize_features(train, test)

        stage1 = fit_played_meaningfully_model(train_z, **fit_kwargs)
        train_played = train_z[train_z["played_meaningfully"].eq(1)].copy()
        if train_played.empty:
            continue
        stage2 = fit_rookie_model(train_played, **fit_kwargs)
        preds = predict_hurdle(stage1, stage2, test_z)
        preds["validation_year"] = int(year)
        all_preds.append(preds)

        eval_df = preds.dropna(subset=["rookie_year_ppr_per_game_full"])
        if eval_df.empty:
            continue
        y = eval_df["rookie_year_ppr_per_game_full"].to_numpy()
        p = eval_df["predicted_rookie_year_ppr_per_game_mean"].to_numpy()

        # Calibration: are the players we predict will play actually playing?
        played_actual = (
            (eval_df["games_played"].fillna(0) >= MIN_GAMES_FOR_TARGET)
            .astype(int)
            .to_numpy()
        )
        played_pred = eval_df["predicted_p_plays_meaningfully"].to_numpy()
        # Simple Brier-style calibration metric: mean squared error of the
        # plays probability against the realized binary outcome.
        plays_brier = float(np.mean((played_pred - played_actual) ** 2))
        plays_actual_rate = float(played_actual.mean())
        plays_pred_rate = float(played_pred.mean())

        metric_rows.append(
            {
                "validation_year": int(year),
                "n_rookies": int(len(eval_df)),
                "n_played_meaningfully": int(played_actual.sum()),
                "rmse_full_target": float(np.sqrt(np.mean((y - p) ** 2))),
                "mae_full_target": float(np.mean(np.abs(y - p))),
                "bias_full_target": float(np.mean(p - y)),
                "interval_50_coverage_full": float(
                    eval_df["rookie_year_ppr_per_game_full"]
                    .between(
                        eval_df["predicted_rookie_year_ppr_per_game_p25"],
                        eval_df["predicted_rookie_year_ppr_per_game_p75"],
                    )
                    .mean()
                ),
                "interval_80_coverage_full": float(
                    eval_df["rookie_year_ppr_per_game_full"]
                    .between(
                        eval_df["predicted_rookie_year_ppr_per_game_p10"],
                        eval_df["predicted_rookie_year_ppr_per_game_p90"],
                    )
                    .mean()
                ),
                "plays_brier": plays_brier,
                "plays_actual_rate": plays_actual_rate,
                "plays_predicted_rate": plays_pred_rate,
            }
        )

    preds_df = (
        pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    )
    metrics_df = pd.DataFrame(metric_rows)
    return preds_df, metrics_df


# ---------------------------------------------------------------------------
# Scoring the live draft class (production fit: no holdout, train on every
# rookie class the rolling validation above has ever tested against)
# ---------------------------------------------------------------------------
def expected_games_if_plays_by_position(train_df: pd.DataFrame) -> dict[str, float]:
    """Mean games_played among rookies who cleared the hurdle (played >= 4
    games), by position, from the training data.

    The hurdle model projects two things: whether a rookie plays meaningfully
    (stage 1) and his per-game rate if he does (stage 2). Neither stage
    predicts *how many* games he plays if he clears the bar, so turning a
    per-game rate into a season total needs this one further, simple,
    documented estimate — a position-level historical average, not a
    player-specific games model. Falls back to the overall average for a
    position with no qualifying rookies in the training window.
    """
    played = train_df[train_df["played_meaningfully"].eq(1)]
    overall_mean = float(played["games_played"].mean()) if not played.empty else 12.0
    by_position = played.groupby("position")["games_played"].mean()
    return {
        pos: float(by_position.get(pos, overall_mean))
        for pos in train_df["position"].unique()
    }


def scale_hurdle_output_to_season_totals(
    scored: pd.DataFrame, games_by_position: dict[str, float]
) -> pd.DataFrame:
    """Pure post-processing: turn ``predict_hurdle``'s per-game columns into
    season totals given a position -> expected-games multiplier.

    Split out from ``score_rookie_class`` so this arithmetic — the one part
    of the scoring path with no PyMC dependency — can be unit-tested
    directly against a synthetic, already-hurdle-scored frame.
    """
    scored = scored.copy()
    games_multiplier = scored["position"].map(games_by_position)
    scored["predicted_games_played"] = (
        games_multiplier * scored["predicted_p_plays_meaningfully"]
    )
    for suffix in ("mean", "p10", "p25", "p75", "p90"):
        per_game_col = f"predicted_rookie_year_ppr_per_game_{suffix}"
        scored[f"predicted_season_ppr_{suffix}"] = scored[per_game_col] * games_multiplier
    return scored


def score_rookie_class(
    modeling_df: pd.DataFrame,
    class_year: int,
    **fit_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score one draft class with a full production fit (no holdout).

    Trains the two-stage hurdle model on every earlier rookie class in
    ``modeling_df`` (rookie_year < ``class_year``) and projects the
    ``class_year`` rookies. Converts the hurdle's per-game rate into a
    season total via ``scale_hurdle_output_to_season_totals``: a constant
    multiplier is percentile-equivariant, so every one of ``predict_hurdle``'s
    existing mean/p10/p25/p75/p90 columns can be scaled directly without
    re-deriving the posterior — no change to that (separately tested)
    function was needed.
    """
    train = modeling_df[modeling_df["rookie_year"].lt(class_year)].copy()
    predict_rows = modeling_df[modeling_df["rookie_year"].eq(class_year)].copy()
    if train.empty:
        raise ValueError(f"No training rookies with rookie_year < {class_year}.")
    if predict_rows.empty:
        raise ValueError(f"No rookies found with rookie_year == {class_year}.")

    train_z, predict_z, _stats = standardize_features(train, predict_rows)

    stage1 = fit_played_meaningfully_model(train_z, **fit_kwargs)
    train_played = train_z[train_z["played_meaningfully"].eq(1)].copy()
    if train_played.empty:
        raise ValueError("No training rookies cleared the played_meaningfully bar.")
    stage2 = fit_rookie_model(train_played, **fit_kwargs)

    scored = predict_hurdle(stage1, stage2, predict_z)
    games_by_position = expected_games_if_plays_by_position(train)
    scored = scale_hurdle_output_to_season_totals(scored, games_by_position)
    return scored, games_by_position


def build_2026_rookie_projection_outputs(
    project_root: str | Path | None = None,
    class_year: int = 2026,
    save_outputs: bool = True,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Score the live draft class: production hurdle fit + season totals.

    Unlike ``build_rookie_bayes_outputs`` (rolling-origin validation on past
    classes, where every outcome is already known), this trains on the
    *entire* history through ``class_year - 1`` and projects a class whose
    outcome is genuinely unknown yet. Requires
    ``data/raw/rookie_class_<class_year>.csv`` (see
    ``scripts/fetch_rookie_class.py``) in addition to the historical rosters
    and player_stats files.

    Imports PyMC inside the fit functions this calls; run from ``.venv-bayes``
    (see ``requirements-bayes.txt``).
    """
    root = (
        find_project_root() if project_root is None else Path(project_root).resolve()
    )
    dirs = ensure_project_dirs(root)
    output_dir = dirs["tables"]

    class_path = root / "data" / "raw" / f"rookie_class_{class_year}.csv"
    if not class_path.exists():
        raise FileNotFoundError(
            f"{class_path} not found. Run "
            f"`python scripts/fetch_rookie_class.py --year {class_year}` first."
        )

    rosters = pd.read_csv(
        root / "data" / "raw" / "rosters_2016_2025.csv", low_memory=False
    )
    player_stats = pd.read_csv(
        root / "data" / "raw" / "player_stats_2016_2025.csv", low_memory=False
    )
    rookie_class = pd.read_csv(class_path)
    combined_rosters = pd.concat([rosters, rookie_class], ignore_index=True)

    modeling_df = build_rookie_modeling_frame(
        combined_rosters, player_stats, project_root=root
    )
    scored, games_by_position = score_rookie_class(modeling_df, class_year, **fit_kwargs)

    keep = [
        "player_id", "player_display_name", "position", "rookie_year",
        "draft_number", "draft_club", "college",
        "predicted_p_plays_meaningfully",
        "predicted_ppr_per_game_if_plays_mean",
        "predicted_games_played",
        "predicted_season_ppr_mean", "predicted_season_ppr_p10",
        "predicted_season_ppr_p25", "predicted_season_ppr_p75",
        "predicted_season_ppr_p90",
    ]
    out = scored[[c for c in keep if c in scored.columns]].sort_values(
        "predicted_season_ppr_mean", ascending=False
    ).reset_index(drop=True)

    summary_text = _build_2026_rookie_summary_text(out, games_by_position, class_year)

    outputs = {
        "projections": out,
        "games_by_position": games_by_position,
        "summary_text": summary_text,
    }
    if save_outputs:
        out.to_csv(
            output_dir / f"{class_year}_rookie_projections.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        report_path = root / "report" / "rookie_2026_projection.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(summary_text)
    return outputs


def _build_2026_rookie_summary_text(
    scored: pd.DataFrame, games_by_position: dict[str, float], class_year: int
) -> str:
    lines = [
        f"# {class_year} rookie class: projected fantasy production",
        "",
        "This scores the incoming draft class with the hierarchical Bayesian",
        "hurdle model, fit on every earlier rookie class (2016 through "
        f"{class_year - 1}) with no held-out year — a production fit, not a",
        "validation run. See `report/rookie_bayes_projection.md` for the",
        "model's out-of-sample accuracy on past classes.",
        "",
        "## From per-game rate to a season total",
        "",
        "The hurdle model projects two things directly: the probability a",
        "rookie plays meaningfully (at least 4 games) and his expected",
        "PPR-per-game rate if he does. Neither predicts how many games he",
        "plays if he clears that bar, so the season total here multiplies the",
        "per-game rate by a further, simple estimate: the historical average",
        "games played among rookies at his position who cleared the hurdle.",
        "This is a documented approximation, not a player-specific games",
        "model — a rookie who wins a starting job in Week 1 and one who",
        "starts Week 9 get the same games multiplier if they share a",
        "position.",
        "",
        "Expected games played, if the hurdle is cleared, by position:",
        "",
        "| Position | Expected games |",
        "| --- | ---: |",
    ]
    for pos in sorted(games_by_position):
        lines.append(f"| {pos} | {games_by_position[pos]:.1f} |")
    lines += [
        "",
        "## Top projected rookies",
        "",
        "| Player | Pos | Pick | P(plays) | Proj PPR | 80% range |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in scored.head(15).iterrows():
        lines.append(
            f"| {row['player_display_name']} | {row['position']} | "
            f"{int(row['draft_number'])} | "
            f"{row['predicted_p_plays_meaningfully']:.0%} | "
            f"{row['predicted_season_ppr_mean']:.0f} | "
            f"{row['predicted_season_ppr_p10']:.0f}"
            f"-{row['predicted_season_ppr_p90']:.0f} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "- Bio data (height, weight) for this class comes from pre-draft",
        "  combine testing joined by name, not a shared ID (nflverse hasn't",
        "  assigned this class stable cross-source IDs yet); the match rate",
        "  is reported at fetch time by `scripts/fetch_rookie_class.py`.",
        "- Age at draft is taken directly from the draft-picks feed (no",
        "  birth_date is published for a class this new); computed the same",
        "  way (age at the ~April draft) as the historical feature, so the",
        "  two sources are consistent.",
        "- The games-played conversion is a position-level historical",
        "  average, not a player-specific projection; see above.",
        "- Incumbent-context features (established starter, recent",
        "  extension, prior-year starting QB production) use the "
        f"{class_year - 1} season, the same leakage-safe convention as the",
        "  historical training data.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def build_rookie_bayes_outputs(
    project_root: str | Path | None = None,
    save_outputs: bool = True,
    validation_years: list[int] | None = None,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Build rolling-validation rookie projection artifacts.

    Imports PyMC inside ``fit_rookie_model``; this top-level entry point
    therefore only succeeds in a venv with PyMC available. The data-prep
    helpers above run in any env.
    """
    root = (
        find_project_root() if project_root is None else Path(project_root).resolve()
    )
    dirs = ensure_project_dirs(root)
    output_dir = dirs["tables"]

    rosters = pd.read_csv(
        root / "data" / "raw" / "rosters_2016_2025.csv", low_memory=False
    )
    player_stats = pd.read_csv(
        root / "data" / "raw" / "player_stats_2016_2025.csv", low_memory=False
    )
    modeling_df = build_rookie_modeling_frame(rosters, player_stats, project_root=root)

    preds, metrics = rolling_rookie_validation(
        modeling_df, validation_years=validation_years, **fit_kwargs
    )

    summary_text = _build_summary_text(modeling_df, preds, metrics)

    outputs = {
        "modeling_frame": modeling_df,
        "validation_predictions": preds,
        "validation_metrics": metrics,
        "summary_text": summary_text,
    }

    if save_outputs:
        modeling_df.to_csv(
            output_dir / "rookie_modeling_frame.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        if not preds.empty:
            preds.to_csv(
                output_dir / "rookie_bayes_validation_predictions.csv",
                index=False,
                float_format=CSV_FLOAT_FORMAT,
            )
        if not metrics.empty:
            metrics.to_csv(
                output_dir / "rookie_bayes_validation_metrics.csv",
                index=False,
                float_format=CSV_FLOAT_FORMAT,
            )
        (root / "report" / "rookie_bayes_projection.md").write_text(summary_text)

    return outputs


def build_rookie_hurdle_outputs(
    project_root: str | Path | None = None,
    save_outputs: bool = True,
    validation_years: list[int] | None = None,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Build hurdle-model artifacts: P(plays) * E[PPR/game | plays] projections.

    Two coupled PyMC models. Stage 1 is logistic; stage 2 is the existing
    Normal regression but trained only on players who played at least
    MIN_GAMES_FOR_TARGET games. The combined projection answers the right
    question: what is this rookie's expected season-long fantasy production
    in their rookie year, accounting for the fact that some players sit
    behind veterans?

    Same dedicated-venv requirements as the PPR/game-only model.
    """
    root = (
        find_project_root() if project_root is None else Path(project_root).resolve()
    )
    dirs = ensure_project_dirs(root)
    output_dir = dirs["tables"]

    rosters = pd.read_csv(
        root / "data" / "raw" / "rosters_2016_2025.csv", low_memory=False
    )
    player_stats = pd.read_csv(
        root / "data" / "raw" / "player_stats_2016_2025.csv", low_memory=False
    )
    modeling_df = build_rookie_modeling_frame(rosters, player_stats, project_root=root)

    preds, metrics = rolling_hurdle_validation(
        modeling_df, validation_years=validation_years, **fit_kwargs
    )

    summary_text = _build_hurdle_summary_text(modeling_df, preds, metrics)

    outputs = {
        "modeling_frame": modeling_df,
        "hurdle_predictions": preds,
        "hurdle_metrics": metrics,
        "summary_text": summary_text,
    }

    if save_outputs:
        modeling_df.to_csv(
            output_dir / "rookie_modeling_frame.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        if not preds.empty:
            preds.to_csv(
                output_dir / "rookie_hurdle_validation_predictions.csv",
                index=False,
                float_format=CSV_FLOAT_FORMAT,
            )
        if not metrics.empty:
            metrics.to_csv(
                output_dir / "rookie_hurdle_validation_metrics.csv",
                index=False,
                float_format=CSV_FLOAT_FORMAT,
            )
        (root / "report" / "rookie_hurdle_projection.md").write_text(summary_text)

    return outputs


def _build_hurdle_summary_text(
    modeling_df: pd.DataFrame,
    preds: pd.DataFrame,
    metrics: pd.DataFrame,
) -> str:
    n_total = len(modeling_df)
    n_played = int(modeling_df.get("played_meaningfully", pd.Series([0])).sum())

    lines = [
        "# Rookie Projection (Hurdle Model)",
        "",
        "Two coupled Bayesian models give a calibrated projection of a rookie's",
        "expected season-long fantasy production. The original PPR/game-only",
        "model silently dropped rookies who didn't play enough games — Jordan",
        "Love in 2020 (behind Rodgers) ended up with a NaN target and was",
        "never learned from. This version handles those cases explicitly.",
        "",
        "**Stage 1** (logistic). P(plays >= 4 games in rookie year) given draft",
        "slot, age, height, weight, position. Trained on every rookie in the",
        "frame, regardless of whether they ended up playing.",
        "",
        "**Stage 2** (Normal). PPR/game *conditional on having played*, with",
        "the same features. Trained only on rookies who cleared the 4-game",
        "threshold.",
        "",
        "**Combined projection**: P(plays) * E[PPR/game | plays]. For Jordan",
        "Love at draft time, this answers: low expected rookie-year production",
        "because he was unlikely to play, not because he was projected as a",
        "bad player.",
        "",
        f"{n_total:,} rookie player-seasons in the frame; {n_played:,} cleared",
        f"the 4-game threshold ({n_played / max(n_total, 1):.0%}).",
        "",
    ]

    if metrics.empty:
        lines.extend(
            [
                "## Validation",
                "",
                "PyMC sampling pass has not been run. From a dedicated venv:",
                "",
                "    python -c \"from src.rookie_bayes import build_rookie_hurdle_outputs; build_rookie_hurdle_outputs()\"",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.append("## Rolling-origin validation against the full target")
    lines.append("")
    lines.append(
        "The 'full target' is rookie-year PPR/game with non-players treated as"
    )
    lines.append("zero rather than dropped from the evaluation.")
    lines.append("")
    lines.append(
        "| Rookie class | n | Played n | RMSE | MAE | 50% cov | 80% cov | "
        "Plays Brier | Plays actual | Plays pred |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for _, row in metrics.iterrows():
        lines.append(
            f"| {int(row['validation_year'])} | {int(row['n_rookies'])} | "
            f"{int(row['n_played_meaningfully'])} | "
            f"{row['rmse_full_target']:.2f} | {row['mae_full_target']:.2f} | "
            f"{row['interval_50_coverage_full']:.0%} | "
            f"{row['interval_80_coverage_full']:.0%} | "
            f"{row['plays_brier']:.3f} | "
            f"{row['plays_actual_rate']:.0%} | "
            f"{row['plays_predicted_rate']:.0%} |"
        )

    lines.append("")
    lines.append("## Spot-check: Jordan Love")
    lines.append("")
    love_rows = preds[preds["player_display_name"].astype(str).str.contains(
        "Jordan Love", case=False, na=False
    )]
    if not love_rows.empty:
        love = love_rows.iloc[0]
        lines.append(
            f"- P(plays >= 4 games as a rookie): "
            f"{love['predicted_p_plays_meaningfully']:.2f}"
        )
        lines.append(
            f"- Conditional E[PPR/game if plays]: "
            f"{love['predicted_ppr_per_game_if_plays_mean']:.1f}"
        )
        lines.append(
            f"- Combined expected rookie-year PPR/game: "
            f"{love['predicted_rookie_year_ppr_per_game_mean']:.1f}"
        )
        lines.append(
            f"- 80% interval: "
            f"{love['predicted_rookie_year_ppr_per_game_p10']:.1f} "
            f"to "
            f"{love['predicted_rookie_year_ppr_per_game_p90']:.1f}"
        )
        games = love.get("games_played", 0)
        games_int = int(games) if pd.notna(games) else 0
        lines.append(
            f"- Actual rookie year ({int(love['rookie_year'])}): "
            f"{love['rookie_year_ppr_per_game_full']:.1f} PPR/game "
            f"({games_int} games)"
        )
    else:
        lines.append(
            "Jordan Love is not in the rookie modeling frame (rookie year"
            " outside the validation window)."
        )
    return "\n".join(lines) + "\n"


def _build_summary_text(
    modeling_df: pd.DataFrame,
    preds: pd.DataFrame,
    metrics: pd.DataFrame,
) -> str:
    lines = [
        "# Bayesian Rookie Projection",
        "",
        "Hierarchical Normal regression on rookie-season PPR per game, partial-",
        "pooled across the four skill positions. Solves the cold-start problem",
        "the existing HGB stack cannot: projecting a rookie before they have any",
        "NFL snaps. Features: log draft pick, age at draft, height, weight (and",
        "an optional college-production score). The model class is the Tier 2 #4",
        "deliverable from `PORTFOLIO_ROADMAP.md`.",
        "",
        "PyMC is isolated in `src/rookie_bayes.fit_rookie_model` because its ABI",
        "conflicts with the rest of the project's pins. Run this section in a",
        "dedicated venv built from `requirements-bayes.txt`.",
        "",
        f"Rookie modeling rows: {len(modeling_df):,}",
        "",
    ]

    if metrics.empty:
        lines.extend(
            [
                "## Validation",
                "",
                "Rolling-origin validation has not been executed yet — install PyMC",
                "via `requirements-bayes.txt` and run",
                "`python -c \"from src.rookie_bayes import build_rookie_bayes_outputs; build_rookie_bayes_outputs()\"`.",
                "",
            ]
        )
    else:
        lines.append("## Validation (rolling-origin by rookie class)")
        lines.append("")
        lines.append(
            "| Rookie class | n | RMSE | MAE | Bias | 50% coverage | 80% coverage |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for _, row in metrics.iterrows():
            lines.append(
                f"| {int(row['validation_year'])} | {int(row['n_rookies']):,} | "
                f"{row['rmse']:.3f} | {row['mae']:.3f} | {row['bias']:+.3f} | "
                f"{row['interval_50_coverage']:.3f} | {row['interval_80_coverage']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Honest caveats",
            "",
            "- College production is not yet wired in. The current model uses draft",
            "  capital and physical features only. Adding a college-translation",
            "  score is the obvious next data-acquisition step (see",
            "  `scripts/fetch_college_production.py` for the stub).",
            "- The Normal likelihood will under-cover for elite rookies (right-",
            "  tailed PPR distributions). A Student-T likelihood is a one-line",
            "  upgrade if calibration looks off.",
            "- Position-specific submodels could plausibly help QB the most (small",
            "  sample, very different production scale). Partial pooling is the",
            "  bet that shared structure outweighs that.",
        ]
    )
    return "\n".join(lines) + "\n"
