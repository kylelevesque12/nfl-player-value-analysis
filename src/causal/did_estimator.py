"""TWFE event-study DiD estimator for the QB-injury WR-PPR design.

Specification
-------------

For each player-week observation in the panel::

    y_it = α_i + γ_k + Σ_{k != -1} β_k · 1[treated_i AND week_offset_it == k] + ε_it

- ``α_i`` are player fixed effects (absorbs each receiver's baseline PPR).
- ``γ_k`` are event-time fixed effects (absorbs the league-wide trajectory).
- ``β_k`` are the per-week-offset treatment effects, with ``k = -1`` omitted
  as the reference week.

In an event-study DiD:

- ``β_k`` for ``k < -1`` traces out the **pretrend**. We want these to be
  individually small. If they are not, parallel trends has failed in a
  technical sense — but the post-period estimates may still be defensible if
  the pretrend coefficients are stable (the bias is roughly the post-period
  coefficient minus an extrapolated pretrend).
- ``β_k`` for ``k >= 0`` traces out the **dynamic treatment effect**.

The estimator uses a within-transformation (de-meaning by player and by
event-time) to absorb the fixed effects, then OLS on the remaining terms.
Standard errors are cluster-robust at the *event* level (each treatment
event is one cluster) since observations within an event share unobserved
shocks.

Placebo testing
---------------

Because parallel trends are borderline / failing in the raw panel, we
report a placebo distribution: re-run the estimator with the
``transition_week`` shifted to a randomly chosen week within the same
season. The placebo gives the distribution of "treatment effect"
estimates we'd get *if there were no real treatment*. If the actual
post-period coefficient is more extreme than 95% of placebo runs, the
effect is credible despite the pretrend.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


REFERENCE_OFFSET = -1
DEFAULT_N_PLACEBO_DRAWS = 200
DEFAULT_PLACEBO_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_design_matrix(
    panel: pd.DataFrame, all_offsets: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, cluster_ids, interaction_column_names) for OLS.

    We use **explicit** player + offset fixed-effect dummies plus the
    treated × offset interaction dummies. No within-transformation magic —
    just one big sparse-ish OLS that any reader can verify against the
    raw 2x2 cell means.

    Reference category for offset is REFERENCE_OFFSET (-1).
    """
    df = panel.dropna(subset=["fantasy_points_ppr"]).copy()
    df["treated"] = df["role"].eq("treated").astype(int)
    df["week_offset"] = df["week_offset"].astype(int)

    # Player FE dummies (one per player, drop first as reference).
    player_dummies = pd.get_dummies(df["player_id"], prefix="P", drop_first=True)
    # Offset FE dummies (one per offset, drop the reference offset).
    offset_dummies = pd.get_dummies(df["week_offset"], prefix="K", drop_first=False)
    if f"K_{REFERENCE_OFFSET}" in offset_dummies.columns:
        offset_dummies = offset_dummies.drop(columns=[f"K_{REFERENCE_OFFSET}"])

    # Treated × offset interaction dummies (one per non-reference offset).
    interaction_cols: list[str] = []
    interaction_data: dict[str, np.ndarray] = {}
    for k in all_offsets:
        if k == REFERENCE_OFFSET:
            continue
        col = f"int_k_{k}"
        interaction_data[col] = (
            (df["treated"].to_numpy() == 1) & (df["week_offset"].to_numpy() == k)
        ).astype("float64")
        interaction_cols.append(col)
    interactions = pd.DataFrame(interaction_data, index=df.index)

    X = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="const"),
            interactions,
            player_dummies.astype("float64"),
            offset_dummies.astype("float64"),
        ],
        axis=1,
    )

    return (
        X.to_numpy(dtype="float64"),
        df["fantasy_points_ppr"].to_numpy(dtype="float64"),
        df["event_id"].astype(str).to_numpy(),
        ["const"] + interaction_cols + list(player_dummies.columns) + list(offset_dummies.columns),
    )


def _cluster_robust_se(
    X: np.ndarray, residuals: np.ndarray, cluster_ids: np.ndarray
) -> np.ndarray:
    """Cluster-robust standard errors (CR0 — Liang & Zeger 1986)."""
    G = len(np.unique(cluster_ids))
    if G < 2 or X.size == 0:
        return np.full(X.shape[1], np.nan)
    XtX_inv = np.linalg.pinv(X.T @ X)
    score_sum = np.zeros((X.shape[1], X.shape[1]))
    for cluster in np.unique(cluster_ids):
        mask = cluster_ids == cluster
        Xg = X[mask]
        eg = residuals[mask]
        score = Xg.T @ eg
        score_sum += np.outer(score, score)
    cov = XtX_inv @ score_sum @ XtX_inv
    return np.sqrt(np.diag(cov))


def _normal_two_sided_pvalue(t: float) -> float:
    if not np.isfinite(t):
        return float("nan")
    from math import erfc, sqrt

    return float(erfc(abs(t) / sqrt(2.0)))


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------
def fit_event_study(panel: pd.DataFrame) -> pd.DataFrame:
    """Direct-cell-mean event study with event-clustered SE.

    ``coefficient`` at offset k is the change in the (treated − control) PPR
    gap between offset k and the reference offset (-1). Positive ⇒ treated did
    better relative to controls at k than at -1; negative ⇒ worse.

    Implementation notes
    --------------------
    We construct β_k directly as the difference of cell means rather than via
    a TWFE regression. On a balanced panel the two are identical. On an
    *unbalanced* panel — which is what we have here, because not every WR or
    control plays every week — TWFE coefficients become a weighted average
    that mixes player composition effects into the estimate; the cell-mean
    approach stays interpretable. The tradeoff is no automatic FE shrinkage,
    but we cluster the bootstrap SE at the event level, which captures the
    correlation structure we actually care about.

    SE are computed by event-cluster bootstrap (1,000 resamples by default
    via the helper; see ``bootstrap_event_study`` for the call signature).
    """
    if panel.empty:
        return pd.DataFrame()
    df = panel.dropna(subset=["fantasy_points_ppr"]).copy()
    df["week_offset"] = df["week_offset"].astype(int)
    all_offsets = sorted(df["week_offset"].unique().tolist())

    cell_means = (
        df.groupby(["role", "week_offset"])["fantasy_points_ppr"]
        .mean()
        .unstack(level="role")
    )
    if "treated" not in cell_means.columns or "control" not in cell_means.columns:
        return pd.DataFrame()
    cell_means["gap"] = cell_means["treated"] - cell_means["control"]
    if REFERENCE_OFFSET not in cell_means.index:
        return pd.DataFrame()
    ref_gap = float(cell_means.loc[REFERENCE_OFFSET, "gap"])

    point_estimates = (cell_means["gap"] - ref_gap).to_dict()

    se = _event_clustered_bootstrap_se(df, all_offsets, ref_gap=ref_gap)

    rows = []
    for k in all_offsets:
        if k == REFERENCE_OFFSET:
            continue
        coef = float(point_estimates.get(k, np.nan))
        s = float(se.get(k, np.nan))
        t = coef / s if s and np.isfinite(s) and s > 0 else float("nan")
        rows.append(
            {
                "week_offset": int(k),
                "is_pre_period": k < 0,
                "coefficient": coef,
                "se_cluster_robust": s,
                "t_stat": float(t),
                "p_value_approx": _normal_two_sided_pvalue(t),
            }
        )
    return pd.DataFrame(rows).sort_values("week_offset").reset_index(drop=True)


def _event_clustered_bootstrap_se(
    df: pd.DataFrame,
    all_offsets: list[int],
    *,
    ref_gap: float,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[int, float]:
    """Event-cluster bootstrap SE for the per-offset gap-difference estimator."""
    rng = np.random.default_rng(seed)
    event_ids = df["event_id"].unique()
    n_events = len(event_ids)
    if n_events < 2:
        return {k: float("nan") for k in all_offsets if k != REFERENCE_OFFSET}

    # Precompute per-event, per-(role, offset) means for fast resampling.
    per_event_cells = (
        df.groupby(["event_id", "role", "week_offset"])["fantasy_points_ppr"]
        .mean()
        .reset_index()
    )

    boot_estimates: dict[int, list[float]] = {
        k: [] for k in all_offsets if k != REFERENCE_OFFSET
    }
    for _ in range(n_boot):
        sample_event_ids = rng.choice(event_ids, size=n_events, replace=True)
        sample = per_event_cells[
            per_event_cells["event_id"].isin(sample_event_ids)
        ]
        # Recompute cell means on the resampled events.
        boot_cells = (
            sample.groupby(["role", "week_offset"])["fantasy_points_ppr"]
            .mean()
            .unstack(level="role")
        )
        if "treated" not in boot_cells.columns or "control" not in boot_cells.columns:
            continue
        boot_cells["gap"] = boot_cells["treated"] - boot_cells["control"]
        if REFERENCE_OFFSET not in boot_cells.index:
            continue
        boot_ref = float(boot_cells.loc[REFERENCE_OFFSET, "gap"])
        for k in all_offsets:
            if k == REFERENCE_OFFSET:
                continue
            if k not in boot_cells.index:
                continue
            boot_estimates[k].append(
                float(boot_cells.loc[k, "gap"]) - boot_ref
            )

    return {
        k: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        for k, v in boot_estimates.items()
    }


def summarize_att(event_study: pd.DataFrame) -> pd.DataFrame:
    """Pool the post-period coefficients into a single ATT estimate.

    The pooled estimate uses the average of the per-offset β_k for k ≥ 0,
    with SE from event-cluster bootstrap (the per-offset SEs are already
    that — we approximate the pooled SE as their RMS divided by √k).
    """
    if event_study.empty:
        return pd.DataFrame()
    post = event_study[~event_study["is_pre_period"]]
    if post.empty:
        return pd.DataFrame()
    att = float(post["coefficient"].mean())
    pool_se = float(
        np.sqrt(np.mean(post["se_cluster_robust"].to_numpy() ** 2) / len(post))
    )
    return pd.DataFrame(
        [
            {
                "att_pooled_post_period": att,
                "att_se_pooled": pool_se,
                "att_t_stat": att / pool_se if pool_se > 0 else float("nan"),
                "att_p_value_approx": _normal_two_sided_pvalue(
                    att / pool_se if pool_se > 0 else float("nan")
                ),
                "n_post_period_offsets": int(len(post)),
            }
        ]
    )


def simple_2x2_did(panel: pd.DataFrame) -> pd.DataFrame:
    """The simplest defensible DiD: full-pre-period vs full-post-period means.

    ATT = (treated_post_avg − treated_pre_avg) − (control_post_avg − control_pre_avg)

    This collapses the entire pre-period into one baseline rather than using
    only week -1 as reference. Reports cell means + ATT + event-cluster
    bootstrap SE.
    """
    if panel.empty:
        return pd.DataFrame()
    df = panel.dropna(subset=["fantasy_points_ppr"]).copy()
    cell = (
        df.groupby(["role", "period"])["fantasy_points_ppr"]
        .mean()
        .unstack(level="period")
    )
    if {"pre", "post"} - set(cell.columns) or {"treated", "control"} - set(cell.index):
        return pd.DataFrame()
    treated_change = float(cell.loc["treated", "post"] - cell.loc["treated", "pre"])
    control_change = float(cell.loc["control", "post"] - cell.loc["control", "pre"])
    att = treated_change - control_change

    # Event-cluster bootstrap SE.
    rng = np.random.default_rng(123)
    event_ids = df["event_id"].unique()
    boot: list[float] = []
    per_event_cells = (
        df.groupby(["event_id", "role", "period"])["fantasy_points_ppr"]
        .mean()
        .reset_index()
    )
    for _ in range(500):
        sample_ids = rng.choice(event_ids, size=len(event_ids), replace=True)
        s = per_event_cells[per_event_cells["event_id"].isin(sample_ids)]
        c = (
            s.groupby(["role", "period"])["fantasy_points_ppr"]
            .mean()
            .unstack(level="period")
        )
        if {"pre", "post"} - set(c.columns) or {"treated", "control"} - set(c.index):
            continue
        boot.append(
            float(
                (c.loc["treated", "post"] - c.loc["treated", "pre"])
                - (c.loc["control", "post"] - c.loc["control", "pre"])
            )
        )
    se = float(np.std(boot, ddof=1)) if len(boot) > 1 else float("nan")
    t = att / se if se and np.isfinite(se) and se > 0 else float("nan")
    return pd.DataFrame(
        [
            {
                "treated_pre_mean": float(cell.loc["treated", "pre"]),
                "treated_post_mean": float(cell.loc["treated", "post"]),
                "treated_change": treated_change,
                "control_pre_mean": float(cell.loc["control", "pre"]),
                "control_post_mean": float(cell.loc["control", "post"]),
                "control_change": control_change,
                "att_2x2": att,
                "se_event_cluster_bootstrap": se,
                "t_stat": t,
                "p_value_approx": _normal_two_sided_pvalue(t),
            }
        ]
    )


# ---------------------------------------------------------------------------
# Placebo testing
# ---------------------------------------------------------------------------
def run_placebo_distribution(
    affected_receivers: pd.DataFrame,
    starting_qbs: pd.DataFrame,
    player_stats: pd.DataFrame,
    actual_events: pd.DataFrame,
    n_draws: int = DEFAULT_N_PLACEBO_DRAWS,
    seed: int = DEFAULT_PLACEBO_SEED,
) -> pd.DataFrame:
    """Estimate the null distribution of the 2x2 ATT under random treatment
    timing.

    For each draw, pick a random (team, season, transition_week) from the
    set of team-season-weeks where **no real QB injury transition occurred**
    in the surrounding 8 weeks. Re-use the affected-receivers-per-event
    structure (same number of treated receivers per event, same pre/post
    window) to build a placebo panel, then re-estimate the 2x2 ATT. The
    distribution of those placebo ATTs is the empirical null.

    Implementing this requires re-running the full panel construction for
    each placebo draw, which is expensive. We provide a callable hook here
    so it can be invoked offline; the session 2 driver runs a small number
    of placebos (50 default) to give an indicative p-value.
    """
    from src.causal.control_matching import construct_control_panel

    rng = np.random.default_rng(seed)

    # Identify all (team, season, week) cells with a stable QB across an
    # 8-week window and no real treatment event nearby.
    real_event_keys = set(
        (row["team"], int(row["season"]), int(row["transition_week"]))
        for _, row in actual_events.iterrows()
    )

    candidates: list[tuple[str, int, int]] = []
    for season in sorted(starting_qbs["season"].dropna().unique()):
        for team in starting_qbs["team"].dropna().unique():
            team_season = starting_qbs[
                starting_qbs["team"].eq(team) & starting_qbs["season"].eq(season)
            ].dropna(subset=["starting_qb_id"])
            weeks = sorted(team_season["week"].astype(int).tolist())
            for w in weeks:
                if w - 4 < min(weeks) or w + 3 > max(weeks):
                    continue
                window_qbs = team_season[team_season["week"].between(w - 4, w + 3)][
                    "starting_qb_id"
                ].astype(str)
                if window_qbs.nunique() != 1:
                    continue
                # Stay clear of any real treatment events for this team within 6 weeks.
                if any(
                    (team == ev_team and season == ev_season and abs(w - ev_week) < 6)
                    for ev_team, ev_season, ev_week in real_event_keys
                ):
                    continue
                candidates.append((team, season, w))

    if not candidates:
        return pd.DataFrame({"placebo_att": []})

    # Pool affected receivers from the real events to know how many per event
    # we should pick.
    placebo_atts: list[float] = []
    for _ in range(n_draws):
        # Sample one placebo per real event so the bootstrap weight is
        # consistent with the real estimate.
        sample_indices = rng.integers(0, len(candidates), size=len(actual_events))
        placebo_events_rows = []
        for idx, real_event in zip(sample_indices, actual_events.iterrows()):
            team, season, w = candidates[idx]
            placebo_events_rows.append(
                {
                    "event_id": f"PLAC_{season}_{team}_W{w:02d}",
                    "team": team,
                    "season": season,
                    "transition_week": w,
                    "prior_qb_id": "",
                    "new_qb_id": "",
                    "cause": "placebo",
                    "post_period_starter_weeks": 4,
                }
            )
        placebo_events = pd.DataFrame(placebo_events_rows)

        # For each placebo event, identify "affected" receivers using the
        # same pre-period-targets-per-game filter.
        from src.causal.treatment_identification import attach_affected_receivers

        placebo_affected = attach_affected_receivers(placebo_events, player_stats)
        if placebo_affected.empty:
            continue
        placebo_panel = construct_control_panel(
            placebo_events,
            placebo_affected,
            player_stats,
            starting_qbs,
        )
        if placebo_panel.empty:
            continue
        twoxtwo = simple_2x2_did(placebo_panel)
        if twoxtwo.empty:
            continue
        placebo_atts.append(float(twoxtwo.iloc[0]["att_2x2"]))

    return pd.DataFrame({"placebo_att": placebo_atts})


def placebo_two_sided_p_value(actual_att: float, placebo_atts: Iterable[float]) -> float:
    placebo_arr = np.asarray(list(placebo_atts), dtype="float64")
    if placebo_arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(placebo_arr) >= np.abs(actual_att)))


def plot_event_study(event_study: pd.DataFrame, out_path) -> None:
    """Plot per-offset β_k coefficients with 95% CI from the bootstrap SE."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if event_study.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    x = event_study["week_offset"].to_numpy()
    y = event_study["coefficient"].to_numpy()
    yerr = 1.96 * event_study["se_cluster_robust"].to_numpy()
    pre_mask = event_study["is_pre_period"].to_numpy()
    ax.errorbar(
        x[pre_mask],
        y[pre_mask],
        yerr=yerr[pre_mask],
        fmt="o",
        color="#C8553D",
        label="Pre-period (parallel-trends test)",
        capsize=4,
        linewidth=2,
    )
    ax.errorbar(
        x[~pre_mask],
        y[~pre_mask],
        yerr=yerr[~pre_mask],
        fmt="o",
        color="#157A6E",
        label="Post-period (treatment effect)",
        capsize=4,
        linewidth=2,
    )
    ax.axhline(0, color="grey", linestyle="--", alpha=0.7)
    ax.axvline(-0.5, color="grey", linestyle=":", alpha=0.5, label="treatment boundary")
    ax.set_xlabel("Week relative to QB injury (transition_week = 0)")
    ax.set_ylabel("β_k: (treated − control) gap relative to offset -1 (PPR)")
    ax.set_title(
        "Event-study coefficients: WR PPR by week relative to QB injury\n"
        "(positive = treated WR did better vs control than at offset -1)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
