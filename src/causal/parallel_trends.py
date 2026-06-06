"""Pre-period parallel-trends check for the QB-injury DiD design.

The DiD estimator (session 2) is only as good as its identification
assumptions, and parallel trends is the central one: treated and control
receivers must have followed parallel PPR trajectories in the pre-period.
This module produces the visual and statistical evidence for that
assumption.

Two specifications are computed:

1. **Visual**: mean PPR/game by ``week_offset`` for treated vs control,
   plotted side-by-side over the pre-period. A reader can eyeball whether
   the trajectories are parallel.

2. **Statistical**: a regression in the pre-period only of PPR on
   ``treated × week_offset`` interactions with player fixed effects. If
   any interaction coefficient is statistically distinguishable from zero
   (p < 0.05), parallel trends has failed and the design needs to be
   rethought before estimation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRE_PERIOD_OFFSETS = (-4, -3, -2, -1)


# ---------------------------------------------------------------------------
# Visual evidence: per-week means
# ---------------------------------------------------------------------------
def compute_pre_period_means(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean / SE of PPR by (role, week_offset) within the pre-period.

    Returned columns: ``role, week_offset, n_observations, mean_ppr,
    se_ppr``.
    """
    if panel.empty:
        return pd.DataFrame()
    pre = panel[panel["period"].eq("pre")].copy()
    agg = (
        pre.groupby(["role", "week_offset"], as_index=False)
        .agg(
            n_observations=("fantasy_points_ppr", "count"),
            mean_ppr=("fantasy_points_ppr", "mean"),
            std_ppr=("fantasy_points_ppr", "std"),
        )
    )
    agg["se_ppr"] = agg["std_ppr"] / np.sqrt(agg["n_observations"].clip(lower=1))
    return agg[["role", "week_offset", "n_observations", "mean_ppr", "se_ppr"]]


def plot_parallel_trends(
    means: pd.DataFrame,
    out_path: Path,
    title: str = "Pre-period parallel-trends check (WR PPR by week relative to QB injury)",
) -> None:
    """Render the parallel-trends plot."""
    if means.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {"treated": "#C8553D", "control": "#3D6B99"}

    for role, group in means.groupby("role"):
        x = group["week_offset"].to_numpy()
        y = group["mean_ppr"].to_numpy()
        yerr = group["se_ppr"].to_numpy()
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=f"{role} (n_obs={int(group['n_observations'].sum()):,})",
            color=palette.get(role, "#157A6E"),
            marker="o",
            capsize=4,
            linewidth=2,
        )

    ax.set_xlabel("Week relative to QB injury (transition_week = 0)")
    ax.set_ylabel("Mean PPR per game")
    ax.set_title(title)
    ax.axvline(-0.5, color="grey", linestyle="--", alpha=0.6, label="treatment boundary")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistical evidence: pre-period interaction coefficients
# ---------------------------------------------------------------------------
def fit_pre_period_interaction_check(panel: pd.DataFrame) -> pd.DataFrame:
    """Regress pre-period PPR on `treated × week_offset` dummies.

    Implements a within-player demeaned OLS (player fixed effects) so we
    measure deviations from each player's own pre-period mean, not raw
    levels. The hypothesis we want to NOT reject: every pre-period
    `treated × week_offset` interaction equals zero.
    """
    if panel.empty:
        return pd.DataFrame()
    pre = panel[panel["period"].eq("pre")].copy()
    pre = pre.dropna(subset=["fantasy_points_ppr"])

    # Demean PPR within each player to absorb player-level baselines.
    pre["ppr_demeaned"] = pre["fantasy_points_ppr"] - pre.groupby("player_id")[
        "fantasy_points_ppr"
    ].transform("mean")

    pre["treated"] = pre["role"].eq("treated").astype(int)
    # We omit week_offset == -1 as the reference category and report the
    # interaction at offsets -4, -3, -2.
    offsets = sorted(o for o in pre["week_offset"].unique() if o != -1)
    rows: list[dict] = []
    for offset in offsets:
        # Subset: rows at offset == -1 (reference) plus rows at this offset.
        sub = pre[pre["week_offset"].isin([-1, offset])].copy()
        if sub.empty:
            continue
        sub["is_offset"] = sub["week_offset"].eq(offset).astype(int)
        sub["interaction"] = sub["treated"] * sub["is_offset"]
        # OLS via least squares with intercept, treated, is_offset, interaction.
        design = np.column_stack(
            [
                np.ones(len(sub)),
                sub["treated"].to_numpy(),
                sub["is_offset"].to_numpy(),
                sub["interaction"].to_numpy(),
            ]
        )
        y = sub["ppr_demeaned"].to_numpy()
        beta, residuals_arr, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ beta
        n = len(sub)
        if n - rank <= 0:
            continue
        sigma2 = float(np.sum(residuals**2) / (n - rank))
        try:
            covariance = np.linalg.inv(design.T @ design) * sigma2
            se = float(np.sqrt(covariance[3, 3]))
        except np.linalg.LinAlgError:
            se = float("nan")
        interaction_coef = float(beta[3])
        t_stat = interaction_coef / se if se > 0 else float("nan")
        rows.append(
            {
                "week_offset_vs_reference_minus1": int(offset),
                "n_observations": int(n),
                "interaction_coefficient": interaction_coef,
                "interaction_se": se,
                "t_stat": t_stat,
                # Two-sided p-value approximation using Gaussian tails;
                # acceptable given n >> 30.
                "p_value_approx": _normal_two_sided_pvalue(t_stat),
            }
        )
    return pd.DataFrame(rows)


def _normal_two_sided_pvalue(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return float("nan")
    # Normal CDF via erfc.
    from math import erfc, sqrt

    return float(erfc(abs(t_stat) / sqrt(2.0)))


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def build_parallel_trends_artifacts(
    panel: pd.DataFrame,
    project_root: Path,
) -> dict[str, pd.DataFrame]:
    """Produce the parallel-trends table + figure and return both."""
    means = compute_pre_period_means(panel)
    pretrend_coefs = fit_pre_period_interaction_check(panel)
    plot_path = project_root / "outputs" / "figures" / "causal_qb_injury_parallel_trends.png"
    plot_parallel_trends(means, plot_path)
    return {
        "pre_period_means": means,
        "pretrend_coefficients": pretrend_coefs,
        "plot_path": plot_path,
    }
