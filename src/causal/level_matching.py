"""Pre-period PPR-level matching for the control panel.

The session-1 parallel-trends check failed because the broad control universe
included receivers on systematically higher-baseline offenses (mean control
PPR ≈ 11.5 vs treated ≈ 9.7). Those high-baseline controls have different
trajectory dynamics than the treated WRs and are not a credible
counterfactual.

This module restricts controls per event to receivers whose own pre-period
PPR average falls within ``half_width`` of the *event-specific* treated
pre-period mean. The result is a level-matched panel that asks "what did
WRs with comparable pre-period production do during the same calendar
weeks?" rather than "what did the average WR do?"

Two convenience implementations:

- ``apply_level_matching`` — given an existing panel and the event-specific
  treated pre-period means, returns a trimmed panel.
- ``compute_treated_pre_means`` — computes those event-specific treated means
  from the panel itself, so the workflow is single-call clean.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_HALF_WIDTH_PPR = 3.0


def compute_treated_pre_means(panel: pd.DataFrame) -> pd.DataFrame:
    """Per event, mean pre-period PPR across the affected receivers.

    Returned columns: ``event_id, treated_pre_period_mean_ppr,
    n_treated_observations``.
    """
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "treated_pre_period_mean_ppr",
                "n_treated_observations",
            ]
        )
    treated_pre = panel[
        panel["role"].eq("treated") & panel["period"].eq("pre")
    ]
    if treated_pre.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "treated_pre_period_mean_ppr",
                "n_treated_observations",
            ]
        )
    grouped = (
        treated_pre.groupby("event_id", as_index=False)
        .agg(
            treated_pre_period_mean_ppr=("fantasy_points_ppr", "mean"),
            n_treated_observations=("fantasy_points_ppr", "count"),
        )
    )
    return grouped


def apply_level_matching(
    panel: pd.DataFrame,
    treated_pre_means: pd.DataFrame,
    *,
    half_width_ppr: float = DEFAULT_HALF_WIDTH_PPR,
) -> pd.DataFrame:
    """Keep only control receivers whose pre-period PPR is within ``half_width``
    of the event's treated pre-period mean.

    Treated rows are kept as-is; only the control set is trimmed.
    """
    if panel.empty:
        return panel

    control_pre = panel[
        panel["role"].eq("control") & panel["period"].eq("pre")
    ]
    control_pre_means = (
        control_pre.groupby(["event_id", "player_id"], as_index=False)["fantasy_points_ppr"]
        .mean()
        .rename(columns={"fantasy_points_ppr": "control_pre_period_mean_ppr"})
    )
    enriched = control_pre_means.merge(
        treated_pre_means[["event_id", "treated_pre_period_mean_ppr"]],
        on="event_id",
        how="left",
    )
    enriched["abs_gap"] = (
        enriched["control_pre_period_mean_ppr"]
        - enriched["treated_pre_period_mean_ppr"]
    ).abs()
    eligible = enriched[enriched["abs_gap"].le(half_width_ppr)][
        ["event_id", "player_id"]
    ]

    treated_panel = panel[panel["role"].eq("treated")].copy()
    control_panel = panel[panel["role"].eq("control")].merge(
        eligible, on=["event_id", "player_id"], how="inner"
    )

    matched = pd.concat([treated_panel, control_panel], ignore_index=True)
    # Re-enforce pre/post balance — a control that previously had both pre and
    # post observations might still pass the filter, but check anyway.
    coverage = (
        matched.groupby(["event_id", "role", "player_id", "period"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    coverage["has_pre"] = coverage.get("pre", 0).gt(0)
    coverage["has_post"] = coverage.get("post", 0).gt(0)
    balanced = coverage[coverage["has_pre"] & coverage["has_post"]][
        ["event_id", "role", "player_id"]
    ]
    matched = matched.merge(balanced, on=["event_id", "role", "player_id"], how="inner")
    return matched.reset_index(drop=True)


def summarize_level_matching(
    pre_matching_panel: pd.DataFrame,
    matched_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Diagnostic table: how many controls were retained per event."""
    if pre_matching_panel.empty:
        return pd.DataFrame()

    def _stats(panel: pd.DataFrame) -> pd.DataFrame:
        ctrl = panel[panel["role"].eq("control")]
        if ctrl.empty:
            return pd.DataFrame(
                columns=["event_id", "n_control_players", "n_control_observations"]
            )
        return (
            ctrl.groupby("event_id", as_index=False)
            .agg(
                n_control_players=("player_id", "nunique"),
                n_control_observations=("fantasy_points_ppr", "count"),
            )
        )

    pre = _stats(pre_matching_panel).rename(
        columns={
            "n_control_players": "n_control_players_unmatched",
            "n_control_observations": "n_control_obs_unmatched",
        }
    )
    post = _stats(matched_panel).rename(
        columns={
            "n_control_players": "n_control_players_matched",
            "n_control_observations": "n_control_obs_matched",
        }
    )
    merged = pre.merge(post, on="event_id", how="left").fillna(0)
    merged["retention_rate"] = (
        merged["n_control_players_matched"]
        / merged["n_control_players_unmatched"].clip(lower=1)
    )
    return merged
