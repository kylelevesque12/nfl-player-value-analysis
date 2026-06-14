"""Replacement-level surplus analysis for the salary track.

Front offices do not think about contracts in absolute dollars — they think
about *surplus over a freely available alternative*. A WR projected for 3 PPR
per game costing $20M is overpaid not because $20M is large in the abstract,
but because a veteran-minimum WR can be expected to deliver ~2 PPR per game
for ~$1M. The cap should be allocated to players who deliver value the
replacement market cannot.

This module adds the replacement-level framing on top of the existing salary
efficiency table. For each (season, position) we estimate two baselines:

  * ``replacement_salary_millions`` — the cap cost of a typical bottom-quartile
    veteran starter at that position. Approximates the price of "next man up."
  * ``replacement_value_score`` — the typical value the same bottom-quartile
    player provides.

For each player-season we then compute:

  * ``cap_over_replacement_millions`` — the premium they cost above replacement.
  * ``value_over_replacement`` — the standardized value they delivered above
    replacement.
  * ``dollar_surplus_millions`` — value over replacement converted into dollars
    via the within-(position, season) slope of salary on value, minus the cap
    premium. Positive surplus means the player out-earned their contract; the
    larger the number, the bigger the deal for the team.

Cap-cost variable: as of Session 4 the underlying ``salary_millions`` is a
season-specific cap hit *reconstructed* from contract terms (prorated signing
bonus + backloaded base; see ``src/cap_hit_reconstruction.py``), replacing the
flat ``inflated_apy`` proxy. It is an estimate, not a parsed cap hit — the
source contract data has no year-by-year cap accounting — so each player-season
carries a ``cap_hit_quality_flag``. This is materially more honest than flat APY
(early-extension years are correctly cheaper than late years), but precision is
still bounded by what the contract terms support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REPLACEMENT_SALARY_PERCENTILE = 0.25  # define replacement as bottom-quartile cost
MIN_REPLACEMENT_SAMPLE = 5  # require at least this many players per position-season
MIN_PRICE_REGRESSION_SAMPLE = 10  # for the per-position-season value-to-$ slope
PREMIUM_FLOOR_MILLIONS = 0.5  # avoid divide-by-zero when cap_over_replacement ~ 0


def compute_replacement_baselines(finding_base: pd.DataFrame) -> pd.DataFrame:
    """Estimate replacement-level salary and value per (season, position)."""
    df = finding_base.copy()
    replacement_pool = df[
        df["salary_percentile"].le(REPLACEMENT_SALARY_PERCENTILE)
    ]
    grouped = replacement_pool.groupby(["season", "position"], as_index=False)
    baselines = grouped.agg(
        replacement_salary_millions=("salary_millions", "median"),
        replacement_value_score=("value_score", "median"),
        replacement_sample_size=("player_id", "count"),
    )
    baselines = baselines[
        baselines["replacement_sample_size"].ge(MIN_REPLACEMENT_SAMPLE)
    ].reset_index(drop=True)
    return baselines


def _position_season_price_per_value_unit(finding_base: pd.DataFrame) -> pd.DataFrame:
    """Per (season, position), slope of ``salary_millions`` on ``value_score``.

    This is the implicit market price of one z-unit of value at that position
    in that season. We use the slope to convert value-over-replacement into a
    dollar-equivalent. Computed on player-seasons where the player had a real
    role (filtered upstream via ``finding_base``).
    """
    rows: list[dict[str, float | str | int]] = []
    for (season, position), group in finding_base.groupby(["season", "position"]):
        clean = group.dropna(subset=["salary_millions", "value_score"])
        if len(clean) < MIN_PRICE_REGRESSION_SAMPLE:
            continue
        y = clean["salary_millions"].to_numpy()
        x = clean["value_score"].to_numpy()
        x_centered = x - x.mean()
        denom = float(np.sum(x_centered ** 2))
        if denom <= 0:
            continue
        slope = float(np.sum(x_centered * (y - y.mean())) / denom)
        rows.append(
            {
                "season": int(season),
                "position": position,
                "price_per_value_unit_millions": slope,
                "price_regression_n": int(len(clean)),
            }
        )
    return pd.DataFrame(rows)


def compute_replacement_level_surplus(
    finding_base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add replacement-level columns to the finding base.

    Returns (enriched_base, replacement_baselines, position_season_prices).
    """
    baselines = compute_replacement_baselines(finding_base)
    prices = _position_season_price_per_value_unit(finding_base)

    enriched = finding_base.merge(
        baselines[
            [
                "season",
                "position",
                "replacement_salary_millions",
                "replacement_value_score",
            ]
        ],
        on=["season", "position"],
        how="left",
    ).merge(
        prices[["season", "position", "price_per_value_unit_millions"]],
        on=["season", "position"],
        how="left",
    )

    enriched["cap_over_replacement_millions"] = (
        enriched["salary_millions"] - enriched["replacement_salary_millions"]
    )
    enriched["value_over_replacement"] = (
        enriched["value_score"] - enriched["replacement_value_score"]
    )

    # Dollarize the above-replacement value via the position-season price.
    enriched["value_over_replacement_dollar_equivalent_millions"] = (
        enriched["value_over_replacement"] * enriched["price_per_value_unit_millions"]
    )
    enriched["dollar_surplus_millions"] = (
        enriched["value_over_replacement_dollar_equivalent_millions"]
        - enriched["cap_over_replacement_millions"]
    )

    # An auxiliary ratio metric useful for tiering within a position-season.
    denom = enriched["cap_over_replacement_millions"].clip(
        lower=PREMIUM_FLOOR_MILLIONS
    )
    enriched["value_per_premium_million"] = (
        enriched["value_over_replacement"] / denom
    )

    return enriched, baselines, prices


def summarize_replacement_level_by_team_season(
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    """Team-season totals of cap premium and dollar surplus over replacement."""
    cols = [
        "cap_over_replacement_millions",
        "value_over_replacement",
        "dollar_surplus_millions",
    ]
    available = [c for c in cols if c in enriched.columns]
    if not available:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    for (season, team), group in enriched.groupby(["season", "team"]):
        rows.append(
            {
                "season": int(season),
                "team": team,
                "player_seasons": int(len(group)),
                "total_cap_over_replacement_millions": float(
                    group["cap_over_replacement_millions"].sum(skipna=True)
                ),
                "total_value_over_replacement": float(
                    group["value_over_replacement"].sum(skipna=True)
                ),
                "total_dollar_surplus_millions": float(
                    group["dollar_surplus_millions"].sum(skipna=True)
                ),
                "mean_dollar_surplus_millions": float(
                    group["dollar_surplus_millions"].mean(skipna=True)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "total_dollar_surplus_millions", ascending=False
    ).reset_index(drop=True)


def summarize_replacement_level_by_position(
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    """Average surplus and price per value unit, grouped by position."""
    if enriched.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | str | int]] = []
    for position, group in enriched.groupby("position"):
        rows.append(
            {
                "position": position,
                "player_seasons": int(len(group)),
                "median_replacement_salary_millions": float(
                    group["replacement_salary_millions"].median(skipna=True)
                ),
                "median_replacement_value_score": float(
                    group["replacement_value_score"].median(skipna=True)
                ),
                "median_price_per_value_unit_millions": float(
                    group["price_per_value_unit_millions"].median(skipna=True)
                ),
                "median_dollar_surplus_millions": float(
                    group["dollar_surplus_millions"].median(skipna=True)
                ),
                "share_positive_surplus": float(
                    (group["dollar_surplus_millions"] > 0).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
