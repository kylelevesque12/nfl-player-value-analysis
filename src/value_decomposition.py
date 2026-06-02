"""Decompose player value into efficiency and opportunity axes.

The project's headline metric, ``value_score``, is the within season-position
z-score of *total* EPA. Total EPA rewards opportunity (snaps, carries, targets)
as much as quality: a high-volume, average-efficiency player can outscore a
low-volume, highly efficient one. For front-office *talent* evaluation that
conflation is a problem, because two very different questions get one number:

1. *How good is this player per opportunity?*  (efficiency / ability)
2. *How much does this player get used?*        (opportunity / role)

This module separates them. For each player-season it computes:

- ``efficiency_raw``  = value EPA per opportunity (per dropback for QBs, per
  carry+target for skill players).
- ``opportunity_raw`` = opportunities per game (volume/role).

Each is then standardized *within season-position groups* (the same peer-set
logic the project already uses for ``value_score``) to give ``efficiency_z`` and
``opportunity_z``. Because total EPA is approximately efficiency times
opportunity, the existing total-value score decomposes cleanly into these two
interpretable axes.

It also builds talent-isolating *rate* features that describe how a player earns
production rather than how much: catch rate, yards per target, air yards per
target (aDOT), YAC per reception, RACR, yards per carry, completion percentage,
yards per attempt, and passing aDOT. These are reconstructed from the season
sum columns already present in the processed data, so no new raw pull is needed.

The module is intentionally dependency-light (pandas + numpy only) so it can run
and be tested without the scientific modeling stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config

GROUP_COLS = list(config.VALUE_GROUP_COLS)  # ["season", "position"]
MIN_OPPORTUNITY = 1.0  # guard against divide-by-zero on opportunity rates

# Efficiency on tiny samples is mostly noise (a player with one target and one
# catch can post an absurd yards-per-target). The efficiency axis is therefore
# only computed for player-seasons that clear a minimum opportunity load, and
# only those rows are used to estimate the group mean/std for standardization.
# QBs are measured per dropback (a different scale), so they get a higher floor.
MIN_EFFICIENCY_OPPORTUNITY = {"QB": 100.0, "RB": 50.0, "WR": 30.0, "TE": 20.0}
DEFAULT_MIN_EFFICIENCY_OPPORTUNITY = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric view of a column, zeros if missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype="float64")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def add_group_zscore(
    df: pd.DataFrame,
    value_col: str,
    z_col: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Standardize ``value_col`` within season-position groups."""
    if group_cols is None:
        group_cols = GROUP_COLS
    out = df.copy()
    grp = out.groupby(group_cols)[value_col]
    mean = grp.transform("mean")
    std = grp.transform("std")
    out[z_col] = (out[value_col] - mean) / std.replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# Opportunity / efficiency axes
# ---------------------------------------------------------------------------
def add_opportunity_and_efficiency(player_season: pd.DataFrame) -> pd.DataFrame:
    """Add raw opportunity, raw efficiency, and their standardized versions.

    Opportunity is position-appropriate:
      - QB:  dropbacks proxy = attempts + carries (``qb_plays``)
      - skill (RB/WR/TE): carries + targets (``scrimmage_touches`` proxy)

    Efficiency is value EPA divided by opportunity, where value EPA matches the
    project's definition (passing+rushing EPA for QBs, rushing+receiving EPA for
    skill players).
    """
    out = player_season.copy()
    games = _num(out, "games_played").replace(0, np.nan)

    is_qb = out["position"].eq("QB")

    qb_opportunity = _num(out, "qb_plays")
    if (qb_opportunity == 0).all():
        qb_opportunity = _num(out, "attempts") + _num(out, "carries")
    skill_opportunity = _num(out, "scrimmage_touches")
    if (skill_opportunity == 0).all():
        skill_opportunity = _num(out, "carries") + _num(out, "targets")

    out["opportunity_total"] = np.where(is_qb, qb_opportunity, skill_opportunity)
    out["opportunity_per_game"] = out["opportunity_total"] / games

    # value EPA already exists as value_epa_total; recompute defensively if not.
    if "value_epa_total" in out.columns:
        value_epa = _num(out, "value_epa_total")
    else:
        qb_epa = _num(out, "qb_epa")
        scrim_epa = _num(out, "scrimmage_epa")
        value_epa = np.where(is_qb, qb_epa, scrim_epa)
    out["value_epa_total"] = value_epa

    safe_opp = out["opportunity_total"].clip(lower=MIN_OPPORTUNITY)
    out["efficiency_per_opportunity"] = out["value_epa_total"] / safe_opp

    # Flag rows with enough volume for efficiency to be a meaningful signal.
    floor = out["position"].map(MIN_EFFICIENCY_OPPORTUNITY).fillna(
        DEFAULT_MIN_EFFICIENCY_OPPORTUNITY
    )
    out["efficiency_qualified"] = out["opportunity_total"].ge(floor)

    # Standardize opportunity over everyone, but standardize efficiency using
    # only qualified rows so small-sample outliers don't distort the group
    # mean/std (or top the rankings). Unqualified rows get NaN efficiency_z.
    out = add_group_zscore(out, "opportunity_per_game", "opportunity_z")
    out = _efficiency_zscore_qualified_only(out)

    return out


def _efficiency_zscore_qualified_only(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize efficiency within groups using only qualified rows."""
    out = df.copy()
    qualified = out[out["efficiency_qualified"]]
    stats = qualified.groupby(GROUP_COLS)["efficiency_per_opportunity"].agg(
        ["mean", "std"]
    )
    out = out.merge(
        stats.rename(columns={"mean": "_eff_mean", "std": "_eff_std"}),
        left_on=GROUP_COLS,
        right_index=True,
        how="left",
    )
    out["efficiency_z"] = (
        out["efficiency_per_opportunity"] - out["_eff_mean"]
    ) / out["_eff_std"].replace(0, np.nan)
    out.loc[~out["efficiency_qualified"], "efficiency_z"] = np.nan
    out = out.drop(columns=["_eff_mean", "_eff_std"])
    return out


# ---------------------------------------------------------------------------
# Talent-isolating rate features
# ---------------------------------------------------------------------------
def add_talent_rate_features(player_season: pd.DataFrame) -> pd.DataFrame:
    """Add efficiency *rate* features that describe how production is earned.

    All rates are reconstructed from season sum columns already present in the
    processed data. Rates are only meaningful given enough volume, so they are
    left as NaN when the denominator is zero (rather than imputed to 0), which
    keeps "no attempts" distinct from "attempts with zero yards".
    """
    out = player_season.copy()

    targets = _num(out, "targets")
    receptions = _num(out, "receptions")
    rec_yards = _num(out, "receiving_yards")
    rec_air_yards = _num(out, "receiving_air_yards")
    rec_yac = _num(out, "receiving_yards_after_catch")

    # Receiving rates (RB/WR/TE, and pass-catching backs)
    out["catch_rate"] = _safe_divide(receptions, targets)
    out["yards_per_target"] = _safe_divide(rec_yards, targets)
    out["yards_per_reception"] = _safe_divide(rec_yards, receptions)
    out["adot"] = _safe_divide(rec_air_yards, targets)  # average depth of target
    out["yac_per_reception"] = _safe_divide(rec_yac, receptions)
    # RACR: receiving yards per air yard (efficiency converting depth to yards)
    out["racr"] = _safe_divide(rec_yards, rec_air_yards)

    # Rushing rates (RB, mobile QB, some WR)
    carries = _num(out, "carries")
    rush_yards = _num(out, "rushing_yards")
    out["yards_per_carry"] = _safe_divide(rush_yards, carries)

    # Passing rates (QB)
    attempts = _num(out, "attempts")
    completions = _num(out, "completions")
    pass_yards = _num(out, "passing_yards")
    pass_air_yards = _num(out, "passing_air_yards")
    out["completion_pct"] = _safe_divide(completions, attempts)
    out["yards_per_attempt"] = _safe_divide(pass_yards, attempts)
    out["passing_adot"] = _safe_divide(pass_air_yards, attempts)
    # PACR: passing yards per air yard
    out["pacr"] = _safe_divide(pass_yards, pass_air_yards)

    return out


TALENT_RATE_FEATURES = [
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


def build_decomposed_player_seasons(player_season: pd.DataFrame) -> pd.DataFrame:
    """Apply the full decomposition + rate-feature pipeline."""
    out = add_opportunity_and_efficiency(player_season)
    out = add_talent_rate_features(out)
    return out


# ---------------------------------------------------------------------------
# Year-over-year stability analysis (talent isolation evidence)
# ---------------------------------------------------------------------------
def _lag1_correlation(df: pd.DataFrame, col: str) -> tuple[float, int]:
    """Pearson correlation between a column and its next-season value per player."""
    s = df.sort_values(["player_id", "season"])[["player_id", "season", col]].copy()
    s["next"] = s.groupby("player_id")[col].shift(-1)
    s["next_season"] = s.groupby("player_id")["season"].shift(-1)
    consecutive = s["next_season"].eq(s["season"] + 1)
    pair = s[consecutive].dropna(subset=[col, "next"])
    if len(pair) < 10:
        return float("nan"), int(len(pair))
    return float(pair[col].corr(pair["next"])), int(len(pair))


def stability_analysis(decomposed: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year persistence of each value axis, overall and by position.

    A higher lag-1 correlation means the signal is more *repeatable* and thus
    more likely to reflect stable ability rather than noise or one-year role.
    This is the evidence for whether efficiency or opportunity is the more
    talent-stable component.
    """
    axes = {
        "total_value": "value_score",
        "efficiency": "efficiency_z",
        "opportunity": "opportunity_z",
    }
    records: list[dict[str, Any]] = []

    for segment_value, frame in [("all", decomposed)] + [
        (pos, grp) for pos, grp in decomposed.groupby("position")
    ]:
        segment = "overall" if segment_value == "all" else "position"
        for axis_name, col in axes.items():
            if col not in frame.columns:
                continue
            corr, n = _lag1_correlation(frame, col)
            records.append(
                {
                    "segment": segment,
                    "segment_value": segment_value,
                    "axis": axis_name,
                    "yoy_correlation": corr,
                    "n_pairs": n,
                }
            )

    return pd.DataFrame(records)


def _fmt(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.3f}"


def build_stability_report_markdown(stability: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Value Decomposition: Efficiency vs Opportunity")
    lines.append("")
    lines.append(
        "The headline `value_score` is the within season-position z-score of "
        "*total* EPA, which blends how good a player is per opportunity with how "
        "much they are used. This report splits value into two standardized "
        "axes — **efficiency** (value EPA per opportunity) and **opportunity** "
        "(usage per game) — and measures how repeatable each is year over year. "
        "A more repeatable signal is more likely to reflect stable ability, "
        "which is what a front office wants to isolate."
    )
    lines.append("")
    lines.append("## Year-over-year persistence (lag-1 correlation)")
    lines.append("")
    lines.append("Higher is more repeatable. Overall, then by position.")
    lines.append("")
    lines.append("| Segment | Total value | Efficiency | Opportunity | n pairs |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")

    pivot = stability.pivot_table(
        index="segment_value", columns="axis", values="yoy_correlation", aggfunc="first"
    )
    npairs = stability.groupby("segment_value")["n_pairs"].max()
    order = ["all", "QB", "RB", "WR", "TE"]
    for seg in [s for s in order if s in pivot.index]:
        row = pivot.loc[seg]
        label = "Overall" if seg == "all" else seg
        lines.append(
            f"| {label} | {_fmt(row.get('total_value', float('nan')))} | "
            f"{_fmt(row.get('efficiency', float('nan')))} | "
            f"{_fmt(row.get('opportunity', float('nan')))} | "
            f"{int(npairs.get(seg, 0))} |"
        )
    lines.append("")

    overall = pivot.loc["all"] if "all" in pivot.index else None
    if overall is not None:
        eff = overall.get("efficiency", float("nan"))
        opp = overall.get("opportunity", float("nan"))
        tot = overall.get("total_value", float("nan"))
        if not (pd.isna(eff) or pd.isna(opp)):
            more = "opportunity" if opp > eff else "efficiency"
            lines.append(
                f"Overall, **{more}** is the more year-over-year stable axis "
                f"(opportunity {_fmt(opp)} vs efficiency {_fmt(eff)}; total value "
                f"{_fmt(tot)}). This is the central finding for talent evaluation: "
                f"if opportunity is the more persistent component, then much of "
                f"what total-EPA value 'predicts' year to year is really role "
                f"stability, not ability. Modeling the two axes separately — a "
                f"role/opportunity forecast times an efficiency forecast — should "
                f"therefore be more honest and more useful than predicting blended "
                f"total value, and it lets the front office ask the two distinct "
                f"questions (How good? vs How used?) independently."
            )
            lines.append("")
    lines.append("## How to use these columns")
    lines.append("")
    lines.append(
        "`efficiency_z` and `opportunity_z` are standardized within each "
        "season-position group, so a value of +1 means roughly one standard "
        "deviation above positional peers that season. The talent rate features "
        "(catch rate, yards per target, aDOT, YAC per reception, RACR, yards per "
        "carry, completion %, yards per attempt, passing aDOT, PACR) describe "
        "*how* production is earned and are the natural inputs for an "
        "efficiency-side model that aims at ability rather than volume."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _load_player_seasons(project_root: Path) -> pd.DataFrame:
    processed_dir = project_root / "data" / "processed"
    value_scores_path = processed_dir / "player_value_scores_2016_2025.csv"
    skill_seasons_path = processed_dir / "skill_player_seasons_2016_2025.csv"
    if value_scores_path.exists():
        return pd.read_csv(value_scores_path)
    if skill_seasons_path.exists():
        return pd.read_csv(skill_seasons_path)
    raise FileNotFoundError(
        "No processed player-season file found in " + str(processed_dir)
    )


def _find_project_root() -> Path:
    current = Path.cwd()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "processed").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root from " + str(current))


def build_value_decomposition_outputs(
    project_root: Path | None = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Build decomposition columns, stability table, and markdown report."""
    if project_root is None:
        project_root = _find_project_root()
    project_root = Path(project_root)

    player_season = _load_player_seasons(project_root)
    decomposed = build_decomposed_player_seasons(player_season)
    stability = stability_analysis(decomposed)
    report_md = build_stability_report_markdown(stability)

    keep_cols = [
        "season",
        "player_id",
        "player_display_name",
        "position",
        "team",
        "games_played",
        "value_epa_total",
        "value_score",
        "opportunity_total",
        "opportunity_per_game",
        "efficiency_per_opportunity",
        "efficiency_qualified",
        "efficiency_z",
        "opportunity_z",
        *TALENT_RATE_FEATURES,
    ]
    keep_cols = [c for c in keep_cols if c in decomposed.columns]
    decomposed_out = decomposed[keep_cols].copy()

    outputs = {
        "decomposed": decomposed_out,
        "stability": stability,
        "report_markdown": report_md,
    }

    if save_outputs:
        processed_dir = project_root / "data" / "processed"
        tables_dir = project_root / "outputs" / "tables"
        report_dir = project_root / "report"
        for d in (processed_dir, tables_dir, report_dir):
            d.mkdir(parents=True, exist_ok=True)

        decomposed_out.to_csv(
            processed_dir / "player_value_decomposition_2016_2025.csv",
            index=False,
            float_format=config.CSV_FLOAT_FORMAT,
        )
        stability.to_csv(
            tables_dir / "value_decomposition_stability.csv",
            index=False,
            float_format=config.CSV_FLOAT_FORMAT,
        )
        (report_dir / "value_decomposition.md").write_text(report_md)

    return outputs


if __name__ == "__main__":
    result = build_value_decomposition_outputs()
    print(result["stability"].to_string(index=False))
