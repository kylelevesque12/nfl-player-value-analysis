"""Finding tables and narrative summaries for salary efficiency analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.load_data import find_project_root, load_csv
from src.replacement_level import (
    compute_replacement_level_surplus,
    summarize_replacement_level_by_position,
    summarize_replacement_level_by_team_season,
)
from src.salary_efficiency import CSV_FLOAT_FORMAT


MIN_FINDING_GAMES = 8
FINDING_OUTPUTS = {
    "summary_metrics": "salary_findings_summary_metrics.csv",
    "top_surplus": "salary_findings_top_surplus_players.csv",
    "high_cost_underperformers": "salary_findings_high_cost_underperformers.csv",
    "rookie_surplus": "salary_findings_rookie_contract_surplus.csv",
    "veteran_values": "salary_findings_veteran_values.csv",
    "team_season": "salary_findings_team_season.csv",
    "team_summary": "salary_findings_team_summary.csv",
    "position_salary_tiers": "salary_findings_position_salary_tiers.csv",
    "season_trends": "salary_findings_season_trends.csv",
    "replacement_baselines": "salary_findings_replacement_baselines.csv",
    "replacement_top_surplus": "salary_findings_replacement_top_surplus.csv",
    "replacement_team_season": "salary_findings_replacement_team_season.csv",
    "replacement_by_position": "salary_findings_replacement_by_position.csv",
}

REPLACEMENT_PLAYER_COLUMNS = [
    "season",
    "player_display_name",
    "position",
    "team",
    "games_played",
    "years_exp",
    "value_score",
    "salary_millions",
    "replacement_salary_millions",
    "replacement_value_score",
    "cap_over_replacement_millions",
    "value_over_replacement",
    "price_per_value_unit_millions",
    "value_over_replacement_dollar_equivalent_millions",
    "dollar_surplus_millions",
]


PLAYER_COLUMNS = [
    "season",
    "player_display_name",
    "position",
    "team",
    "games_played",
    "years_exp",
    "value_score",
    "salary_millions",
    "salary_percentile",
    "value_above_expected_salary",
    "salary_efficiency_percentile",
]


def _resolve_project_root(project_root: str | Path | None = None) -> Path:
    return find_project_root() if project_root is None else Path(project_root).resolve()


def _format_number(value: Any, column: str | None = None) -> str:
    if pd.isna(value):
        return ""
    if column == "season":
        return str(int(value))
    if column in {
        "games_played",
        "player_seasons",
        "high_efficiency_players",
        "low_efficiency_players",
        "best_team_season",
    }:
        return f"{int(value):,}"
    if column == "years_exp":
        return f"{float(value):.0f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.2f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str], rows: int = 8) -> str:
    table = df[columns].head(rows).copy()
    headers = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in table.iterrows():
        body.append(
            "| "
            + " | ".join(_format_number(row[col], column=col) for col in columns)
            + " |"
        )
    return "\n".join([headers, separator, *body])


def prepare_salary_finding_base(
    salary_efficiency: pd.DataFrame,
    min_games: int = MIN_FINDING_GAMES,
) -> pd.DataFrame:
    """Filter salary-efficiency rows to a cleaner findings sample."""
    df = salary_efficiency.copy()
    df["has_salary"] = df["has_salary"].astype(bool)
    df = df[
        df["has_salary"]
        & df["games_played"].ge(min_games)
        & df["value_above_expected_salary"].notna()
    ].copy()

    df["salary_tier"] = pd.cut(
        df["salary_percentile"],
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["Low Cost", "Below Median Cost", "Above Median Cost", "High Cost"],
    ).astype("string")

    df["career_stage"] = np.select(
        [
            df["years_exp"].le(3),
            df["years_exp"].between(4, 7),
            df["years_exp"].ge(8),
        ],
        ["Rookie-Contract Proxy", "Prime/Veteran", "Late-Career Veteran"],
        default="Unknown",
    )

    return df


def build_salary_finding_tables(
    project_root: str | Path | None = None,
    min_games: int = MIN_FINDING_GAMES,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Build salary-efficiency finding tables and an optional narrative report."""
    root = _resolve_project_root(project_root)
    output_dir = root / "outputs" / "tables"
    report_dir = root / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    salary_efficiency = load_csv(
        "outputs/tables/salary_efficiency_2016_2025.csv",
        root,
    )
    finding_base = prepare_salary_finding_base(salary_efficiency, min_games=min_games)

    top_surplus = (
        finding_base[PLAYER_COLUMNS]
        .sort_values("value_above_expected_salary", ascending=False)
        .head(25)
        .copy()
    )

    high_cost_underperformers = (
        finding_base[finding_base["salary_percentile"].ge(0.75)][PLAYER_COLUMNS]
        .sort_values("value_above_expected_salary", ascending=True)
        .head(25)
        .copy()
    )

    rookie_surplus = (
        finding_base[finding_base["years_exp"].le(3)][PLAYER_COLUMNS]
        .sort_values("value_above_expected_salary", ascending=False)
        .head(25)
        .copy()
    )

    veteran_values = (
        finding_base[
            finding_base["years_exp"].ge(4)
            & finding_base["salary_percentile"].le(0.50)
        ][PLAYER_COLUMNS]
        .sort_values("value_above_expected_salary", ascending=False)
        .head(25)
        .copy()
    )

    team_season = (
        finding_base
        .groupby(["season", "team"], as_index=False)
        .agg(
            player_seasons=("player_id", "count"),
            total_value_above_expected_salary=("value_above_expected_salary", "sum"),
            mean_value_above_expected_salary=("value_above_expected_salary", "mean"),
            median_value_above_expected_salary=("value_above_expected_salary", "median"),
            total_salary_millions=("salary_millions", "sum"),
            median_salary_millions=("salary_millions", "median"),
            mean_value_score=("value_score", "mean"),
            high_efficiency_players=("salary_efficiency_percentile", lambda s: int(s.ge(0.75).sum())),
            low_efficiency_players=("salary_efficiency_percentile", lambda s: int(s.le(0.25).sum())),
        )
        .sort_values("total_value_above_expected_salary", ascending=False)
    )

    team_summary_base = (
        team_season
        .sort_values("total_value_above_expected_salary", ascending=False)
        .drop_duplicates("team")
        [["team", "season", "total_value_above_expected_salary"]]
        .rename(
            columns={
                "season": "best_team_season",
                "total_value_above_expected_salary": "best_team_season_surplus",
            }
        )
    )
    team_summary = (
        finding_base
        .groupby("team", as_index=False)
        .agg(
            player_seasons=("player_id", "count"),
            seasons=("season", "nunique"),
            total_value_above_expected_salary=("value_above_expected_salary", "sum"),
            mean_value_above_expected_salary=("value_above_expected_salary", "mean"),
            total_salary_millions=("salary_millions", "sum"),
            median_salary_millions=("salary_millions", "median"),
            high_efficiency_players=("salary_efficiency_percentile", lambda s: int(s.ge(0.75).sum())),
            low_efficiency_players=("salary_efficiency_percentile", lambda s: int(s.le(0.25).sum())),
        )
        .merge(team_summary_base, on="team", how="left")
        .sort_values("total_value_above_expected_salary", ascending=False)
    )

    position_salary_tiers = (
        finding_base
        .groupby(["position", "salary_tier"], observed=True, as_index=False)
        .agg(
            player_seasons=("player_id", "count"),
            median_salary_millions=("salary_millions", "median"),
            mean_value_score=("value_score", "mean"),
            mean_value_above_expected_salary=("value_above_expected_salary", "mean"),
            median_value_above_expected_salary=("value_above_expected_salary", "median"),
        )
        .sort_values(["position", "median_salary_millions"])
    )

    season_trends = (
        finding_base
        .groupby("season", as_index=False)
        .agg(
            player_seasons=("player_id", "count"),
            median_salary_millions=("salary_millions", "median"),
            mean_salary_millions=("salary_millions", "mean"),
            mean_value_above_expected_salary=("value_above_expected_salary", "mean"),
            median_value_above_expected_salary=("value_above_expected_salary", "median"),
        )
        .sort_values("season")
    )

    summary_metrics = pd.DataFrame(
        [
            {
                "metric": "finding_sample_rows",
                "value": len(finding_base),
                "note": f"Matched salary rows with at least {min_games} games played.",
            },
            {
                "metric": "seasons_covered",
                "value": finding_base["season"].nunique(),
                "note": "Number of seasons in the findings sample.",
            },
            {
                "metric": "median_salary_millions",
                "value": finding_base["salary_millions"].median(),
                "note": "Median inflated APY in the findings sample.",
            },
            {
                "metric": "median_value_above_expected_salary",
                "value": finding_base["value_above_expected_salary"].median(),
                "note": "Median residual from the salary-efficiency model.",
            },
            {
                "metric": "top_team_season",
                "value": f"{team_season.iloc[0]['season']} {team_season.iloc[0]['team']}",
                "note": "Highest total value above expected salary among team-seasons.",
            },
            {
                "metric": "top_team_season_surplus",
                "value": team_season.iloc[0]["total_value_above_expected_salary"],
                "note": "Total residual for that team-season.",
            },
        ]
    )

    # ------------------------------------------------------------------
    # Replacement-level surplus (front-office framing)
    # ------------------------------------------------------------------
    enriched_base, replacement_baselines, _price_table = (
        compute_replacement_level_surplus(finding_base)
    )
    replacement_top_surplus = (
        enriched_base.dropna(subset=["dollar_surplus_millions"])
        [REPLACEMENT_PLAYER_COLUMNS]
        .sort_values("dollar_surplus_millions", ascending=False)
        .head(25)
        .copy()
    )
    replacement_team_season = summarize_replacement_level_by_team_season(
        enriched_base
    )
    replacement_by_position = summarize_replacement_level_by_position(
        enriched_base
    )

    tables = {
        "summary_metrics": summary_metrics,
        "top_surplus": top_surplus,
        "high_cost_underperformers": high_cost_underperformers,
        "rookie_surplus": rookie_surplus,
        "veteran_values": veteran_values,
        "team_season": team_season,
        "team_summary": team_summary,
        "position_salary_tiers": position_salary_tiers,
        "season_trends": season_trends,
        "replacement_baselines": replacement_baselines,
        "replacement_top_surplus": replacement_top_surplus,
        "replacement_team_season": replacement_team_season,
        "replacement_by_position": replacement_by_position,
        "finding_base": finding_base,
    }

    report_markdown = build_salary_findings_report(tables, min_games=min_games)

    if save_outputs:
        for key, filename in FINDING_OUTPUTS.items():
            tables[key].to_csv(
                output_dir / filename,
                index=False,
                float_format=CSV_FLOAT_FORMAT,
            )
        (report_dir / "salary_efficiency_findings.md").write_text(
            report_markdown,
            encoding="utf-8",
        )

    return {
        "tables": tables,
        "report_markdown": report_markdown,
        "output_dir": output_dir,
        "report_path": report_dir / "salary_efficiency_findings.md",
    }


def build_salary_findings_report(
    tables: dict[str, pd.DataFrame],
    min_games: int = MIN_FINDING_GAMES,
) -> str:
    """Create a concise written findings report from salary-efficiency tables."""
    summary = tables["summary_metrics"]
    top_surplus = tables["top_surplus"]
    high_cost_underperformers = tables["high_cost_underperformers"]
    rookie_surplus = tables["rookie_surplus"]
    team_season = tables["team_season"]
    team_summary = tables["team_summary"]
    position_salary_tiers = tables["position_salary_tiers"]

    rows = int(summary.loc[summary["metric"].eq("finding_sample_rows"), "value"].iloc[0])
    median_salary = float(summary.loc[summary["metric"].eq("median_salary_millions"), "value"].iloc[0])
    top_team_season = str(summary.loc[summary["metric"].eq("top_team_season"), "value"].iloc[0])
    top_team_surplus = float(summary.loc[summary["metric"].eq("top_team_season_surplus"), "value"].iloc[0])

    top_player = top_surplus.iloc[0]
    top_rookie = rookie_surplus.iloc[0]
    lowest_high_cost = high_cost_underperformers.iloc[0]
    top_team = team_summary.iloc[0]

    return "\n\n".join(
        [
            "# Salary Efficiency Findings",
            (
                "This report turns the salary-efficiency tables into a smaller set of "
                "interpretable findings. The analysis uses matched contract rows with "
                f"at least {min_games} games played, leaving {rows:,} player-seasons "
                "for the main findings sample."
            ),
            "## Key Takeaways",
            "\n".join(
                [
                    (
                        f"- The median salary in the findings sample is about "
                        f"${median_salary:,.1f} million in inflated APY."
                    ),
                    (
                        f"- The strongest individual surplus season is "
                        f"{top_player['season']} {top_player['player_display_name']} "
                        f"({top_player['position']}, {top_player['team']}), with a "
                        f"value-above-expected-salary score of "
                        f"{top_player['value_above_expected_salary']:.2f}."
                    ),
                    (
                        f"- The strongest rookie-contract proxy season is "
                        f"{top_rookie['season']} {top_rookie['player_display_name']} "
                        f"({top_rookie['position']}, {top_rookie['team']})."
                    ),
                    (
                        f"- Among high-cost player-seasons, the lowest residual in this "
                        f"sample is {lowest_high_cost['season']} "
                        f"{lowest_high_cost['player_display_name']} "
                        f"({lowest_high_cost['position']}, {lowest_high_cost['team']})."
                    ),
                    (
                        f"- The top team-season by total surplus is {top_team_season}, "
                        f"with total value above expected salary of {top_team_surplus:.2f}."
                    ),
                    (
                        f"- Across all seasons, {top_team['team']} has the highest total "
                        "surplus in this filtered skill-position sample."
                    ),
                ]
            ),
            "## Top Surplus Player-Seasons",
            _markdown_table(
                top_surplus,
                [
                    "season",
                    "player_display_name",
                    "position",
                    "team",
                    "games_played",
                    "salary_millions",
                    "value_score",
                    "value_above_expected_salary",
                ],
                rows=10,
            ),
            "## High-Cost Underperformers",
            (
                "This table is limited to players at or above the 75th salary percentile "
                "within their season-position group. It should be read as contract-cost "
                "underperformance, not exact cap-hit underperformance."
            ),
            _markdown_table(
                high_cost_underperformers,
                [
                    "season",
                    "player_display_name",
                    "position",
                    "team",
                    "games_played",
                    "salary_millions",
                    "value_score",
                    "value_above_expected_salary",
                ],
                rows=10,
            ),
            "## Rookie-Contract Proxy Surplus",
            _markdown_table(
                rookie_surplus,
                [
                    "season",
                    "player_display_name",
                    "position",
                    "team",
                    "games_played",
                    "years_exp",
                    "salary_millions",
                    "value_above_expected_salary",
                ],
                rows=10,
            ),
            "## Team-Season Leaderboard",
            _markdown_table(
                team_season,
                [
                    "season",
                    "team",
                    "player_seasons",
                    "total_salary_millions",
                    "total_value_above_expected_salary",
                    "high_efficiency_players",
                    "low_efficiency_players",
                ],
                rows=10,
            ),
            "## Position And Salary-Tier Pattern",
            _markdown_table(
                position_salary_tiers,
                [
                    "position",
                    "salary_tier",
                    "player_seasons",
                    "median_salary_millions",
                    "mean_value_score",
                    "mean_value_above_expected_salary",
                ],
                rows=16,
            ),
            "## Replacement-Level Surplus (Front-Office Framing)",
            (
                "Front offices do not think about contracts in absolute dollars. "
                "They think about *surplus over a freely available alternative*: "
                "a player only earns their cap hit if they out-produce the player a team "
                "could sign at the veteran minimum. This section computes, for each "
                "(season, position), a replacement-level cap cost (median of bottom-quartile "
                "salaries) and a replacement-level value (median value-score of those players). "
                "The surplus per player-season converts above-replacement value into dollars "
                "via the within-(position, season) slope of salary on value, then subtracts the "
                "cap premium paid. Positive surplus means the player out-earned their contract; "
                "the higher, the bigger the deal for the team."
            ),
            "### Top 10 Replacement-Level Surplus Player-Seasons",
            _markdown_table(
                tables["replacement_top_surplus"],
                [
                    "season",
                    "player_display_name",
                    "position",
                    "team",
                    "games_played",
                    "salary_millions",
                    "value_score",
                    "cap_over_replacement_millions",
                    "value_over_replacement",
                    "dollar_surplus_millions",
                ],
                rows=10,
            ),
            "### Replacement-Level Snapshot by Position",
            (
                "The price-per-value-unit column is the implicit market rate of one "
                "z-unit of value at that position. RB occasionally shows a negative slope, "
                "reflecting the well-documented running-back-market irrationality: paying RBs "
                "more is not consistently associated with getting more production."
            ),
            _markdown_table(
                tables["replacement_by_position"],
                [
                    "position",
                    "player_seasons",
                    "median_replacement_salary_millions",
                    "median_replacement_value_score",
                    "median_price_per_value_unit_millions",
                    "median_dollar_surplus_millions",
                    "share_positive_surplus",
                ],
                rows=8,
            ),
            "### Top Team-Seasons by Total Replacement-Level Surplus",
            _markdown_table(
                tables["replacement_team_season"],
                [
                    "season",
                    "team",
                    "player_seasons",
                    "total_cap_over_replacement_millions",
                    "total_value_over_replacement",
                    "total_dollar_surplus_millions",
                ],
                rows=10,
            ),
            (
                "Honesty caveat: the cost variable is still `inflated_apy`, not "
                "year-by-year cap hit. Replacing APY with true cap hit is Tier 1 item #3 "
                "of `PORTFOLIO_ROADMAP.md` and the next upgrade for this analysis."
            ),
            "## Method Notes",
            "\n".join(
                [
                    "- Salary is measured using `inflated_apy`, so this is contract-cost efficiency rather than exact salary-cap accounting.",
                    "- The residual metric compares actual value score to expected value score after accounting for salary, position, age, experience, draft slot, and games played.",
                    "- The findings filter removes very small samples, but sports performance is still noisy and context-dependent.",
                    "- Tight ends remain harder to evaluate because blocking value is not fully captured in the production data.",
                ]
            ),
        ]
    )
