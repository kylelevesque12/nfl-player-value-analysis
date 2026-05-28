"""Utilities for NFL salary-efficiency analysis.

This module joins player value scores to historical contract data and creates
descriptive salary-efficiency metrics. The contract data is useful, but it is
not the same thing as a precise season-level cap-hit file. The functions keep
that distinction visible by using an explicit salary_source column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
DEFAULT_CONTRACTS_FILE = "historical_contracts.csv"
CONTRACTS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/contracts/"
    "historical_contracts.rds"
)


def find_project_root(expected_file: str = "data/processed/player_value_scores_2016_2025.csv") -> Path:
    """Find the project root from common terminal or notebook working dirs."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / expected_file).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find " + expected_file + " from working directory " + str(Path.cwd())
    )


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_contracts(project_root: Path | None = None, filename: str = DEFAULT_CONTRACTS_FILE) -> pd.DataFrame:
    """Load locally saved nflverse historical contracts.

    The expected local file is data/raw/historical_contracts.csv. It is ignored
    by Git because raw data can be regenerated. If the file is missing, the
    error message explains where it comes from.
    """
    if project_root is None:
        project_root = find_project_root()

    contracts_path = project_root / "data" / "raw" / filename
    if not contracts_path.exists():
        raise FileNotFoundError(
            "Missing contract data at "
            + str(contracts_path)
            + ". Download nflverse historical contracts from "
            + CONTRACTS_URL
            + " and convert/save them as data/raw/historical_contracts.csv."
        )

    return pd.read_csv(contracts_path)


def prepare_contracts(contracts: pd.DataFrame) -> pd.DataFrame:
    """Clean historical contract rows and create a comparable cost metric."""
    contracts = contracts.copy()
    required_cols = ["player", "position", "team", "year_signed", "years", "gsis_id"]
    missing_cols = [col for col in required_cols if col not in contracts.columns]
    if missing_cols:
        raise ValueError("Missing required contract columns: " + ", ".join(missing_cols))

    numeric_cols = [
        "year_signed",
        "years",
        "value",
        "apy",
        "guaranteed",
        "apy_cap_pct",
        "inflated_value",
        "inflated_apy",
        "inflated_guaranteed",
        "draft_year",
        "draft_round",
        "draft_overall",
    ]
    contracts = _to_numeric(contracts, numeric_cols)

    contracts = contracts[
        contracts["position"].isin(SKILL_POSITIONS)
        & contracts["gsis_id"].notna()
        & contracts["year_signed"].notna()
        & contracts["years"].notna()
        & contracts["year_signed"].ge(1990)
        & contracts["years"].gt(0)
    ].copy()

    contracts["contract_start_season"] = contracts["year_signed"].astype(int)
    contracts["contract_end_season"] = (
        contracts["year_signed"] + np.ceil(contracts["years"]) - 1
    ).astype(int)

    if "inflated_apy" in contracts.columns and contracts["inflated_apy"].notna().any():
        contracts["salary_millions"] = contracts["inflated_apy"]
        contracts["salary_source"] = "inflated_apy"
    elif "apy" in contracts.columns:
        contracts["salary_millions"] = contracts["apy"]
        contracts["salary_source"] = "apy"
    else:
        raise ValueError("Contract data must include either inflated_apy or apy.")

    contracts["salary_millions"] = pd.to_numeric(
        contracts["salary_millions"], errors="coerce"
    )
    contracts = contracts[contracts["salary_millions"].gt(0)].copy()

    keep_cols = [
        "player",
        "position",
        "team",
        "is_active",
        "year_signed",
        "years",
        "contract_start_season",
        "contract_end_season",
        "value",
        "apy",
        "guaranteed",
        "apy_cap_pct",
        "inflated_value",
        "inflated_apy",
        "inflated_guaranteed",
        "salary_millions",
        "salary_source",
        "player_page",
        "otc_id",
        "gsis_id",
    ]
    keep_cols = [col for col in keep_cols if col in contracts.columns]

    return contracts[keep_cols].rename(
        columns={
            "player": "contract_player",
            "position": "contract_position",
            "team": "contract_team",
        }
    )


def expand_contracts_to_player_seasons(
    contracts: pd.DataFrame,
    seasons: list[int] | range,
) -> pd.DataFrame:
    """Approximate active player-season contracts from year signed and term."""
    contracts = prepare_contracts(contracts)
    records = []

    for season in seasons:
        active = contracts[
            contracts["contract_start_season"].le(season)
            & contracts["contract_end_season"].ge(season)
        ].copy()
        active["season"] = int(season)
        records.append(active)

    if not records:
        return pd.DataFrame()

    expanded = pd.concat(records, ignore_index=True)
    expanded = expanded.sort_values(
        ["season", "gsis_id", "year_signed", "salary_millions"],
        ascending=[True, True, False, False],
    )
    expanded = expanded.drop_duplicates(["season", "gsis_id"], keep="first")
    expanded["contract_match_method"] = "gsis_id_active_contract_window"
    expanded["salary_interpretation"] = (
        "Approximate annual contract cost from nflverse/OverTheCap historical "
        "contracts; not a precise season cap hit."
    )

    return expanded


def merge_value_and_salary(
    value_scores: pd.DataFrame,
    contracts: pd.DataFrame,
    seasons: list[int] | range | None = None,
) -> pd.DataFrame:
    """Merge value scores with expanded contract-season rows."""
    value_scores = value_scores.copy()
    if seasons is None:
        seasons = range(int(value_scores["season"].min()), int(value_scores["season"].max()) + 1)

    contract_seasons = expand_contracts_to_player_seasons(contracts, seasons)
    merged = value_scores.merge(
        contract_seasons,
        left_on=["season", "player_id"],
        right_on=["season", "gsis_id"],
        how="left",
        suffixes=("", "_contract"),
    )

    merged["salary_match_status"] = np.where(
        merged["salary_millions"].notna(),
        "matched_contract",
        "missing_contract",
    )
    merged["has_salary"] = merged["salary_millions"].notna()

    return merged


def add_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple and model-adjusted salary-efficiency metrics."""
    df = df.copy()
    df["salary_millions"] = pd.to_numeric(df["salary_millions"], errors="coerce")
    df["salary_dollars"] = df["salary_millions"] * 1_000_000
    df["log_salary_millions"] = np.log1p(df["salary_millions"])

    df["value_per_million"] = df["value_score"] / df["salary_millions"]
    df["epa_per_million"] = df["value_epa_total"] / df["salary_millions"]

    group_cols = ["season", "position"]
    df["salary_percentile"] = df.groupby(group_cols)["salary_millions"].rank(pct=True)
    df["value_cost_percentile_gap"] = (
        df["position_season_percentile"] - df["salary_percentile"]
    )

    model_df = df[
        df["has_salary"]
        & df["value_score"].notna()
        & df["salary_millions"].gt(0)
    ].copy()

    feature_cols = [
        "log_salary_millions",
        "position",
        "age",
        "years_exp",
        "draft_number",
        "games_played",
    ]
    feature_cols = [col for col in feature_cols if col in model_df.columns]

    df["expected_value_given_salary"] = np.nan
    df["value_above_expected_salary"] = np.nan

    if len(model_df) >= 100 and feature_cols:
        categorical_cols = [col for col in feature_cols if col == "position"]
        numeric_cols = [col for col in feature_cols if col not in categorical_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                ),
            ],
            remainder="drop",
        )
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=10.0)),
            ]
        )
        model.fit(model_df[feature_cols], model_df["value_score"])
        expected = model.predict(model_df[feature_cols])
        df.loc[model_df.index, "expected_value_given_salary"] = expected
        df.loc[model_df.index, "value_above_expected_salary"] = (
            model_df["value_score"] - expected
        )

    df["salary_efficiency_percentile"] = (
        df.groupby(group_cols)["value_above_expected_salary"].rank(pct=True)
    )
    df["salary_efficiency_tier"] = pd.cut(
        df["salary_efficiency_percentile"],
        bins=[-np.inf, 0.25, 0.50, 0.75, 0.90, np.inf],
        labels=[
            "Low Efficiency",
            "Below Average",
            "Average",
            "Above Average",
            "High Efficiency",
        ],
    ).astype("string")

    return df


def summarize_salary_merge(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create row-count, match-rate, and salary summary diagnostics."""
    overall = pd.DataFrame(
        [
            {
                "rows": len(df),
                "matched_salary_rows": int(df["has_salary"].sum()),
                "missing_salary_rows": int((~df["has_salary"]).sum()),
                "match_rate": float(df["has_salary"].mean()) if len(df) else np.nan,
                "salary_source": (
                    ", ".join(sorted(df["salary_source"].dropna().unique()))
                    if "salary_source" in df.columns
                    else None
                ),
            }
        ]
    )

    by_position = (
        df.groupby("position", as_index=False)
        .agg(
            rows=("player_id", "count"),
            matched_salary_rows=("has_salary", "sum"),
            match_rate=("has_salary", "mean"),
            median_salary_millions=("salary_millions", "median"),
            median_value_score=("value_score", "median"),
            median_value_above_expected_salary=(
                "value_above_expected_salary",
                "median",
            ),
        )
        .sort_values("position")
    )

    by_season = (
        df.groupby("season", as_index=False)
        .agg(
            rows=("player_id", "count"),
            matched_salary_rows=("has_salary", "sum"),
            match_rate=("has_salary", "mean"),
            median_salary_millions=("salary_millions", "median"),
            median_value_above_expected_salary=(
                "value_above_expected_salary",
                "median",
            ),
        )
        .sort_values("season")
    )

    return {
        "overall": overall,
        "by_position": by_position,
        "by_season": by_season,
    }


def build_salary_efficiency_tables(
    project_root: Path | None = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Build salary-efficiency tables from value scores and contracts."""
    if project_root is None:
        project_root = find_project_root()

    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    value_scores = pd.read_csv(processed_dir / "player_value_scores_2016_2025.csv")
    contracts = load_contracts(project_root)
    salary_efficiency = merge_value_and_salary(value_scores, contracts)
    salary_efficiency = add_efficiency_metrics(salary_efficiency)

    diagnostics = summarize_salary_merge(salary_efficiency)

    matched = salary_efficiency[salary_efficiency["has_salary"]].copy()
    top_efficient = (
        matched.sort_values("value_above_expected_salary", ascending=False)
        .head(25)
        .copy()
    )
    lowest_efficiency = (
        matched.sort_values("value_above_expected_salary", ascending=True)
        .head(25)
        .copy()
    )

    summary_cols = [
        "season",
        "player_id",
        "player_display_name",
        "position",
        "team",
        "games_played",
        "age",
        "years_exp",
        "draft_number",
        "value_score",
        "value_epa_total",
        "position_season_percentile",
        "salary_millions",
        "salary_source",
        "contract_player",
        "contract_team",
        "year_signed",
        "years",
        "apy",
        "apy_cap_pct",
        "inflated_apy",
        "value_per_million",
        "epa_per_million",
        "salary_percentile",
        "value_cost_percentile_gap",
        "expected_value_given_salary",
        "value_above_expected_salary",
        "salary_efficiency_percentile",
        "salary_efficiency_tier",
        "has_salary",
        "salary_match_status",
        "salary_interpretation",
    ]
    summary_cols = [col for col in summary_cols if col in salary_efficiency.columns]
    salary_efficiency_export = salary_efficiency[summary_cols].copy()

    if save_outputs:
        salary_efficiency_export.to_csv(
            output_dir / "salary_efficiency_2016_2025.csv",
            index=False,
        )
        diagnostics["overall"].to_csv(
            output_dir / "salary_efficiency_merge_diagnostics.csv",
            index=False,
        )
        diagnostics["by_position"].to_csv(
            output_dir / "salary_efficiency_by_position.csv",
            index=False,
        )
        diagnostics["by_season"].to_csv(
            output_dir / "salary_efficiency_by_season.csv",
            index=False,
        )
        top_efficient[summary_cols].to_csv(
            output_dir / "salary_efficiency_top_players.csv",
            index=False,
        )
        lowest_efficiency[summary_cols].to_csv(
            output_dir / "salary_efficiency_lowest_players.csv",
            index=False,
        )

    return {
        "salary_efficiency": salary_efficiency_export,
        "diagnostics": diagnostics,
        "top_efficient": top_efficient[summary_cols],
        "lowest_efficiency": lowest_efficiency[summary_cols],
        "output_dir": output_dir,
    }
