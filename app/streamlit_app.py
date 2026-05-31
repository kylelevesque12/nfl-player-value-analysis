"""Streamlit dashboard for the NFL player value analysis project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"


st.set_page_config(
    page_title="NFL Player Value Dashboard",
    page_icon="NFL",
    layout="wide",
)


@st.cache_data
def load_csv(filename: str, modified_at: float) -> pd.DataFrame:
    path = TABLE_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_markdown(relative_path: str, modified_at: float) -> str:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return ""
    return path.read_text()


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def fmt_number(value: float | int | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def fmt_percent(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def multiselect_filter(
    df: pd.DataFrame,
    column: str,
    label: str,
    default: list[str] | None = None,
) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    options = sorted(df[column].dropna().astype(str).unique())
    return st.multiselect(label, options, default=default)


def apply_filter(df: pd.DataFrame, column: str, selected: list[str]) -> pd.DataFrame:
    if df.empty or not selected or column not in df.columns:
        return df
    return df[df[column].astype(str).isin(selected)].copy()


def _available_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def card_row(metrics: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics):
        column.metric(label, value, help=help_text)


def show_missing_data_warning(missing: list[str]) -> None:
    if missing:
        st.warning(
            "Some expected output tables are missing: "
            + ", ".join(missing)
            + ". Rebuild outputs with `python scripts/run_pipeline.py`."
        )


def load_all_data() -> dict[str, pd.DataFrame]:
    files = {
        "predictions": "2026_player_value_predictions.csv",
        "salary": "salary_efficiency_2016_2025.csv",
        "salary_top": "salary_findings_top_surplus_players.csv",
        "salary_team": "salary_findings_team_season.csv",
        "salary_diag": "salary_efficiency_merge_diagnostics.csv",
        "value_validation": "2026_value_validation_by_position.csv",
        "interval_validation": "2026_prediction_interval_validation.csv",
        "availability_validation": "2026_availability_validation_metrics.csv",
        "methodology": "methodology_checks.csv",
        "feature_importance": "model_interpretation_feature_importance.csv",
        "position_models": "position_model_comparison_summary.csv",
        "context_summary": "context_feature_group_summary.csv",
        "fantasy": "2026_fantasy_football_projections.csv",
        "fantasy_validation": "fantasy_projection_validation_by_position.csv",
        "weekly_wins": "weekly_win_projection_games.csv",
        "weekly_win_validation": "weekly_win_projection_validation.csv",
    }
    return {
        name: load_csv(filename, file_mtime(TABLE_DIR / filename))
        for name, filename in files.items()
    }


def overview_page(data: dict[str, pd.DataFrame]) -> None:
    predictions = data["predictions"]
    salary_diag = data["salary_diag"]
    interval = data["interval_validation"]
    methodology = data["methodology"]
    value_validation = data["value_validation"]

    st.title("NFL Player Value Dashboard")
    st.caption(
        "Portfolio dashboard with three product views: front-office player value, "
        "fantasy-football projections, and weekly game-pick probabilities."
    )

    overall_interval = interval[interval.get("segment", pd.Series(dtype=str)).eq("overall")]
    interval_row = overall_interval.iloc[0] if not overall_interval.empty else None
    salary_match_rate = (
        float(salary_diag["match_rate"].iloc[0])
        if not salary_diag.empty and "match_rate" in salary_diag.columns
        else None
    )
    methodology_passes = (
        int(methodology["status"].eq("PASS").sum())
        if not methodology.empty and "status" in methodology.columns
        else None
    )
    methodology_total = len(methodology) if not methodology.empty else None

    card_row(
        [
            (
                "2026 projections",
                f"{len(predictions):,}" if not predictions.empty else "N/A",
                "Players in the 2026 projection table.",
            ),
            (
                "Rolling RMSE",
                fmt_number(interval_row["rmse"], 2) if interval_row is not None else "N/A",
                "Average rolling-validation RMSE for next-season value score.",
            ),
            (
                "Interval coverage",
                fmt_percent(interval_row["coverage_rate"]) if interval_row is not None else "N/A",
                "Historical coverage for the approximate central prediction interval.",
            ),
            (
                "Salary match rate",
                fmt_percent(salary_match_rate),
                "Share of value-score rows matched to salary data.",
            ),
            (
                "Methodology checks",
                (
                    f"{methodology_passes}/{methodology_total}"
                    if methodology_passes is not None and methodology_total is not None
                    else "N/A"
                ),
                "Passing project-quality checks.",
            ),
        ]
    )

    st.divider()

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Top Projected 2026 Players")
        if predictions.empty:
            st.info("Prediction table is missing.")
        else:
            top_players = predictions.nlargest(12, "predicted_2026_value_score")
            fig = px.bar(
                top_players.sort_values("predicted_2026_value_score"),
                x="predicted_2026_value_score",
                y="player_display_name",
                color="position",
                orientation="h",
                labels={
                    "predicted_2026_value_score": "Predicted 2026 value",
                    "player_display_name": "Player",
                },
                title="Top projected player values",
            )
            fig.update_layout(height=520, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Validation By Position")
        if value_validation.empty:
            st.info("Validation table is missing.")
        else:
            fig = px.bar(
                value_validation.sort_values("rmse"),
                x="position",
                y="rmse",
                text="rmse",
                labels={"rmse": "RMSE", "position": "Position"},
                title="Next-season value model error",
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Project Notes")
        st.markdown(
            "- The main value metric is position-season standardized total EPA.\n"
            "- The prediction model is intended for tiering and screening.\n"
            "- Salary efficiency uses `inflated_apy`, not exact cap hit.\n"
            "- Fantasy and weekly-win sections are first-pass draft models.\n"
            "- GitHub-friendly notebook mirrors are available in `notebooks_markdown/`."
        )


def predictions_page(data: dict[str, pd.DataFrame]) -> None:
    predictions = data["predictions"]
    st.title("2026 Player Predictions")
    with st.expander("How to read the front-office value board", expanded=True):
        st.markdown(
            "- `predicted_2026_value_score` is the expected position-adjusted EPA value for 2026. Around `0` is average for that position; positive is above average.\n"
            "- `prediction_interval_low` and `prediction_interval_high` show a rough model range, not a guarantee.\n"
            "- `confidence_level` is about projection stability, while `availability_risk_level` is about the chance of a qualifying season.\n"
            "- `prediction_driver` is the fastest plain-English explanation of why the model sees the player that way."
        )

    if predictions.empty:
        st.info("Prediction table is missing. Run `python scripts/run_pipeline.py`.")
        return

    with st.sidebar:
        st.subheader("Prediction Filters")
        positions = multiselect_filter(predictions, "position", "Position")
        teams = multiselect_filter(predictions, "primary_team_2025", "2025 team")
        tiers = multiselect_filter(predictions, "predicted_2026_value_tier", "Projected tier")
        confidence = multiselect_filter(predictions, "confidence_level", "Confidence")
        risk = multiselect_filter(predictions, "availability_risk_level", "Availability risk")
        min_games = st.slider(
            "Minimum 2025 games",
            0,
            int(predictions["games_played_2025"].max()),
            0,
        )

    filtered = predictions.copy()
    for column, selected in [
        ("position", positions),
        ("primary_team_2025", teams),
        ("predicted_2026_value_tier", tiers),
        ("confidence_level", confidence),
        ("availability_risk_level", risk),
    ]:
        filtered = apply_filter(filtered, column, selected)
    filtered = filtered[filtered["games_played_2025"].ge(min_games)].copy()

    card_row(
        [
            ("Players", f"{len(filtered):,}", None),
            (
                "Average projection",
                fmt_number(filtered["predicted_2026_value_score"].mean(), 2),
                None,
            ),
            (
                "High confidence",
                f"{filtered['confidence_level'].eq('High').sum():,}",
                None,
            ),
            (
                "Low availability risk",
                f"{filtered['availability_risk_level'].eq('Low').sum():,}",
                None,
            ),
        ]
    )

    left, right = st.columns([1.1, 1])
    with left:
        chart_df = filtered.nlargest(20, "predicted_2026_value_score")
        fig = px.bar(
            chart_df.sort_values("predicted_2026_value_score"),
            x="predicted_2026_value_score",
            y="player_display_name",
            color="position",
            orientation="h",
            labels={
                "predicted_2026_value_score": "Predicted 2026 value",
                "player_display_name": "Player",
            },
            title="Top filtered projections",
        )
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            filtered,
            x="predicted_2026_value_score",
            y="prediction_uncertainty",
            color="position",
            hover_data=[
                "player_display_name",
                "primary_team_2025",
                "confidence_level",
                "availability_risk_level",
            ],
            labels={
                "predicted_2026_value_score": "Predicted 2026 value",
                "prediction_uncertainty": "Prediction uncertainty",
            },
            title="Projection vs uncertainty",
        )
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        "player_display_name",
        "position",
        "primary_team_2025",
        "games_played_2025",
        "value_score_2025",
        "predicted_2026_value_score",
        "prediction_interval_low",
        "prediction_interval_high",
        "confidence_level",
        "availability_risk_level",
        "prediction_driver",
    ]
    st.subheader("Filtered Player Table")
    st.dataframe(
        filtered[_available_columns(filtered, display_cols)].sort_values("predicted_2026_value_score", ascending=False),
        width="stretch",
    )
    st.download_button(
        "Download filtered predictions",
        filtered.to_csv(index=False),
        file_name="filtered_2026_player_predictions.csv",
        mime="text/csv",
    )


def player_lookup_page(data: dict[str, pd.DataFrame]) -> None:
    predictions = data["predictions"]
    st.title("Player Lookup")

    if predictions.empty:
        st.info("Prediction table is missing. Run `python scripts/run_pipeline.py`.")
        return

    names = predictions["player_display_name"].sort_values().unique()
    selected_name = st.selectbox("Choose a player", names)
    matches = predictions[predictions["player_display_name"].eq(selected_name)].copy()
    if matches.empty:
        st.info("No player found.")
        return

    if len(matches) > 1:
        selected_team = st.selectbox(
            "Choose row",
            matches["primary_team_2025"].astype(str).tolist(),
        )
        row = matches[matches["primary_team_2025"].astype(str).eq(selected_team)].iloc[0]
    else:
        row = matches.iloc[0]

    st.subheader(f"{row['player_display_name']} - {row['position']} - {row['primary_team_2025']}")
    card_row(
        [
            ("2025 value score", fmt_number(row["value_score_2025"], 2), None),
            ("Predicted 2026 value", fmt_number(row["predicted_2026_value_score"], 2), None),
            ("Position percentile", fmt_percent(row["predicted_2026_position_percentile"]), None),
            ("Qualifying probability", fmt_percent(row["predicted_2026_qualifying_probability"]), None),
            ("Confidence", str(row["confidence_level"]), None),
        ]
    )

    st.markdown("### Prediction Interval")
    interval_df = pd.DataFrame(
        {
            "label": ["Low", "Prediction", "High"],
            "value": [
                row["prediction_interval_low"],
                row["predicted_2026_value_score"],
                row["prediction_interval_high"],
            ],
        }
    )
    fig = px.scatter(
        interval_df,
        x="value",
        y=["Projected range"] * len(interval_df),
        text="label",
        labels={"value": "Value score", "y": ""},
        title="Approximate prediction range",
    )
    fig.update_traces(textposition="top center", marker_size=14)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(height=260)
    st.plotly_chart(fig, use_container_width=True)

    detail_cols = [
        "games_played_2025",
        "age_2025",
        "projected_age_2026",
        "years_exp_2025",
        "projected_years_exp_2026",
        "value_epa_total_2025",
        "value_epa_per_game_2025",
        "yards_per_game_2025",
        "tds_per_game_2025",
        "value_score_last2_avg",
        "value_score_last3_avg",
        "games_played_last2_sum",
    ]
    st.markdown("### Player Details")
    st.dataframe(
        pd.DataFrame(
            [{"metric": col, "value": row[col]} for col in detail_cols if col in row.index]
        ),
        width="stretch",
    )

    st.markdown("### Plain-English Driver")
    st.write(row["prediction_driver"])
    st.caption(row["confidence_note"])


def salary_page(data: dict[str, pd.DataFrame]) -> None:
    salary = data["salary"]
    salary_top = data["salary_top"]
    salary_team = data["salary_team"]

    st.title("Salary Efficiency")
    st.caption(
        "This section uses inflated APY as an approximate contract-cost metric. "
        "It should be read as contract efficiency, not exact cap accounting."
    )

    if salary.empty:
        st.info("Salary-efficiency table is missing.")
        return

    with st.sidebar:
        st.subheader("Salary Filters")
        seasons = st.multiselect(
            "Season",
            sorted(salary["season"].dropna().unique()),
            default=[],
        )
        positions = multiselect_filter(salary, "position", "Position")
        teams = multiselect_filter(salary, "team", "Team")
        tiers = multiselect_filter(salary, "salary_efficiency_tier", "Efficiency tier")
        min_games = st.slider("Minimum games", 0, int(salary["games_played"].max()), 8)

    filtered = salary.copy()
    if seasons:
        filtered = filtered[filtered["season"].isin(seasons)].copy()
    for column, selected in [
        ("position", positions),
        ("team", teams),
        ("salary_efficiency_tier", tiers),
    ]:
        filtered = apply_filter(filtered, column, selected)
    filtered = filtered[filtered["games_played"].ge(min_games)].copy()

    card_row(
        [
            ("Player-seasons", f"{len(filtered):,}", None),
            ("Median salary", "$" + fmt_number(filtered["salary_millions"].median(), 2) + "M", None),
            (
                "Median surplus",
                fmt_number(filtered["value_above_expected_salary"].median(), 2),
                None,
            ),
            (
                "High efficiency rows",
                f"{filtered['salary_efficiency_tier'].eq('High Efficiency').sum():,}",
                None,
            ),
        ]
    )

    left, right = st.columns([1.1, 1])
    with left:
        top_filtered = filtered.nlargest(20, "value_above_expected_salary")
        fig = px.bar(
            top_filtered.sort_values("value_above_expected_salary"),
            x="value_above_expected_salary",
            y="player_display_name",
            color="position",
            orientation="h",
            labels={
                "value_above_expected_salary": "Value above expected salary",
                "player_display_name": "Player",
            },
            title="Top surplus player-seasons",
        )
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            filtered,
            x="salary_millions",
            y="value_score",
            color="position",
            hover_data=[
                "season",
                "player_display_name",
                "team",
                "value_above_expected_salary",
                "salary_efficiency_tier",
            ],
            labels={
                "salary_millions": "Salary, millions",
                "value_score": "Standardized value score",
            },
            title="Salary vs value",
        )
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Team-Season Salary Efficiency")
    if salary_team.empty:
        st.info("Team-season salary table is missing.")
    else:
        team_chart = salary_team.nlargest(20, "total_value_above_expected_salary")
        team_chart = team_chart.assign(team_season=team_chart["season"].astype(str) + " " + team_chart["team"].astype(str))
        fig = px.bar(
            team_chart.sort_values("total_value_above_expected_salary"),
            x="total_value_above_expected_salary",
            y="team_season",
            orientation="h",
            labels={
                "total_value_above_expected_salary": "Total surplus",
                "team_season": "Team-season",
            },
            title="Top team-season surplus",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        "season",
        "player_display_name",
        "position",
        "team",
        "games_played",
        "salary_millions",
        "value_score",
        "value_above_expected_salary",
        "salary_efficiency_percentile",
        "salary_efficiency_tier",
    ]
    st.subheader("Filtered Salary Table")
    st.dataframe(
        filtered[display_cols].sort_values("value_above_expected_salary", ascending=False),
        width="stretch",
    )

    st.subheader("Published Top Surplus Sample")
    if not salary_top.empty:
        st.dataframe(salary_top, width="stretch")


def validation_page(data: dict[str, pd.DataFrame]) -> None:
    st.title("Model Validation And Interpretation")

    value_validation = data["value_validation"]
    interval = data["interval_validation"]
    availability = data["availability_validation"]
    feature_importance = data["feature_importance"]
    position_models = data["position_models"]
    context_summary = data["context_summary"]

    left, right = st.columns(2)
    with left:
        st.subheader("Value Model Error By Position")
        if value_validation.empty:
            st.info("Value validation table is missing.")
        else:
            fig = px.bar(
                value_validation,
                x="position",
                y=["mae", "rmse"],
                barmode="group",
                labels={"value": "Error", "variable": "Metric"},
                title="Rolling-validation error",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(value_validation, width="stretch")

    with right:
        st.subheader("Prediction Interval Coverage")
        if interval.empty:
            st.info("Interval validation table is missing.")
        else:
            fig = px.bar(
                interval,
                x="segment_value",
                y="coverage_rate",
                color="segment",
                labels={"segment_value": "Segment", "coverage_rate": "Coverage"},
                title="Approximate interval coverage",
            )
            fig.add_hline(y=0.80, line_dash="dash", annotation_text="Target 80%")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(interval, width="stretch")

    st.subheader("Feature Importance")
    if feature_importance.empty:
        st.info("Feature-importance table is missing.")
    else:
        top_features = feature_importance.nlargest(15, "importance_mean")
        fig = px.bar(
            top_features.sort_values("importance_mean"),
            x="importance_mean",
            y="feature",
            color="feature_group",
            orientation="h",
            labels={"importance_mean": "Permutation importance", "feature": "Feature"},
            title="Top 2024 permutation importance signals",
        )
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(feature_importance, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Pooled vs Position-Specific Models")
        if position_models.empty:
            st.info("Position-model comparison table is missing.")
        else:
            fig = px.bar(
                position_models,
                x="position",
                y="avg_rmse",
                color="model_type",
                barmode="group",
                labels={"avg_rmse": "Average RMSE"},
                title="Model comparison by position",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(position_models, width="stretch")

    with right:
        st.subheader("Context Feature Experiment")
        if context_summary.empty:
            st.info("Context feature summary is missing.")
        else:
            fig = px.bar(
                context_summary.sort_values("avg_rmse"),
                x="avg_rmse",
                y="feature_set",
                orientation="h",
                labels={"avg_rmse": "Average RMSE", "feature_set": "Feature set"},
                title="Context feature group comparison",
            )
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(context_summary, width="stretch")

    st.subheader("Availability Model")
    if availability.empty:
        st.info("Availability validation table is missing.")
    else:
        fig = px.line(
            availability.sort_values("valid_year"),
            x="valid_year",
            y="roc_auc",
            markers=True,
            labels={"valid_year": "Validation season", "roc_auc": "ROC AUC"},
            title="Availability-model rolling ROC AUC",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(availability, width="stretch")


def front_office_page(data: dict[str, pd.DataFrame]) -> None:
    st.sidebar.subheader("Front Office")
    office_view = st.sidebar.radio(
        "Choose a front-office view",
        [
            "Executive Summary",
            "2026 Value Board",
            "Player Lookup",
            "Salary Efficiency",
            "Model Validation",
        ],
    )

    if office_view == "Executive Summary":
        overview_page(data)
    elif office_view == "2026 Value Board":
        predictions_page(data)
    elif office_view == "Player Lookup":
        player_lookup_page(data)
    elif office_view == "Salary Efficiency":
        salary_page(data)
    elif office_view == "Model Validation":
        validation_page(data)


def fantasy_page(data: dict[str, pd.DataFrame]) -> None:
    fantasy = data["fantasy"]
    validation = data["fantasy_validation"]

    st.title("Fantasy Football Perspective")
    st.caption(
        "A draft model for 2026 season-long PPR fantasy points. This is a "
        "model-driven starting board, not a final draft kit: it does not yet "
        "include rookies, manual depth-chart changes, injuries, or offseason news."
    )
    with st.expander("How to read the fantasy board", expanded=True):
        st.markdown(
            "- `predicted_2026_fantasy_points_ppr` is projected season-long PPR scoring.\n"
            "- `projection_change_from_2025` shows whether the model expects the player to rise or regress from 2025.\n"
            "- `usage_profile` translates targets, receptions, and carries into a football role label.\n"
            "- `confidence_level` describes projection stability, not upside. A low-confidence player can still be a high-upside target.\n"
            "- `fantasy_explanation` gives the one-sentence reason to care about the row."
        )

    if fantasy.empty:
        st.info(
            "Fantasy projection table is missing. Run "
            "`python scripts/run_pipeline.py --steps fantasy`."
        )
        return

    with st.sidebar:
        st.subheader("Fantasy Filters")
        positions = multiselect_filter(fantasy, "position", "Position")
        teams = multiselect_filter(fantasy, "primary_team_2025", "2025 team")
        tiers = multiselect_filter(fantasy, "fantasy_projection_tier", "Projection tier")
        confidence = multiselect_filter(fantasy, "confidence_level", "Confidence")
        max_rank = st.slider(
            "Maximum overall fantasy rank",
            1,
            int(fantasy["fantasy_overall_rank"].max()),
            min(120, int(fantasy["fantasy_overall_rank"].max())),
        )

    filtered = fantasy.copy()
    for column, selected in [
        ("position", positions),
        ("primary_team_2025", teams),
        ("fantasy_projection_tier", tiers),
        ("confidence_level", confidence),
    ]:
        filtered = apply_filter(filtered, column, selected)
    filtered = filtered[filtered["fantasy_overall_rank"].le(max_rank)].copy()

    if filtered.empty:
        st.warning("No players match the selected fantasy filters.")
        return

    stable_projection_count = filtered["confidence_level"].isin(["Medium", "High"]).sum()
    card_row(
        [
            ("Players", f"{len(filtered):,}", None),
            (
                "Avg projected PPR",
                fmt_number(filtered["predicted_2026_fantasy_points_ppr"].mean(), 1),
                None,
            ),
            (
                "Medium/high confidence",
                f"{stable_projection_count:,}",
                "Confidence describes projection stability, not player upside.",
            ),
            (
                "Top projection",
                fmt_number(filtered["predicted_2026_fantasy_points_ppr"].max(), 1),
                None,
            ),
        ]
    )

    left, right = st.columns([1.1, 1])
    with left:
        chart_df = filtered.nsmallest(25, "fantasy_overall_rank")
        fig = px.bar(
            chart_df.sort_values("predicted_2026_fantasy_points_ppr"),
            x="predicted_2026_fantasy_points_ppr",
            y="player_display_name",
            color="position",
            orientation="h",
            labels={
                "predicted_2026_fantasy_points_ppr": "Projected 2026 PPR points",
                "player_display_name": "Player",
            },
            title="Top filtered fantasy projections",
        )
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            filtered,
            x="fantasy_points_ppr_2025",
            y="predicted_2026_fantasy_points_ppr",
            color="position",
            hover_data=[
                "player_display_name",
                "primary_team_2025",
                "fantasy_position_rank",
                "projection_change_label",
                "usage_profile",
                "confidence_level",
            ],
            labels={
                "fantasy_points_ppr_2025": "2025 PPR points",
                "predicted_2026_fantasy_points_ppr": "Projected 2026 PPR points",
            },
            title="Current fantasy production vs next-season projection",
        )
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fantasy Model Validation")
    if validation.empty:
        st.info("Fantasy validation summary is missing.")
    else:
        position_validation = validation[validation["segment"].eq("position")].copy()
        if not position_validation.empty:
            fig = px.bar(
                position_validation,
                x="segment_value",
                y=["mae", "rmse"],
                barmode="group",
                labels={"segment_value": "Position", "value": "PPR points", "variable": "Metric"},
                title="Rolling validation error by position",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(validation, width="stretch")

    display_cols = [
        "fantasy_overall_rank",
        "fantasy_position_rank",
        "player_display_name",
        "position",
        "primary_team_2025",
        "games_played_2025",
        "fantasy_points_ppr_2025",
        "predicted_2026_fantasy_points_ppr",
        "projection_change_from_2025",
        "projection_change_label",
        "prediction_interval_low",
        "prediction_interval_high",
        "fantasy_projection_tier",
        "usage_profile",
        "confidence_level",
        "fantasy_explanation",
    ]
    st.subheader("Filtered Fantasy Board")
    st.dataframe(
        filtered[_available_columns(filtered, display_cols)].sort_values("fantasy_overall_rank"),
        width="stretch",
    )
    st.download_button(
        "Download filtered fantasy board",
        filtered.to_csv(index=False),
        file_name="filtered_2026_fantasy_projections.csv",
        mime="text/csv",
    )


def weekly_win_projection_page(data: dict[str, pd.DataFrame]) -> None:
    games = data["weekly_wins"]
    validation = data["weekly_win_validation"]

    st.title("Weekly Win Projection")
    st.caption(
        "A draft game-pick section that estimates home-win probability from "
        "market context, rest, divisional status, weather, and recent team form. "
        "The current table is a rolling historical backtest, so it is useful for "
        "validating the approach before adding future schedule rows."
    )
    with st.expander("How to read weekly win projections", expanded=True):
        st.markdown(
            "- `winner_probability` is the model's probability for the team listed in `predicted_winner`.\n"
            "- The model is market-informed because it uses `spread_line` and `total_line`, so this is not a pure team-strength rating.\n"
            "- `market_signal` translates the spread into which side the betting market leaned toward.\n"
            "- `pick_explanation` combines the market lean, recent form, and rest edge into a short explanation.\n"
            "- `correct_prediction` is only available because this table is currently a historical backtest."
        )

    if games.empty:
        st.info(
            "Weekly win projection table is missing. Run "
            "`python scripts/run_pipeline.py --steps weekly_wins`."
        )
        return

    with st.sidebar:
        st.subheader("Game Filters")
        seasons = st.multiselect(
            "Season",
            sorted(games["season"].dropna().unique(), reverse=True),
            default=[games["season"].max()],
        )
        weeks = st.multiselect(
            "Week",
            sorted(games["week"].dropna().astype(int).unique()),
            default=[],
        )
        teams = sorted(
            set(games["home_team"].dropna().astype(str))
            | set(games["away_team"].dropna().astype(str))
        )
        selected_teams = st.multiselect("Team", teams)
        confidence = multiselect_filter(games, "confidence_level", "Confidence")

    filtered = games.copy()
    if seasons:
        filtered = filtered[filtered["season"].isin(seasons)].copy()
    if weeks:
        filtered = filtered[filtered["week"].isin(weeks)].copy()
    if selected_teams:
        filtered = filtered[
            filtered["home_team"].astype(str).isin(selected_teams)
            | filtered["away_team"].astype(str).isin(selected_teams)
        ].copy()
    filtered = apply_filter(filtered, "confidence_level", confidence)

    if filtered.empty:
        st.warning("No games match the selected filters.")
        return

    accuracy = (
        filtered["correct_prediction"].mean()
        if "correct_prediction" in filtered.columns
        else np.nan
    )
    card_row(
        [
            ("Games", f"{len(filtered):,}", None),
            ("Filtered accuracy", fmt_percent(accuracy), "Accuracy only applies to completed backtest games."),
            (
                "Avg winner probability",
                fmt_percent(filtered["winner_probability"].mean()),
                None,
            ),
            (
                "High-confidence picks",
                f"{filtered['confidence_level'].eq('High').sum():,}",
                "High confidence means the predicted winner probability is at least 65%.",
            ),
        ]
    )

    left, right = st.columns([1.15, 1])
    with left:
        chart_df = filtered.sort_values(["season", "week", "game_id"]).head(40)
        fig = px.bar(
            chart_df,
            x="winner_probability",
            y="matchup",
            color="predicted_winner",
            orientation="h",
            labels={"winner_probability": "Predicted winner probability", "matchup": "Game"},
            title="Projected winners for selected games",
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if validation.empty:
            st.info("Weekly validation table is missing.")
        else:
            season_validation = validation[
                ~validation["season"].astype(str).eq("overall")
            ].copy()
            fig = px.line(
                season_validation,
                x="season",
                y="accuracy",
                markers=True,
                labels={"season": "Validation season", "accuracy": "Accuracy"},
                title="Rolling backtest accuracy by season",
            )
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(validation, width="stretch")

        fig = px.histogram(
            filtered,
            x="winner_probability",
            nbins=12,
            labels={"winner_probability": "Predicted winner probability"},
            title="Confidence distribution",
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        "season",
        "week",
        "gameday",
        "away_team",
        "home_team",
        "predicted_winner",
        "winner_probability",
        "actual_winner",
        "correct_prediction",
        "confidence_level",
        "market_signal",
        "recent_point_diff_diff",
        "recent_win_rate_diff",
        "rest_advantage",
        "spread_line",
        "total_line",
        "pick_explanation",
    ]
    st.subheader("Filtered Game Table")
    sorted_games = filtered.sort_values(["season", "week", "game_id"])
    st.dataframe(
        sorted_games[_available_columns(sorted_games, display_cols)],
        width="stretch",
    )
    st.download_button(
        "Download filtered game projections",
        filtered.to_csv(index=False),
        file_name="filtered_weekly_win_projections.csv",
        mime="text/csv",
    )


def methodology_page(data: dict[str, pd.DataFrame]) -> None:
    methodology = data["methodology"]
    st.title("Methodology Checks")
    st.caption(
        "These checks do not prove the model is correct. They catch common "
        "project-quality and leakage risks."
    )

    if methodology.empty:
        st.info("Methodology checks are missing. Run `python scripts/run_pipeline.py --steps checks`.")
        return

    pass_count = methodology["status"].eq("PASS").sum()
    fail_count = methodology["status"].eq("FAIL").sum()
    warn_count = methodology["status"].eq("WARN").sum()
    card_row(
        [
            ("Checks", f"{len(methodology):,}", None),
            ("Passed", f"{pass_count:,}", None),
            ("Warnings", f"{warn_count:,}", None),
            ("Failed", f"{fail_count:,}", None),
        ]
    )

    fig = px.histogram(
        methodology,
        x="status",
        color="status",
        title="Methodology check status",
    )
    fig.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(methodology, width="stretch")

    st.subheader("Report Text")
    methodology_path = PROJECT_ROOT / "report" / "methodology_checks.md"
    report_text = load_markdown(
        "report/methodology_checks.md",
        file_mtime(methodology_path),
    )
    if report_text:
        st.markdown(report_text)


def reports_page() -> None:
    st.title("Reports And Artifacts")
    st.markdown(
        "Use this page as a quick index for the project outputs. The dashboard "
        "uses the CSV outputs, while the reports provide the written explanation."
    )

    links = [
        ("Final project report", "report/final_project_report.md"),
        ("Methodology checks", "report/methodology_checks.md"),
        ("Model interpretation", "report/model_interpretation.md"),
        ("Salary-efficiency findings", "report/salary_efficiency_findings.md"),
        ("Fantasy football projection summary", "report/fantasy_football_projection_summary.md"),
        ("Weekly win projection summary", "report/weekly_win_projection_summary.md"),
        ("Context feature impact", "report/context_feature_impact.md"),
        ("Prediction report summary", "report/2026_prediction_report_summary.md"),
    ]
    for label, relative_path in links:
        path = PROJECT_ROOT / relative_path
        status = "available" if path.exists() else "missing"
        st.write(f"- `{relative_path}`: {status}")

    st.subheader("Final Project Report Preview")
    final_report_path = PROJECT_ROOT / "report" / "final_project_report.md"
    report_text = load_markdown(
        "report/final_project_report.md",
        file_mtime(final_report_path),
    )
    if report_text:
        st.markdown(report_text[:6000] + "\n\n...")
    else:
        st.info("Final project report is missing.")


def main() -> None:
    data = load_all_data()
    missing = [
        name
        for name, df in data.items()
        if df.empty
        and name
        in {
            "predictions",
            "salary",
            "value_validation",
            "interval_validation",
            "methodology",
            "feature_importance",
            "fantasy",
            "weekly_wins",
        }
    ]
    show_missing_data_warning(missing)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a section",
        [
            "Front Office Perspective",
            "Fantasy Football Perspective",
            "Weekly Win Projection",
            "Methodology And Reports",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Built from committed output tables. Rebuild data with "
        "`python scripts/run_pipeline.py`."
    )

    if page == "Front Office Perspective":
        front_office_page(data)
    elif page == "Fantasy Football Perspective":
        fantasy_page(data)
    elif page == "Weekly Win Projection":
        weekly_win_projection_page(data)
    elif page == "Methodology And Reports":
        methodology_page(data)
        st.divider()
        reports_page()


if __name__ == "__main__":
    main()
