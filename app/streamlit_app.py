"""Streamlit dashboard for the NFL player value analysis project."""

from __future__ import annotations

from pathlib import Path

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
def load_csv(filename: str) -> pd.DataFrame:
    path = TABLE_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_markdown(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return ""
    return path.read_text()


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
    }
    return {name: load_csv(filename) for name, filename in files.items()}


def overview_page(data: dict[str, pd.DataFrame]) -> None:
    predictions = data["predictions"]
    salary_diag = data["salary_diag"]
    interval = data["interval_validation"]
    methodology = data["methodology"]
    value_validation = data["value_validation"]

    st.title("NFL Player Value Dashboard")
    st.caption(
        "Portfolio dashboard for player value, 2026 projections, salary efficiency, "
        "model validation, and methodology checks."
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
            "- GitHub-friendly notebook mirrors are available in `notebooks_markdown/`."
        )


def predictions_page(data: dict[str, pd.DataFrame]) -> None:
    predictions = data["predictions"]
    st.title("2026 Player Predictions")

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
        filtered[display_cols].sort_values("predicted_2026_value_score", ascending=False),
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
    report_text = load_markdown("report/methodology_checks.md")
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
        ("Context feature impact", "report/context_feature_impact.md"),
        ("Prediction report summary", "report/2026_prediction_report_summary.md"),
    ]
    for label, relative_path in links:
        path = PROJECT_ROOT / relative_path
        status = "available" if path.exists() else "missing"
        st.write(f"- `{relative_path}`: {status}")

    st.subheader("Final Project Report Preview")
    report_text = load_markdown("report/final_project_report.md")
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
        }
    ]
    show_missing_data_warning(missing)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        [
            "Overview",
            "2026 Player Predictions",
            "Player Lookup",
            "Salary Efficiency",
            "Model Validation",
            "Methodology",
            "Reports",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Built from committed output tables. Rebuild data with "
        "`python scripts/run_pipeline.py`."
    )

    if page == "Overview":
        overview_page(data)
    elif page == "2026 Player Predictions":
        predictions_page(data)
    elif page == "Player Lookup":
        player_lookup_page(data)
    elif page == "Salary Efficiency":
        salary_page(data)
    elif page == "Model Validation":
        validation_page(data)
    elif page == "Methodology":
        methodology_page(data)
    elif page == "Reports":
        reports_page()


if __name__ == "__main__":
    main()
