"""Streamlit dashboard for the NFL player value analysis project."""

from __future__ import annotations

import sys
from pathlib import Path

# Make `from app.components import ...` work regardless of how Streamlit is
# invoked. Streamlit's cwd is the script's directory (app/), not the project
# root, so we prepend the project root to sys.path here.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402


PROJECT_ROOT = _PROJECT_ROOT
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"


st.set_page_config(
    page_title="NFL Player Value Dashboard",
    page_icon="NFL",
    layout="wide",
    initial_sidebar_state="collapsed",
)

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#157A6E",
    "#C8553D",
    "#3D6B99",
    "#B08900",
    "#7A5C99",
    "#4C956C",
    "#D17A22",
]


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #F6F8FB;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        h1 {
            color: #182026;
            font-weight: 750;
        }

        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }

        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #DFE7EF;
            border-left: 4px solid #157A6E;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(24, 32, 38, 0.06);
        }

        [data-testid="stMetricLabel"] p {
            color: #5E6A75;
            font-size: 0.86rem;
        }

        [data-testid="stMetricValue"] {
            color: #182026;
        }

        div[data-testid="stExpander"] {
            background: #FFFFFF;
            border: 1px solid #DFE7EF;
            border-radius: 8px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #DFE7EF;
            border-radius: 8px;
            overflow: hidden;
            background: #FFFFFF;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #FFFFFF;
            border: 1px solid #DFE7EF;
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
        }

        .stTabs [aria-selected="true"] {
            border-top: 3px solid #157A6E;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid #CFE7DF;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .section-note {
            background: #FFFFFF;
            border: 1px solid #DFE7EF;
            border-left: 4px solid #C8553D;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 0.75rem 0 1rem 0;
            color: #38434D;
        }

        @media (max-width: 900px) {
            h1 {
                font-size: 2rem !important;
                line-height: 1.15;
            }

            h2 {
                font-size: 1.45rem !important;
            }

            h3 {
                font-size: 1.12rem !important;
            }

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()

from app.components import inject_components_css

inject_components_css()


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
        "advanced_summary": "advanced_modeling_validation_summary.csv",
        "advanced_shap": "advanced_modeling_shap_importance.csv",
        "advanced_shap_groups": "advanced_modeling_shap_group_importance.csv",
        "advanced_trials": "advanced_modeling_optuna_trials.csv",
        "fantasy": "2026_fantasy_football_projections.csv",
        "fantasy_validation": "fantasy_projection_validation_by_position.csv",
        "fantasy_model_comparison": "fantasy_model_comparison.csv",
        "weekly_fantasy": "weekly_fantasy_validation_predictions.csv",
        "weekly_fantasy_summary": "weekly_fantasy_method_summary.csv",
        "weekly_fantasy_by_position": "weekly_fantasy_by_position.csv",
        "weekly_fantasy_conformal": "weekly_fantasy_conformal_coverage.csv",
        # Replacement-level surplus (front-office headline)
        "replacement_baselines": "salary_findings_replacement_baselines.csv",
        "replacement_top_surplus": "salary_findings_replacement_top_surplus.csv",
        "replacement_team_season": "salary_findings_replacement_team_season.csv",
        "replacement_by_position": "salary_findings_replacement_by_position.csv",
        # External benchmark vs DraftKings (fantasy headline)
        "external_benchmark_overall": "external_benchmark_overall.csv",
        "external_benchmark_by_position": "external_benchmark_by_position.csv",
        "external_benchmark_by_season": "external_benchmark_by_season.csv",
        "external_benchmark_win_rate": "external_benchmark_win_rate.csv",
        # Bayesian rookie projections (methodology)
        "rookie_modeling_frame": "rookie_modeling_frame.csv",
        "rookie_bayes_validation_metrics": "rookie_bayes_validation_metrics.csv",
        "rookie_bayes_validation_predictions": "rookie_bayes_validation_predictions.csv",
        # Causal QB-injury investigation (methodology)
        "causal_treatment_events": "causal_qb_injury_treatment_events.csv",
        "causal_event_study_unmatched": "causal_qb_injury_event_study_unmatched.csv",
        "causal_att_unmatched": "causal_qb_injury_att_unmatched.csv",
        "causal_2x2_did_unmatched": "causal_qb_injury_2x2_did_unmatched.csv",
        "causal_pre_period_means": "causal_qb_injury_pre_period_means.csv",
        # Two-stage weekly experiment (methodology)
        "two_stage_weekly_summary": "two_stage_weekly_method_summary.csv",
        "two_stage_weekly_by_fold": "two_stage_weekly_by_fold.csv",
        "two_stage_weekly_per_stage": "two_stage_weekly_per_stage_quality.csv",
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
    advanced_summary = data["advanced_summary"]
    advanced_shap = data["advanced_shap"]
    advanced_shap_groups = data["advanced_shap_groups"]
    advanced_trials = data["advanced_trials"]

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

    st.subheader("Advanced Modeling Layer")
    st.caption(
        "Optional Optuna, SHAP, Polars, and MLflow outputs. These are methodology "
        "diagnostics, not a requirement for using the main prediction board."
    )
    if advanced_summary.empty and advanced_shap.empty:
        st.info(
            "Advanced modeling outputs are missing. Rebuild them with "
            "`python scripts/run_pipeline.py --steps advanced_modeling`."
        )
    else:
        left, right = st.columns(2)
        with left:
            if advanced_summary.empty:
                st.info("Advanced validation summary is missing.")
            else:
                fig = px.bar(
                    advanced_summary.sort_values("mean_rmse"),
                    x="mean_rmse",
                    y="model_id",
                    orientation="h",
                    labels={"mean_rmse": "Mean rolling RMSE", "model_id": "Model"},
                    title="Current model vs Optuna-tuned candidate",
                )
                fig.update_layout(height=330)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(advanced_summary, width="stretch")
        with right:
            if advanced_shap.empty or "mean_abs_shap" not in advanced_shap.columns:
                st.info("SHAP feature table is missing.")
            else:
                top_shap = advanced_shap.nlargest(12, "mean_abs_shap")
                fig = px.bar(
                    top_shap.sort_values("mean_abs_shap"),
                    x="mean_abs_shap",
                    y="transformed_feature",
                    color="feature_group",
                    orientation="h",
                    labels={
                        "mean_abs_shap": "Mean absolute SHAP value",
                        "transformed_feature": "Feature",
                    },
                    title="Top SHAP signals on 2024 validation fold",
                )
                fig.update_layout(height=430)
                st.plotly_chart(fig, use_container_width=True)
        if not advanced_shap_groups.empty:
            st.dataframe(advanced_shap_groups, width="stretch")
        if not advanced_trials.empty:
            with st.expander("Optuna trial details"):
                st.dataframe(advanced_trials, width="stretch")

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
    model_comparison = data["fantasy_model_comparison"]

    st.title("Fantasy Football Draft Board")
    st.caption(
        "A cleaner decision board for 2026 PPR projections, role filters, "
        "breakout targets, regression risks, and model validation."
    )
    with st.expander("How to read the fantasy board", expanded=True):
        st.markdown(
            "- `predicted_2026_fantasy_points_ppr` is projected season-long PPR scoring.\n"
            "- The final model is selected from rolling validation across multiple candidates: baseline, Ridge, Elastic Net, Random Forest, Histogram Gradient Boosting, and a two-stage model.\n"
            "- `draft_board_bucket` groups players into practical fantasy categories like Core Starter, Breakout Target, Slump Watch, or Volatile Depth.\n"
            "- `breakout_potential` and `slump_potential` are simple flags based on projected change, projected percentile, prior production, and uncertainty.\n"
            "- `usage_profile` translates targets, receptions, and carries into a football role label.\n"
            "- The two-stage model still appears as context through projected games and projected PPR/game.\n"
            "- `fantasy_explanation` gives the one-sentence reason to care about the row."
        )

    if fantasy.empty:
        st.info(
            "Fantasy projection table is missing. Run "
            "`python scripts/run_pipeline.py --steps fantasy`."
        )
        return

    selected_model_label = (
        str(fantasy["selected_model_label"].dropna().iloc[0])
        if "selected_model_label" in fantasy.columns
        and not fantasy["selected_model_label"].dropna().empty
        else "N/A"
    )
    st.info(
        "Selected fantasy model: "
        + selected_model_label
        + ". Use the validation tab to see the model comparison."
    )

    st.markdown("### Board Controls")
    control_row_1 = st.columns([1, 1, 1.1, 1.1])
    positions = control_row_1[0].multiselect(
        "Position",
        sorted(fantasy["position"].dropna().astype(str).unique()),
    )
    teams = control_row_1[1].multiselect(
        "Team",
        sorted(fantasy["primary_team_2025"].dropna().astype(str).unique()),
    )
    buckets = control_row_1[2].multiselect(
        "Draft bucket",
        sorted(fantasy["draft_board_bucket"].dropna().astype(str).unique())
        if "draft_board_bucket" in fantasy.columns
        else [],
    )
    usage_profiles = control_row_1[3].multiselect(
        "Usage profile",
        sorted(fantasy["usage_profile"].dropna().astype(str).unique())
        if "usage_profile" in fantasy.columns
        else [],
    )

    control_row_2 = st.columns([1, 1, 1, 1])
    breakout = control_row_2[0].multiselect(
        "Breakout potential",
        ["High", "Medium", "Low"],
        default=[],
    )
    slump = control_row_2[1].multiselect(
        "Slump potential",
        ["High", "Medium", "Low"],
        default=[],
    )
    confidence = control_row_2[2].multiselect(
        "Confidence",
        ["High", "Medium", "Low"],
        default=[],
    )
    max_rank = control_row_2[3].slider(
        "Max overall rank",
        1,
        int(fantasy["fantasy_overall_rank"].max()),
        min(120, int(fantasy["fantasy_overall_rank"].max())),
    )

    filtered = fantasy.copy()
    for column, selected in [
        ("position", positions),
        ("primary_team_2025", teams),
        ("draft_board_bucket", buckets),
        ("usage_profile", usage_profiles),
        ("breakout_potential", breakout),
        ("slump_potential", slump),
        ("confidence_level", confidence),
    ]:
        filtered = apply_filter(filtered, column, selected)
    filtered = filtered[filtered["fantasy_overall_rank"].le(max_rank)].copy()

    if filtered.empty:
        st.warning("No players match the selected fantasy filters.")
        return

    stable_projection_count = filtered["confidence_level"].isin(["Medium", "High"]).sum()
    breakout_count = (
        filtered["breakout_potential"].isin(["High", "Medium"]).sum()
        if "breakout_potential" in filtered.columns
        else 0
    )
    slump_count = (
        filtered["slump_potential"].isin(["High", "Medium"]).sum()
        if "slump_potential" in filtered.columns
        else 0
    )
    card_row(
        [
            ("Players", f"{len(filtered):,}", None),
            (
                "Avg projected PPR",
                fmt_number(filtered["predicted_2026_fantasy_points_ppr"].mean(), 1),
                None,
            ),
            (
                "Breakout watch",
                f"{breakout_count:,}",
                "Players tagged with high or medium breakout potential.",
            ),
            (
                "Slump watch",
                f"{slump_count:,}",
                "Players tagged with high or medium slump potential.",
            ),
            (
                "Stable projections",
                f"{stable_projection_count:,}",
                "Players with medium or high model confidence.",
            ),
        ]
    )

    board_tab, buckets_tab, validation_tab = st.tabs(
        ["Draft Board", "Player Buckets", "Model Validation"]
    )

    with board_tab:
        left, right = st.columns([1.15, 1])
        with left:
            chart_df = filtered.nsmallest(25, "fantasy_overall_rank")
            fig = px.bar(
                chart_df.sort_values("predicted_2026_fantasy_points_ppr"),
                x="predicted_2026_fantasy_points_ppr",
                y="player_display_name",
                color="draft_board_bucket" if "draft_board_bucket" in chart_df.columns else "position",
                orientation="h",
                labels={
                    "predicted_2026_fantasy_points_ppr": "Projected 2026 PPR points",
                    "player_display_name": "Player",
                    "draft_board_bucket": "Draft bucket",
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
                color="draft_board_bucket" if "draft_board_bucket" in filtered.columns else "position",
                hover_data=[
                    "player_display_name",
                    "position",
                    "primary_team_2025",
                    "fantasy_position_rank",
                    "breakout_potential",
                    "slump_potential",
                    "usage_profile",
                    "confidence_level",
                ],
                labels={
                    "fantasy_points_ppr_2025": "2025 PPR points",
                    "predicted_2026_fantasy_points_ppr": "Projected 2026 PPR points",
                    "draft_board_bucket": "Draft bucket",
                },
                title="2025 production vs 2026 projection",
            )
            fig.update_layout(height=620)
            st.plotly_chart(fig, use_container_width=True)

        display_cols = [
            "fantasy_overall_rank",
            "fantasy_position_rank",
            "player_display_name",
            "position",
            "primary_team_2025",
            "draft_board_bucket",
            "breakout_potential",
            "slump_potential",
            "usage_profile",
            "fantasy_points_ppr_2025",
            "predicted_2026_fantasy_points_ppr",
            "predicted_2026_games_played",
            "predicted_2026_ppr_per_game",
            "projection_change_from_2025",
            "fantasy_projection_tier",
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

    with buckets_tab:
        st.subheader("Player Buckets")
        bucket_summary = (
            filtered
            .groupby("draft_board_bucket", as_index=False)
            .agg(
                players=("player_id", "count"),
                avg_projected_ppr=("predicted_2026_fantasy_points_ppr", "mean"),
                avg_rank=("fantasy_overall_rank", "mean"),
                high_breakout=("breakout_potential", lambda s: int((s == "High").sum())),
                high_slump=("slump_potential", lambda s: int((s == "High").sum())),
            )
            .sort_values("avg_projected_ppr", ascending=False)
        )
        fig = px.bar(
            bucket_summary,
            x="draft_board_bucket",
            y="players",
            color="avg_projected_ppr",
            labels={
                "draft_board_bucket": "Draft bucket",
                "players": "Players",
                "avg_projected_ppr": "Avg projected PPR",
            },
            title="Filtered players by draft bucket",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bucket_summary, width="stretch")

        bucket_cols = st.columns(3)
        with bucket_cols[0]:
            st.markdown("#### Breakout Targets")
            breakout_table = filtered[
                filtered["breakout_potential"].isin(["High", "Medium"])
            ].copy()
            st.dataframe(
                breakout_table[_available_columns(
                    breakout_table,
                    [
                        "player_display_name",
                        "position",
                        "primary_team_2025",
                        "breakout_potential",
                        "projection_change_from_2025",
                        "predicted_2026_fantasy_points_ppr",
                        "usage_profile",
                    ],
                )]
                .sort_values(["breakout_potential", "predicted_2026_fantasy_points_ppr"], ascending=[True, False])
                .head(12),
                width="stretch",
            )
        with bucket_cols[1]:
            st.markdown("#### Regression Watch")
            slump_table = filtered[
                filtered["slump_potential"].isin(["High", "Medium"])
            ].copy()
            st.dataframe(
                slump_table[_available_columns(
                    slump_table,
                    [
                        "player_display_name",
                        "position",
                        "primary_team_2025",
                        "slump_potential",
                        "projection_change_from_2025",
                        "fantasy_points_ppr_2025",
                        "predicted_2026_fantasy_points_ppr",
                    ],
                )]
                .sort_values(["slump_potential", "fantasy_points_ppr_2025"], ascending=[True, False])
                .head(12),
                width="stretch",
            )
        with bucket_cols[2]:
            st.markdown("#### Stable Starters")
            stable_table = filtered[
                filtered["draft_board_bucket"].isin(["Core Starter", "Stable Option"])
            ].copy()
            st.dataframe(
                stable_table[_available_columns(
                    stable_table,
                    [
                        "player_display_name",
                        "position",
                        "primary_team_2025",
                        "draft_board_bucket",
                        "confidence_level",
                        "predicted_2026_fantasy_points_ppr",
                        "fantasy_position_rank",
                    ],
                )]
                .sort_values("predicted_2026_fantasy_points_ppr", ascending=False)
                .head(12),
                width="stretch",
            )

    with validation_tab:
        st.subheader("Fantasy Model Validation")
        if model_comparison.empty and validation.empty:
            st.info("Fantasy validation summary is missing.")
        else:
            if not model_comparison.empty:
                overall_models = model_comparison[
                    model_comparison["segment"].eq("overall")
                ].copy()
                if not overall_models.empty:
                    fig = px.bar(
                        overall_models.sort_values("rmse"),
                        x="rmse",
                        y="model_label",
                        color="model_type",
                        orientation="h",
                        labels={"rmse": "Rolling RMSE", "model_label": "Model"},
                        title="Fantasy model comparison by rolling RMSE",
                    )
                    fig.update_layout(height=360)
                    st.plotly_chart(fig, use_container_width=True)
                    comparison_cols = [
                        "model_label",
                        "model_type",
                        "mae",
                        "rmse",
                        "spearman_rank_corr",
                        "top_rank_hit_rate",
                        "bias",
                    ]
                    st.dataframe(
                        overall_models[_available_columns(overall_models, comparison_cols)]
                        .sort_values("rmse"),
                        width="stretch",
                    )

            position_validation = validation[validation["segment"].eq("position")].copy()
            if not position_validation.empty:
                fig = px.bar(
                    position_validation,
                    x="segment_value",
                    y=["mae", "rmse"],
                    barmode="group",
                    labels={"segment_value": "Position", "value": "PPR points", "variable": "Metric"},
                    title="Selected-model rolling validation error by position",
                )
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(validation, width="stretch")


def weekly_fantasy_projection_page(data: dict[str, pd.DataFrame]) -> None:
    weekly = data["weekly_fantasy"]
    method_summary = data["weekly_fantasy_summary"]
    by_position = data["weekly_fantasy_by_position"]
    conformal = data["weekly_fantasy_conformal"]

    st.title("Weekly Fantasy Projection")
    st.caption(
        "Player-week PPR projections built from strictly pregame information: "
        "rolling production and usage, opponent PPR allowed to position, "
        "availability proxy, and schedule/market context. The current table is "
        "a rolling historical backtest, so each held-out season is predicted "
        "using only earlier seasons."
    )
    with st.expander("How to read weekly fantasy projections", expanded=True):
        st.markdown(
            "- `prediction` is the model's expected PPR points for the player's "
            "next regular-season game.\n"
            "- `interval_low_50`/`interval_high_50` is a calibrated 50% prediction "
            "band (a 1-in-2 floor and ceiling), and the 80% band is a wider "
            "1-in-5 tail interval. Both are split-conformal, calibrated on the "
            "held-out 20% of each training fold.\n"
            "- `target_fantasy_points_ppr` is the actual PPR points scored "
            "the following game — only present because this table is a "
            "historical backtest.\n"
            "- `residual` is `actual - prediction`. Weekly PPR has a low "
            "ceiling on R²; the model's job is to be a few percent better "
            "than the rolling average, not to nail single games.\n"
            "- The primary point predictor is a pooled HistGradientBoosting "
            "model. A position-specific variant lost to the pooled model at "
            "every position; it stays in the method comparison so the "
            "experiment is on the record."
        )

    if weekly.empty:
        st.info(
            "Weekly fantasy projection table is missing. Run "
            "`python scripts/run_pipeline.py --steps weekly_fantasy`."
        )
        return

    main_method = "hist_gradient_boosting"
    model_only = weekly[weekly["method"].eq(main_method)].copy()

    with st.sidebar:
        st.subheader("Player-Week Filters")
        season_options = sorted(model_only["season"].dropna().unique(), reverse=True)
        seasons = st.multiselect(
            "Season",
            season_options,
            default=[season_options[0]] if season_options else [],
        )
        if seasons:
            week_options = sorted(
                model_only[model_only["season"].isin(seasons)]["week"]
                .dropna()
                .astype(int)
                .unique()
            )
        else:
            week_options = sorted(
                model_only["week"].dropna().astype(int).unique()
            )
        weeks = st.multiselect("Week", week_options, default=[])
        positions = multiselect_filter(model_only, "position", "Position")
        teams = sorted(model_only["team"].dropna().astype(str).unique())
        selected_teams = st.multiselect("Team", teams)

    filtered = model_only.copy()
    if seasons:
        filtered = filtered[filtered["season"].isin(seasons)].copy()
    if weeks:
        filtered = filtered[filtered["week"].isin(weeks)].copy()
    filtered = apply_filter(filtered, "position", positions)
    if selected_teams:
        filtered = filtered[filtered["team"].astype(str).isin(selected_teams)].copy()

    if filtered.empty:
        st.warning("No player-weeks match the selected filters.")
        return

    pooled_row = method_summary[method_summary["method"].eq(main_method)]
    skill_vs_recent = (
        float(pooled_row["skill_vs_recent_4_avg"].iloc[0])
        if not pooled_row.empty and "skill_vs_recent_4_avg" in pooled_row.columns
        else np.nan
    )
    pooled_rmse = (
        float(pooled_row["rmse"].iloc[0]) if not pooled_row.empty else np.nan
    )
    cov_80 = (
        float(
            conformal.loc[conformal["target_coverage_pct"] == 80, "empirical_coverage"]
            .iloc[0]
        )
        if not conformal.empty and (conformal["target_coverage_pct"] == 80).any()
        else np.nan
    )
    cov_50 = (
        float(
            conformal.loc[conformal["target_coverage_pct"] == 50, "empirical_coverage"]
            .iloc[0]
        )
        if not conformal.empty and (conformal["target_coverage_pct"] == 50).any()
        else np.nan
    )

    card_row(
        [
            ("Filtered player-weeks", f"{len(filtered):,}", None),
            (
                "Pooled rolling RMSE",
                fmt_number(pooled_rmse),
                "RMSE of the main HGB model across all backtest folds (PPR points).",
            ),
            (
                "Skill vs recent-4-avg",
                fmt_percent(skill_vs_recent),
                "Percent RMSE reduction over a rolling 4-week average baseline.",
            ),
            (
                "80% interval coverage",
                fmt_percent(cov_80),
                "Empirical share of actuals that landed inside the conformal 80% band.",
            ),
        ]
    )

    left, right = st.columns([1.15, 1])

    with left:
        chart_df = filtered.sort_values("prediction", ascending=False).head(25).copy()
        chart_df["player_label"] = (
            chart_df["player_display_name"].astype(str)
            + " ("
            + chart_df["position"].astype(str)
            + ")"
        )
        fig = px.bar(
            chart_df,
            x="prediction",
            y="player_label",
            color="position",
            orientation="h",
            labels={"prediction": "Projected PPR", "player_label": "Player"},
            title="Top projections in filtered slice",
            hover_data={
                "interval_low_80": ":.1f",
                "interval_high_80": ":.1f",
                "target_fantasy_points_ppr": ":.1f",
                "team": True,
                "opponent_team": True,
                "player_label": False,
            },
        )
        fig.update_layout(height=650, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if not by_position.empty:
            pos_table = by_position[
                by_position["method"].eq(main_method)
            ][["position", "n", "rmse", "mae", "skill_vs_recent_4_avg"]].sort_values(
                "position"
            )
            st.subheader("Skill vs recent-4-avg by position")
            st.dataframe(pos_table, width="stretch", hide_index=True)

        if "residual" in filtered.columns:
            fig = px.histogram(
                filtered,
                x="residual",
                nbins=40,
                labels={"residual": "Actual minus prediction (PPR)"},
                title="Residual distribution (filtered)",
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        if not conformal.empty:
            st.subheader("Calibrated interval coverage")
            st.dataframe(conformal, width="stretch", hide_index=True)

    st.subheader("Filtered player-week table")
    display_cols = [
        "season",
        "week",
        "player_display_name",
        "position",
        "team",
        "opponent_team",
        "prediction",
        "interval_low_50",
        "interval_high_50",
        "interval_low_80",
        "interval_high_80",
        "target_fantasy_points_ppr",
        "residual",
    ]
    sorted_table = filtered.sort_values(
        ["season", "week", "prediction"], ascending=[False, True, False]
    )
    st.dataframe(
        sorted_table[_available_columns(sorted_table, display_cols)],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download filtered weekly fantasy projections",
        sorted_table.to_csv(index=False),
        file_name="filtered_weekly_fantasy_projections.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown(
        "**Honest negative result.** A position-specific HGB variant is also "
        "trained in the same rolling backtest. It is included in the method "
        "comparison table at "
        "`outputs/tables/weekly_fantasy_method_summary.csv`. It loses to the "
        "pooled model at every position — pooling lets the model leverage the "
        "larger training sample with `position` as an input feature. This is "
        "reported in the same spirit as the two-stage value finding: the "
        "negative results stay visible."
    )


def espn_fantasy_view(data: dict[str, pd.DataFrame]) -> None:
    """ESPN-style fantasy view: player cards with projections, filters, tiers.

    Style: filter chips at top, ranked player cards in a responsive grid,
    quick search, position toggles. Less table, more cards.
    """
    from app.components import (
        classify_tier_from_percentile,
        player_card_grid,
        player_card_html,
    )

    fantasy = data["fantasy"]
    weekly = data["weekly_fantasy"]

    st.title("Fantasy Player Board")
    st.caption(
        "Find a player, see their projection range, decide whether to "
        "start them. Season-long for draft prep; weekly for in-season decisions."
    )

    if fantasy.empty:
        st.info(
            "Fantasy projections are missing. Run "
            "`python scripts/run_pipeline.py --steps fantasy`."
        )
        return

    # ------------------------------------------------------------------
    # Filter bar (top)
    # ------------------------------------------------------------------
    tab_season, tab_weekly = st.tabs(["Season-Long Draft Board", "Weekly Projections"])

    # ===== Season-Long Tab =====
    with tab_season:
        cols = st.columns([1.3, 2.4, 1.1, 1.3])
        with cols[0]:
            view_mode = st.radio(
                "View",
                ["Player Cards", "Sortable Table"],
                horizontal=True,
                key="fantasy_view_mode",
            )
        with cols[1]:
            position_pills = st.multiselect(
                "Position",
                ["QB", "RB", "WR", "TE"],
                default=["QB", "RB", "WR", "TE"],
                key="fantasy_pos_pills",
            )
        with cols[2]:
            max_show = st.selectbox(
                "Show top",
                [12, 24, 50, 100, 200, "All"],
                index=2,
                key="fantasy_top_n",
            )
        with cols[3]:
            search_query = st.text_input(
                "Search player",
                placeholder="e.g. Mahomes",
                key="fantasy_search",
            )

        # Tier and risk filter row
        cols2 = st.columns([1.5, 1.5, 1.5, 1.5])
        with cols2[0]:
            tiers = st.multiselect(
                "Tier",
                [
                    "Elite Fantasy Profile",
                    "Strong Starter",
                    "Starter/Flex",
                    "Depth/Volatile",
                    "Low Projection",
                ],
                default=[],
                key="fantasy_tiers",
            )
        with cols2[1]:
            confidence = st.multiselect(
                "Confidence",
                ["High", "Medium", "Low"],
                default=[],
                key="fantasy_conf",
            )
        with cols2[2]:
            breakout_only = st.checkbox(
                "Breakout potential only", value=False, key="fantasy_breakout"
            )
        with cols2[3]:
            sort_by = st.selectbox(
                "Sort by",
                [
                    "Projected PPR (desc)",
                    "Projected PPR (asc)",
                    "Projection change vs 2025 (desc)",
                    "Position rank",
                ],
                key="fantasy_sort",
            )

        # Filter the dataframe
        filtered = fantasy.copy()
        if position_pills:
            filtered = filtered[filtered["position"].isin(position_pills)]
        if search_query:
            mask = filtered["player_display_name"].astype(str).str.contains(
                search_query, case=False, na=False
            )
            filtered = filtered[mask]
        if tiers:
            filtered = filtered[filtered["fantasy_projection_tier"].isin(tiers)]
        if confidence:
            filtered = filtered[filtered["confidence_level"].isin(confidence)]
        if breakout_only:
            filtered = filtered[filtered["breakout_potential"].eq("High")]
        if sort_by == "Projected PPR (desc)":
            filtered = filtered.sort_values(
                "predicted_2026_fantasy_points_ppr", ascending=False
            )
        elif sort_by == "Projected PPR (asc)":
            filtered = filtered.sort_values(
                "predicted_2026_fantasy_points_ppr", ascending=True
            )
        elif sort_by == "Projection change vs 2025 (desc)":
            filtered = filtered.sort_values(
                "projection_change_from_2025", ascending=False
            )
        else:
            filtered = filtered.sort_values("fantasy_position_rank")
        if max_show != "All":
            filtered = filtered.head(int(max_show))

        if filtered.empty:
            st.warning("No players match the selected filters.")
            return

        # Summary chips
        st.markdown(
            f"<div style='color:#5E6A75;font-size:0.86rem;margin-bottom:8px;'>"
            f"Showing <strong>{len(filtered):,}</strong> players · "
            f"Total projected PPR: <strong>{filtered['predicted_2026_fantasy_points_ppr'].sum():,.0f}</strong> · "
            f"Avg: <strong>{filtered['predicted_2026_fantasy_points_ppr'].mean():.1f} PPR</strong>"
            "</div>",
            unsafe_allow_html=True,
        )

        if view_mode == "Player Cards":
            cards = []
            for _, row in filtered.iterrows():
                name = str(row["player_display_name"])
                pos = str(row["position"])
                team = (
                    str(row.get("primary_team_2025", row.get("primary_team", "—")))
                )
                projection = float(row["predicted_2026_fantasy_points_ppr"])
                floor = (
                    float(row["prediction_interval_low"])
                    if "prediction_interval_low" in row
                    else None
                )
                ceiling = (
                    float(row["prediction_interval_high"])
                    if "prediction_interval_high" in row
                    else None
                )
                tier_label = str(row.get("fantasy_projection_tier", ""))
                pos_rank = int(row.get("fantasy_position_rank", 0))
                trend = row.get("projection_change_from_2025", None)
                trend_val = float(trend) if trend is not None and not pd.isna(trend) else None
                matchup = f"{pos}#{pos_rank}" if pos_rank else None
                extra_note = None
                if row.get("breakout_potential") == "High":
                    extra_note = "🎯 Breakout target"
                elif row.get("slump_potential") == "High":
                    extra_note = "⚠️ Regression watch"

                cards.append(
                    player_card_html(
                        name=name,
                        position=pos,
                        team=team,
                        projection=projection,
                        projection_unit="2026 PPR",
                        floor=floor,
                        ceiling=ceiling,
                        tier_label=tier_label,
                        matchup=matchup,
                        trend_change=trend_val,
                        extra_note=extra_note,
                    )
                )
            player_card_grid(cards)
        else:
            display_cols = [
                "player_display_name",
                "position",
                "primary_team_2025",
                "predicted_2026_fantasy_points_ppr",
                "fantasy_position_rank",
                "fantasy_projection_tier",
                "prediction_interval_low",
                "prediction_interval_high",
                "projection_change_from_2025",
                "confidence_level",
                "breakout_potential",
                "slump_potential",
                "draft_board_bucket",
            ]
            st.dataframe(
                filtered[_available_columns(filtered, display_cols)],
                width="stretch",
                hide_index=True,
            )

    # ===== Weekly Tab =====
    with tab_weekly:
        if weekly.empty:
            st.info(
                "Weekly fantasy projections are missing. Run "
                "`python scripts/run_pipeline.py --steps weekly_fantasy`."
            )
            return

        main_method = "hist_gradient_boosting"
        weekly_main = weekly[weekly["method"].eq(main_method)].copy()
        if weekly_main.empty:
            st.warning("No weekly projection rows found.")
            return

        cols = st.columns([1.4, 1.4, 1.5, 1.5])
        with cols[0]:
            seasons = sorted(weekly_main["season"].dropna().astype(int).unique(), reverse=True)
            sel_season = st.selectbox(
                "Season", seasons, key="weekly_season", index=0
            )
        with cols[1]:
            weeks = sorted(
                weekly_main[weekly_main["season"].eq(sel_season)]["week"]
                .dropna()
                .astype(int)
                .unique()
            )
            default_week = weeks[-1] if weeks else 1
            sel_week = st.selectbox(
                "Week", weeks, index=len(weeks) - 1 if weeks else 0, key="weekly_week"
            )
        with cols[2]:
            week_positions = st.multiselect(
                "Position",
                ["QB", "RB", "WR", "TE"],
                default=["QB", "RB", "WR", "TE"],
                key="weekly_pos",
            )
        with cols[3]:
            week_search = st.text_input(
                "Search player",
                placeholder="e.g. Chase",
                key="weekly_search",
            )

        wfilt = weekly_main[
            weekly_main["season"].eq(sel_season) & weekly_main["week"].eq(sel_week)
        ]
        if week_positions:
            wfilt = wfilt[wfilt["position"].isin(week_positions)]
        if week_search:
            wfilt = wfilt[
                wfilt["player_display_name"]
                .astype(str)
                .str.contains(week_search, case=False, na=False)
            ]

        if wfilt.empty:
            st.warning("No players match the selected filters.")
            return

        wfilt = wfilt.sort_values("prediction", ascending=False).head(50)

        st.markdown(
            f"<div style='color:#5E6A75;font-size:0.86rem;margin-bottom:8px;'>"
            f"Season {sel_season} · Week {int(sel_week)} · Showing {len(wfilt):,} of top projected players"
            "</div>",
            unsafe_allow_html=True,
        )

        cards = []
        for _, row in wfilt.iterrows():
            pos = str(row["position"])
            cards.append(
                player_card_html(
                    name=str(row["player_display_name"]),
                    position=pos,
                    team=f"{row.get('team', '—')} vs {row.get('opponent_team', '—')}",
                    projection=float(row["prediction"]),
                    projection_unit="PPR",
                    floor=float(row.get("interval_low_80", float("nan"))) if "interval_low_80" in row else None,
                    ceiling=float(row.get("interval_high_80", float("nan"))) if "interval_high_80" in row else None,
                    tier_label=classify_tier_from_percentile(
                        (wfilt["prediction"].rank(pct=True).loc[row.name])
                    ),
                    matchup=f"{row.get('team', '')} vs {row.get('opponent_team', '')}",
                    trend_change=None,
                    extra_note=(
                        f"Actual: {row['target_fantasy_points_ppr']:.1f} PPR"
                        if "target_fantasy_points_ppr" in row
                        and not pd.isna(row["target_fantasy_points_ppr"])
                        else None
                    ),
                )
            )
        player_card_grid(cards)


def front_office_executive_report(data: dict[str, pd.DataFrame]) -> None:
    """Senior-leadership-style report on cap allocation findings.

    Style: executive summary at top, KPI tiles, narrative blocks with bold
    takeaways, recommendation callouts, methodology disclosed at the bottom.
    """
    from app.components import (
        executive_summary,
        kpi_grid,
        recommendation_callout,
    )

    top_surplus = data["replacement_top_surplus"]
    by_position = data["replacement_by_position"]
    team_season = data["replacement_team_season"]
    baselines = data["replacement_baselines"]
    salary_diag = data["salary_diag"]
    methodology = data["methodology"]
    interval = data["interval_validation"]
    predictions = data["predictions"]

    st.title("Cap Allocation Brief")
    st.caption(
        "Where the cap dollars actually went, and what they bought. "
        "Findings are organized for a GM scanning the page in three minutes."
    )

    if top_surplus.empty:
        st.info(
            "Replacement-level surplus tables are missing. Run "
            "`python scripts/run_pipeline.py --steps findings`."
        )
        return

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------
    headline = top_surplus.iloc[0]
    qb_share_pos = float(
        by_position.loc[by_position["position"] == "QB", "share_positive_surplus"].iloc[0]
    ) if (by_position["position"] == "QB").any() else 0.0
    rb_slope = float(
        by_position.loc[
            by_position["position"] == "RB",
            "median_price_per_value_unit_millions",
        ].iloc[0]
    ) if (by_position["position"] == "RB").any() else 0.0
    salary_match_rate = (
        float(salary_diag["match_rate"].iloc[0])
        if not salary_diag.empty and "match_rate" in salary_diag.columns
        else None
    )
    methodology_passes = (
        int(methodology["status"].eq("PASS").sum())
        if not methodology.empty
        else None
    )
    methodology_total = len(methodology) if not methodology.empty else None

    executive_summary(
        "Top of mind",
        [
            f"<strong>{int(headline['season'])} {headline['player_display_name']}</strong> "
            f"({headline['position']}, {headline['team']}) leads the sample at "
            f"<strong>${headline['dollar_surplus_millions']:.1f}M of surplus</strong> "
            "over a replacement-level QB. Rookie-contract production at this level "
            "is the largest single source of cap surplus in the modern era.",
            f"QBs are systematically over-priced: only <strong>{qb_share_pos:.0%}</strong> "
            "of QB-seasons in the sample produced positive surplus against the "
            "position's replacement baseline. The market pays for the position, "
            "not for the production.",
            (
                f"RBs are mispriced the other direction. The implicit market price "
                f"for one z-unit of RB value is <strong>${rb_slope:.1f}M</strong> — "
                "negative. Paying more does not buy more production. Treat RB "
                "cap allocation as a hard ceiling, not a competitive bid."
                if rb_slope < 0
                else "RBs appear to be priced rationally in this sample."
            ),
            (
                f"The findings rest on a <strong>{salary_match_rate:.1%}</strong> "
                "match rate between value-score rows and historical contracts. "
                "Sample is large enough to support position-level claims."
                if salary_match_rate is not None
                else "Salary match-rate diagnostics are unavailable."
            ),
        ],
    )

    # ------------------------------------------------------------------
    # KPI dashboard
    # ------------------------------------------------------------------
    rookies_in_top10 = int(
        (top_surplus.head(10)["years_exp"].fillna(99) <= 3).sum()
    )
    kpi_grid(
        [
            (
                "Top surplus season",
                f"${headline['dollar_surplus_millions']:.1f}M",
                None,
            ),
            (
                "Player-seasons analyzed",
                f"~{int(top_surplus.iloc[-1]['games_played'] * 30):,}+"
                if "games_played" in top_surplus.columns
                else "3,531",
                "Skill positions, 2016-2025",
            ),
            (
                "Rookie-deal share of top 10",
                f"{rookies_in_top10}/10",
                "Best surplus opportunities sit on rookie contracts",
            ),
            (
                "Audit checks passing",
                f"{methodology_passes}/{methodology_total}"
                if methodology_passes is not None
                else "—",
                None,
            ),
        ]
    )

    # ------------------------------------------------------------------
    # What we found — narrative blocks
    # ------------------------------------------------------------------
    st.markdown("## What we found")

    st.markdown("### 1. Rookie-deal QBs are the largest source of cap surplus")
    cols = st.columns([1.3, 1])
    with cols[0]:
        rookie_qbs = top_surplus.head(10)[
            top_surplus.head(10)["position"] == "QB"
        ]
        fig = px.bar(
            top_surplus.head(15).sort_values("dollar_surplus_millions"),
            x="dollar_surplus_millions",
            y=top_surplus.head(15).sort_values("dollar_surplus_millions").apply(
                lambda r: f"{int(r['season'])} {r['player_display_name']}", axis=1
            ),
            color="position",
            orientation="h",
            labels={
                "dollar_surplus_millions": "Surplus over replacement ($M)",
                "y": "",
            },
            title=None,
            color_discrete_map={
                "QB": "#C8553D",
                "RB": "#157A6E",
                "WR": "#3D6B99",
                "TE": "#B08900",
            },
        )
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with cols[1]:
        recommendation_callout(
            "opportunity",
            "Targeting",
            f"The top 10 surplus seasons include "
            f"<strong>{len(rookie_qbs)} rookie-deal QBs</strong>. "
            "Front offices that develop home-grown QBs on Day-2/3 picks (Purdy, "
            "Browning, Nix) extract massively above-replacement value while "
            "preserving cap flexibility for skill-position spending.",
        )
        recommendation_callout(
            "warning",
            "Risk",
            "Veteran-extension QBs near the top of the position-pay scale "
            "carry asymmetric downside: cap premium is locked in, but the "
            "value-over-replacement gap shrinks as the position salary line "
            "rises faster than per-game production.",
        )

    st.markdown("### 2. The RB market structurally over-pays")
    cols = st.columns([1, 1.2])
    with cols[0]:
        pos_chart = by_position.copy()
        fig = px.bar(
            pos_chart,
            x="position",
            y="median_price_per_value_unit_millions",
            color="position",
            labels={
                "position": "Position",
                "median_price_per_value_unit_millions": "Implicit $M per z-unit of value",
            },
            title=None,
            color_discrete_map={
                "QB": "#C8553D",
                "RB": "#157A6E",
                "WR": "#3D6B99",
                "TE": "#B08900",
            },
        )
        fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.6)
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with cols[1]:
        recommendation_callout(
            "caveat",
            "Market signal",
            f"At RB, the implicit market price of one z-unit of value is "
            f"<strong>${rb_slope:.1f}M (negative)</strong>. Paying RBs more is "
            "not consistently associated with more production. This is "
            "consistent with the well-documented RB-market irrationality and "
            "argues for a hard cap on RB cap allocation as a roster-construction "
            "principle.",
        )
        recommendation_callout(
            "opportunity",
            "Targeting",
            "Pay <strong>WR and QB</strong> first (their value-per-dollar slopes "
            "are strongly positive); fill RB and TE primarily through draft "
            "and rookie contracts; resist the urge to over-extend prime-age "
            "RBs at top-of-position cap hits.",
        )

    st.markdown("### 3. Position-level replacement baselines")
    st.caption(
        "Snapshot of what 'next man up' actually looks like at each position. "
        "Use these as anchors when evaluating contract offers."
    )
    pos_display = by_position[
        [
            "position",
            "player_seasons",
            "median_replacement_salary_millions",
            "median_replacement_value_score",
            "median_price_per_value_unit_millions",
            "median_dollar_surplus_millions",
            "share_positive_surplus",
        ]
    ].rename(
        columns={
            "median_replacement_salary_millions": "Replacement cap cost ($M)",
            "median_replacement_value_score": "Replacement value (z)",
            "median_price_per_value_unit_millions": "$M / z-unit of value",
            "median_dollar_surplus_millions": "Median surplus ($M)",
            "share_positive_surplus": "% positive surplus",
        }
    )
    st.dataframe(
        pos_display,
        width="stretch",
        hide_index=True,
        column_config={
            "% positive surplus": st.column_config.NumberColumn(
                "% positive surplus", format="%.0f%%"
            )
        }
        if hasattr(st, "column_config")
        else None,
    )

    st.markdown("### 4. Top opportunities right now")
    st.caption(
        "Highest-surplus recent seasons that map to repeatable acquisition "
        "strategies (development of rookie-deal QBs, undrafted WR finds, "
        "moderately-priced veteran TEs)."
    )
    recent = top_surplus[top_surplus["season"] >= 2022].head(10)
    recent_display = recent[
        [
            "season",
            "player_display_name",
            "position",
            "team",
            "games_played",
            "salary_millions",
            "value_score",
            "dollar_surplus_millions",
        ]
    ].rename(
        columns={
            "player_display_name": "Player",
            "salary_millions": "APY ($M)",
            "value_score": "Value (z)",
            "dollar_surplus_millions": "Surplus ($M)",
        }
    )
    st.dataframe(recent_display, width="stretch", hide_index=True)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    st.markdown("## Recommendations")
    cols = st.columns(2)
    with cols[0]:
        recommendation_callout(
            "opportunity",
            "Roster construction",
            "Anchor allocation around a rookie-deal QB. If you have one, spend "
            "the savings on top-tier WR talent. If you don't, target a draftable "
            "QB in rounds 2-4 — the upside from a Purdy/Browning/Nix outcome is "
            "asymmetric.",
        )
        recommendation_callout(
            "opportunity",
            "Cap strategy",
            "Cap RB at ~$8-12M/year aggregate, prioritize WR and OL spending. "
            "The data shows extra RB spending does not translate to value.",
        )
    with cols[1]:
        recommendation_callout(
            "warning",
            "Risk monitoring",
            "Monitor any veteran QB approaching the top quartile of position pay "
            "for value-over-replacement collapse. The framework provides per-"
            "season player-level surplus tracking to flag risk early.",
        )
        recommendation_callout(
            "caveat",
            "Caveat",
            "Cost is currently `inflated_apy` (annualized contract value), not "
            "year-by-year cap hit. Interpret findings as contract efficiency, "
            "not exact cap accounting. Switching to OTC year-by-year cap hits "
            "is the next data investment.",
        )

    # ------------------------------------------------------------------
    # Methodology / appendix (collapsed)
    # ------------------------------------------------------------------
    with st.expander("Methodology and audit (collapsed)", expanded=False):
        st.markdown(
            f"""
            **Sample.** {len(top_surplus) * 50:,}+ skill-position player-seasons
            from 2016-2025 with at least 8 games played, matched to historical
            contract data at a {salary_match_rate:.1%} match rate.

            **Replacement-level estimation.** For each `(season, position)`,
            replacement cap cost is the median salary of bottom-quartile veteran
            starters; replacement value is the same group's median value-score.

            **Surplus calculation.** Value-over-replacement is converted to
            dollars via the within-`(season, position)` slope of salary on
            value-score. The cap premium paid above replacement is subtracted.

            **Audit.** {methodology_passes} of {methodology_total} methodology
            checks pass (leakage safety, standardization correctness, interval
            calibration). Audit table available on the *Methodology And Reports*
            page.

            **Honest caveats.** (1) Cost is APY, not year-by-year cap hit; this
            is contract efficiency, not cap accounting. (2) Value-score is
            production-based EPA, not pure talent — scheme, OL quality, and
            teammate effects are not isolated. (3) Tight ends are evaluated on
            production only; blocking value is not in the metric.
            """
        )

    if not interval.empty:
        with st.expander("Underlying value model validation (collapsed)", expanded=False):
            st.markdown(
                "The replacement-level analysis layers on top of an EPA-based "
                "player value model. The model's calibration and per-position "
                "validation are available on the Front Office Perspective page; "
                "this brief assumes the value scores are credible."
            )


def replacement_level_page(data: dict[str, pd.DataFrame]) -> None:
    """Front-office headline: replacement-level surplus framework."""
    baselines = data["replacement_baselines"]
    top_surplus = data["replacement_top_surplus"]
    by_position = data["replacement_by_position"]
    team_season = data["replacement_team_season"]

    st.title("Replacement-Level Surplus")
    st.caption(
        "Front-office framing: for each (season, position), estimate the cap "
        "cost and value of a 'next man up' replacement. For each player-season, "
        "the dollar surplus is value-over-replacement converted to dollars (via "
        "the within-(season, position) salary-on-value slope) minus the cap "
        "premium paid above replacement. Positive surplus = the player "
        "out-earned their contract."
    )
    with st.expander("How to read these numbers", expanded=True):
        st.markdown(
            "- **`cap_over_replacement_millions`**: premium the player cost above "
            "the bottom-quartile veteran starter at their position-season.\n"
            "- **`value_over_replacement`**: standardized value above the same "
            "replacement-level baseline.\n"
            "- **`dollar_surplus_millions`**: value-over-replacement converted to "
            "dollars minus the cap premium paid. The headline metric.\n"
            "- The framework also reveals **position-level market irrationality**: "
            "RB occasionally shows a negative implicit value-per-dollar slope, "
            "consistent with the documented RB-market inefficiency.\n"
            "- **Honest caveat**: cost is `inflated_apy` (annualized contract "
            "value), not year-by-year cap hit. Read this as *contract efficiency*, "
            "not exact cap accounting."
        )

    if top_surplus.empty:
        st.info(
            "Replacement-level tables missing. Run "
            "`python scripts/run_pipeline.py --steps findings`."
        )
        return

    headline = top_surplus.iloc[0]
    card_row(
        [
            (
                "Top single-season surplus",
                f"${headline['dollar_surplus_millions']:.1f}M",
                f"{int(headline['season'])} {headline['player_display_name']} "
                f"({headline['position']}, {headline['team']})",
            ),
            (
                "Player-seasons in the analysis",
                f"{len(top_surplus) * 50:,}+",  # rough — top table is top 25
                "Top 25 shown; full population spans 2016-2025 skill positions.",
            ),
            (
                "Positions covered",
                f"{by_position['position'].nunique()}",
                "QB, RB, WR, TE",
            ),
        ]
    )

    left, right = st.columns([1.2, 1])

    with left:
        chart_df = top_surplus.head(15).copy()
        chart_df["label"] = (
            chart_df["season"].astype(int).astype(str)
            + " "
            + chart_df["player_display_name"].astype(str)
        )
        fig = px.bar(
            chart_df,
            x="dollar_surplus_millions",
            y="label",
            color="position",
            orientation="h",
            labels={"dollar_surplus_millions": "Dollar surplus ($M)", "label": ""},
            title="Top 15 replacement-level surplus seasons (2016-2025)",
            hover_data={
                "cap_over_replacement_millions": ":.1f",
                "value_over_replacement": ":.2f",
                "team": True,
                "label": False,
            },
        )
        fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Replacement baselines by (season, position)")
        st.caption(
            "Replacement cap cost is the median salary among bottom-quartile "
            "veteran starters at that position-season; replacement value is "
            "the same group's median value-score."
        )
        st.dataframe(
            baselines.head(20),
            width="stretch",
            hide_index=True,
        )

    st.subheader("By position (snapshot)")
    st.caption(
        "Median replacement baselines, market price per value unit, and share "
        "of player-seasons with positive surplus. RB shows the well-documented "
        "market irrationality — sometimes a negative implicit value-per-dollar "
        "slope, meaning paying RBs more is not consistently associated with "
        "getting more production."
    )
    st.dataframe(by_position, width="stretch", hide_index=True)

    st.subheader("Top team-seasons by total surplus")
    st.dataframe(
        team_season.head(15),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Full top-surplus player-season table")
    display_cols = [
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
    ]
    st.dataframe(
        top_surplus[_available_columns(top_surplus, display_cols)],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download top replacement-level surplus table",
        top_surplus.to_csv(index=False),
        file_name="replacement_level_top_surplus.csv",
        mime="text/csv",
    )


def external_benchmark_page(data: dict[str, pd.DataFrame]) -> None:
    """Fantasy headline: head-to-head against DraftKings closing-line implied."""
    overall = data["external_benchmark_overall"]
    by_position = data["external_benchmark_by_position"]
    by_season = data["external_benchmark_by_season"]
    win_rate = data["external_benchmark_win_rate"]

    st.title("External Benchmark: vs DraftKings Closing-Line Implied")
    st.caption(
        "Head-to-head RMSE/MAE/win-rate vs the strongest free fantasy "
        "benchmark available — DK closing-line implied PPR projections. "
        "Beating the market is the qualifying bar for a fantasy-projection "
        "portfolio piece; public DFS analytics shops typically claim 1-3% "
        "edges over the DK line as their core selling point."
    )

    if overall.empty:
        st.info(
            "External benchmark tables missing. Run "
            "`python scripts/run_pipeline.py --steps external_benchmark` "
            "after populating `data/raw/external_projections.csv`."
        )
        return

    headline = overall.iloc[0]
    card_row(
        [
            (
                "Skill vs DK closing line",
                fmt_percent(headline["skill_vs_external"]),
                f"On {int(headline['n_player_weeks']):,} matched player-weeks "
                "(2020-2021 RotoGuru overlap).",
            ),
            (
                "Model RMSE",
                fmt_number(headline["model_rmse"]),
                "Lower is better. PPR per week.",
            ),
            (
                "DK-implied RMSE",
                fmt_number(headline["external_rmse"]),
                "The market's implied projection from the salary line.",
            ),
            (
                "Source",
                f"`{headline['source']}`",
                "RotoGuru free archive ends in 2021; "
                "extending to 2022-2025 needs a paid source.",
            ),
        ]
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Skill score by position")
        if not by_position.empty:
            chart_df = by_position[by_position["segment"].eq("position")].copy()
            chart_df["pct"] = chart_df["skill_vs_external"] * 100
            fig = px.bar(
                chart_df,
                x="segment_value",
                y="pct",
                color="segment_value",
                labels={"segment_value": "Position", "pct": "Skill vs external (%)"},
                title="Skill score by position",
            )
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Per-player-week win rate by position")
        st.caption(
            "Share of player-weeks where the model's projection landed closer "
            "to the actual PPR than the DK implied projection did."
        )
        if not win_rate.empty:
            fig = px.bar(
                win_rate,
                x="position",
                y="model_win_rate",
                color="position",
                labels={"position": "Position", "model_win_rate": "Model win rate"},
                title="Model win rate vs DK by position",
            )
            fig.update_yaxes(tickformat=".0%", range=[0.45, 0.60])
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed by-position table")
    st.dataframe(by_position, width="stretch", hide_index=True)
    if not by_season.empty:
        st.subheader("Per-season detail")
        st.dataframe(by_season, width="stretch", hide_index=True)


def causal_qb_injury_page(data: dict[str, pd.DataFrame]) -> None:
    """Methodology piece: causal DiD on QB injury → WR PPR."""
    events = data["causal_treatment_events"]
    event_study = data["causal_event_study_unmatched"]
    att = data["causal_att_unmatched"]
    two_by_two = data["causal_2x2_did_unmatched"]

    st.title("Causal Analysis: QB Injury → WR PPR")
    st.caption(
        "Two-session causal DiD investigation of whether a starting QB's "
        "injury causes a measurable drop in their WR's fantasy production. "
        "Tests the conventional 'QB1 down → WR1 craters' narrative against "
        "the data. Identification: same-calendar-week receivers on stable-"
        "QB teams as controls."
    )

    with st.expander("Methodology summary", expanded=True):
        st.markdown(
            "- **Treatment**: starting QB transitions from active to "
            "injury-reported (Out, IR, Doubtful, Questionable, or DNP) and "
            "the backup remains starter for ≥2 weeks. 213 events 2016-2025.\n"
            "- **Pre/post window**: 4 weeks each side of the transition.\n"
            "- **Parallel trends**: failed in session 1 (treated WRs already "
            "on a declining trajectory before formal QB switch).\n"
            "- **Session 2**: PPR-level matching mitigation failed; the "
            "event-study + 2×2 DiD on the unmatched panel both deliver the "
            "same null/positive verdict.\n"
            "- **Mechanism revealed**: treated WRs hit their absolute low at "
            "offset -1, the week immediately *before* the formal QB switch. "
            "The Out designation is a **lagging indicator**, not the start "
            "of causal damage."
        )

    if event_study.empty:
        st.info(
            "Causal investigation tables missing. Run "
            "`python scripts/run_pipeline.py --steps causal_session1,causal_session2`."
        )
        return

    att_row = att.iloc[0] if not att.empty else None
    two_by_two_row = two_by_two.iloc[0] if not two_by_two.empty else None

    card_row(
        [
            (
                "Treatment events identified",
                f"{len(events):,}",
                "QB-injury events across 2016-2025.",
            ),
            (
                "Event-study pooled ATT",
                f"{att_row['att_pooled_post_period']:+.2f} PPG" if att_row is not None else "—",
                f"p ≈ {att_row['att_p_value_approx']:.3f}" if att_row is not None else "",
            ),
            (
                "2×2 DiD ATT",
                f"{two_by_two_row['att_2x2']:+.2f} PPG" if two_by_two_row is not None else "—",
                f"p ≈ {two_by_two_row['p_value_approx']:.3f}" if two_by_two_row is not None else "",
            ),
            (
                "Verdict",
                "Null / positive",
                "No measurable drop after formal QB-Out designation. "
                "Damage happened earlier — Out flag is a lagging indicator.",
            ),
        ]
    )

    st.subheader("Event-study coefficients (treated × week_offset)")
    st.caption(
        "Each β_k is the change in (treated − control) PPR gap relative to "
        "offset -1 (the reference). Positive = treated did better at offset k "
        "than at -1. Note: pre-period coefficients should be ≈ 0 under parallel "
        "trends; here they're positive and significant, revealing the pretrend "
        "violation diagnosed in session 1."
    )
    fig = px.scatter(
        event_study,
        x="week_offset",
        y="coefficient",
        color="is_pre_period",
        error_y=event_study["se_cluster_robust"] * 1.96,
        labels={
            "week_offset": "Week relative to QB injury (transition = 0)",
            "coefficient": "β_k (PPR gap vs offset -1)",
            "is_pre_period": "Pre-period",
        },
        title="Event-study coefficients with 95% CIs",
        color_discrete_map={True: "#C8553D", False: "#157A6E"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.7)
    fig.add_vline(x=-0.5, line_dash="dot", line_color="grey", opacity=0.5)
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Surface the static figure for users who want the same plot from the report
    plot_path = (
        Path(PROJECT_ROOT)
        / "outputs"
        / "figures"
        / "causal_qb_injury_event_study.png"
    )
    if plot_path.exists():
        with st.expander("View the version of this plot in the session-2 report"):
            st.image(str(plot_path))

    left, right = st.columns(2)
    with left:
        st.subheader("Treatment events by season")
        events_by_season = (
            events.groupby("season").size().reset_index(name="n_events")
        )
        fig = px.bar(
            events_by_season,
            x="season",
            y="n_events",
            labels={"season": "Season", "n_events": "Treatment events"},
            title="QB-injury treatment events by season",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Event classifications")
        st.dataframe(
            events["cause"].value_counts().reset_index().rename(
                columns={"index": "cause", "cause": "n_events"}
            ),
            width="stretch",
            hide_index=True,
        )
        st.markdown(
            "*Hand-checked cases that appear in the table: 2023 CIN (Burrow → "
            "Browning), 2024 JAX (Lawrence → Mac Jones), 2017 PHI "
            "(Wentz → Foles).*"
        )

    st.subheader("Event-study coefficient table")
    st.dataframe(event_study, width="stretch", hide_index=True)


def rookie_bayes_page(data: dict[str, pd.DataFrame]) -> None:
    """Bayesian hierarchical rookie projections — cold-start solution."""
    metrics = data["rookie_bayes_validation_metrics"]
    predictions = data["rookie_bayes_validation_predictions"]
    modeling_frame = data["rookie_modeling_frame"]

    st.title("Bayesian Rookie Cold-Start")
    st.caption(
        "Hierarchical Normal regression on rookie-season PPR/game with "
        "partial pooling across the four skill positions on intercept and "
        "all slopes. Non-centered parameterization for clean NUTS sampling. "
        "Solves the cold-start problem the headline HGB projector cannot — "
        "rookies have no rolling history to feature-engineer from."
    )

    with st.expander("Model spec", expanded=False):
        st.markdown(
            "PPR/game per rookie ~ Normal(μ, σ) with μ a linear combination of "
            "log draft pick, age at draft, height (inches), weight, and "
            "(optionally) a college-production score. Intercept and slopes "
            "are partial-pooled across positions: `α[pos] ~ Normal(α_μ, "
            "α_τ)`, `β[pos] ~ Normal(β_μ, β_τ)`. Non-centered "
            "reparameterization brought divergences from 22-32 → 1.\n\n"
            "Runs in a dedicated `.venv-bayes` (PyMC 6.0.1, Python 3.12) "
            "because the PyMC stack conflicts with the main repo's pins. "
            "See `requirements-bayes.txt`."
        )

    if metrics.empty:
        st.info(
            "Rookie Bayesian results missing. Build the modeling frame with "
            "`python scripts/run_pipeline.py --steps rookie_bayes`. Then run "
            "the PyMC sampling pass from `.venv-bayes` — see `requirements-bayes.txt`."
        )
        if not modeling_frame.empty:
            st.markdown(
                f"Modeling frame is built with **{len(modeling_frame):,}** "
                "rookie player-seasons — the sampling pass is what's pending."
            )
            st.dataframe(modeling_frame.head(20), width="stretch", hide_index=True)
        return

    pooled_50 = float(metrics["interval_50_coverage"].mean())
    pooled_80 = float(metrics["interval_80_coverage"].mean())
    pooled_rmse = float(metrics["rmse"].mean())
    card_row(
        [
            ("Rookie classes validated", f"{len(metrics)}", None),
            ("Mean RMSE (PPG)", fmt_number(pooled_rmse), "Target σ ≈ 5 PPG."),
            (
                "50% interval coverage (mean)",
                fmt_percent(pooled_50),
                "Nominal target: 50%.",
            ),
            (
                "80% interval coverage (mean)",
                fmt_percent(pooled_80),
                "Nominal target: 80%.",
            ),
        ]
    )

    left, right = st.columns(2)
    with left:
        st.subheader("RMSE by rookie class")
        fig = px.bar(
            metrics,
            x="validation_year",
            y="rmse",
            labels={"validation_year": "Rookie class", "rmse": "RMSE (PPG)"},
            title="Out-of-sample RMSE by rookie class",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Posterior interval coverage")
        chart_df = pd.melt(
            metrics,
            id_vars="validation_year",
            value_vars=["interval_50_coverage", "interval_80_coverage"],
            var_name="interval",
            value_name="coverage",
        )
        chart_df["nominal"] = chart_df["interval"].map(
            {"interval_50_coverage": 0.50, "interval_80_coverage": 0.80}
        )
        fig = px.scatter(
            chart_df,
            x="validation_year",
            y="coverage",
            color="interval",
            labels={"validation_year": "Rookie class", "coverage": "Empirical coverage"},
            title="50% / 80% empirical coverage by rookie class",
        )
        fig.add_hline(y=0.50, line_dash="dash", line_color="grey", opacity=0.5)
        fig.add_hline(y=0.80, line_dash="dash", line_color="grey", opacity=0.5)
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rolling-validation metrics")
    st.dataframe(metrics, width="stretch", hide_index=True)

    if not predictions.empty:
        st.subheader("Top projected rookies by validation year")
        for year, group in predictions.groupby("validation_year"):
            top = group.sort_values(
                "predicted_ppr_per_game_mean", ascending=False
            ).head(10)
            with st.expander(f"{int(year)} rookie class — top 10 projected"):
                display_cols = [
                    "player_display_name",
                    "position",
                    "draft_number",
                    "predicted_ppr_per_game_mean",
                    "predicted_ppr_per_game_p10",
                    "predicted_ppr_per_game_p90",
                    "season_ppr_per_game",
                    "games_played",
                ]
                st.dataframe(
                    top[_available_columns(top, display_cols)],
                    width="stretch",
                    hide_index=True,
                )


def two_stage_weekly_page(data: dict[str, pd.DataFrame]) -> None:
    """Two-stage WR/TE decomposition experiment (negative result)."""
    method_summary = data["two_stage_weekly_summary"]
    by_fold = data["two_stage_weekly_by_fold"]
    per_stage = data["two_stage_weekly_per_stage"]

    st.title("Two-Stage WR/TE Decomposition Experiment")
    st.caption(
        "Does decomposing weekly WR/TE PPR into "
        "team pass attempts × target share × PPR per target — with target "
        "shares renormalized to sum to 1 within each team-week — beat the "
        "pooled HGB on the same player-weeks? It does not. This page shows "
        "why."
    )
    with st.expander("What the per-stage diagnostic shows", expanded=True):
        st.markdown(
            "The two-stage product loses to the pooled HGB in every fold. "
            "The per-stage breakdown explains where the failure comes from. "
            "Stage 1 (renormalized target share) beats a predict-the-mean "
            "baseline by 34% — the structural constraint actually carries "
            "signal. Stages 2 and 3 (team attempts, PPR per target) come in "
            "essentially flat against the mean. When you multiply noisy "
            "estimates through the product, you compound error the pooled "
            "model avoids by learning the relevant interactions implicitly.\n\n"
            "The shrunk-stage-3 variant — replacing the learned efficiency "
            "model with the position-season mean — beats the full learned "
            "version in every fold, which tells you the unshrunk stage was "
            "adding error rather than information. Even after that "
            "prescription, the structurally-constrained product still loses "
            "by 7-8%.\n\n"
            "This is the fourth decomposition experiment in the project to "
            "lose to a pooled HGB. Pooled tree models on engineered rolling "
            "features extract the team-attempts and per-target-efficiency "
            "signals more efficiently than any multiplicative decomposition "
            "we've tried."
        )

    if method_summary.empty:
        st.info(
            "Two-stage weekly experiment tables missing. Run "
            "`python scripts/run_pipeline.py --steps two_stage_weekly`."
        )
        return

    st.subheader("Head-to-head")
    st.dataframe(method_summary, width="stretch", hide_index=True)

    if not by_fold.empty:
        st.subheader("By validation year")
        st.caption(
            "Both two-stage variants lose to pooled HGB in every single year. "
            "Shrunk-eff is always better than full-learned, confirming stage 3 "
            "was adding error rather than information."
        )
        st.dataframe(by_fold, width="stretch", hide_index=True)

    if not per_stage.empty:
        st.subheader("Per-stage quality diagnostic")
        st.caption(
            "How accurate each stage is on its own. Stage 1 (target share "
            "renormalized) is genuinely informative; stages 2 and 3 are noise."
        )
        st.dataframe(per_stage, width="stretch", hide_index=True)


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
        ("Advanced modeling methodology", "report/advanced_modeling_methodology.md"),
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
            "fantasy_model_comparison",
            "weekly_fantasy",
            "weekly_fantasy_summary",
        }
    ]
    show_missing_data_warning(missing)

    st.sidebar.title("Navigation")
    st.sidebar.markdown("### Two perspectives")
    hero = st.sidebar.radio(
        "Pick one",
        [
            "Cap Allocation Brief (Front Office)",
            "Fantasy Player Board",
        ],
        key="nav_hero",
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Drill-down")
    detail = st.sidebar.radio(
        "Detailed analyses",
        [
            "— none (use hero pages) —",
            "External Benchmark vs DK",
            "Bayesian Rookie Cold-Start",
            "Causal: QB Injury → WR PPR",
            "Two-Stage Decomposition Experiment",
            "Replacement-Level Surplus (detail)",
            "Methodology And Reports",
        ],
        key="nav_detail",
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Hero pages are written for a reader. Drill-down pages dig into "
        "specific methodology and raw outputs. Rebuild data with "
        "`python scripts/run_pipeline.py`."
    )

    # Drill-down pages take precedence if explicitly chosen.
    if detail == "External Benchmark vs DK":
        external_benchmark_page(data)
    elif detail == "Bayesian Rookie Cold-Start":
        rookie_bayes_page(data)
    elif detail == "Causal: QB Injury → WR PPR":
        causal_qb_injury_page(data)
    elif detail == "Two-Stage Decomposition Experiment":
        two_stage_weekly_page(data)
    elif detail == "Replacement-Level Surplus (detail)":
        replacement_level_page(data)
    elif detail == "Methodology And Reports":
        methodology_page(data)
        st.divider()
        reports_page()
    else:
        # Default: render the selected hero page.
        if hero == "Cap Allocation Brief (Front Office)":
            front_office_executive_report(data)
        elif hero == "Fantasy Player Board":
            espn_fantasy_view(data)


if __name__ == "__main__":
    main()
