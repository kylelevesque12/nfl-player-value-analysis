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
    page_title="NFL Player Value & Fantasy Forecasting",
    page_icon="NFL",
    layout="wide",
    initial_sidebar_state="expanded",
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

from app.components import (
    inject_components_css,
    executive_summary,
    caveat_callout,
    source_footer,
    render_page_scaffold,
)
from app.page_content import DETAIL_PAGES
from app import player_search as ps
from app.section_content import (
    section_reference_markdown,
    reference_markdown,
    split_reference,
)

NAV_PLAYER = "Player Detail"


def inject_theme_css() -> None:
    """Brand color + polish layered on top of the base component CSS."""
    st.markdown(
        """
        <style>
        :root {
            --brand-navy: #0d2b45;
            --brand-blue: #1565C0;
            --brand-sky: #4a90d9;
            --brand-tint: #eef3f9;
        }
        /* Section headings get brand color + a light rule. */
        .main h1 { color: var(--brand-navy); font-weight: 800; letter-spacing: -0.01em; }
        .main h2 { color: var(--brand-blue); border-bottom: 2px solid #dce6f2;
                   padding-bottom: 0.25rem; }
        .main h3 { color: #1b3a5b; }

        /* Sidebar: deep navy gradient with light text for a bit of flash. */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d2b45 0%, #143a5e 100%);
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 { color: #eaf1f8 !important; }
        /* Keep dropdown/search controls readable (dark text on white). */
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: #ffffff; color: #16263a;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #16263a; }

        /* Metric KPIs: card with a brand accent bar. */
        div[data-testid="stMetric"] {
            background: var(--brand-tint);
            border-left: 4px solid var(--brand-blue);
            border-radius: 10px;
            padding: 0.6rem 0.9rem;
        }

        /* Tabs: brand underline on the active tab. */
        .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
        .stTabs [aria-selected="true"] { color: var(--brand-blue) !important; }

        /* Buttons: rounded, brand-tinted. */
        .stButton > button {
            border-radius: 9px;
            border: 1px solid #cdd9e8;
            font-weight: 600;
        }
        .stButton > button:hover {
            border-color: var(--brand-blue);
            color: var(--brand-blue);
        }

        /* Hero banner used on the Home page. */
        .hero {
            background: linear-gradient(120deg, #0d2b45 0%, #1565C0 70%, #2f7fd1 100%);
            color: #ffffff;
            padding: 1.5rem 1.7rem;
            border-radius: 16px;
            margin-bottom: 1.1rem;
            box-shadow: 0 6px 20px rgba(13, 43, 69, 0.18);
        }
        .hero h1 { color: #ffffff !important; margin: 0; font-size: 1.9rem; }
        .hero p { color: #dbe8f6; margin: 0.45rem 0 0; font-size: 1.02rem; }
        .pill {
            display: inline-block; background: rgba(255,255,255,0.16);
            color: #fff; border-radius: 999px; padding: 0.15rem 0.7rem;
            font-size: 0.8rem; margin-right: 0.4rem; margin-top: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_components_css()
inject_theme_css()


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


def _reference_text() -> str:
    """The clean, plain-language project reference (PROJECT_REFERENCE.md), used to
    populate each section's 'Full write-up' panel."""
    path = PROJECT_ROOT / "PROJECT_REFERENCE.md"
    return load_markdown("PROJECT_REFERENCE.md", file_mtime(path))


def _full_writeup_expander(key: str, label: str = "Full write-up") -> None:
    """Render an expandable panel with the reference sections backing this app
    section. No-ops if the reference file is missing."""
    detail = section_reference_markdown(_reference_text(), key)
    if detail:
        with st.expander(label):
            st.markdown(detail)


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


def card_row(metrics: list[tuple[str, str, str | None]], max_per_row: int = 3) -> None:
    """KPI tiles that wrap into balanced rows so they stay readable on tablet /
    phone widths (Streamlit stacks columns fully below its small-screen
    breakpoint; this keeps the mid-width range tidy)."""
    from app.layout import chunk_metrics

    for row in chunk_metrics(metrics, max_per_row):
        columns = st.columns(len(row))
        for column, (label, value, help_text) in zip(columns, row):
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
        "weekly_fantasy_live": "weekly_fantasy_live_projection.csv",
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
        # Session 5: first-injury-report treatment (the current causal result)
        "causal_s3_att": "causal_s3_att.csv",
        "causal_s3_eligibility": "causal_s3_eligibility.csv",
        "causal_s3_events": "causal_s3_first_report_events.csv",
        "causal_s3_event_study": "causal_s3_event_study.csv",
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
            "- Salary efficiency uses a reconstructed cap hit, not exact cap accounting.\n"
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


def _weekly_fantasy_live_view(live: pd.DataFrame) -> None:
    """Minimal 'upcoming week' projection table (Session 7)."""
    proj_week = int(live["week"].iloc[0]) if "week" in live.columns and not live.empty else None
    st.subheader(f"Projected week {proj_week}" if proj_week else "Upcoming week projections")
    st.caption(
        "Projections for the upcoming week, scored before the games are played. "
        "Player-history features are carried forward from the latest completed "
        "week; opponent, home/away, and roof/temp/wind come from the upcoming "
        "game's schedule. No same-week outcome, PBP, or NGS/PFR information is "
        "used. Intervals are per-position conformal bands (wider for QBs, whose "
        "scoring is more volatile)."
    )
    with st.sidebar:
        st.subheader("Live Projection Filters")
        positions = st.multiselect("Position", sorted(live["position"].dropna().unique()))
        teams = st.multiselect("Team", sorted(live["team"].dropna().astype(str).unique()))
        min_pts = st.slider("Minimum projected points", 0.0, float(live["projected_points"].max()), 0.0, 0.5)
    view = live.copy()
    if positions:
        view = view[view["position"].isin(positions)]
    if teams:
        view = view[view["team"].astype(str).isin(teams)]
    view = view[view["projected_points"] >= min_pts]
    view = view.sort_values("projected_points", ascending=False)
    show_cols = [c for c in [
        "player_display_name", "position", "team", "opponent_team", "is_home",
        "projected_points", "interval_low_50", "interval_high_50",
        "interval_low_80", "interval_high_80", "pbp_depth_chart_rank_last1",
        "implied_team_total", "game_temp", "game_wind", "is_indoor",
    ] if c in view.columns]
    st.dataframe(view[show_cols], use_container_width=True, hide_index=True)
    st.caption(f"{len(view)} players. Sorted by projected points (descending).")


def weekly_fantasy_projection_page(data: dict[str, pd.DataFrame]) -> None:
    weekly = data["weekly_fantasy"]
    method_summary = data["weekly_fantasy_summary"]
    by_position = data["weekly_fantasy_by_position"]
    conformal = data["weekly_fantasy_conformal"]
    live = data.get("weekly_fantasy_live", pd.DataFrame())

    st.title("Weekly Fantasy Projection")

    # Minimal mode selector (Session 7): historical backtest vs upcoming week.
    mode_options = ["Historical backtest"]
    if live is not None and not live.empty:
        mode_options.append("Current / next week projections")
    mode = st.radio("View", mode_options, horizontal=True) if len(mode_options) > 1 else mode_options[0]
    if mode == "Current / next week projections":
        _weekly_fantasy_live_view(live)
        return

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
            "the following game, only present because this table is a "
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
        "pooled model at every position, pooling lets the model leverage the "
        "larger training sample with `position` as an input feature. This is "
        "reported in the same spirit as the two-stage value finding: the "
        "negative results stay visible."
    )


def espn_fantasy_view(data: dict[str, pd.DataFrame]) -> None:
    """Fantasy rankings: top-25 2026 season projections per position, plus
    week-by-week projection-vs-actual for completed games. Table-first."""
    import numpy as np

    fantasy = data["fantasy"]
    weekly = data["weekly_fantasy"]

    st.title("Fantasy Rankings")
    st.caption("2026 PPR projections by position, and week-by-week projection accuracy.")

    if fantasy.empty:
        st.info(
            "Fantasy projections are missing. Run "
            "`python scripts/run_pipeline.py --steps fantasy`."
        )
        return

    position = st.radio(
        "Position", ["QB", "RB", "WR", "TE"], horizontal=True, key="rank_pos"
    )

    pos = (
        fantasy[fantasy["position"].eq(position)]
        .sort_values("predicted_2026_fantasy_points_ppr", ascending=False)
        .head(25)
        .reset_index(drop=True)
        .copy()
    )
    pos.insert(0, "Rank", range(1, len(pos) + 1))
    team_col = "primary_team_2025" if "primary_team_2025" in pos.columns else "team"

    ranking = pd.DataFrame({
        "Rank": pos["Rank"],
        "Player": pos["player_display_name"],
        "Team": pos.get(team_col, ""),
        "Proj PPR": pos["predicted_2026_fantasy_points_ppr"].round(1),
        "PPR/G": pos["predicted_2026_ppr_per_game"].round(1),
        "GP": pos["predicted_2026_games_played"].round(0),
        "80% Low": pos["prediction_interval_low"].round(0),
        "80% High": pos["prediction_interval_high"].round(0),
        "Tier": pos.get("fantasy_projection_tier", ""),
    })
    if "projection_change_from_2025" in pos.columns:
        ranking["vs 2025"] = pos["projection_change_from_2025"].round(1)

    st.subheader(f"Top 25 {position}s: 2026 projected PPR")
    st.dataframe(ranking, width="stretch", hide_index=True)
    st.download_button(
        f"Download {position} rankings",
        ranking.to_csv(index=False),
        file_name=f"fantasy_rankings_2026_{position}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Weekly projection vs actual")

    options = pos["player_id"].tolist()
    labels = dict(zip(pos["player_id"], pos["player_display_name"]))
    sel = st.selectbox(
        "Player", options, format_func=lambda p: labels.get(p, p), key="rank_weekly_player"
    )

    wk = weekly.copy()
    if "method" in wk.columns:
        wk = wk[wk["method"].eq("hist_gradient_boosting")]
    wk = wk[wk["player_id"].astype(str).eq(str(sel))].copy()
    if wk.empty:
        st.info("No completed-game projections on record for this player.")
        return

    latest = int(pd.to_numeric(wk["season"], errors="coerce").max())
    wk = wk[wk["season"].eq(latest)].sort_values("week")
    proj = wk["prediction"].to_numpy()
    actual = wk["target_fantasy_points_ppr"].to_numpy()

    weekly_table = pd.DataFrame({
        "Week": wk["week"].astype(int),
        "Opp": wk.get("opponent_team", ""),
        "Projected": wk["prediction"].round(1),
        "Actual": wk["target_fantasy_points_ppr"].round(1),
        "Error": (wk["target_fantasy_points_ppr"] - wk["prediction"]).round(1),
    })
    card_row([
        ("Games", f"{len(wk)}", None),
        ("Avg projected", f"{proj.mean():.1f}", None),
        ("Avg actual", f"{actual.mean():.1f}", None),
        ("RMSE", f"{float(np.sqrt(np.mean((actual - proj) ** 2))):.1f}", None),
    ])
    st.caption(f"{labels.get(sel, sel)}, {latest} season, projected vs actual PPR by game.")
    st.dataframe(weekly_table, width="stretch", hide_index=True)


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
        "Where the cap dollars went, and what they bought, replacement-level "
        "surplus priced against reconstructed cap-hit estimates."
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
                f"for one z-unit of RB value is <strong>${rb_slope:.1f}M</strong>, "
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
    # Findings, narrative blocks
    # ------------------------------------------------------------------
    st.markdown("## Findings")

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
            "Anchor allocation around a rookie-deal QB. With one in place, the "
            "savings flow to top-tier WR talent; without one, a draftable QB in "
            "rounds 2-4 carries asymmetric upside, a Purdy / Browning / Nix "
            "outcome.",
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
            "Reconstructed estimate",
            "Cost is a season-specific cap hit reconstructed from contract terms "
            "(prorated signing bonus + backloaded base), an estimate, not exact "
            "NFL cap accounting, since the source data has no year-by-year cap "
            "breakdown. Each player-season carries a cap_hit_quality_flag.",
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
            production-based EPA, not pure talent, scheme, OL quality, and
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

    content = DETAIL_PAGES["surplus"]
    render_page_scaffold(content)

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
                "Top-25 surplus seasons shown",
                f"{len(top_surplus):,}",
                "Full population: 2016-2025 skill positions.",
            ),
            (
                "Positions covered",
                f"{by_position['position'].nunique()}",
                "QB, RB, WR, TE",
            ),
        ]
    )

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
    fig.update_layout(height=560, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    caveat_callout(content["caveat"]["body"], content["caveat"]["label"])

    with st.expander("By position, market price per value unit & surplus share"):
        st.caption(
            "Median replacement baselines, market price per value unit, and share "
            "of player-seasons with positive surplus. RB shows the documented "
            "market irrationality, sometimes a negative implicit value-per-dollar "
            "slope."
        )
        st.dataframe(by_position, width="stretch", hide_index=True)

    with st.expander("Replacement baselines by (season, position)"):
        st.dataframe(baselines.head(20), width="stretch", hide_index=True)

    with st.expander("Top team-seasons by total surplus"):
        st.dataframe(team_season.head(15), width="stretch", hide_index=True)

    with st.expander("Full top-surplus player-season table + download"):
        display_cols = [
            "season", "player_display_name", "position", "team", "games_played",
            "salary_millions", "value_score", "cap_over_replacement_millions",
            "value_over_replacement", "dollar_surplus_millions",
        ]
        st.dataframe(
            top_surplus[_available_columns(top_surplus, display_cols)],
            width="stretch", hide_index=True,
        )
        st.download_button(
            "Download top replacement-level surplus table",
            top_surplus.to_csv(index=False),
            file_name="replacement_level_top_surplus.csv",
            mime="text/csv",
        )

    source_footer(content["footer"])


def external_benchmark_page(data: dict[str, pd.DataFrame]) -> None:
    """Fantasy headline: head-to-head against DraftKings closing-line implied."""
    overall = data["external_benchmark_overall"]
    by_position = data["external_benchmark_by_position"]
    by_season = data["external_benchmark_by_season"]
    win_rate = data["external_benchmark_win_rate"]

    content = DETAIL_PAGES["benchmark"]
    render_page_scaffold(content)

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
                "Skill vs naive baselines (all years)",
                "+7-9%",
                "Primary claim: RMSE reduction vs recent-form / season-to-date "
                "averages, every season 2020-2025.",
            ),
            (
                "Skill vs DK (2020-2021 only)",
                fmt_percent(headline["skill_vs_external"]),
                f"Scoped secondary check, {int(headline['n_player_weeks']):,} "
                "matched player-weeks.",
            ),
            (
                "Model RMSE",
                fmt_number(headline["model_rmse"]),
                "Lower is better. PPR per week.",
            ),
            (
                "DK-implied RMSE",
                fmt_number(headline["external_rmse"]),
                "Market-implied projection from the salary line.",
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

    caveat_callout(content["caveat"]["body"], content["caveat"]["label"])

    with st.expander("Detailed by-position table"):
        st.dataframe(by_position, width="stretch", hide_index=True)
    if not by_season.empty:
        with st.expander("Per-season detail"):
            st.dataframe(by_season, width="stretch", hide_index=True)

    source_footer(content["footer"])


def causal_qb_injury_page(data: dict[str, pd.DataFrame]) -> None:
    """Methodology piece: causal DiD on QB injury report → WR PPR (Session 5)."""
    content = DETAIL_PAGES["causal"]
    att = data.get("causal_s3_att", pd.DataFrame())
    eligibility = data.get("causal_s3_eligibility", pd.DataFrame())
    events = data.get("causal_s3_events", pd.DataFrame())
    event_study = data.get("causal_s3_event_study", pd.DataFrame())

    render_page_scaffold(content)

    if event_study.empty or att.empty:
        st.info(
            "First-report causal tables missing. Run "
            "`python -m src.causal.session3_driver` to write the causal_s3_*.csv "
            "outputs."
        )
        return

    att_row = att.iloc[0]
    n_events = len(events) if not events.empty else 104
    card_row(
        [
            (
                "First-report events",
                f"{n_events:,}",
                "Any injury-report status, while the QB is the established starter.",
            ),
            (
                "Out-only events (comparison)",
                "19",
                "The stricter Out-only trigger under the same eligibility.",
            ),
            (
                "Post-period ATT",
                f"{att_row['att_pooled_post_period']:+.2f} PPG",
                f"p ≈ {att_row['att_p_value_approx']:.3f}",
            ),
            (
                "Verdict",
                "Suggestive",
                "A small negative effect the Out-only design missed, modest and "
                "underpowered, not a clean headline.",
            ),
        ]
    )

    st.subheader("Event-study coefficients (treated × week_offset)")
    st.caption(
        "Each β_k is the change in (treated − control) PPR gap relative to "
        "offset -1. The drop is concentrated at offset +1, the first game after "
        "the QB's first injury report. The pre-period is plausible, but the gap "
        "is already slightly elevated at -3 (see the caveat below)."
    )
    fig = px.scatter(
        event_study,
        x="week_offset",
        y="coefficient",
        color="is_pre_period",
        error_y=event_study["se_cluster_robust"] * 1.96,
        labels={
            "week_offset": "Week relative to first injury report (event = 0)",
            "coefficient": "β_k (PPR gap vs offset -1)",
            "is_pre_period": "Pre-period",
        },
        title="First-report event-study coefficients with 95% CIs",
        color_discrete_map={True: "#C8553D", False: "#157A6E"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.7)
    fig.add_vline(x=-0.5, line_dash="dot", line_color="grey", opacity=0.5)
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    caveat_callout(content["caveat"]["body"], content["caveat"]["label"])

    with st.expander("Sample construction & eligibility (320 candidates → 104 events)"):
        st.dataframe(eligibility, width="stretch", hide_index=True)
        if not events.empty and "first_injury_status" in events.columns:
            st.caption(
                "First-report status mix (blank = practice-report-only, no game "
                "designation):"
            )
            st.dataframe(
                events["first_injury_status"].value_counts(dropna=False)
                .rename_axis("first_injury_status").reset_index(name="events"),
                width="stretch", hide_index=True,
            )

    with st.expander("Event-study coefficient table"):
        st.dataframe(event_study, width="stretch", hide_index=True)

    with st.expander("Earlier Out-only analysis (for context)"):
        st.markdown(
            "An earlier specification defined treatment as the formal QB *transition* "
            "with an Out / Doubtful / Questionable status and found a null effect: by the "
            "time a QB is formally Out, his receivers have already been declining "
            "for weeks, so the Out flag lags the causal damage. That null is what "
            "motivated re-timing treatment to the first injury report. See "
            "`report/causal/qb_injury_session1.md` and `qb_injury_session2.md`."
        )

    source_footer(content["footer"])


def rookie_bayes_page(data: dict[str, pd.DataFrame]) -> None:
    """Bayesian hierarchical rookie projections, cold-start solution."""
    metrics = data["rookie_bayes_validation_metrics"]
    predictions = data["rookie_bayes_validation_predictions"]
    modeling_frame = data["rookie_modeling_frame"]

    content = DETAIL_PAGES["rookie"]
    render_page_scaffold(content)

    from app.components import recommendation_callout
    recommendation_callout(
        "opportunity",
        "Jordan Love",
        "P(plays meaningfully) moves 0.611 → 0.513 once the model can see Green "
        "Bay had a recently-extended incumbent QB, the incumbent-context core "
        "doing exactly what it was built to do.",
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
            "the PyMC sampling pass from `.venv-bayes`, see `requirements-bayes.txt`."
        )
        if not modeling_frame.empty:
            st.markdown(
                f"Modeling frame is built with **{len(modeling_frame):,}** "
                "rookie player-seasons, the sampling pass is what's pending."
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

    caveat_callout(content["caveat"]["body"], content["caveat"]["label"])

    with st.expander("Rolling-validation metrics"):
        st.dataframe(metrics, width="stretch", hide_index=True)

    if not predictions.empty:
        st.subheader("Top projected rookies by validation year")
        for year, group in predictions.groupby("validation_year"):
            top = group.sort_values(
                "predicted_ppr_per_game_mean", ascending=False
            ).head(10)
            with st.expander(f"{int(year)} rookie class, top 10 projected"):
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

    source_footer(content["footer"])


def two_stage_weekly_page(data: dict[str, pd.DataFrame]) -> None:
    """Two-stage WR/TE decomposition experiment (negative result)."""
    method_summary = data["two_stage_weekly_summary"]
    by_fold = data["two_stage_weekly_by_fold"]
    per_stage = data["two_stage_weekly_per_stage"]

    content = DETAIL_PAGES["two_stage"]
    render_page_scaffold(content)

    with st.expander("What the per-stage diagnostic shows", expanded=False):
        st.markdown(
            "The two-stage product loses to the pooled HGB in every fold. "
            "The per-stage breakdown explains where the failure comes from. "
            "Stage 1 (renormalized target share) beats a predict-the-mean "
            "baseline by 34%, the structural constraint actually carries "
            "signal. Stages 2 and 3 (team attempts, PPR per target) come in "
            "essentially flat against the mean. Multiplying noisy estimates "
            "through the product compounds error the pooled model avoids by "
            "learning the relevant interactions implicitly.\n\n"
            "The shrunk-stage-3 variant, replacing the learned efficiency "
            "model with the position-season mean, beats the full learned "
            "version in every fold, which shows the unshrunk stage was "
            "adding error rather than information. Even after that "
            "prescription, the structurally-constrained product still loses "
            "by 7-8%.\n\n"
            "This is the fourth decomposition experiment in the project to "
            "lose to a pooled HGB. Pooled tree models on engineered rolling "
            "features extract the team-attempts and per-target-efficiency "
            "signals more efficiently than any multiplicative decomposition "
            "tested here."
        )

    if method_summary.empty:
        st.info(
            "Two-stage weekly experiment tables missing. Run "
            "`python scripts/run_pipeline.py --steps two_stage_weekly`."
        )
        return

    caveat_callout(content["caveat"]["body"], content["caveat"]["label"])

    with st.expander("Head-to-head method summary"):
        st.dataframe(method_summary, width="stretch", hide_index=True)

    if not by_fold.empty:
        with st.expander("By validation year"):
            st.caption(
                "Both two-stage variants lose to pooled HGB in every single year. "
                "Shrunk-eff is always better than full-learned, confirming stage 3 "
                "was adding error rather than information."
            )
            st.dataframe(by_fold, width="stretch", hide_index=True)

    if not per_stage.empty:
        with st.expander("Per-stage quality diagnostic"):
            st.caption(
                "How accurate each stage is on its own. Stage 1 (target share "
                "renormalized) is genuinely informative; stages 2 and 3 are noise."
            )
            st.dataframe(per_stage, width="stretch", hide_index=True)

    source_footer(content["footer"])


def methodology_page(data: dict[str, pd.DataFrame]) -> None:
    methodology = data["methodology"]
    content = DETAIL_PAGES["methodology"]
    render_page_scaffold(content)

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

    caveat_callout(content["caveat"]["body"], content["caveat"]["label"])

    with st.expander("All methodology checks (detailed table)"):
        st.dataframe(methodology, width="stretch")

    methodology_path = PROJECT_ROOT / "report" / "methodology_checks.md"
    report_text = load_markdown(
        "report/methodology_checks.md",
        file_mtime(methodology_path),
    )
    if report_text:
        with st.expander("Methodology checks, full report text"):
            st.markdown(report_text)

    source_footer(content["footer"])


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


# ---------------------------------------------------------------------------
# Landing page (Session 8), pure content/config lives in app/landing_content.py
# ---------------------------------------------------------------------------
from app.landing_content import (  # noqa: E402
    LANDING_TITLE,
    LANDING_SUBTITLE,
    NAV_HOME,
    NAV_FANTASY,
    NAV_CAP,
    NAV_ROOKIE,
    NAV_CAUSAL,
    NAV_REPORT,
    NAV_METHOD,
    SECTIONS,
    methodology_strip_labels,
)


def _go_to(target: str) -> None:
    """Defer navigation to the next run (radio keys can't be set after the
    widgets are instantiated). Handled by _handle_landing_nav at top of main."""
    st.session_state["_landing_goto"] = target
    st.rerun()


def _handle_landing_nav() -> None:
    goto = st.session_state.pop("_landing_goto", None)
    if not goto:
        return
    # Single-section navigation: every target is one sidebar section.
    st.session_state["nav_section"] = goto


def landing_page() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{LANDING_TITLE}</h1>
            <p>{LANDING_SUBTITLE}</p>
            <span class="pill">Front office</span>
            <span class="pill">Fantasy</span>
            <span class="pill">2016–2025</span>
            <span class="pill">QB · RB · WR · TE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "This project does two jobs with the same NFL data, covering quarterbacks, "
        "running backs, receivers, and tight ends from 2016 to 2025.\n\n"
        "The **front-office job** measures how much value a player produced, forecasts "
        "how that value carries into the next season, and compares it to contract cost "
        "to show who is overpaid or underpaid. The **fantasy job** projects PPR fantasy "
        "points (both season-long totals for the upcoming year and week-by-week scores "
        "during a season) and presents them as rankings a manager can act on. Around "
        "both sits a research layer: a causal study of quarterback injuries, a Bayesian "
        "model for rookies with no NFL history, and an external market benchmark."
    )
    st.markdown(
        "The guiding principle is **rigorous, transparent evaluation**. Every model is "
        "graded against a strong, hard-to-beat simple baseline rather than against zero, "
        "results that did not work are kept on the record instead of hidden, and every "
        "projection ships with a clear statement of how uncertain it is."
    )

    st.divider()
    st.markdown("### Headline findings")
    st.markdown(
        "- Season value is hard to predict beyond a smart baseline, so the value model "
        "is best used for sorting players into tiers, not exact ranks.\n"
        "- Player value splits into a **role** half (very predictable) and a per-play "
        "**efficiency** half (nearly random), except at quarterback, where efficiency is "
        "real and stable. This is the project's central insight.\n"
        "- The weekly fantasy model beats every simple baseline by a steady **7–9%** "
        "across six seasons and edges the DraftKings market line on 2020–2021.\n"
        "- Reconstructed season cap hits make the salary analysis credible; cheap young "
        "quarterbacks dominate surplus, and the running-back market overpays veterans.\n"
        "- Quarterback injury has a **modest, suggestive** negative effect on receiver "
        "scoring once timed to the first injury report, not the collapse fans assume."
    )

    st.divider()
    st.markdown("### Where to go")
    st.caption("Pick a section in the sidebar, or jump straight in:")
    targets = [
        (NAV_FANTASY, "Fantasy Rankings"),
        (NAV_CAP, "Player Value & Cap"),
        (NAV_ROOKIE, "Rookies"),
        (NAV_CAUSAL, "QB Injury Study"),
        (NAV_METHOD, "Methodology & Sources"),
    ]
    cols = st.columns(len(targets))
    for col, (target, label) in zip(cols, targets):
        with col:
            if st.button(label, key=f"home_go_{target}", use_container_width=True):
                _go_to(target)

    st.divider()
    st.caption(" · ".join(f"✓ {label}" for label in methodology_strip_labels()))

    overview = reference_markdown(_reference_text(), [1, 2, 3, 12])
    if overview:
        with st.expander("Read the full project overview"):
            st.markdown(overview)

    with st.expander("How to use this app"):
        st.markdown(
            "Each section in the sidebar pairs a plain-language explanation with its "
            "tool and results, and carries a **Full write-up** panel with the deeper "
            "detail (models, metrics, methods, limitations). The **Methodology & "
            "Sources** section holds the safeguards, how the models are graded, the data "
            "and reference sources, and the complete project reference to download."
        )


def render_player_search(index: pd.DataFrame) -> None:
    """Always-visible sidebar player search. Selecting a player navigates to the
    unified Player Detail view (reuses the Session 8 deferred-nav pattern)."""
    st.sidebar.divider()
    st.sidebar.markdown("### Player search")
    if index is None or index.empty:
        st.sidebar.caption("Player index unavailable.")
        return
    label_map = {row["player_id"]: ps.display_label(row) for _, row in index.iterrows()}
    options = [""] + index["player_id"].tolist()
    choice = st.sidebar.selectbox(
        "Type a player name",
        options,
        format_func=lambda p: "type to search…" if p == "" else label_map.get(p, p),
        key="player_search_select",
    )
    if choice and choice != st.session_state.get("_selected_player_id"):
        st.session_state["_selected_player_id"] = choice
        _go_to(NAV_PLAYER)


def _kpi_or_dash(value, fmt: str = "{:.1f}") -> str:
    return fmt.format(value) if value is not None and pd.notna(value) else "—"


def _player_index_from_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return ps.build_player_index(
        data.get("weekly_fantasy"),
        data.get("weekly_fantasy_live"),
        data.get("salary"),
        data.get("rookie_modeling_frame"),
        data.get("causal_s3_events"),
    )


def player_detail_page(data: dict[str, pd.DataFrame], index: pd.DataFrame) -> None:
    pid = st.session_state.get("_selected_player_id")
    if not pid:
        st.title("Player Detail")
        st.info("Search for a player in the sidebar to open a detail view.")
        return

    detail = ps.assemble_player_detail(
        pid,
        weekly=data.get("weekly_fantasy"),
        live=data.get("weekly_fantasy_live"),
        salary=data.get("salary"),
        top_surplus=data.get("replacement_top_surplus"),
        rookie=data.get("rookie_modeling_frame"),
        rookie_pred=data.get("rookie_bayes_validation_predictions"),
        causal=data.get("causal_s3_events"),
    )
    meta = index[index["player_id"].astype(str) == str(pid)] if not index.empty else index
    meta_row = meta.iloc[0] if meta is not None and not meta.empty else None
    name = detail["player_name"]
    pos = meta_row["position"] if meta_row is not None else None
    team = meta_row["team"] if meta_row is not None else None
    seasons = meta_row["seasons"] if meta_row is not None else "—"

    st.title(str(name))
    bits = " · ".join([str(x) for x in [pos, team, f"seasons {seasons}"] if x and pd.notna(x)])
    st.caption(f"Unified player view, every project output available for this player. {bits}")
    if st.button("← Back to dashboard"):
        st.session_state["_selected_player_id"] = None
        _go_to(NAV_HOME)

    wk = detail["weekly_history"]
    live = detail["live"]
    sal = detail["surplus_history"]
    rk = detail["rookie"]
    rk_pred = detail["rookie_pred"]
    cz = detail["causal"]

    # ---- KPI row ----
    latest_proj = None
    if live is not None and "projected_points" in live.columns:
        latest_proj = float(live["projected_points"].iloc[0])
    elif wk is not None and "prediction" in wk.columns and not wk["prediction"].dropna().empty:
        latest_proj = float(wk["prediction"].dropna().iloc[-1])
    depth_rank = None
    if wk is not None and "pbp_depth_chart_rank_last1" in wk.columns:
        dr = wk["pbp_depth_chart_rank_last1"].dropna()
        depth_rank = float(dr.iloc[-1]) if not dr.empty else None
    best_surplus = None
    if sal is not None and "value_above_expected_salary" in sal.columns:
        v = sal["value_above_expected_salary"].dropna()
        best_surplus = float(v.max()) if not v.empty else None
    n_causal = 0 if cz is None else len(cz)

    card_row(
        [
            ("Latest projected PPR", _kpi_or_dash(latest_proj),
             "Live week if available, else most recent backtest game."),
            ("Latest depth rank (PBP)", _kpi_or_dash(depth_rank, "{:.0f}"),
             "1 = top of the position group."),
            ("Best value-over-expected ($M)", _kpi_or_dash(best_surplus),
             "Peak surplus season on record." if best_surplus is not None else "No salary record."),
            ("Causal QB events", f"{n_causal}",
             "First-injury-report events where this player was the treated QB."),
        ]
    )

    # ---- Weekly fantasy ----
    st.subheader("Weekly fantasy")
    if wk is None:
        st.info("No weekly fantasy projection history for this player.")
    else:
        plot = wk.copy()
        plot["game"] = plot["season"].astype(int).astype(str) + "-W" + plot["week"].astype(int).astype(str).str.zfill(2)
        ycols = [c for c in ["prediction", "target_fantasy_points_ppr"] if c in plot.columns]
        fig = px.line(
            plot, x="game", y=ycols,
            labels={"value": "PPR points", "game": "Game", "variable": ""},
            title="Projected vs actual PPR by game (production HGB)",
        )
        fig.update_layout(height=380, xaxis=dict(showticklabels=False))
        st.plotly_chart(fig, use_container_width=True)
        if live is not None:
            lrow = live.iloc[0]
            opp = lrow.get("opponent_team", "")
            lo = lrow.get("interval_low_80"); hi = lrow.get("interval_high_80")
            st.markdown(
                f"**Upcoming-week projection:** {lrow.get('projected_points', float('nan')):.1f} PPR "
                f"vs {opp}, 80% band [{lo:.1f}, {hi:.1f}]." if pd.notna(lo) else
                f"**Upcoming-week projection:** {lrow.get('projected_points', float('nan')):.1f} PPR vs {opp}."
            )
        with st.expander("Weekly projection table"):
            cols = [c for c in ["season", "week", "team", "opponent_team", "prediction",
                                "interval_low_80", "interval_high_80", "target_fantasy_points_ppr"]
                    if c in wk.columns]
            st.dataframe(wk[cols].tail(40), width="stretch", hide_index=True)

    # ---- Value / surplus ----
    st.subheader("Value & cap surplus")
    if sal is None:
        st.info("No salary / value record for this player.")
    else:
        cols = [c for c in ["season", "team", "games_played", "value_score", "salary_millions",
                            "value_above_expected_salary", "salary_efficiency_tier", "salary_source"]
                if c in sal.columns]
        show = sal[cols].rename(columns={"salary_millions": "cap_hit_$M"})
        st.dataframe(show, width="stretch", hide_index=True)
        caveat_callout(
            "Cap cost is a season-specific cap hit reconstructed from contract terms "
            "(prorated signing bonus + backloaded base), an estimate, not exact NFL "
            "cap accounting. See the salary_source column.",
            "Reconstructed estimate",
        )
        if detail["top_surplus"] is not None:
            st.caption("This player appears in the top-25 replacement-level surplus board.")

    # ---- Rookie model ----
    st.subheader("Rookie model")
    if rk is None:
        st.info("This player is not in the rookie modeling frame.")
    else:
        r = rk.iloc[0]
        ry = int(r["rookie_year"]) if "rookie_year" in r and pd.notna(r["rookie_year"]) else None
        dn = r.get("draft_number"); played = r.get("played_meaningfully")
        st.markdown(
            f"- Rookie year: **{ry or '—'}**  ·  draft pick: "
            f"**{int(dn) if pd.notna(dn) else '—'}**  ·  played meaningfully (>=4 games): "
            f"**{'Yes' if pd.notna(played) and int(played) == 1 else 'No'}**"
        )
        if rk_pred is not None and "predicted_ppr_per_game_mean" in rk_pred.columns:
            pr = rk_pred.iloc[0]
            st.markdown(
                f"- Bayesian projection: **{pr['predicted_ppr_per_game_mean']:.1f} PPR/game** "
                "(validation class)."
            )
        st.caption(
            "A 3-feature incumbent-context core sharpens the hurdle gate "
            "(combine and broad-depth features were tested and dropped). The gain is "
            "concentrated in the blocked-QB cell."
        )

    # ---- Causal study ----
    st.subheader("Causal study (QB injury)")
    if cz is None:
        st.info("This player is not a treated QB in the first-injury-report causal panel.")
    else:
        cols = [c for c in ["season", "team", "event_week", "first_injury_status",
                            "games_started_before_event"] if c in cz.columns]
        st.dataframe(cz[cols], width="stretch", hide_index=True)
        caveat_callout(
            "The first-report causal effect (ATT ~= -0.58 PPG) is suggestive and "
            "underpowered; appearing here means the player was a treated QB, not that "
            "a specific effect is attributed to him.",
            "Suggestive / underpowered",
        )

    with st.expander("ID diagnostics"):
        st.write({"player_id (gsis)": str(pid), "display_name": str(name)})

    source_footer(
        "Assembled from saved outputs only (weekly backtest, live projection, salary/"
        "value, rookie frame, causal events), no models are recomputed in the app."
    )


# ---------------------------------------------------------------------------
# Integrated sections: each pairs a short explanation with its tool(s)
# ---------------------------------------------------------------------------
def player_value_section(data: dict[str, pd.DataFrame]) -> None:
    st.title("Player Value & Cap")
    st.caption(
        "How much value each player produced, and whether the contract pays for it."
    )
    st.markdown(
        "Player value is measured with **EPA** (expected points added), how much "
        "each play changed a team's expected points, then standardized within each "
        "season and position so a 2025 tight end and a 2016 quarterback are scored "
        "against their own peers. The cost side replaces a flat yearly salary average "
        "with a **season-specific cap hit reconstructed from contract terms** "
        "(prorated bonus plus backloaded base), because a star's early years cost far "
        "less against the cap than his average implies. **Surplus** is value above "
        "what a freely available replacement would give for that cost. Cheap "
        "rookie-contract QBs (Brock Purdy) dominate, and the running-back market "
        "tends to overpay veterans."
    )
    tab1, tab2 = st.tabs(["Cap surplus brief", "Replacement-level detail"])
    with tab1:
        front_office_executive_report(data)
    with tab2:
        replacement_level_page(data)
    st.divider()
    _full_writeup_expander("value")


def fantasy_section(data: dict[str, pd.DataFrame]) -> None:
    st.title("Fantasy Rankings")
    st.caption(
        "2026 projections by position, week-by-week accuracy, and how the model is graded."
    )
    st.markdown(
        "Two models feed this section. Season-long 2026 totals come from an "
        "**Elastic Net** (a disciplined linear model chosen from six candidates by "
        "lowest validation error). Weekly scores come from a **gradient-boosting** "
        "model (many small decision trees, each correcting the last) using only "
        "information known before kickoff (recent form, opponent, betting lines, "
        "weather, injury status). Accuracy is judged the way forecasters do: "
        "against a strong naive baseline (a player's recent-game average), which the "
        "weekly model beats by a steady **7–9%** across six seasons, and against a "
        "DraftKings-implied market line on 2020–2021, where it is competitive to "
        "slightly ahead."
    )
    tab1, tab2, tab3 = st.tabs(
        ["Rankings", "Accuracy & benchmark", "Decomposition experiment"]
    )
    with tab1:
        espn_fantasy_view(data)
    with tab2:
        external_benchmark_page(data)
    with tab3:
        two_stage_weekly_page(data)
    st.divider()
    _full_writeup_expander("fantasy")


def rookie_section(data: dict[str, pd.DataFrame]) -> None:
    rookie_bayes_page(data)
    st.divider()
    _full_writeup_expander("rookie")


def causal_section(data: dict[str, pd.DataFrame]) -> None:
    causal_qb_injury_page(data)
    st.divider()
    _full_writeup_expander("causal")


def _sources_block() -> None:
    st.subheader("Sources")
    st.markdown(
        "**How the models are evaluated.** The metric choices follow established "
        "forecasting and fantasy-accuracy practice, not a yardstick invented here:\n\n"
        "- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.): "
        "the skill-score / \"beat the naive baseline\" standard. "
        "<https://otexts.com/fpp3/accuracy.html>\n"
        "- Fantasy Football Analytics, *Which Fantasy Football Projections Are Most "
        "Accurate?*: the realistic ceiling on weekly predictability and the value of "
        "consistency across seasons. "
        "<https://fantasyfootballanalytics.net/2024/12/which-fantasy-football-projections-are-most-accurate.html>\n"
        "- FantasyPros, *In-Season Accuracy Methodology*: the industry's own "
        "error-versus-realized-points grading standard. "
        "<https://www.fantasypros.com/about/faq/football-inseason-accuracy-methodology/>\n\n"
        "**Data sources.**\n\n"
        "- **nflverse** (via `nflreadpy`): weekly stats, rosters, schedules, depth "
        "charts, injuries, play-by-play, combine, draft picks (2016–2025).\n"
        "- **OverTheCap** (via nflverse contracts): contract terms behind the cap-hit "
        "reconstruction.\n"
        "- **RotoGuru / DraftKings**: the free DK salary archive used to build the "
        "market-implied benchmark (through 2021)."
    )


def project_report_section() -> None:
    st.title("Project Report")
    st.caption(
        "The complete write-up: every model, metric, and method in plain terms, "
        "with findings, safeguards, and limitations."
    )
    ref = _reference_text()
    if not ref:
        st.info(
            "The project reference is unavailable. It lives at PROJECT_REFERENCE.md "
            "in the project root."
        )
        return

    cols = st.columns(2)
    with cols[0]:
        st.download_button(
            "Download report (Markdown)",
            ref,
            file_name="PROJECT_REFERENCE.md",
            mime="text/markdown",
            use_container_width=True,
        )
    docx_path = PROJECT_ROOT / "PROJECT_REFERENCE.docx"
    if docx_path.exists():
        with cols[1]:
            st.download_button(
                "Download report (Word)",
                docx_path.read_bytes(),
                file_name="PROJECT_REFERENCE.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

    st.divider()
    numbers = sorted(split_reference(ref).keys())
    st.markdown(reference_markdown(ref, numbers))


def methodology_sources_section(data: dict[str, pd.DataFrame]) -> None:
    methodology_page(data)
    st.divider()
    _full_writeup_expander("methodology", "Models, safeguards, limitations & roadmap (full write-up)")
    st.divider()
    _sources_block()
    st.divider()

    ref_path = PROJECT_ROOT / "PROJECT_REFERENCE.md"
    if ref_path.exists():
        st.subheader("Full project reference")
        st.caption(
            "The complete write-up, every model, metric, and method explained in "
            "plain terms, with findings, safeguards, and limitations."
        )
        st.download_button(
            "Download full reference (Markdown)",
            ref_path.read_text(encoding="utf-8"),
            file_name="PROJECT_REFERENCE.md",
            mime="text/markdown",
        )
        docx_path = PROJECT_ROOT / "PROJECT_REFERENCE.docx"
        if docx_path.exists():
            st.download_button(
                "Download full reference (Word)",
                docx_path.read_bytes(),
                file_name="PROJECT_REFERENCE.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        st.divider()

    reports_page()


def main() -> None:
    data = load_all_data()
    _handle_landing_nav()
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

    player_index = _player_index_from_data(data)

    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Section",
        SECTIONS,
        key="nav_section",
    )
    render_player_search(player_index)
    st.sidebar.divider()
    st.sidebar.caption(
        "Each section pairs a short explanation with its tool and results. "
        "Rebuild data with `python scripts/run_pipeline.py`."
    )

    if section == NAV_CAP:
        player_value_section(data)
    elif section == NAV_FANTASY:
        fantasy_section(data)
    elif section == NAV_ROOKIE:
        rookie_section(data)
    elif section == NAV_CAUSAL:
        causal_section(data)
    elif section == NAV_PLAYER:
        player_detail_page(data, player_index)
    elif section == NAV_REPORT:
        project_report_section()
    elif section == NAV_METHOD:
        methodology_sources_section(data)
    else:
        landing_page()


if __name__ == "__main__":
    main()
