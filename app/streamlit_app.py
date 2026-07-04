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
import streamlit.components.v1 as components  # noqa: E402


PROJECT_ROOT = _PROJECT_ROOT
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"


st.set_page_config(
    page_title="NFL Player Value & Fantasy Forecasting",
    page_icon="🏈",
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
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 { color: #eaf1f8 !important; }
        /* Sidebar captions are quieter than headings, but still legible. */
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
            color: #b9cbe0 !important;
        }
        /* Keep dropdown/search controls readable (dark text on white). */
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: #ffffff; color: #16263a;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #16263a; }
        /* Inline code chips in the sidebar: readable on navy. */
        section[data-testid="stSidebar"] code {
            background: rgba(255,255,255,0.12); color: #eaf1f8;
        }

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

        /* Branded section header (eyebrow + title + subtitle). */
        .sec-head {
            border-left: 5px solid var(--brand-blue);
            padding: 0.15rem 0 0.15rem 0.85rem;
            margin: 0.2rem 0 1.0rem;
        }
        .sec-eyebrow {
            text-transform: uppercase; letter-spacing: 0.08em;
            font-size: 0.72rem; font-weight: 700; color: var(--brand-sky);
        }
        .sec-title { font-size: 1.7rem; font-weight: 800; color: var(--brand-navy);
                     line-height: 1.15; }
        .sec-sub { color: #5b6b7c; font-size: 1.0rem; margin-top: 0.15rem; }

        /* A clear callout box for the 'full write-up' pointer. */
        .writeup-hint {
            background: var(--brand-tint);
            border: 1px solid #d6e2f0;
            border-radius: 10px;
            padding: 0.6rem 0.85rem;
            font-size: 0.92rem;
            color: #284b6e;
            margin: 0.4rem 0 0.2rem;
        }

        /* Dataframe header: brand tint so tables read cleanly. */
        .stDataFrame thead tr th { background: var(--brand-tint) !important; }

        /* Expanders: subtle border + tinted header so they're noticeable. */
        details, .streamlit-expanderHeader, [data-testid="stExpander"] {
            border-radius: 10px;
        }
        [data-testid="stExpander"] summary { font-weight: 600; }

        /* Links pick up the brand color. */
        .main a { color: var(--brand-blue); }

        /* --- Sidebar navigation: radio restyled as nav rows. --- */
        /* Brand block at the top of the sidebar. */
        .side-brand {
            font-size: 1.08rem; font-weight: 800; color: #ffffff;
            letter-spacing: -0.01em; padding: 0.35rem 0 0.1rem;
            line-height: 1.25; white-space: nowrap;
        }
        .side-brand .sub {
            display: block; font-size: 0.76rem; font-weight: 500;
            color: #b9cbe0; letter-spacing: 0.02em; margin-top: 0.15rem;
        }
        /* Hide the radio circles; options become full-width rows. */
        section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
            display: none;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            width: 100%; margin: 2px 0; padding: 0.5rem 0.75rem;
            border-radius: 9px; cursor: pointer;
            transition: background 0.15s ease;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.08);
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: rgba(255, 255, 255, 0.15);
            box-shadow: inset 3px 0 0 var(--brand-sky);
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label p {
            font-size: 0.95rem; font-weight: 500;
        }

        /* --- Home-page section cards. --- */
        .home-card {
            background: #ffffff;
            border: 1px solid #dfe7ef;
            border-radius: 14px;
            padding: 1.1rem 1.2rem 0.9rem;
            height: 100%;
            box-shadow: 0 1px 3px rgba(13, 43, 69, 0.06);
        }
        .home-card .tag {
            display: inline-block; font-size: 0.7rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.07em;
            color: var(--brand-blue); background: var(--brand-tint);
            border-radius: 999px; padding: 0.14rem 0.6rem; margin-bottom: 0.5rem;
        }
        .home-card h4 {
            margin: 0 0 0.5rem; font-size: 1.04rem; font-weight: 700;
            color: var(--brand-navy); line-height: 1.3;
        }
        .home-card ul {
            margin: 0 0 0.25rem 1.05rem; padding: 0;
            color: #38434d; font-size: 0.88rem; line-height: 1.45;
        }
        .home-card li { margin-bottom: 0.3rem; }

        /* --- Home-page player tiles (top projected players strip). --- */
        .player-tile {
            background: #ffffff;
            border: 1px solid #dfe7ef;
            border-top: 4px solid var(--team-color, #0d2b45);
            border-radius: 12px;
            padding: 0.75rem 0.85rem 0.7rem;
            box-shadow: 0 1px 3px rgba(13, 43, 69, 0.06);
        }
        .player-tile .rank {
            font-size: 0.7rem; font-weight: 700; color: #8a97a5;
            letter-spacing: 0.05em;
        }
        .player-tile .name {
            font-size: 0.98rem; font-weight: 800; color: var(--brand-navy);
            line-height: 1.2; margin: 0.1rem 0 0.15rem;
        }
        .player-tile .meta {
            font-size: 0.76rem; font-weight: 600; color: #5b6b7c;
            text-transform: uppercase; letter-spacing: 0.03em;
        }
        .player-tile .points {
            font-size: 1.35rem; font-weight: 800; color: var(--brand-blue);
            margin-top: 0.35rem; line-height: 1;
        }
        .player-tile .points span {
            font-size: 0.72rem; font-weight: 600; color: #8a97a5;
            margin-left: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_components_css()
inject_theme_css()


def section_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    """Styled section header: a small colored eyebrow label, a bold title, and an
    optional one-line subtitle. Gives each section a consistent, branded top."""
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="sec-head">
            <div class="sec-eyebrow">{eyebrow}</div>
            <div class="sec-title">{title}</div>
            {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _scroll_top_on_tab_change() -> None:
    """Scroll the main panel back to the top when a tab is clicked, so switching
    tabs doesn't leave the reader stranded mid-page.

    Wrapped in a broad guard: components.html is deprecated upstream and
    st.iframe only accepts a src URL (no raw HTML+script), so there is no
    drop-in replacement yet. This helper is purely cosmetic — if a future
    Streamlit release removes the API, the app must keep working without it."""
    try:
        _render_scroll_script()
    except Exception:
        pass


def _render_scroll_script() -> None:
    components.html(
        """
        <script>
        const doc = window.parent.document;
        doc.querySelectorAll('button[data-baseweb="tab"]').forEach(function (btn) {
            if (!btn.dataset.scrollbound) {
                btn.dataset.scrollbound = '1';
                btn.addEventListener('click', function () {
                    setTimeout(function () {
                        const main = doc.querySelector('section.main')
                            || doc.querySelector('[data-testid="stMain"]')
                            || doc.querySelector('[data-testid="stAppViewContainer"]');
                        if (main && main.scrollTo) {
                            main.scrollTo({ top: 0, behavior: 'smooth' });
                        } else {
                            window.parent.scrollTo({ top: 0, behavior: 'smooth' });
                        }
                    }, 60);
                });
            }
        });
        </script>
        """,
        height=0,
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


def _reference_text() -> str:
    """The clean, plain-language project reference (PROJECT_REFERENCE.md), used to
    populate each section's 'Full write-up' panel."""
    path = PROJECT_ROOT / "PROJECT_REFERENCE.md"
    return load_markdown("PROJECT_REFERENCE.md", file_mtime(path))


def _full_writeup_expander(
    key: str,
    label: str = "Open the full write-up: models, metrics, methods & limitations",
) -> None:
    """Render a clear pointer plus an expandable panel with the reference sections
    backing this app section. No-ops if the reference file is missing."""
    detail = section_reference_markdown(_reference_text(), key)
    if detail:
        st.markdown(
            '<div class="writeup-hint">Want the depth? The full write-up below '
            "explains the models, metrics, methods, and limitations for this section "
            "in plain terms.</div>",
            unsafe_allow_html=True,
        )
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
        "salary": "salary_efficiency_2016_2025.csv",
        "methodology": "methodology_checks.csv",
        "fantasy": "2026_fantasy_football_projections.csv",
        "weekly_fantasy": "weekly_fantasy_validation_predictions.csv",
        "weekly_fantasy_live": "weekly_fantasy_live_projection.csv",
        # Season value decomposition: powers the stable/shaky role badge and
        # the regression watch (efficiency_variance_share per player).
        "two_stage_projection": "two_stage_2026_projection.csv",
        # Replacement-level surplus: headline for the research card + the
        # surplus history in Player Detail.
        "replacement_top_surplus": "salary_findings_replacement_top_surplus.csv",
        # External benchmark vs DraftKings (Draft Board accuracy tab)
        "external_benchmark_overall": "external_benchmark_overall.csv",
        "external_benchmark_by_position": "external_benchmark_by_position.csv",
        "external_benchmark_by_season": "external_benchmark_by_season.csv",
        "external_benchmark_win_rate": "external_benchmark_win_rate.csv",
        # Bayesian rookie projections (player index / Player Detail)
        "rookie_modeling_frame": "rookie_modeling_frame.csv",
        "rookie_bayes_validation_predictions": "rookie_bayes_validation_predictions.csv",
        # Causal QB-injury study: headline ATT for the research-studies card,
        # events for the player index / Player Detail view. The full analysis
        # lives in the repo reports, not as an app page.
        "causal_s3_att": "causal_s3_att.csv",
        "causal_s3_events": "causal_s3_first_report_events.csv",
    }
    return {
        name: load_csv(filename, file_mtime(TABLE_DIR / filename))
        for name, filename in files.items()
    }


def espn_fantasy_view(data: dict[str, pd.DataFrame]) -> None:
    """Fantasy rankings: top-25 2026 season projections per position, plus
    week-by-week projection-vs-actual for completed games. Table-first."""
    import numpy as np

    fantasy = data["fantasy"]
    weekly = data["weekly_fantasy"]

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

    low = pos["prediction_interval_low"].round(0)
    high = pos["prediction_interval_high"].round(0)
    profile_short = (
        pos.get("fantasy_projection_tier", pd.Series([""] * len(pos)))
        .astype(str)
        .str.replace(" Fantasy Profile", "", regex=False)
        .str.replace(" Profile", "", regex=False)
    )
    tiers = fc.assign_tiers(pos)
    badges = fc.stability_labels(data.get("two_stage_projection", pd.DataFrame()))
    role = (
        pos.merge(badges, on="player_id", how="left")["role_badge"].fillna("")
        if not badges.empty
        else pd.Series([""] * len(pos))
    )
    ranking = pd.DataFrame({
        "Rank": pos["Rank"].astype(int),
        "Tier": tiers.reindex(pos.index).astype(int),
        "Player": pos["player_display_name"],
        "Team": pos.get(team_col, ""),
        "Proj PPR": pos["predicted_2026_fantasy_points_ppr"].round(1),
        "PPR/G": pos["predicted_2026_ppr_per_game"].round(1),
        "GP": pos["predicted_2026_games_played"].round(0).astype(int),
        "80% range": [f"{lo:.0f}–{hi:.0f}" for lo, hi in zip(low, high)],
        "Role": role,
        "Profile": profile_short,
    })
    if "projection_change_from_2025" in pos.columns:
        ranking["Δ vs '25"] = pos["projection_change_from_2025"].round(1)

    proj_max = float(pos["predicted_2026_fantasy_points_ppr"].max() or 1.0)
    st.subheader(f"Top 25 {position}s: 2026 projected PPR")
    st.dataframe(
        ranking,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Proj PPR": st.column_config.ProgressColumn(
                "Proj PPR",
                help="Projected 2026 PPR points",
                format="%.1f",
                min_value=0.0,
                max_value=proj_max,
            ),
            "PPR/G": st.column_config.NumberColumn("PPR/G", width="small", format="%.1f"),
            "GP": st.column_config.NumberColumn("GP", width="small", format="%d"),
            "80% range": st.column_config.TextColumn(
                "80% range",
                width="small",
                help="80% prediction interval: the actual total should land in "
                "this range about 8 times in 10. Season totals are genuinely "
                "hard to predict, so honest ranges are wide.",
            ),
            "Tier": st.column_config.NumberColumn(
                "Tier",
                width="small",
                help="Players in the same tier are close enough that the model "
                "cannot confidently order them — take the tier, not the exact "
                "rank. A new tier starts where the drop in projected points is "
                "large relative to the projections' own uncertainty.",
            ),
            "Role": st.column_config.TextColumn(
                "Role",
                width="small",
                help="Stable = the projection rests on role (targets, snaps, "
                "carries), which repeats strongly year to year. Shaky = it "
                "leans on per-play efficiency, which barely repeats for "
                "RB/WR/TE. QBs get no badge: QB efficiency genuinely repeats, "
                "so the label would mislead there.",
            ),
            "Profile": st.column_config.TextColumn("Profile", width="medium"),
            "Δ vs '25": st.column_config.NumberColumn(
                "Δ vs '25",
                width="small",
                format="%+.1f",
                help="Projected 2026 total minus the player's actual 2025 total. "
                "Negative for most top players by design: career-best seasons "
                "tend to regress toward the mean, so the model projects below "
                "last year's peak.",
            ),
        },
    )
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
            st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

    caveat_callout(content["caveat"]["body"], content["caveat"]["label"])

    with st.expander("Detailed by-position table"):
        st.dataframe(by_position, width="stretch", hide_index=True)
    if not by_season.empty:
        with st.expander("Per-season detail"):
            st.dataframe(by_season, width="stretch", hide_index=True)

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
    st.plotly_chart(fig, width="stretch")

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


GITHUB_BLOB_BASE = "https://github.com/kylelevesque12/nfl-player-value-analysis/blob/main"


def reports_page() -> None:
    st.subheader("Underlying research notes")
    st.caption(
        "The original per-topic research notes behind this app, hosted in the "
        "GitHub repository. The methodology depth lives there; this app ships "
        "the results."
    )

    links = [
        ("Final project report", "report/final_project_report.md"),
        ("Methodology checks", "report/methodology_checks.md"),
        ("Model interpretation", "report/model_interpretation.md"),
        ("Salary-efficiency findings", "report/salary_efficiency_findings.md"),
        ("Season fantasy projection summary", "report/fantasy_football_projection_summary.md"),
        ("Weekly fantasy projection summary", "report/weekly_fantasy_projection_summary.md"),
        ("External benchmark (vs DraftKings)", "report/external_benchmark.md"),
        ("Rookie Bayesian projection", "report/rookie_bayes_projection.md"),
        ("QB injury causal study", "report/causal/qb_injury_session3.md"),
        ("Two-stage weekly decomposition (negative result)", "report/two_stage_weekly.md"),
        ("2026 prediction report summary", "report/2026_prediction_report_summary.md"),
    ]
    available = [(label, rel) for label, rel in links if (PROJECT_ROOT / rel).exists()]
    for label, relative_path in available:
        st.write(f"- [{label}]({GITHUB_BLOB_BASE}/{relative_path})")


# ---------------------------------------------------------------------------
# Landing page: pure content/config lives in app/landing_content.py
# ---------------------------------------------------------------------------
from app.landing_content import (  # noqa: E402
    LANDING_TITLE,
    LANDING_SUBTITLE,
    NAV_HOME,
    NAV_FANTASY,
    NAV_DRAFTROOM,
    NAV_METHOD,
    NAV_CAPTIONS,
    SECTIONS,
    TEAM_COLORS,
    DEFAULT_TEAM_COLOR,
    methodology_strip_labels,
)
from app import fantasy_content as fc  # noqa: E402


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


def _top_projected_strip(data: dict[str, pd.DataFrame]) -> None:
    """ESPN-style strip: the top five projected players for 2026 as team-color
    tiles, with position quick-links into the Draft Board."""
    fantasy = data.get("fantasy", pd.DataFrame())
    if fantasy.empty or "predicted_2026_fantasy_points_ppr" not in fantasy.columns:
        return

    st.markdown("### Top projected players, 2026")
    top = fantasy.sort_values(
        "predicted_2026_fantasy_points_ppr", ascending=False
    ).head(5)
    team_col = "primary_team_2025" if "primary_team_2025" in top.columns else "team"
    cols = st.columns(len(top))
    for col, (rank, (_, row)) in zip(cols, enumerate(top.iterrows(), start=1)):
        team = str(row.get(team_col, "") or "")
        color = TEAM_COLORS.get(team, DEFAULT_TEAM_COLOR)
        ppg = row.get("predicted_2026_ppr_per_game")
        ppg_txt = f"{ppg:.1f} per game" if pd.notna(ppg) else ""
        with col:
            st.markdown(
                f"""
                <div class="player-tile" style="--team-color:{color}">
                    <div class="rank">#{rank} · {row['position']}</div>
                    <div class="name">{row['player_display_name']}</div>
                    <div class="meta">{team} · {ppg_txt}</div>
                    <div class="points">{row['predicted_2026_fantasy_points_ppr']:.0f}
                    <span>proj PPR</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("Full rankings by position, with floors and ceilings:")
    pos_cols = st.columns(4)
    for col, pos in zip(pos_cols, ["QB", "RB", "WR", "TE"]):
        with col:
            if st.button(f"{pos} rankings", key=f"home_pos_{pos}", width="stretch"):
                st.session_state["rank_pos"] = pos
                _go_to(NAV_FANTASY)


def _player_content_modules(data: dict[str, pd.DataFrame]) -> None:
    """Home player-content modules: projected risers and the regression watch.

    A rookie-fliers module joins these once the 2026 rookie class is scored
    (roadmap: rookies into the season rankings)."""
    fantasy = data.get("fantasy", pd.DataFrame())
    two_stage = data.get("two_stage_projection", pd.DataFrame())

    risers = fc.risers_frame(fantasy)
    watch = fc.regression_watch_frame(fantasy, two_stage)
    if risers.empty and watch.empty:
        return

    left, right = st.columns(2)
    with left:
        st.markdown("### Projected risers")
        st.caption(
            "Projected to beat last season's total by the most. Several are "
            "returns from injury-shortened seasons."
        )
        if risers.empty:
            st.info("Riser data unavailable.")
        else:
            show = pd.DataFrame(
                {
                    "Player": risers["player_display_name"],
                    "Pos": risers["position"],
                    "Team": risers["team"],
                    "Proj PPR": risers[fc.PROJ_COL].round(0).astype(int),
                    "vs 2025": risers[fc.DELTA_COL].round(0).astype(int),
                }
            )
            st.dataframe(show, width="stretch", hide_index=True)

    with right:
        st.markdown("### Regression watch")
        st.caption(
            "Big seasons that leaned on per-play efficiency, the part of "
            "production that barely repeats year to year. Role-driven players "
            "are safer; these are the opposite."
        )
        if watch.empty:
            st.info("Regression-watch data unavailable.")
        else:
            show = pd.DataFrame(
                {
                    "Player": watch["player_display_name"],
                    "Pos": watch["position"],
                    "Team": watch["team"],
                    "Proj PPR": watch[fc.PROJ_COL].round(0).astype(int),
                    "vs 2025": watch[fc.DELTA_COL].round(0).astype(int),
                }
            )
            st.dataframe(show, width="stretch", hide_index=True)
    st.caption(
        "Why no quarterbacks on the regression watch: QB efficiency is the "
        "one kind that genuinely repeats, so the fade-the-fluke logic does "
        "not apply at that position."
    )


def landing_page(data: dict[str, pd.DataFrame]) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{LANDING_TITLE}</h1>
            <p>{LANDING_SUBTITLE}</p>
            <span class="pill">2026 draft prep</span>
            <span class="pill">Honest ranges</span>
            <span class="pill">QB · RB · WR · TE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _top_projected_strip(data)

    st.divider()
    _player_content_modules(data)

    st.divider()
    st.caption(" · ".join(f"✓ {label}" for label in methodology_strip_labels()))

    overview = reference_markdown(_reference_text(), [1, 2, 3, 12])
    if overview:
        with st.expander("Read the full project overview"):
            st.markdown(overview)

    with st.expander("How to use this app"):
        st.markdown(
            "The Draft Board has the 2026 rankings with tiers, floors, and "
            "ceilings; the Draft Room has the positional scarcity picture and "
            "will host the whole-draft planner; Player Detail assembles "
            "everything on one player. Methodology & Research holds the "
            "safeguards audit, the research study summaries, sources, and the "
            "full project report — the technical depth lives in the "
            "[GitHub repository](https://github.com/kylelevesque12/nfl-player-value-analysis)."
        )


def render_player_search(index: pd.DataFrame) -> None:
    """Always-visible sidebar player search. Selecting a player navigates to the
    unified Player Detail view (reuses the deferred-nav pattern)."""
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
        st.plotly_chart(fig, width="stretch")
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
def draft_room_section(data: dict[str, pd.DataFrame]) -> None:
    """Draft prep home: the positional scarcity picture now, the whole-draft
    planner (VORP + dropoff optimization over every remaining pick) next."""
    section_header(
        "Draft prep",
        "Draft Room",
        "Plan the whole draft, not just the next pick.",
    )

    fantasy = data.get("fantasy", pd.DataFrame())
    scarcity = fc.scarcity_frame(fantasy)
    if scarcity.empty:
        st.info(
            "Season projections are missing. Run "
            "`python scripts/run_pipeline.py --steps fantasy`."
        )
        return

    st.subheader("Positional scarcity: where the cliffs are")
    st.markdown(
        "Each line is one position's projected points by positional rank. A "
        "steep line means the position falls off fast, so waiting a round "
        "costs real points. A flat line means you can wait. This chart is the "
        "foundation the draft planner builds on."
    )
    fig = px.line(
        scarcity,
        x="positional_rank",
        y="projected_points",
        color="position",
        hover_data={"player": True, "positional_rank": True, "projected_points": ":.0f"},
        labels={
            "positional_rank": "Positional rank",
            "projected_points": "Projected 2026 PPR",
            "position": "",
        },
        color_discrete_map={
            "QB": "#C8553D", "RB": "#157A6E", "WR": "#3D6B99", "TE": "#B08900",
        },
    )
    fig.update_layout(height=460, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    dropoffs = fc.starter_window_dropoffs(scarcity)
    if not dropoffs.empty:
        st.subheader("The cost of drafting the 12th-best instead of the best")
        show = dropoffs.rename(
            columns={
                "position": "Position",
                "top_projection": "Best (proj PPR)",
                "rank12_projection": "12th-best (proj PPR)",
                "dropoff": "Points lost",
            }
        )
        st.dataframe(
            show.round(0),
            width="stretch",
            hide_index=True,
        )
        flattest = dropoffs.iloc[-1]["position"]
        st.caption(
            f"The flattest position ({flattest}) is the one to wait on. The "
            "tier column on the Draft Board shows where each position's "
            "cliffs fall, player by player."
        )

    st.divider()
    st.markdown(
        "**Coming before August drafts: the whole-draft planner.** Set your "
        "league (teams, pick slot, roster), track picks live, and get a plan "
        "for every remaining pick — built on value over replacement, the "
        "dropoff math above, and ADP-based projections of who will still be "
        "available at each of your turns. The build plan lives in the "
        f"[roadmap]({GITHUB_BLOB_BASE}/PORTFOLIO_ROADMAP.md)."
    )


def fantasy_section(data: dict[str, pd.DataFrame]) -> None:
    section_header(
        "Fantasy",
        "Draft Board",
        "2026 PPR projections by position, with week-by-week accuracy and the market benchmark.",
    )
    tab1, tab2 = st.tabs(["Rankings", "Accuracy & benchmark"])
    with tab1:
        espn_fantasy_view(data)
    with tab2:
        external_benchmark_page(data)
    _scroll_top_on_tab_change()
    st.divider()
    with st.expander("How these projections are built and graded"):
        st.markdown(
            "Two models feed this section. Season-long 2026 totals come from an "
            "Elastic Net (a disciplined linear model chosen from six candidates by "
            "lowest validation error). Weekly scores come from a gradient-boosting "
            "model (many small decision trees, each correcting the last) using only "
            "information known before kickoff (recent form, opponent, betting lines, "
            "weather, injury status). Accuracy is judged the way forecasters do: "
            "against a strong naive baseline (a player's recent-game average), which the "
            "weekly model beats by a steady 7–9% across six seasons, and against a "
            "DraftKings-implied market line on 2020–2021, where it is competitive to "
            "slightly ahead.\n\n"
            "**Tier** groups players the model cannot confidently separate: a new "
            "tier starts only where the drop in projected points is large relative "
            "to the projections' own uncertainty, so within a tier you should take "
            "the player you prefer, not the higher row. **Role** comes from the "
            "season value model's decomposition: production built on role "
            "(targets, snaps, carries) repeats strongly year to year, while "
            "per-play efficiency barely repeats for RB/WR/TE — so *Stable* marks "
            "role-driven projections and *Shaky* marks efficiency-driven ones. "
            "QBs get no Role badge because QB efficiency is the documented "
            "exception that does repeat."
        )
    _full_writeup_expander("fantasy")


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


def _project_report_tab() -> None:
    """Project report downloads + the full reference readable in-app, plus the
    sources block and links to the per-topic research notes on GitHub."""
    st.subheader("Project report")
    st.caption(
        "The complete write-up: every model, metric, and method in plain terms, "
        "with findings, safeguards, and limitations."
    )
    ref = _reference_text()
    if ref:
        cols = st.columns(2)
        with cols[0]:
            st.download_button(
                "Download report (Markdown)",
                ref,
                file_name="PROJECT_REFERENCE.md",
                mime="text/markdown",
                width="stretch",
            )
        docx_path = PROJECT_ROOT / "PROJECT_REFERENCE.docx"
        if docx_path.exists():
            with cols[1]:
                st.download_button(
                    "Download report (Word)",
                    docx_path.read_bytes(),
                    file_name="PROJECT_REFERENCE.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    width="stretch",
                )
        with st.expander("Read the full report in-app"):
            numbers = sorted(split_reference(ref).keys())
            st.markdown(reference_markdown(ref, numbers))
    else:
        st.info(
            "The project reference is unavailable. It lives at PROJECT_REFERENCE.md "
            "in the project root."
        )

    st.divider()
    _sources_block()
    st.divider()
    reports_page()


def _study_card(content: dict, report_path: str, headline: str | None = None) -> None:
    """Compact summary card for a research study whose full analysis lives in
    the GitHub repo: title, purpose, key findings, headline number, and a link.
    The app ships the conclusion; the repo carries the depth."""
    st.markdown(f"#### {content['title']}")
    st.caption(content["purpose"])
    if headline:
        st.markdown(headline)
    st.markdown("\n".join(f"- {point}" for point in content["summary"]))
    caveat_callout(content["caveat"]["body"], content["caveat"]["label"])
    st.markdown(f"[Read the full analysis on GitHub]({GITHUB_BLOB_BASE}/{report_path})")


def _research_studies_tab(data: dict[str, pd.DataFrame]) -> None:
    st.caption(
        "Research studies behind the product pages, summarized. Full write-ups, "
        "diagnostics, and code live in the GitHub repository."
    )
    att = data.get("causal_s3_att", pd.DataFrame())
    att_headline = None
    if not att.empty:
        att_row = att.iloc[0]
        att_headline = (
            f"**Headline estimate:** {att_row['att_pooled_post_period']:+.2f} PPG "
            f"pooled post-period effect (p ≈ {att_row['att_p_value_approx']:.3f}), "
            "on 104 first-report events."
        )
    top_surplus = data.get("replacement_top_surplus", pd.DataFrame())
    surplus_headline = None
    if not top_surplus.empty:
        lead = top_surplus.iloc[0]
        surplus_headline = (
            f"**Headline finding:** {int(lead['season'])} "
            f"{lead['player_display_name']} produced "
            f"**${lead['dollar_surplus_millions']:.1f}M** of surplus over a "
            "replacement-level player — the front-office study whose "
            "role-vs-efficiency machinery now powers the stable/shaky badges "
            "on the Draft Board."
        )
    _study_card(
        DETAIL_PAGES["surplus"], "report/salary_efficiency_findings.md", surplus_headline
    )
    st.divider()
    _study_card(DETAIL_PAGES["rookie"], "report/rookie_bayes_projection.md")
    st.divider()
    _study_card(DETAIL_PAGES["causal"], "report/causal/qb_injury_session3.md", att_headline)
    st.divider()
    _study_card(DETAIL_PAGES["two_stage"], "report/two_stage_weekly.md")
    st.divider()
    _full_writeup_expander("causal", "QB injury study, plain-language write-up")


def methodology_research_section(data: dict[str, pd.DataFrame]) -> None:
    """One section for everything methodology: the safeguards audit, summaries
    of the research studies, and the full project report. The product sections
    stay lean; the research depth lives in the GitHub repo."""
    section_header(
        "Research",
        "Methodology & Research",
        "How the models are built and checked, and the research studies behind the product pages.",
    )
    tab1, tab2, tab3 = st.tabs(
        ["Safeguards & checks", "Research studies", "Report & sources"]
    )
    with tab1:
        methodology_page(data)
        st.divider()
        _full_writeup_expander(
            "methodology", "Models, safeguards, limitations & roadmap (full write-up)"
        )
    with tab2:
        _research_studies_tab(data)
    with tab3:
        _project_report_tab()
    _scroll_top_on_tab_change()


def main() -> None:
    data = load_all_data()
    _handle_landing_nav()
    missing = [
        name
        for name, df in data.items()
        if df.empty
        and name
        in {
            "salary",
            "methodology",
            "fantasy",
            "weekly_fantasy",
            "two_stage_projection",
        }
    ]
    show_missing_data_warning(missing)

    player_index = _player_index_from_data(data)

    st.sidebar.markdown(
        """
        <div class="side-brand">🏈&nbsp;NFL Player Value
        <span class="sub">&amp; Fantasy Forecasting</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    section = st.sidebar.radio(
        "Section",
        SECTIONS,
        key="nav_section",
        label_visibility="collapsed",
        captions=[NAV_CAPTIONS.get(s, "") for s in SECTIONS],
    )
    render_player_search(player_index)
    st.sidebar.divider()
    st.sidebar.caption(
        "Projections cover the 2016-2025 seasons and the 2026 outlook. "
        "Code, data pipeline, and research notes: "
        "[GitHub](https://github.com/kylelevesque12/nfl-player-value-analysis)."
    )

    if section == NAV_FANTASY:
        fantasy_section(data)
    elif section == NAV_DRAFTROOM:
        draft_room_section(data)
    elif section == NAV_PLAYER:
        player_detail_page(data, player_index)
    elif section == NAV_METHOD:
        methodology_research_section(data)
    else:
        landing_page(data)


if __name__ == "__main__":
    main()
