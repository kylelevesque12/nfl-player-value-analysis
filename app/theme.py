"""All app chrome: the CSS cascade and the shared section header.

The three CSS injectors must run in this order — base, then components, then
theme — because each layer is written to override the one before it. The entry
script calls them in that order; nothing else should.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from app.components import inject_components_css


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



def inject_all_css() -> None:
    """Inject the full cascade in the one order that renders correctly."""
    inject_custom_css()
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

