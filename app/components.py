"""Shared design components for the NFL player value Streamlit dashboard.

Reusable primitives so the perspective-specific pages (front-office report,
ESPN-style fantasy view, ESPN-style weekly games) share a consistent visual
language. Each function takes pandas inputs and renders Streamlit HTML.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# CSS injection (call once at app start; safe to call repeatedly)
# ---------------------------------------------------------------------------
def inject_components_css() -> None:
    st.markdown(
        """
        <style>
        /* Card primitives -----------------------------------------------*/
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
            margin-top: 8px;
            margin-bottom: 16px;
        }

        .player-card {
            background: #FFFFFF;
            border: 1px solid #DFE7EF;
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(24,32,38,0.06);
            display: flex;
            flex-direction: column;
            gap: 8px;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .player-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 14px rgba(24,32,38,0.12);
        }
        .player-card .player-name {
            font-size: 1.05rem;
            font-weight: 700;
            color: #182026;
            line-height: 1.1;
            margin-bottom: 2px;
        }
        .player-card .player-meta {
            font-size: 0.78rem;
            color: #5E6A75;
            margin-bottom: 6px;
        }
        .player-card .projection {
            font-size: 1.75rem;
            font-weight: 800;
            color: #157A6E;
            line-height: 1;
            margin: 4px 0;
        }
        .player-card .projection-unit {
            font-size: 0.7rem;
            font-weight: 500;
            color: #5E6A75;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .player-card .ceiling-floor {
            font-size: 0.78rem;
            color: #5E6A75;
            margin-top: 2px;
        }
        .player-card .meta-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid #EEF1F4;
            padding-top: 8px;
            margin-top: 4px;
            font-size: 0.78rem;
            color: #5E6A75;
        }

        /* Tier badge ---------------------------------------------------*/
        .tier-badge {
            display: inline-block;
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 3px 9px;
            border-radius: 999px;
            margin-right: 4px;
        }
        .tier-elite        { background: #157A6E; color: #FFFFFF; }
        .tier-strong       { background: #3D6B99; color: #FFFFFF; }
        .tier-flex         { background: #B08900; color: #FFFFFF; }
        .tier-bench        { background: #BCC4CC; color: #182026; }

        /* Position badge -----------------------------------------------*/
        .pos-badge {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 5px;
            margin-right: 6px;
            background: #182026;
            color: #FFFFFF;
        }
        .pos-QB { background: #C8553D; }
        .pos-RB { background: #157A6E; }
        .pos-WR { background: #3D6B99; }
        .pos-TE { background: #B08900; }

        /* Game card ----------------------------------------------------*/
        .game-card {
            background: #FFFFFF;
            border: 1px solid #DFE7EF;
            border-radius: 12px;
            padding: 14px 18px;
            box-shadow: 0 1px 3px rgba(24,32,38,0.05);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .game-card .matchup {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 1.05rem;
            font-weight: 700;
        }
        .game-card .matchup .team {
            flex: 1;
            text-align: center;
        }
        .game-card .matchup .at {
            font-size: 0.85rem;
            font-weight: 500;
            color: #BCC4CC;
            padding: 0 14px;
        }
        .game-card .meta {
            font-size: 0.78rem;
            color: #5E6A75;
            display: flex;
            justify-content: space-between;
            border-top: 1px solid #EEF1F4;
            padding-top: 8px;
        }
        .game-card .pick {
            font-size: 0.86rem;
            color: #182026;
            background: #F6F8FB;
            border-left: 3px solid #157A6E;
            padding: 8px 12px;
            border-radius: 0 6px 6px 0;
        }
        .game-card .confidence-bar {
            display: flex;
            height: 12px;
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid #DFE7EF;
        }
        .game-card .conf-segment {
            text-align: center;
            font-size: 0.62rem;
            font-weight: 700;
            color: #FFFFFF;
            padding: 1px 0;
        }
        .game-card .conf-away   { background: #3D6B99; }
        .game-card .conf-home   { background: #157A6E; }

        /* Executive summary block --------------------------------------*/
        .exec-summary {
            background: linear-gradient(135deg, #F6F8FB 0%, #FFFFFF 100%);
            border: 1px solid #DFE7EF;
            border-left: 5px solid #157A6E;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 18px;
        }
        .exec-summary h4 {
            margin: 0 0 10px 0;
            font-size: 0.85rem;
            color: #5E6A75;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }
        .exec-summary ul {
            margin: 0;
            padding-left: 18px;
            color: #182026;
        }
        .exec-summary ul li {
            margin: 6px 0;
            line-height: 1.5;
            font-size: 0.94rem;
        }
        .exec-summary ul li strong {
            color: #157A6E;
        }

        /* Recommendation callout ---------------------------------------*/
        .reco-callout {
            background: #FFFFFF;
            border: 1px solid #DFE7EF;
            border-radius: 8px;
            padding: 14px 16px;
            margin: 12px 0;
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }
        .reco-callout .reco-tag {
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            background: #157A6E;
            color: #FFFFFF;
            padding: 3px 8px;
            border-radius: 4px;
            white-space: nowrap;
        }
        .reco-callout.warning .reco-tag  { background: #C8553D; }
        .reco-callout.opportunity .reco-tag { background: #157A6E; }
        .reco-callout.caveat .reco-tag      { background: #B08900; }
        .reco-callout .reco-body {
            font-size: 0.92rem;
            color: #182026;
            line-height: 1.45;
        }

        /* Filter bar ---------------------------------------------------*/
        .filter-bar {
            background: #FFFFFF;
            border: 1px solid #DFE7EF;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------
def tier_badge_html(tier_label: str | None) -> str:
    if not tier_label or pd.isna(tier_label):
        return ""
    label = str(tier_label)
    css_map = {
        "Elite Fantasy Profile": "tier-elite",
        "Elite Upside": "tier-elite",
        "Core Starter": "tier-elite",
        "Strong Starter": "tier-strong",
        "Starter/Flex": "tier-flex",
        "Depth/Volatile": "tier-bench",
        "Low Projection": "tier-bench",
        "Volatile Depth": "tier-bench",
    }
    css = css_map.get(label, "tier-flex")
    short = {
        "Elite Fantasy Profile": "Elite",
        "Elite Upside": "Elite",
        "Core Starter": "Must Start",
        "Strong Starter": "Strong",
        "Starter/Flex": "Flex",
        "Depth/Volatile": "Depth",
        "Volatile Depth": "Volatile",
        "Low Projection": "Bench",
    }
    text = short.get(label, label)
    return f'<span class="tier-badge {css}">{text}</span>'


def position_badge_html(position: str | None) -> str:
    if not position or pd.isna(position):
        return ""
    pos = str(position).upper()
    return f'<span class="pos-badge pos-{pos}">{pos}</span>'


def trend_arrow(change: float | None) -> str:
    if change is None or pd.isna(change):
        return ""
    if change >= 30:
        return '<span style="color:#157A6E;font-weight:700;">▲</span>'
    if change <= -30:
        return '<span style="color:#C8553D;font-weight:700;">▼</span>'
    return '<span style="color:#5E6A75;">—</span>'


# ---------------------------------------------------------------------------
# Player card (ESPN fantasy style)
# ---------------------------------------------------------------------------
def player_card_html(
    *,
    name: str,
    position: str,
    team: str,
    projection: float,
    projection_unit: str = "PPR pts",
    floor: float | None = None,
    ceiling: float | None = None,
    tier_label: str | None = None,
    matchup: str | None = None,
    trend_change: float | None = None,
    extra_note: str | None = None,
) -> str:
    tier = tier_badge_html(tier_label)
    pos = position_badge_html(position)
    trend = trend_arrow(trend_change)
    floor_ceiling_str = ""
    if floor is not None and ceiling is not None and not pd.isna(floor) and not pd.isna(ceiling):
        floor_ceiling_str = (
            f'<div class="ceiling-floor">Range {floor:.0f} – {ceiling:.0f}</div>'
        )
    matchup_html = f'<span>{matchup}</span>' if matchup else "<span></span>"
    extra_html = (
        f'<div style="font-size:0.78rem;color:#5E6A75;margin-top:4px;">{extra_note}</div>'
        if extra_note
        else ""
    )
    return f"""
    <div class="player-card">
      <div>{pos}{tier}{trend}</div>
      <div class="player-name">{name}</div>
      <div class="player-meta">{team}</div>
      <div>
        <span class="projection">{projection:.1f}</span>
        <span class="projection-unit">{projection_unit}</span>
      </div>
      {floor_ceiling_str}
      <div class="meta-row">
        {matchup_html}
        <span>{tier_label or ""}</span>
      </div>
      {extra_html}
    </div>
    """


def player_card_grid(cards_html: Iterable[str]) -> None:
    html = '<div class="card-grid">' + "".join(cards_html) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Game card (ESPN weekly games style)
# ---------------------------------------------------------------------------
def game_card_html(
    *,
    away_team: str,
    home_team: str,
    away_score: float | None,
    home_score: float | None,
    predicted_winner: str,
    winner_probability: float,
    spread_line: float | None,
    total_line: float | None,
    market_signal: str | None,
    pick_explanation: str,
    gameday: str | None,
    actual_winner: str | None = None,
    correct_prediction: bool | None = None,
) -> str:
    # Confidence bar split between away & home
    away_pct = (
        (1 - winner_probability) * 100
        if predicted_winner == home_team
        else winner_probability * 100
    )
    home_pct = 100 - away_pct

    # Status badge
    status_html = ""
    if actual_winner and not pd.isna(actual_winner):
        if correct_prediction:
            status_html = (
                '<span style="background:#157A6E;color:#fff;font-size:0.66rem;'
                'font-weight:700;text-transform:uppercase;padding:2px 8px;'
                'border-radius:4px;">✓ Correct</span>'
            )
        else:
            status_html = (
                '<span style="background:#C8553D;color:#fff;font-size:0.66rem;'
                'font-weight:700;text-transform:uppercase;padding:2px 8px;'
                'border-radius:4px;">✗ Missed</span>'
            )

    score_str = ""
    if (
        away_score is not None
        and home_score is not None
        and not pd.isna(away_score)
        and not pd.isna(home_score)
    ):
        score_str = f'<div style="font-size:0.85rem;color:#5E6A75;">Final: {int(away_score)}–{int(home_score)}</div>'

    spread_str = f"Spread {spread_line:+.1f}" if spread_line is not None and not pd.isna(spread_line) else ""
    total_str = f"Total {total_line:.1f}" if total_line is not None and not pd.isna(total_line) else ""
    market_str = market_signal or ""

    return f"""
    <div class="game-card">
      <div class="matchup">
        <span class="team">{away_team}</span>
        <span class="at">@</span>
        <span class="team">{home_team}</span>
      </div>
      {score_str}
      <div class="confidence-bar">
        <div class="conf-segment conf-away" style="width:{away_pct:.0f}%;">{away_team}</div>
        <div class="conf-segment conf-home" style="width:{home_pct:.0f}%;">{home_team}</div>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:0.9rem;">
          <strong style="color:#157A6E;">{predicted_winner}</strong>
          <span style="color:#5E6A75;"> at {winner_probability:.0%}</span>
        </div>
        {status_html}
      </div>
      <div class="pick">{pick_explanation}</div>
      <div class="meta">
        <span>{gameday or ""}</span>
        <span>{spread_str}{" · " if spread_str and total_str else ""}{total_str}</span>
      </div>
      <div style="font-size:0.74rem;color:#5E6A75;">{market_str}</div>
    </div>
    """


def game_card_grid(cards_html: Iterable[str]) -> None:
    html = '<div class="card-grid">' + "".join(cards_html) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Executive summary box (front office leadership-report style)
# ---------------------------------------------------------------------------
def executive_summary(title: str, bullets: Iterable[str]) -> None:
    bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(
        f"""
        <div class="exec-summary">
          <h4>{title}</h4>
          <ul>{bullets_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendation_callout(category: str, label: str, body: str) -> None:
    css = {"opportunity": "opportunity", "warning": "warning", "caveat": "caveat"}.get(
        category.lower(), "opportunity"
    )
    st.markdown(
        f"""
        <div class="reco-callout {css}">
          <span class="reco-tag">{label}</span>
          <span class="reco-body">{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# KPI grid (Streamlit-native metrics in a clean row)
# ---------------------------------------------------------------------------
def kpi_grid(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render N st.metric tiles in a single row."""
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            if delta is not None:
                st.metric(label, value, delta)
            else:
                st.metric(label, value)


# ---------------------------------------------------------------------------
# Tier classification (used by player cards when fantasy projection table
# doesn't already carry a tier label)
# ---------------------------------------------------------------------------
def classify_tier_from_percentile(percentile: float | None) -> str:
    if percentile is None or pd.isna(percentile):
        return "Flex"
    if percentile >= 0.90:
        return "Elite Fantasy Profile"
    if percentile >= 0.75:
        return "Strong Starter"
    if percentile >= 0.50:
        return "Starter/Flex"
    if percentile >= 0.25:
        return "Depth/Volatile"
    return "Low Projection"
