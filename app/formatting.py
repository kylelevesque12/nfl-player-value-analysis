"""Small pure-ish display helpers shared by every section: number formatting,
dataframe filtering, safe column access, and the KPI tile row.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.layout import chunk_metrics


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


def col_or_na(df: pd.DataFrame, col: str) -> pd.Series:
    """``df[col]`` if present, else an index-aligned all-NaN Series.

    Safer than ``df.get(col, pd.Series(...))``: an unaligned default Series
    (e.g. the empty ``pd.Series(dtype="float64")`` this replaces) has the
    wrong length and breaks ``pd.DataFrame({...})`` construction the moment a
    column that's normally present (like a market-data field on an older,
    pre-fallback board) goes missing.
    """
    return df[col] if col in df.columns else pd.Series(pd.NA, index=df.index)


def card_row(metrics: list[tuple[str, str, str | None]], max_per_row: int = 3) -> None:
    """KPI tiles that wrap into balanced rows so they stay readable on tablet /
    phone widths (Streamlit stacks columns fully below its small-screen
    breakpoint; this keeps the mid-width range tidy)."""
    for row in chunk_metrics(metrics, max_per_row):
        columns = st.columns(len(row))
        for column, (label, value, help_text) in zip(columns, row):
            column.metric(label, value, help=help_text)



def kpi_or_dash(value, fmt: str = "{:.1f}") -> str:
    return fmt.format(value) if value is not None and pd.notna(value) else "—"
