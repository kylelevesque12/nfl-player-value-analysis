"""Leakage-safe Next Gen Stats (NGS) and Pro-Football-Reference (PFR) weekly
features for the weekly fantasy model.

NGS and PFR weekly stats describe what *happened* in a game — average
separation, broken tackles, drop rate, time to throw. They are post-game
information. Using week-t NGS/PFR to predict week-t fantasy points would leak
the outcome into the prediction. So every metric here is converted into a
prior-game feature before it is exposed to the model:

- ``<metric>_lag1`` : the value from the player's previous game
- ``<metric>_roll3``: the mean of the player's previous three games

Both are built with ``groupby(player_id).shift(1)`` over a (season, week)-sorted
frame, so week t's own value can never enter week t's feature. Carryover across
a season boundary is allowed — that is still *past* information, not leakage,
and it keeps week 1 from being all-NaN.

NGS files carry ``player_gsis_id`` directly, so they join to the weekly model
on ``player_id`` with no bridge. PFR files are keyed on ``pfr_player_id``; we
hop through the rosters table (``pfr_id`` -> ``gsis_id``) exactly like the snap
counts join in the weekly model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Requested NGS metrics, grouped by source file. Only columns that actually
# exist in the local data are kept (see ``_select_existing``); a missing column
# is reported rather than invented.
NGS_RECEIVING_METRICS = {
    "avg_separation": "ngs_avg_separation",
    "avg_cushion": "ngs_avg_cushion",
    "avg_yac_above_expectation": "ngs_avg_yac_above_expectation",
    "percent_share_of_intended_air_yards": "ngs_pct_share_intended_air_yards",
}
NGS_RUSHING_METRICS = {
    "efficiency": "ngs_rush_efficiency",
    "percent_attempts_gte_eight_defenders": "ngs_pct_attempts_gte_eight_def",
}
NGS_PASSING_METRICS = {
    "avg_time_to_throw": "ngs_avg_time_to_throw",
    "avg_completed_air_yards": "ngs_avg_completed_air_yards",
}

# Requested PFR metrics, grouped by source file.
PFR_REC_METRICS = {
    "receiving_broken_tackles": "pfr_receiving_broken_tackles",
    "receiving_drop_pct": "pfr_receiving_drop_pct",
}
PFR_RUSH_METRICS = {
    "rushing_broken_tackles": "pfr_rushing_broken_tackles",
}
PFR_PASS_METRICS = {
    "passing_drops": "pfr_passing_drops",
}

# All renamed base metric columns (same-week), used downstream to know which
# columns to lag/roll.
NGS_BASE_COLS = list(
    {**NGS_RECEIVING_METRICS, **NGS_RUSHING_METRICS, **NGS_PASSING_METRICS}.values()
)
PFR_BASE_COLS = list(
    {**PFR_REC_METRICS, **PFR_RUSH_METRICS, **PFR_PASS_METRICS}.values()
)


def _find_file(project_root: Path, stem: str) -> Path | None:
    raw_dir = project_root / "data" / "raw"
    if not raw_dir.exists():
        return None
    matches = sorted(raw_dir.glob(f"{stem}_*.csv"))
    return matches[0] if matches else None


def _select_existing(df: pd.DataFrame, rename_map: dict[str, str]) -> dict[str, str]:
    """Keep only the requested columns that exist; warn (comment) on the rest."""
    present = {src: dst for src, dst in rename_map.items() if src in df.columns}
    missing = [src for src in rename_map if src not in df.columns]
    if missing:
        # Reported, not invented — the model simply goes without these.
        print(f"[external_features] columns unavailable, skipped: {missing}")
    return present


def _read_ngs(path: Path, rename_map: dict[str, str]) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path, low_memory=False)
    if "season_type" in df.columns:
        df = df[df["season_type"].eq("REG")]
    # NGS weekly files include a week-0 row that is the season-to-date
    # aggregate; drop it so it can't pollute the weekly ordering.
    df = df[pd.to_numeric(df["week"], errors="coerce").fillna(0) > 0]
    keep = _select_existing(df, rename_map)
    if not keep:
        return None
    out = df[["player_gsis_id", "season", "week", *keep.keys()]].rename(
        columns={"player_gsis_id": "player_id", **keep}
    )
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out.dropna(subset=["player_id", "season", "week"])
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out


def load_ngs_features(project_root: Path) -> pd.DataFrame | None:
    """One same-week row per (player_id, season, week) of renamed NGS metrics."""
    parts = []
    for stem, rename_map in [
        ("ngs_receiving", NGS_RECEIVING_METRICS),
        ("ngs_rushing", NGS_RUSHING_METRICS),
        ("ngs_passing", NGS_PASSING_METRICS),
    ]:
        part = _read_ngs(_find_file(project_root, stem), rename_map)
        if part is not None:
            parts.append(part)
    if not parts:
        return None
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=["player_id", "season", "week"], how="outer")
    # A player can be in receiving + rushing NGS in the same week (e.g. a
    # pass-catching back); the outer merge keeps one row per player-week.
    assert not merged.duplicated(["player_id", "season", "week"]).any()
    return merged


def _read_pfr(path: Path, rename_map: dict[str, str]) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path, low_memory=False)
    if "game_type" in df.columns:
        df = df[df["game_type"].eq("REG")]
    keep = _select_existing(df, rename_map)
    if not keep:
        return None
    out = df[["pfr_player_id", "season", "week", *keep.keys()]].rename(columns=keep)
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out.dropna(subset=["pfr_player_id", "season", "week"])
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out


def load_pfr_weekly_features(
    project_root: Path, rosters: pd.DataFrame | None = None
) -> pd.DataFrame | None:
    """One same-week row per (player_id, season, week) of renamed PFR metrics,
    bridged from ``pfr_player_id`` to ``gsis_id`` through the rosters table."""
    parts = []
    for stem, rename_map in [
        ("pfr_weekly_rec", PFR_REC_METRICS),
        ("pfr_weekly_rush", PFR_RUSH_METRICS),
        ("pfr_weekly_pass", PFR_PASS_METRICS),
    ]:
        part = _read_pfr(_find_file(project_root, stem), rename_map)
        if part is not None:
            parts.append(part)
    if not parts:
        return None
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(
            part, on=["pfr_player_id", "season", "week"], how="outer"
        )
    assert not merged.duplicated(["pfr_player_id", "season", "week"]).any()

    # Bridge pfr_player_id -> gsis_id via rosters (dedupe on (season, pfr_id) so
    # a mid-season team change doesn't fan a single player-week into many).
    if rosters is None:
        rosters_path = _find_file(project_root, "rosters")
        if rosters_path is None:
            return None
        rosters = pd.read_csv(
            rosters_path, usecols=["season", "gsis_id", "pfr_id"], low_memory=False
        )
    bridge = rosters[["season", "gsis_id", "pfr_id"]].copy()
    bridge["season"] = pd.to_numeric(bridge["season"], errors="coerce")
    bridge = bridge.dropna(subset=["season", "gsis_id", "pfr_id"])
    bridge["season"] = bridge["season"].astype(int)
    bridge = bridge.drop_duplicates(subset=["season", "pfr_id"])

    merged = merged.merge(
        bridge,
        left_on=["season", "pfr_player_id"],
        right_on=["season", "pfr_id"],
        how="left",
    )
    merged = merged.dropna(subset=["gsis_id"]).rename(columns={"gsis_id": "player_id"})
    merged = merged.drop(columns=["pfr_player_id", "pfr_id"], errors="ignore")

    # After the bridge two pfr_ids could (very rarely) map to one gsis_id in a
    # season; collapse to one row per player-week by averaging the metrics.
    metric_cols = [c for c in merged.columns if c.startswith("pfr_")]
    merged = (
        merged.groupby(["player_id", "season", "week"], as_index=False)[metric_cols]
        .mean()
    )
    assert not merged.duplicated(["player_id", "season", "week"]).any()
    return merged


def _add_lag_and_roll(
    same_week: pd.DataFrame, base_cols: list[str]
) -> pd.DataFrame:
    """For each base metric add ``_lag1`` and ``_roll3`` leakage-safe columns.

    Sorted by (player_id, season, week); ``shift(1)`` guarantees the current
    game's value never enters the current game's feature. Only the lagged and
    rolled columns are returned — the raw same-week columns are dropped so they
    cannot be added to the model by accident.
    """
    df = same_week.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grp = df.groupby("player_id", group_keys=False)
    out_cols = ["player_id", "season", "week"]
    for col in base_cols:
        if col not in df.columns:
            continue
        df[f"{col}_lag1"] = grp[col].shift(1)
        df[f"{col}_roll3"] = grp[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        out_cols += [f"{col}_lag1", f"{col}_roll3"]
    return df[out_cols].reset_index(drop=True)


def build_external_weekly_features(
    project_root: Path,
    rosters: pd.DataFrame | None = None,
    include_ngs: bool = True,
    include_pfr: bool = True,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Return (feature_table, leakage_safe_column_names).

    The table has exactly one row per (player_id, season, week) and contains
    only ``_lag1`` / ``_roll3`` columns. The second element lists those columns
    so the caller can register them with the model.
    """
    tables: list[pd.DataFrame] = []
    feature_cols: list[str] = []

    if include_ngs:
        ngs = load_ngs_features(project_root)
        if ngs is not None:
            present_base = [c for c in NGS_BASE_COLS if c in ngs.columns]
            ngs_lagged = _add_lag_and_roll(ngs, present_base)
            tables.append(ngs_lagged)
            feature_cols += [
                c for c in ngs_lagged.columns
                if c not in ("player_id", "season", "week")
            ]

    if include_pfr:
        pfr = load_pfr_weekly_features(project_root, rosters)
        if pfr is not None:
            present_base = [c for c in PFR_BASE_COLS if c in pfr.columns]
            pfr_lagged = _add_lag_and_roll(pfr, present_base)
            tables.append(pfr_lagged)
            feature_cols += [
                c for c in pfr_lagged.columns
                if c not in ("player_id", "season", "week")
            ]

    if not tables:
        return None, []

    feature_table = tables[0]
    for tbl in tables[1:]:
        feature_table = feature_table.merge(
            tbl, on=["player_id", "season", "week"], how="outer"
        )
    assert not feature_table.duplicated(["player_id", "season", "week"]).any(), (
        "external feature table must be one row per player-week"
    )
    return feature_table, feature_cols


# ---------------------------------------------------------------------------
# NGS coverage flags — the production features (see session-2 report).
#
# A permutation test showed the NGS metric *values* (separation, cushion, air
# yards, efficiency, time to throw) add essentially nothing once lagged: shuffle
# the values while keeping the NaN pattern and the RMSE gain is unchanged. The
# whole signal is COVERAGE — whether a player was a tracked receiver / rusher /
# passer in their previous game, which is a clean prior-game role/playing-time
# proxy. Three explicit binary flags capture that better than 16 noisy imputed
# value columns, so those flags are what the model actually uses. PFR is dropped
# entirely (its values were neutral-to-negative and its join is fragile).
# ---------------------------------------------------------------------------
NGS_COVERAGE_SOURCES = {
    "ngs_receiving": "ngs_rec_tracked_lag1",
    "ngs_rushing": "ngs_rush_tracked_lag1",
    "ngs_passing": "ngs_pass_tracked_lag1",
}
NGS_COVERAGE_FLAGS = list(NGS_COVERAGE_SOURCES.values())


def _load_ngs_presence(project_root: Path, stem: str) -> pd.DataFrame | None:
    """Unique (player_id, season, week) rows a player was NGS-tracked in,
    regular season only, week-0 season aggregate dropped."""
    path = _find_file(project_root, stem)
    if path is None:
        return None
    df = pd.read_csv(
        path,
        low_memory=False,
        usecols=lambda c: c in {"player_gsis_id", "season", "week", "season_type"},
    )
    if "season_type" in df.columns:
        df = df[df["season_type"].eq("REG")]
    df = df.rename(columns={"player_gsis_id": "player_id"})
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["player_id", "season", "week"])
    df = df[df["week"] > 0]
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return df[["player_id", "season", "week"]].drop_duplicates()


def attach_ngs_coverage_flags(
    modeling_frame: pd.DataFrame, project_root: Path | None = None
) -> pd.DataFrame:
    """Attach the three leakage-safe NGS coverage flags to the modeling frame.

    For each NGS source, mark whether the player was tracked in the *same* week
    (a row exists in that NGS file), then ``groupby(player_id).shift(1)`` over
    the player's game sequence so the flag reflects the PREVIOUS game only —
    week t's own tracking status never enters week t. A player's first game gets
    0 (not previously tracked). No-op if NGS files are absent.
    """
    if project_root is None:
        from src.load_data import find_project_root

        project_root = find_project_root()
    project_root = Path(project_root)

    out = modeling_frame.copy()
    out["_orig_order"] = range(len(out))
    sorted_view = out.sort_values(["player_id", "season", "week"])

    attached_any = False
    for stem, flag in NGS_COVERAGE_SOURCES.items():
        presence = _load_ngs_presence(project_root, stem)
        if presence is None:
            continue
        presence = presence.assign(_tracked_now=1.0)
        merged = sorted_view.merge(
            presence, on=["player_id", "season", "week"], how="left"
        )
        merged["_tracked_now"] = merged["_tracked_now"].fillna(0.0)
        # shift(1) over the player's game order -> "tracked in previous game".
        merged[flag] = (
            merged.groupby("player_id")["_tracked_now"].shift(1).fillna(0.0)
        )
        sorted_view[flag] = merged[flag].to_numpy()
        attached_any = True

    if not attached_any:
        return modeling_frame

    restored = sorted_view.sort_values("_orig_order").drop(columns=["_orig_order"])
    assert len(restored) == len(modeling_frame), "coverage-flag attach changed row count"
    return restored.reset_index(drop=True)


def attach_external_weekly_features(
    modeling_frame: pd.DataFrame,
    project_root: Path | None = None,
    rosters: pd.DataFrame | None = None,
    include_ngs: bool = True,
    include_pfr: bool = True,
) -> pd.DataFrame:
    """Left-join leakage-safe NGS/PFR *value* features (lag1/roll3) onto the
    modeling frame.

    This is the value-based path retained for the session-2 diagnostics and
    ablation scripts. The production model uses ``attach_ngs_coverage_flags``
    instead (the values were shown to add nothing beyond coverage). A no-op if
    no external data is present; asserts the join does not add rows.
    """
    if project_root is None:
        from src.load_data import find_project_root

        project_root = find_project_root()
    project_root = Path(project_root)

    feature_table, _ = build_external_weekly_features(
        project_root, rosters, include_ngs=include_ngs, include_pfr=include_pfr
    )
    if feature_table is None:
        return modeling_frame

    before = len(modeling_frame)
    merged = modeling_frame.merge(
        feature_table, on=["player_id", "season", "week"], how="left"
    )
    assert len(merged) == before, (
        f"external feature join changed row count: {before} -> {len(merged)}"
    )
    return merged
