"""Merge the current draft class's rookie projections into the season
fantasy rankings table, so rookies compete on equal footing with veterans
instead of being invisible to the Draft Board and VORP.

No PyMC dependency — this only combines two already-built CSVs (the season
fantasy table and the rookie projections from
``build_2026_rookie_projection_outputs``, run separately via ``.venv-bayes``)
and can run in the main venv / standard pipeline. If the rookie projections
file doesn't exist yet, the merge is a no-op: veterans pass through
unchanged, the same graceful-degradation principle used for the ADP-optional
draft board.

Rank, percentile, and tier columns are recomputed over the *combined* set
(they are position/pool-relative, so adding rookies genuinely changes them
for veterans too — a rookie who projects inside the top 10 overall should
push the 10th-best veteran down one slot). Breakout/slump potential and the
draft-board bucket are also recomputed for the same reason: they read the
recomputed percentile. Everything else (usage profile, notes, explanations,
model attribution) is either carried through unchanged for veterans or
purpose-written for rookies — a rookie's "profile" prose has no natural
reading in the veteran template (there is no 2025 season to compare against).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.fantasy_projection import (
    PREDICTION_INTERVAL_MULTIPLIER,
    _assign_confidence_level,
    _assign_fantasy_tier,
    _breakout_potential,
    _draft_board_bucket,
    _slump_potential,
)
from src.load_data import ensure_project_dirs, find_project_root, load_csv

ROOKIE_MODEL_LABEL = "Bayesian Rookie Hurdle Model"
ROOKIE_SELECTION_REASON = (
    "Rookie with no prior NFL data; projected from draft capital, age, "
    "physical profile, and incumbent context via a hierarchical Bayesian "
    "hurdle model (P(plays meaningfully) x PPR/game if he plays)."
)
ROOKIE_PROJECTION_CHANGE_LABEL = "Rookie — no 2025 comparison"


def _rookie_usage_profile(position: str) -> str:
    return f"Rookie {position} — no NFL usage yet"


def _rookie_fantasy_note(row: pd.Series) -> str:
    notes = ["rookie season, no NFL track record"]
    p_plays = row.get("predicted_p_plays_meaningfully", np.nan)
    if pd.notna(p_plays) and p_plays < 0.6:
        notes.append("meaningful chance he doesn't win a real role in 2026")
    return "; ".join(notes)


def _rookie_fantasy_explanation(row: pd.Series) -> str:
    proj = row.get("predicted_2026_fantasy_points_ppr", np.nan)
    draft_num = row.get("draft_number", np.nan)
    p_plays = row.get("predicted_p_plays_meaningfully", np.nan)
    games = row.get("predicted_2026_games_played", np.nan)
    ppg = row.get("predicted_2026_ppr_per_game", np.nan)
    lo = row.get("prediction_interval_low", np.nan)
    hi = row.get("prediction_interval_high", np.nan)

    if pd.notna(proj) and pd.notna(draft_num):
        opening = f"Rookie projection ({int(draft_num)} overall pick): {proj:.1f} PPR points in 2026."
    else:
        opening = "Rookie projection based on draft capital, age, and profile."

    details = [opening]
    if pd.notna(p_plays):
        details.append(
            f"Modeled {p_plays:.0%} chance of playing a meaningful role (4+ games)."
        )
    if pd.notna(games) and pd.notna(ppg):
        details.append(
            f"Two-stage context: {games:.1f} projected games at {ppg:.1f} PPR/game if he plays."
        )
    if pd.notna(lo) and pd.notna(hi):
        details.append(f"Reasonable model range: {lo:.0f}-{hi:.0f} PPR.")
    details.append("No 2025 NFL history to compare against.")
    return " ".join(details)


def build_rookie_rows(rookie_projections: pd.DataFrame) -> pd.DataFrame:
    """Construct season-table-schema rows from the scored rookie class.

    ``rookie_projections`` is the output of
    ``build_2026_rookie_projection_outputs`` (columns: player_id,
    player_display_name, position, draft_number, draft_club,
    predicted_p_plays_meaningfully, predicted_ppr_per_game_if_plays_mean,
    predicted_games_played, predicted_season_ppr_mean/p10/p25/p75/p90).
    """
    r = rookie_projections
    rows = pd.DataFrame(
        {
            "player_id": r["player_id"],
            "player_display_name": r["player_display_name"],
            "position": r["position"],
            "primary_team_2025": r.get("draft_club"),
            "teams_2025": pd.NA,
            # Literally true (not in the NFL in 2025) and what lets the
            # shared confidence-score formula treat rookies as zero-sample,
            # zero-history without any special-casing.
            "games_played_2025": 0,
            "age_2025": pd.NA,
            "years_exp_2025": 0,
            "draft_number": r["draft_number"],
            # Left NaN rather than 0: these mean "not applicable," not "he
            # scored zero." _projection_change_label and friends already
            # branch on pd.isna(...) for exactly this reason.
            "fantasy_points_ppr_2025": np.nan,
            "fantasy_points_ppr_per_game_2025": np.nan,
            "targets_2025": np.nan,
            "receptions_2025": np.nan,
            "carries_2025": np.nan,
            "value_score": np.nan,
            "predicted_2026_fantasy_points_ppr": r["predicted_season_ppr_mean"],
            "predicted_2026_games_played": r["predicted_games_played"],
            "predicted_2026_ppr_per_game": r["predicted_ppr_per_game_if_plays_mean"],
            "projection_change_from_2025": np.nan,
            "projection_change_label": ROOKIE_PROJECTION_CHANGE_LABEL,
            "prediction_interval_low": r["predicted_season_ppr_p10"].clip(lower=0),
            "prediction_interval_high": r["predicted_season_ppr_p90"],
            "model_disagreement": np.nan,
            "selected_model": "rookie_bayes_hurdle",
            "selected_model_label": ROOKIE_MODEL_LABEL,
            "model_selection_reason": ROOKIE_SELECTION_REASON,
            "predicted_p_plays_meaningfully": r["predicted_p_plays_meaningfully"],
            "is_rookie_projection": True,
        }
    )
    rows["prediction_uncertainty"] = (
        rows["prediction_interval_high"] - rows["prediction_interval_low"]
    ) / (2 * PREDICTION_INTERVAL_MULTIPLIER)
    rows["usage_profile"] = rows["position"].map(_rookie_usage_profile)
    rows["fantasy_note"] = rows.apply(_rookie_fantasy_note, axis=1)
    rows["fantasy_explanation"] = rows.apply(_rookie_fantasy_explanation, axis=1)
    return rows


def merge_rookies_into_season_table(
    veterans: pd.DataFrame, rookie_projections: pd.DataFrame | None
) -> pd.DataFrame:
    """Combine the veteran season table with the scored rookie class.

    Graceful no-op (returns ``veterans`` unchanged) if ``rookie_projections``
    is ``None`` or empty — the rookie-scoring step is a separate, optional
    ``.venv-bayes`` run, and the season table must stay usable without it.
    """
    if rookie_projections is None or rookie_projections.empty:
        return veterans.copy()

    rookie_rows = build_rookie_rows(rookie_projections)
    combined = pd.concat([veterans, rookie_rows], ignore_index=True, sort=False)
    # Veterans never had this column, so concat introduces NaN for their
    # rows; .eq(True) (NaN != True -> False) gives a clean bool Series
    # without the object-dtype fillna deprecation path.
    combined["is_rookie_projection"] = combined["is_rookie_projection"].eq(True)

    # Rank/percentile are position-and-pool-relative: adding rookies changes
    # them for veterans too, so recompute over the combined set rather than
    # just appending the rookies' own numbers.
    combined["fantasy_overall_rank"] = combined["predicted_2026_fantasy_points_ppr"].rank(
        ascending=False, method="min"
    )
    combined["fantasy_position_rank"] = combined.groupby("position")[
        "predicted_2026_fantasy_points_ppr"
    ].rank(ascending=False, method="min")
    combined["predicted_2026_overall_percentile"] = combined[
        "predicted_2026_fantasy_points_ppr"
    ].rank(pct=True)
    combined["predicted_2026_position_percentile"] = combined.groupby("position")[
        "predicted_2026_fantasy_points_ppr"
    ].rank(pct=True)
    combined["fantasy_projection_tier"] = combined[
        "predicted_2026_position_percentile"
    ].apply(_assign_fantasy_tier)

    # confidence_score mirrors build_fantasy_projection_outputs's formula
    # exactly, with one documented substitution: the veteran build used an
    # intermediate `prior_qualifying_seasons` feature that isn't part of the
    # 40-column season-table schema this merge step reads, so years_exp_2025
    # (clipped 0-4) stands in for the same "how much NFL history does he
    # have" signal. The two are highly correlated in practice, and a rookie's
    # value is unambiguous either way: 0.
    uncertainty_pct = combined["prediction_uncertainty"].rank(pct=True)
    sample_score = combined["games_played_2025"].fillna(0).clip(lower=0, upper=17) / 17
    history_score = combined["years_exp_2025"].fillna(0).clip(lower=0, upper=4) / 4
    combined["confidence_score"] = (
        (1 - uncertainty_pct) * 45 + sample_score * 30 + history_score * 25
    ).round(1)
    combined["confidence_level"] = combined["confidence_score"].apply(
        _assign_confidence_level
    )

    # breakout/slump/bucket read predicted_2026_position_percentile, which
    # just changed for everyone, so recompute uniformly rather than leaving
    # veterans on a stale, pre-merge percentile.
    combined["breakout_potential"] = combined.apply(_breakout_potential, axis=1)
    combined["slump_potential"] = combined.apply(_slump_potential, axis=1)
    combined["draft_board_bucket"] = combined.apply(_draft_board_bucket, axis=1)

    return combined.sort_values(
        "predicted_2026_fantasy_points_ppr", ascending=False
    ).reset_index(drop=True)


def build_rookie_rankings_merge_outputs(
    project_root: str | Path | None = None,
    year: int = 2026,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Pipeline entry point: merge rookies into the season fantasy table.

    Reads ``{year}_fantasy_football_projections.csv`` (veterans) and
    ``{year}_rookie_projections.csv`` (rookies, optional — produced
    separately via ``.venv-bayes``); overwrites the season table in place
    with the combined result so every downstream reader (VORP, the app)
    picks up rookies automatically, with no code changes needed there.
    """
    root = find_project_root() if project_root is None else Path(project_root).resolve()
    dirs = ensure_project_dirs(root)
    table_dir = dirs["tables"]

    veterans = load_csv(f"outputs/tables/{year}_fantasy_football_projections.csv", root)
    rookie_path = table_dir / f"{year}_rookie_projections.csv"
    # load_csv raises if the file is missing rather than degrading
    # gracefully; the rookie file genuinely may not exist yet (it's a
    # separate, optional .venv-bayes run), so check explicitly.
    rookies = pd.read_csv(rookie_path) if rookie_path.exists() else None

    combined = merge_rookies_into_season_table(veterans, rookies)
    n_rookies_added = int(combined.get("is_rookie_projection", pd.Series(dtype=bool)).sum())

    if save_outputs:
        combined.to_csv(
            table_dir / f"{year}_fantasy_football_projections.csv",
            index=False,
            float_format="%.12g",
        )
    return {"combined": combined, "n_rookies_added": n_rookies_added}
