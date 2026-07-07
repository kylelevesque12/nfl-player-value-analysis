"""Build a scorable frame for a brand-new draft class (pure, no network calls).

The historical rookie training data (``build_rookie_modeling_frame`` in
``src/rookie_bayes.py``) reads bio data off nflverse's roster feed, which is
populated once a player has an NFL roster entry. A class freshly drafted a
few months ago doesn't have that yet — nflverse's roster snapshot carries the
player's identity but not birth_date/height/weight for weeks after the draft.

Three feeds are involved:

- ``load_draft_picks()``: draft slot, team, college, and age (computed at
  draft time, so no birth_date lookup is needed) for everyone drafted.
- ``load_combine()``: height and weight from pre-draft testing.
- ``load_rosters()`` for the target season: the identity anchor.
  ``load_draft_picks()`` *also* carries a ``gsis_id`` field, but verified
  against the real 2026 class it is a different, non-standard identifier
  scheme — every value disagreed with the roster-sourced gsis_id for the 67
  players where both were populated. Using it directly would have silently
  assigned every 2026 rookie the wrong player_id, unusable for joining to
  their own stats once they start accumulating them. ``load_rosters()``'s
  gsis_id is the one used everywhere else in this project, so it is the only
  identity source trusted here.

None of the three feeds share a reliable ID with each other for a class this
new (combine's ``pfr_id``/``cfb_id`` are typically unpopulated until a player
has an NFL profile page), so every join here is on normalized player name,
with the match rate reported rather than assumed.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.adp import normalize_name

SKILL_POSITIONS = list(config.SKILL_POSITIONS)


def build_rookie_class_frame(
    draft_picks: pd.DataFrame,
    combine: pd.DataFrame,
    rosters: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, dict]:
    """Build a rosters-shaped frame for one draft class.

    ``draft_picks``, ``combine``, and ``rosters`` should already be filtered
    to the target season by the caller (``rosters`` to that season's rows;
    it does not need to be pre-filtered to rookies). Returns a frame with the
    same column shape ``build_rookie_modeling_frame`` expects from a rosters
    table (season, rookie_year, entry_year, draft_number, weight, height,
    birth_date, draft_club, position, player_display_name), plus
    ``age_at_draft_hint`` — draft-day age, used as a fallback when birth_date
    (unavailable for a class this new) can't produce one — and match-rate
    diagnostics.
    """
    picks = draft_picks[draft_picks["position"].isin(SKILL_POSITIONS)].copy()
    total_picks = len(picks)
    # Drop draft_picks' own gsis_id-labeled column outright — it is a
    # different, non-standard scheme for this class (see module docstring)
    # and must not collide with (or silently shadow) the roster-sourced one
    # merged in next.
    picks = picks.drop(columns=["gsis_id"], errors="ignore")
    picks["_key"] = picks["pfr_player_name"].map(normalize_name)

    # Identity anchor: the roster feed's gsis_id, not draft_picks' own
    # gsis_id field (see module docstring — verified to be a different,
    # non-standard ID for this class).
    roster_ids = rosters.dropna(subset=["gsis_id"]).copy()
    roster_ids["_key"] = roster_ids["full_name"].map(normalize_name)
    roster_ids = roster_ids.drop_duplicates(subset="_key")
    picks = picks.merge(roster_ids[["_key", "gsis_id"]], on="_key", how="left")

    missing_gsis = picks[picks["gsis_id"].isna()]
    picks = picks.dropna(subset=["gsis_id"]).copy()
    picks = picks.drop_duplicates(subset=["gsis_id"])

    combine_keyed = combine.copy()
    combine_keyed["_key"] = combine_keyed["player_name"].map(normalize_name)
    combine_keyed = combine_keyed.drop_duplicates(subset=["_key"])

    merged = picks.merge(
        combine_keyed[["_key", "ht", "wt"]], on="_key", how="left"
    )
    combine_matched = int(merged["ht"].notna().sum())

    out = pd.DataFrame(
        {
            "gsis_id": merged["gsis_id"],
            "player_display_name": merged["pfr_player_name"],
            "position": merged["position"],
            "season": year,
            "rookie_year": year,
            "entry_year": year,
            "draft_number": pd.to_numeric(merged["pick"], errors="coerce"),
            "draft_club": merged["team"],
            "college": merged.get("college"),
            "height": merged["ht"],
            "weight": pd.to_numeric(merged["wt"], errors="coerce"),
            "birth_date": pd.NaT,
            "age_at_draft_hint": pd.to_numeric(merged["age"], errors="coerce"),
        }
    )

    diagnostics = {
        "total_picks": total_picks,
        "gsis_id_coverage": len(picks),
        "gsis_id_rate": len(picks) / total_picks if total_picks else 0.0,
        "missing_gsis_id_names": missing_gsis["pfr_player_name"].tolist(),
        "combine_matched": combine_matched,
        "combine_match_rate": combine_matched / len(out) if len(out) else 0.0,
    }
    return out, diagnostics
