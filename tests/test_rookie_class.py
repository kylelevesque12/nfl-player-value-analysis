"""Tests for building a scorable frame from a brand-new draft class
(src/rookie_class.py) — pure, no network calls."""

from __future__ import annotations

import pandas as pd

from src.rookie_class import build_rookie_class_frame


def _draft_picks_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026] * 4,
            "round": [1, 1, 1, 2],
            "pick": [1, 3, 4, 33],
            "team": ["ARI", "DAL", "NYJ", "SF"],
            # draft_picks' own gsis_id-labeled field is deliberately a
            # different, WRONG scheme here (mirrors the real 2026 data,
            # where it never matched the roster-sourced gsis_id) — the
            # frame builder must ignore this column entirely.
            "gsis_id": ["WRONG-1", "WRONG-2", "WRONG-3", "WRONG-4"],
            "pfr_player_name": ["Alpha One", "Beta Two", "Gamma Three", "Delta Four"],
            "position": ["QB", "RB", "WR", "TE"],
            "college": ["State U", "Tech", "A&M", "Coastal"],
            "age": [22.0, 21.0, 20.0, 23.0],
        }
    )


def _combine_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "player_name": ["Alpha One", "Beta Two", "Someone Else"],
            "pos": ["QB", "RB", "OT"],
            "ht": ["6-2", "5-11", "6-5"],
            "wt": [215.0, 205.0, 310.0],
        }
    )


def _rosters_fixture() -> pd.DataFrame:
    # The correct identity source. Gamma Three has no roster entry yet (not
    # signed / not yet rostered under a real gsis_id) -> must be dropped.
    return pd.DataFrame(
        {
            "full_name": ["Alpha One", "Beta Two", "Delta Four", "Some Vet"],
            "gsis_id": ["00-0041001", "00-0041002", "00-0041004", "00-0009999"],
        }
    )


def test_uses_roster_gsis_id_not_draft_picks_gsis_id():
    frame, _ = build_rookie_class_frame(
        _draft_picks_fixture(), _combine_fixture(), _rosters_fixture(), year=2026
    )
    row = frame.set_index("player_display_name")
    # Must be the roster-sourced id, never the "WRONG-*" placeholder.
    assert row.loc["Alpha One", "gsis_id"] == "00-0041001"
    assert row.loc["Beta Two", "gsis_id"] == "00-0041002"
    assert not frame["gsis_id"].astype(str).str.startswith("WRONG").any()


def test_builds_rosters_shaped_frame_with_expected_columns():
    frame, _ = build_rookie_class_frame(
        _draft_picks_fixture(), _combine_fixture(), _rosters_fixture(), year=2026
    )
    expected_cols = {
        "gsis_id", "player_display_name", "position", "season", "rookie_year",
        "entry_year", "draft_number", "draft_club", "college", "height",
        "weight", "birth_date", "age_at_draft_hint",
    }
    assert expected_cols <= set(frame.columns)


def test_drops_rows_with_no_roster_identity_and_reports_it():
    frame, diag = build_rookie_class_frame(
        _draft_picks_fixture(), _combine_fixture(), _rosters_fixture(), year=2026
    )
    # Gamma Three has no entry in the roster feed -> dropped, not kept with
    # a null or borrowed-from-elsewhere id.
    assert "Gamma Three" not in frame["player_display_name"].tolist()
    assert diag["total_picks"] == 4
    assert diag["gsis_id_coverage"] == 3
    assert diag["missing_gsis_id_names"] == ["Gamma Three"]


def test_combine_join_by_normalized_name_reports_match_rate():
    frame, diag = build_rookie_class_frame(
        _draft_picks_fixture(), _combine_fixture(), _rosters_fixture(), year=2026
    )
    row = frame.set_index("player_display_name")
    assert row.loc["Alpha One", "height"] == "6-2"
    assert row.loc["Alpha One", "weight"] == 215.0
    # Delta Four wasn't in the combine fixture -> height/weight stay missing.
    assert pd.isna(row.loc["Delta Four", "height"])
    assert diag["combine_matched"] == 2
    assert diag["combine_match_rate"] == 2 / 3  # 2 of the 3 rows with a roster id


def test_age_hint_carries_draft_day_age_for_fallback():
    frame, _ = build_rookie_class_frame(
        _draft_picks_fixture(), _combine_fixture(), _rosters_fixture(), year=2026
    )
    row = frame.set_index("player_display_name")
    assert row.loc["Alpha One", "age_at_draft_hint"] == 22.0
    assert pd.isna(row.loc["Alpha One", "birth_date"])  # never available for a new class


def test_non_skill_positions_excluded():
    picks = pd.concat(
        [
            _draft_picks_fixture(),
            pd.DataFrame(
                {
                    "season": [2026], "round": [3], "pick": [70], "team": ["KC"],
                    "gsis_id": ["WRONG-5"], "pfr_player_name": ["Epsilon Five"],
                    "position": ["OT"], "college": ["Big U"], "age": [22.0],
                }
            ),
        ],
        ignore_index=True,
    )
    frame, diag = build_rookie_class_frame(
        picks, _combine_fixture(), _rosters_fixture(), year=2026
    )
    assert "Epsilon Five" not in frame["player_display_name"].tolist()
    assert diag["total_picks"] == 4  # the OT is filtered before the picks count
