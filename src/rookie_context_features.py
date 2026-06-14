"""Pre-rookie-season features for the rookie hurdle model.

The hurdle model's stage 1 asks: will this rookie *play meaningfully* in year 1?
Draft position, age, and size get you part of the way, but they miss the single
biggest determinant of early playing time — opportunity. A third-round QB drafted
behind a freshly-extended franchise starter (Jordan Love) and a third-round QB
drafted onto a team with a hole at the position are night and day, and the
baseline model can't tell them apart.

This module adds three families of strictly *pre-season* information:

1. Combine athletic testing (forty, vertical, broad jump, bench, cone, shuttle).
2. Prior-season team context (pass rate, the prior starting QB's production).
3. Incumbent / depth at the rookie's position entering their rookie season.

LEAKAGE DISCIPLINE. Nothing here may use the rookie's own first-season outcomes.
Combine happens pre-draft. Team-context and incumbent features are all computed
from the season *before* the rookie year (``rookie_year - 1``) and from roster
composition known at training camp. The rookie himself is always excluded from
his own position's veteran counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COMBINE_METRICS = ["forty", "vertical", "broad_jump", "bench", "cone", "shuttle"]

# Feature columns this module contributes to the rookie model, grouped so the
# benchmark can switch them on and off.
COMBINE_FEATURES = COMBINE_METRICS + ["bmi"]
TEAM_CONTEXT_FEATURES = [
    "prior_team_pass_rate",
    "prior_qb_pprpg",
    "prior_qb_games",
    "pos_vet_count",
    "pos_vet_max_pprpg",
    "established_incumbent",
    "incumbent_recent_extension",
]
ALL_CONTEXT_FEATURES = COMBINE_FEATURES + TEAM_CONTEXT_FEATURES


def _find(project_root: Path, stem: str) -> Path | None:
    matches = sorted((project_root / "data" / "raw").glob(f"{stem}_*.csv"))
    if matches:
        return matches[0]
    single = project_root / "data" / "raw" / f"{stem}.csv"
    return single if single.exists() else None


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def _combine_id_bridges(rosters: pd.DataFrame, draft: pd.DataFrame | None):
    """Return (pfr->gsis, cfb->gsis) lookup frames built from the cleanest
    available sources. draft_picks is purpose-built for id mapping; rosters'
    pfr_id is a fallback."""
    pfr_parts, cfb_parts = [], []
    if draft is not None:
        d = draft.rename(columns={"gsis_id": "player_id"})
        if {"pfr_player_id", "player_id"}.issubset(d.columns):
            pfr_parts.append(
                d.dropna(subset=["pfr_player_id", "player_id"])[
                    ["pfr_player_id", "player_id"]
                ].rename(columns={"pfr_player_id": "pfr_id"})
            )
        if {"cfb_player_id", "player_id"}.issubset(d.columns):
            cfb_parts.append(
                d.dropna(subset=["cfb_player_id", "player_id"])[
                    ["cfb_player_id", "player_id"]
                ].rename(columns={"cfb_player_id": "cfb_id"})
            )
    if {"pfr_id", "gsis_id"}.issubset(rosters.columns):
        pfr_parts.append(
            rosters.dropna(subset=["pfr_id", "gsis_id"])[["pfr_id", "gsis_id"]].rename(
                columns={"gsis_id": "player_id"}
            )
        )
    pfr = (
        pd.concat(pfr_parts, ignore_index=True).drop_duplicates("pfr_id")
        if pfr_parts
        else pd.DataFrame(columns=["pfr_id", "player_id"])
    )
    cfb = (
        pd.concat(cfb_parts, ignore_index=True).drop_duplicates("cfb_id")
        if cfb_parts
        else pd.DataFrame(columns=["cfb_id", "player_id"])
    )
    return pfr, cfb


def load_combine_features(
    project_root: Path,
    rosters: pd.DataFrame,
    draft: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """One row per ``player_id`` of combine athletic measurements.

    Bridges combine's ``pfr_id`` (then ``cfb_id``) to ``gsis_id`` through
    draft_picks + rosters. No fuzzy name matching. Returns None if absent.
    """
    path = _find(project_root, "combine")
    if path is None:
        return None
    comb = pd.read_csv(path, low_memory=False)
    pfr, cfb = _combine_id_bridges(rosters, draft)

    comb = comb.merge(pfr, on="pfr_id", how="left")
    if "cfb_id" in comb.columns and not cfb.empty:
        comb = comb.merge(cfb, on="cfb_id", how="left", suffixes=("", "_cfb"))
        comb["player_id"] = comb["player_id"].fillna(comb["player_id_cfb"])
        comb = comb.drop(columns=["player_id_cfb"], errors="ignore")

    comb = comb.dropna(subset=["player_id"])
    keep = ["player_id"] + [c for c in COMBINE_METRICS if c in comb.columns]
    comb = comb[keep].copy()
    for c in COMBINE_METRICS:
        if c in comb.columns:
            comb[c] = pd.to_numeric(comb[c], errors="coerce")
    # A player can appear once; if duplicated across bridges keep the row with
    # the most non-null measurements.
    comb["_n"] = comb[[c for c in COMBINE_METRICS if c in comb.columns]].notna().sum(axis=1)
    comb = comb.sort_values("_n", ascending=False).drop_duplicates("player_id")
    return comb.drop(columns="_n").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Player-season and team-season production (prior-year context)
# ---------------------------------------------------------------------------
def build_player_season_pprpg(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Per (player_id, season): PPR/game, games, position, primary team."""
    df = player_stats.copy()
    df = df[df["season_type"].eq("REG")]
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["fantasy_points_ppr"] = pd.to_numeric(df["fantasy_points_ppr"], errors="coerce")
    df = df.dropna(subset=["season", "player_id"])
    df["season"] = df["season"].astype(int)
    if "position" not in df.columns:
        df["position"] = np.nan
    agg = df.groupby(["player_id", "season"], as_index=False).agg(
        ppr_total=("fantasy_points_ppr", "sum"),
        games=("week", "nunique"),
        position=("position", "first"),
    )
    # Primary team = team with the most weeks that season. Degrade gracefully if
    # no team column is present (minimal/synthetic frames).
    team_col = "recent_team" if "recent_team" in df.columns else (
        "team" if "team" in df.columns else None
    )
    if team_col is not None:
        team = (
            df.groupby(["player_id", "season", team_col])["week"].nunique().reset_index()
            .sort_values("week", ascending=False)
            .drop_duplicates(["player_id", "season"])
            .rename(columns={team_col: "team"})[["player_id", "season", "team"]]
        )
        agg = agg.merge(team, on=["player_id", "season"], how="left")
    else:
        agg["team"] = np.nan
    agg["pprpg"] = agg["ppr_total"] / agg["games"].clip(lower=1)
    return agg


def build_team_season_context(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Per (team, season): pass rate from team pass attempts vs rush attempts."""
    df = player_stats.copy()
    df = df[df["season_type"].eq("REG")]
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df.dropna(subset=["season"])
    df["season"] = df["season"].astype(int)
    team_col = "recent_team" if "recent_team" in df.columns else (
        "team" if "team" in df.columns else None
    )
    if team_col is None:
        return pd.DataFrame(columns=["team", "season", "prior_team_pass_rate"])
    for c in ("attempts", "carries"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else 0.0
    grp = df.groupby([team_col, "season"], as_index=False).agg(
        team_pass_att=("attempts", "sum"),
        team_rush_att=("carries", "sum"),
    ).rename(columns={team_col: "team"})
    denom = (grp["team_pass_att"] + grp["team_rush_att"]).clip(lower=1)
    grp["prior_team_pass_rate"] = grp["team_pass_att"] / denom
    return grp


def build_team_starting_qb(player_season_pprpg: pd.DataFrame) -> pd.DataFrame:
    """Per (team, season): the QB with the most games = the de-facto starter,
    with his PPR/game and games. Used as prior-year incumbent-QB context."""
    qbs = player_season_pprpg[player_season_pprpg["position"].eq("QB")].copy()
    qbs = qbs.dropna(subset=["team"])
    qbs = qbs.sort_values("games", ascending=False).drop_duplicates(["team", "season"])
    return qbs.rename(
        columns={"player_id": "starting_qb_id", "pprpg": "prior_qb_pprpg", "games": "prior_qb_games"}
    )[["team", "season", "starting_qb_id", "prior_qb_pprpg", "prior_qb_games"]]


# ---------------------------------------------------------------------------
# Incumbent extension proxy from contracts
# ---------------------------------------------------------------------------
def _load_contracts(project_root: Path) -> pd.DataFrame | None:
    path = project_root / "data" / "raw" / "historical_contracts.csv"
    if not path.exists():
        return None
    c = pd.read_csv(path, low_memory=False)
    for col in ("year_signed", "years", "apy_cap_pct"):
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors="coerce")
    return c


def _incumbent_extension_lookup(contracts: pd.DataFrame) -> pd.DataFrame:
    """Per (gsis_id, year_signed) the contract end year and a 'meaningful'
    flag (non-trivial APY share). Generic, not hand-tuned to any player."""
    c = contracts.dropna(subset=["gsis_id", "year_signed"]).copy()
    c = c[c["year_signed"] > 0]
    c["years"] = c["years"].fillna(1).clip(lower=1)
    c["end_year"] = c["year_signed"] + c["years"]
    c["meaningful"] = (c.get("apy_cap_pct", pd.Series(0, index=c.index)).fillna(0) >= 0.05).astype(float)
    return c[["gsis_id", "year_signed", "end_year", "meaningful"]]


# ---------------------------------------------------------------------------
# Master attach
# ---------------------------------------------------------------------------
def attach_rookie_context_features(
    rookie_frame: pd.DataFrame,
    rosters: pd.DataFrame,
    player_stats: pd.DataFrame,
    project_root: Path | None = None,
    draft: pd.DataFrame | None = None,
    contracts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach combine + prior-season team-context + incumbent/depth features.

    One row in, one row out: asserts row count and rookie-id uniqueness are
    preserved. All non-combine features are derived from ``rookie_year - 1`` and
    from training-camp roster composition; the rookie is excluded from his own
    position's veteran tallies.
    """
    if project_root is None:
        from src.load_data import find_project_root

        project_root = find_project_root()
    project_root = Path(project_root)
    if draft is None:
        dp = _find(project_root, "draft_picks")
        draft = pd.read_csv(dp, low_memory=False) if dp is not None else None
    if contracts is None:
        contracts = _load_contracts(project_root)

    out = rookie_frame.copy()
    before = len(out)
    out["ctx_uid"] = range(before)

    # Rookie team: drafting club (unambiguously pre-season); fall back to the
    # rookie-year roster team for UDFAs.
    out["rookie_team"] = out.get("draft_club")
    ros = rosters.copy()
    ros["season"] = pd.to_numeric(ros["season"], errors="coerce")
    # Fall back to the rookie-year roster team for UDFAs (only if rosters carry
    # a team column — minimal/synthetic frames may not).
    if "team" in ros.columns:
        roster_team = (
            ros.dropna(subset=["gsis_id", "season"])
            .rename(columns={"gsis_id": "player_id", "team": "_roster_team"})
            .drop_duplicates(["player_id", "season"])[["player_id", "season", "_roster_team"]]
        )
        out = out.merge(
            roster_team, left_on=["player_id", "rookie_year"],
            right_on=["player_id", "season"], how="left",
        ).drop(columns=["season"], errors="ignore")
        out["rookie_team"] = out["rookie_team"].fillna(out["_roster_team"])
        out = out.drop(columns=["_roster_team"], errors="ignore")

    # --- Combine ---
    combine = load_combine_features(project_root, rosters, draft)
    if combine is not None:
        out = out.merge(combine, on="player_id", how="left")
    # BMI from the size data already on the frame (height_inches, weight).
    if {"height_inches", "weight"}.issubset(out.columns):
        h = pd.to_numeric(out["height_inches"], errors="coerce")
        w = pd.to_numeric(out["weight"], errors="coerce")
        out["bmi"] = (w / (h * h)) * 703.0

    # --- Prior-season production tables ---
    psp = build_player_season_pprpg(player_stats)
    team_ctx = build_team_season_context(player_stats)
    starting_qb = build_team_starting_qb(psp)
    out["_prior"] = out["rookie_year"] - 1

    out = out.merge(
        team_ctx[["team", "season", "prior_team_pass_rate"]],
        left_on=["rookie_team", "_prior"], right_on=["team", "season"], how="left",
    ).drop(columns=["team", "season"], errors="ignore")
    out = out.merge(
        starting_qb[["team", "season", "starting_qb_id", "prior_qb_pprpg", "prior_qb_games"]],
        left_on=["rookie_team", "_prior"], right_on=["team", "season"], how="left",
    ).drop(columns=["team", "season"], errors="ignore")

    # --- Incumbent / depth at the rookie's position ---
    depth = _build_position_depth(out, ros, psp, contracts)
    out = out.merge(depth, on="ctx_uid", how="left")
    for col in ["pos_vet_count", "pos_vet_max_pprpg", "pos_vet_sum_pprpg",
                "established_incumbent", "incumbent_recent_extension"]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    out = out.drop(columns=["_prior", "ctx_uid", "starting_qb_id"], errors="ignore")

    assert len(out) == before, f"context join changed row count: {before} -> {len(out)}"
    assert not out.duplicated("player_id").any(), "duplicate rookie player_id after join"
    return out.reset_index(drop=True)


def _build_position_depth(
    rookies: pd.DataFrame,
    rosters: pd.DataFrame,
    psp: pd.DataFrame,
    contracts: pd.DataFrame | None,
) -> pd.DataFrame:
    """Veteran depth at each rookie's position on his rookie-year team.

    Veterans = players rostered at the same position on the rookie's team in his
    rookie season who ENTERED the league before that season (entry_year <
    rookie_year), explicitly excluding the rookie himself. Their prior-year
    PPR/game (max, sum) and counts summarize the wall the rookie has to climb.
    """
    default_cols = ["ctx_uid", "pos_vet_count", "pos_vet_max_pprpg", "pos_vet_sum_pprpg",
                    "established_incumbent", "incumbent_recent_extension"]
    if "team" not in rosters.columns or "position" not in rosters.columns:
        # Insufficient roster detail to compute depth; emit defaults so the
        # attach still succeeds on minimal/synthetic frames.
        d = pd.DataFrame({"ctx_uid": rookies["ctx_uid"].to_numpy()})
        for c in default_cols[1:]:
            d[c] = 0.0
        return d

    ros = rosters.copy()
    for c in ("season", "entry_year", "rookie_year"):
        if c in ros.columns:
            ros[c] = pd.to_numeric(ros[c], errors="coerce")
    entry = ros["entry_year"]
    if entry.isna().all() and "rookie_year" in ros.columns:
        entry = ros["rookie_year"]
    ros = ros.assign(_entry=entry).rename(columns={"gsis_id": "rid"})

    ext = _incumbent_extension_lookup(contracts) if contracts is not None else None

    recs = []
    # Index prior-year production for quick lookup.
    psp_idx = psp.set_index(["player_id", "season"])["pprpg"].to_dict()

    for row in rookies.itertuples(index=False):
        uid = row.ctx_uid
        team = getattr(row, "rookie_team", None)
        pos = row.position
        ryear = row.rookie_year
        prior = ryear - 1
        rec = {"ctx_uid": uid, "pos_vet_count": 0, "pos_vet_max_pprpg": 0.0,
               "pos_vet_sum_pprpg": 0.0, "established_incumbent": 0.0,
               "incumbent_recent_extension": 0.0}
        if team is None or (isinstance(team, float) and np.isnan(team)):
            recs.append(rec); continue

        mates = ros[(ros["team"] == team) & (ros["season"] == ryear)
                    & (ros["position"] == pos) & (ros["rid"] != row.player_id)
                    & (ros["_entry"] < ryear)]
        mates = mates.drop_duplicates("rid")
        vet_pprpg = [psp_idx.get((rid, prior), 0.0) for rid in mates["rid"]]
        vet_pprpg = [v for v in vet_pprpg if pd.notna(v)]
        rec["pos_vet_count"] = float(len(mates))
        if vet_pprpg:
            rec["pos_vet_max_pprpg"] = float(np.max(vet_pprpg))
            rec["pos_vet_sum_pprpg"] = float(np.sum(vet_pprpg))
        # Established incumbent: a returning vet at the position who was a real
        # contributor last year (>= 8 PPR/game ~ a clear starter/rotation piece).
        rec["established_incumbent"] = float(rec["pos_vet_max_pprpg"] >= 8.0)

        # Incumbent extension: did the position's top returning vet sign a
        # meaningful deal recently (within the prior 2 years) that still runs
        # into the rookie year? Generic, derived only from contracts.
        if ext is not None and vet_pprpg:
            top_rid = mates.iloc[int(np.argmax([psp_idx.get((rid, prior), 0.0) for rid in mates["rid"]]))]["rid"]
            ec = ext[(ext["gsis_id"] == top_rid)
                     & (ext["year_signed"] >= ryear - 2) & (ext["year_signed"] <= ryear - 1)
                     & (ext["end_year"] >= ryear) & (ext["meaningful"] > 0)]
            rec["incumbent_recent_extension"] = float(len(ec) > 0)
        recs.append(rec)

    return pd.DataFrame(recs)
