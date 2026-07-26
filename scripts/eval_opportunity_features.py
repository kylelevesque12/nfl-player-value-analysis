"""A/B test: do share-normalized opportunity features earn their place?

Step 1 of the 2026-07 review (report/project_review_2026_07.md §1B.7). The season
fantasy model uses raw counts (targets, receptions, carries), which conflate a
player's role with his team's play volume. This evaluates adding share-normalized
opportunity — target_share, air_yards_share, wopr, carry_share and one-season
lags — against the frozen V1 feature set.

Rules, matching the project's existing ablation discipline
(report/fantasy/injury_return_features.md):
  * same rolling validation years, same models, same target
  * the bar is the project's ~0.2% RMSE improvement threshold
  * the result is reported either way; a negative result is a finding

Nothing is registered into production by this script. It writes a comparison
table and prints a verdict.

Run:  .venv/bin/python scripts/eval_opportunity_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fantasy_projection import (  # noqa: E402
    FANTASY_FEATURES_V1,
    OPPORTUNITY_FEATURES,
    create_fantasy_modeling_frame,
    rolling_fantasy_validation,
    summarize_fantasy_model_comparison,
)
from src.load_data import load_csv  # noqa: E402
from src.opportunity_features import attach_opportunity_features  # noqa: E402

SKILL_SEASONS = ROOT / "data" / "processed" / "skill_player_seasons_2016_2025.csv"
OUT = ROOT / "outputs" / "tables" / "opportunity_feature_ablation.csv"
IMPROVEMENT_BAR = 0.002  # 0.2%, the project's existing ablation threshold


def _overall(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[summary["segment"].eq("overall")].set_index("model_name")


def main() -> None:
    if not SKILL_SEASONS.exists():
        raise SystemExit(
            f"missing {SKILL_SEASONS.relative_to(ROOT)} — run the pipeline first"
        )
    skill_seasons = load_csv(SKILL_SEASONS)
    print(f"skill player-season-team rows: {len(skill_seasons):,}")

    # Baseline frame (V1) and the same frame with share features attached.
    base_frame = create_fantasy_modeling_frame(skill_seasons)
    share_frame = attach_opportunity_features(base_frame, skill_seasons)

    available_shares = [c for c in OPPORTUNITY_FEATURES if c in share_frame.columns]
    missing = sorted(set(OPPORTUNITY_FEATURES) - set(available_shares))
    print(f"share features built: {available_shares}")
    if missing:
        print(f"  (unavailable, skipped: {missing})")

    coverage = (
        share_frame[available_shares].notna().mean().round(3).to_dict()
        if available_shares
        else {}
    )
    print(f"non-null coverage: {coverage}")

    # A third arm, and the one that actually tests the review's thesis: a small,
    # deliberately-motivated feature set that uses SHARES INSTEAD OF raw counts,
    # rather than piling shares on top of 44 redundant columns. Every member is
    # here for a stated reason — profile, durability, scoring rate, history,
    # efficiency, and role — with no volume/rate/lag duplicates of the same
    # quantity.
    lean_features = [
        # profile
        "position", "age", "years_exp", "draft_number",
        # durability
        "games_played",
        # scoring rate (not total: total conflates rate with availability)
        "fantasy_points_ppr_per_game",
        # short history
        "fantasy_points_ppr_prev", "fantasy_points_ppr_last2_avg",
        # efficiency signal, per game
        "value_epa_per_game",
        # role, share-normalized
        "target_share", "air_yards_share", "carry_share", "target_share_prev",
    ]

    # Control arm: the SAME lean design but with raw per-game counts where the
    # lean arm uses shares. Without this, "lean matches V1" is confounded — it
    # changes size, scaling and normalization at once, so it cannot say whether
    # share-normalization specifically contributed anything.
    lean_raw_features = [
        "position", "age", "years_exp", "draft_number",
        "games_played",
        "fantasy_points_ppr_per_game",
        "fantasy_points_ppr_prev", "fantasy_points_ppr_last2_avg",
        "value_epa_per_game",
        # role, raw (the counterpart of the share block above)
        "targets_per_game", "receptions_per_game", "carries_per_game", "targets_prev",
    ]

    runs = {
        "v1_baseline": (base_frame, list(FANTASY_FEATURES_V1)),
        "v1_plus_shares": (share_frame, list(FANTASY_FEATURES_V1) + available_shares),
        "lean_raw": (share_frame, lean_raw_features),
        "lean_shares": (share_frame, lean_features),
    }

    summaries = {}
    for label, (frame, features) in runs.items():
        feats = [c for c in features if c in frame.columns]
        print(f"\n=== {label}: {len(feats)} features ===")
        preds = rolling_fantasy_validation(frame, feats)
        summaries[label] = _overall(summarize_fantasy_model_comparison(preds))

    base = summaries["v1_baseline"]
    rows = []
    for model in base.index:
        row = {
            "model": model,
            "n_features_v1": len(FANTASY_FEATURES_V1),
            "v1_rmse": base.loc[model, "rmse"],
            "v1_spearman": base.loc[model, "spearman_rank_corr"],
        }
        for label in ("v1_plus_shares", "lean_raw", "lean_shares"):
            s = summaries[label]
            if model in s.index:
                row[f"{label}_rmse"] = s.loc[model, "rmse"]
                row[f"{label}_spearman"] = s.loc[model, "spearman_rank_corr"]
                row[f"{label}_pct_improvement"] = (
                    base.loc[model, "rmse"] - s.loc[model, "rmse"]
                ) / base.loc[model, "rmse"]
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values("v1_rmse")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT, index=False)

    print("\n=== RMSE by arm (lower is better) ===")
    cols = [
        "model", "v1_rmse", "v1_plus_shares_rmse", "lean_raw_rmse", "lean_shares_rmse",
        "lean_shares_pct_improvement", "lean_shares_spearman",
    ]
    print(
        comparison[[c for c in cols if c in comparison.columns]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    n_lean = len([c for c in lean_features if c in share_frame.columns])
    for label, n_feat in (("v1_plus_shares", len(FANTASY_FEATURES_V1) + len(available_shares)),
                          ("lean_shares", n_lean)):
        col = f"{label}_pct_improvement"
        if col not in comparison.columns:
            continue
        best = comparison.loc[comparison[f"{label}_rmse"].idxmin()]
        print(
            f"\n[{label}] {n_feat} features | best {best['model']}: "
            f"RMSE {best[f'{label}_rmse']:.3f} vs V1 {best['v1_rmse']:.3f} "
            f"({best[col]:+.3%})"
        )
        if label == "lean_shares":
            # The lean arm is a SIMPLIFICATION test: matching V1 with a third of
            # the features is the win, so judge it on parity, not improvement.
            within_noise = abs(best[col]) < IMPROVEMENT_BAR
            print(
                "  VERDICT: matches V1 within noise using "
                f"{n_feat} vs {len(FANTASY_FEATURES_V1)} features — "
                "SIMPLIFICATION JUSTIFIED" if within_noise
                else ("  VERDICT: BEATS V1 outright" if best[col] > 0
                      else "  VERDICT: loses more than the bar — keep more features")
            )
        else:
            print(
                f"  VERDICT: {'EARNS ITS PLACE' if best[col] >= IMPROVEMENT_BAR else 'DOES NOT clear the bar'}"
            )
    # Isolate the share contribution: lean_shares vs lean_raw, same design size.
    if {"lean_raw_rmse", "lean_shares_rmse"}.issubset(comparison.columns):
        print("\n=== does share-normalization itself help? (lean_shares vs lean_raw) ===")
        for _, r in comparison.iterrows():
            if pd.isna(r.get("lean_raw_rmse")) or pd.isna(r.get("lean_shares_rmse")):
                continue
            delta = (r["lean_raw_rmse"] - r["lean_shares_rmse"]) / r["lean_raw_rmse"]
            print(f"  {r['model']:<34} raw {r['lean_raw_rmse']:.3f} -> shares "
                  f"{r['lean_shares_rmse']:.3f}  ({delta:+.3%})")

    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
