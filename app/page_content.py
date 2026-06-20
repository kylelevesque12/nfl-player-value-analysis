"""Pure (Streamlit-free) copy + caveat config for the detail pages.

Every detail page uses one structure: title + purpose, executive summary, KPI
row, a concise visual/table, a caveat callout, optional detail expanders, and a
footer. The textual content lives here so it can be unit-tested without a
Streamlit runtime and kept consistent with the latest committed results.

Each entry: title, purpose (one sentence), summary (executive-summary bullets),
caveat {label, body} (a limitation that must stay visible), footer (source note).
"""

from __future__ import annotations

DETAIL_PAGES: dict[str, dict] = {
    "surplus": {
        "title": "Replacement-Level Surplus",
        "purpose": "Front-office view: value a player delivered above a freely available replacement, priced against a season-specific reconstructed cap hit.",
        "summary": [
            "For each (season, position), the framework estimates the cap cost and value of a 'next man up' replacement; a player's dollar surplus is their value-over-replacement (priced via the within-(season, position) salary-on-value slope) minus the cap premium they cost above replacement.",
            "Brock Purdy's 2023 season is the largest single-season surplus in 2016-2025; rookie-deal QBs dominate the top of the board because their reconstructed cap hits are tiny relative to their production.",
            "The RB market shows a negative implicit price for value at the position level, the well-documented RB-market inefficiency.",
        ],
        "caveat": {
            "label": "Reconstructed estimate",
            "body": "Cap cost is a season-specific cap hit reconstructed from contract terms (prorated signing bonus + backloaded base, in inflation-adjusted millions), an estimate, not exact NFL cap accounting, since the source data has no year-by-year cap breakdown. Every player-season carries a cap_hit_quality_flag.",
        },
        "footer": "Source: nflverse / OverTheCap historical contracts + player_value_scores. Cap hits via src/cap_hit_reconstruction.py.",
    },
    "benchmark": {
        "title": "External Benchmark: weekly model vs the market",
        "purpose": "How the weekly model compares to a market-implied projection on the one window where a free benchmark exists.",
        "summary": [
            "The model's primary, all-years claim is a 7-9% RMSE reduction versus the naive forecasting baselines (recent-form and season-to-date averages), the standard bar any forecast must clear, sustained in every season 2020-2025.",
            "On the 2020-2021 window where a free market-implied benchmark exists (DraftKings closing-line salaries via RotoGuru), the model is competitive-to-slightly-ahead of the DK-implied projection on 11,191 matched player-weeks.",
            "This is a scoped, secondary check, not a claim of beating live DraftKings, FantasyPros, or ESPN in recent years.",
        ],
        "caveat": {
            "label": "Limited benchmark",
            "body": "The DraftKings comparison covers only 2020-2021 matched player-weeks (RotoGuru's free archive ends in 2021) and is a market-implied proxy whose salary→points conversion is fit on in-season actuals, making it a deliberately tough bar. It cannot be extended to recent years without paid projection data. See report/fantasy/external_projection_benchmark_feasibility.md.",
        },
        "footer": "Source: external_benchmark.py vs draftkings_implied_via_rotoguru. Baseline skill scores: report/weekly_fantasy_projection_summary.md.",
    },
    "causal": {
        "title": "Causal: QB injury report → WR PPR",
        "purpose": "Does a starting QB's first injury-report appearance cause a measurable change in his receivers' fantasy production?",
        "summary": [
            "Defining treatment as the formal Out designation produces a null: by the time a QB is ruled Out he has usually been playing hurt for weeks, so the Out flag lags the causal damage.",
            "Re-timing treatment to the first week the established starter appears on the injury report at all, any status, expands the event set from 19 (Out-only) to 104 and surfaces a small post-period decline.",
            "Pooled post-period ATT is about -0.58 PPG (p ~= 0.04), concentrated one game after the first report, consistent with the mechanism that damage clusters around when QB health first becomes shaky.",
        ],
        "caveat": {
            "label": "Suggestive / underpowered",
            "body": "With ~104 events the design is moderately powered and the estimate sits near the 5% significance border. The fixed-effect parallel-trends test passes, but the cell-mean event study shows a slightly elevated treated-minus-control gap at offset -3, so part of the post drop may be continuation of a pre-existing decline rather than pure causal effect. Reported as suggestive, not a headline causal estimate.",
        },
        "footer": "Source: src/causal/first_report_treatment.py. Report: report/causal/qb_injury_session3.md.",
    },
    "rookie": {
        "title": "Bayesian Rookie Cold-Start + incumbent context",
        "purpose": "Project rookies who have no rolling history, and sharpen the 'will he play?' gate with pre-season incumbent context.",
        "summary": [
            "A hierarchical Bayesian model (partial pooling across positions) solves the cold-start problem and hits near-nominal posterior interval coverage in every rookie class.",
            "A focused incumbent-context core (established incumbent, recent extension, prior-year starting-QB production) sharpens the hurdle stage, the QB 'is he blocked?' cell.",
            "Jordan Love's modeled P(plays) moves the right way once the model can see Green Bay had a recently-extended incumbent: 0.611 -> 0.513.",
        ],
        "caveat": {
            "label": "Scope of the gain",
            "body": "Combine athletic-testing features and the broader veteran-depth features were tested and dropped, they did not beat draft capital. Only a 3-feature incumbent core was kept. The aggregate QB AUC gain is small; the value is concentrated in the rare blocked-QB cell (Love, Mahomes), so this is a targeted improvement, not a big across-the-board lift.",
        },
        "footer": "Source: src/rookie_bayes.py + src/rookie_context_features.py. Report: report/rookie/session3_combine_team_context.md.",
    },
    "two_stage": {
        "title": "Two-Stage WR/TE Decomposition (negative result)",
        "purpose": "Does decomposing weekly WR/TE PPR into team attempts x target share x efficiency beat the pooled model? It does not.",
        "summary": [
            "The two-stage multiplicative product loses to the pooled HGB in every validation fold.",
            "The per-stage diagnostic shows why: stage 1 (renormalized target share) carries real signal, but stages 2-3 are near-noise, and multiplying noisy estimates compounds error the pooled model avoids.",
            "This is one of several decomposition experiments kept on the record specifically because they lost, negative results are documented, not hidden.",
        ],
        "caveat": {
            "label": "Documented negative result",
            "body": "Nothing from this experiment is in production. It is retained as evidence of the modeling discipline: pooled tree models on engineered rolling features beat every multiplicative decomposition tried here.",
        },
        "footer": "Source: src/two_stage_weekly.py. Report: report/two_stage_weekly.md.",
    },
    "methodology": {
        "title": "Methodology & Trust Signals",
        "purpose": "The models behind each result, and the checks that make those results defensible, not a proof of correctness, but the guardrails against common failure modes.",
        "summary": [
            "Models by task: weekly fantasy points use histogram-based gradient-boosted regression trees (HistGradientBoosting) on engineered rolling role features; rookie projections use a hierarchical Bayesian model with partial pooling across positions; the QB-injury question uses difference-in-differences / event-study estimation. Gradient boosting is strong on tabular, non-linear interactions; the Bayesian model is strong at the cold-start problem and honest interval coverage; DiD isolates a causal effect under stated assumptions.",
            "Known limitations: weekly intervals under-cover heavy-tailed QB scoring even after per-position calibration; the causal estimate is moderately powered and sits near the 5% border; reconstructed cap hits are estimates, not exact NFL accounting.",
            "Leakage-safe feature construction: every rolling/lagged feature is shift(1)-safe; NGS/PFR features were rejected because their availability/missingness leaked same-week status.",
            "Time-based validation only: rolling-origin backtests predict each season from strictly earlier data, so recent seasons are genuine out-of-sample tests.",
            "Negative results documented: NGS/PFR, the ensemble/quantile experiment, and the two-stage decomposition all lost and are kept on the record.",
        ],
        "caveat": {
            "label": "What checks can and can't do",
            "body": "These checks catch project-quality and leakage risks; they do not prove the model is correct. Important limitations stay visible on each page rather than being summarized away here.",
        },
        "footer": "Source: src/methodology_checks.py + the project test suite. Report: report/methodology_checks.md.",
    },
}

# Caveat keywords that must remain present on specific pages (used by tests to
# guard against a future edit silently dropping a required disclosure).
REQUIRED_CAVEAT_TOKENS: dict[str, list[str]] = {
    "surplus": ["reconstructed", "not exact"],
    "benchmark": ["2020-2021", "market-implied"],
    "causal": ["suggestive", "underpowered", "parallel-trends"],
    "rookie": ["tested and dropped", "incumbent"],
    "two_stage": ["negative result"],
    "methodology": ["leakage"],
}


def page(key: str) -> dict:
    return DETAIL_PAGES[key]
