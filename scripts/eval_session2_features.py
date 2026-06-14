"""Deprecated — superseded by ``scripts/eval_session2_diagnostic.py``.

The original three-arm benchmark (baseline / +NGS / +NGS+PFR) was replaced by
the fuller diagnostic, which also runs the shuffled-value permutation test and
the leak-free coverage-flag arm that together established the Session 2 negative
result. See ``report/fantasy/session2_ngs_pfr_features.md``.
"""

if __name__ == "__main__":
    raise SystemExit(
        "Use: python -m scripts.eval_session2_diagnostic "
        "(this script was replaced; see report/fantasy/session2_ngs_pfr_features.md)"
    )
