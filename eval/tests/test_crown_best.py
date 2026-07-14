"""CPU-only tests for `_crown_best` (issue #82): eval must not crown a "best"
transform when nothing actually beats exact.

The evaluator sets a transform's `score` to 0 unless it is a genuine improvement
over exact. Previously `best` was just the top-ranked row, so a table where
every score is 0 (all gated, or nothing dominates exact) still reported a
"best" — implying an improvement that isn't there.

Run:  python eval/tests/test_crown_best.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.evaluator import _crown_best


def _rank(*pairs):
    # pairs: (name, score) in already-sorted (descending) order.
    return [(name, {"score": score}) for name, score in pairs]


def test_no_best_when_all_scores_zero():
    # Every transform gated / not an improvement -> no winner.
    assert _crown_best(_rank(("rsvd", 0.0), ("nystrom", 0.0))) is None


def test_no_best_for_empty_ranking():
    assert _crown_best([]) is None


def test_best_is_the_positive_top_scorer():
    assert _crown_best(_rank(("nystrom", 12.3), ("rsvd", 0.0))) == "nystrom"


def test_single_positive_transform_is_crowned():
    assert _crown_best(_rank(("rsvd", 4.5))) == "rsvd"


def test_single_zero_transform_is_not_crowned():
    assert _crown_best(_rank(("rsvd", 0.0))) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
