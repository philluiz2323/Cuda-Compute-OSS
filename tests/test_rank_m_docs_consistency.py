"""CPU-only tests that the advertised default rank_m matches the real
default (issue #98): strategy/subspace.default_rank_m is floored at 64,
but strategy/cli.py, strategy/README.md and CONTRIBUTING.md previously
advertised a bare n//8 with no floor.

Run:  python tests/test_rank_m_docs_consistency.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.cli import build_parser
from strategy.subspace import default_rank_m

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The stale, unfloored claim these three surfaces used to make.
_STALE_PATTERNS = [
    re.compile(r"default n//8"),
    re.compile(r"n // 8`\s*=\s*1500"),
    re.compile(r"M\s*=\s*N//8 grow"),
]


def _read(relpath: str) -> str:
    with open(os.path.join(_ROOT, relpath), encoding="utf-8") as f:
        return f.read()


def test_default_rank_m_has_the_64_floor():
    # Sanity check on the real function these docs must describe.
    assert default_rank_m(256) == 64
    assert default_rank_m(256) != 256 // 8
    assert default_rank_m(12000) == 12000 // 8  # floor inactive once N >= 512


def test_cli_help_states_the_real_default():
    actions = {a.dest: a for a in build_parser()._actions}
    help_text = actions["rank_m"].help
    for pat in _STALE_PATTERNS:
        assert not pat.search(help_text), f"stale claim in --rank-m help: {help_text!r}"
    assert "max(64" in help_text


def test_strategy_readme_states_the_real_default():
    text = _read("strategy/README.md")
    for pat in _STALE_PATTERNS:
        assert not pat.search(text), f"stale claim in strategy/README.md: {pat.pattern!r}"
    assert "max(64" in text


def test_contributing_states_the_real_default():
    text = _read("CONTRIBUTING.md")
    for pat in _STALE_PATTERNS:
        assert not pat.search(text), f"stale claim in CONTRIBUTING.md: {pat.pattern!r}"
    assert "max(64" in text


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
