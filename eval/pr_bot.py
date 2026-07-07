"""PR evaluation gate chain -- oldest-PR-first, adapted from sparkinfer's
``eval/pr_eval_bot.py`` (see docs/sn74-emission-strategy.md,
docs/testing-strategy.md).

Gate chain per open PR, in order:
    draft skip -> blocked-contributor check -> copycat check
    (eval.copycat_guard) -> idempotency marker -> self-scorecard greenlight
    -> [GPU eval -- stubbed in Phase 1, wired to eval.runner in Phase 2]

The DECISION (:func:`process_pr`) is a pure function of already-fetched data
-- no GitHub I/O happens inside it, so the whole gate chain is unit-testable
with plain fixtures (see eval/tests/test_pr_bot.py). :class:`GitHubClient` is
the only place that talks to ``gh``; :func:`run_once` wires the two together
and is the only place writes (comment/label/close) happen, and only when
``dry_run=False``.

Phase 1 keeps write-back OFF by default -- there is no live bot identity or
GPU runner yet (Phase 2/3). The write path below is implemented and tested
now so it's ready to flip on deliberately later, not exercised against the
real repo by default.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import copycat_guard
from .github_client import GitHubClient, PRInfo

REPO_DEFAULT = "zeokin/Cuda-Compute-OSS"
BLOCKED_CONTRIBUTORS_PATH = ".github/blocked-contributors.txt"
IDEMPOTENCY_MARKER = "<!-- cco-eval:{sha} -->"

# Matches .github/workflows/labeler.yml's existing status:needs-scorecard
# detector exactly (kept as one Python regex so the two never drift): treat
# the PR as missing its scorecard unless the body carries an actual filled-in
# metric, not just an empty template or a checked-but-unsubstantiated box.
# Phase 2's real runner is the actual authority on whether a PR improved
# anything -- this is only a courtesy pre-filter.
SCORECARD_RE = re.compile(
    r"accuracy\s*\|?\s*[0-9]|latency\s*\|?\s*[0-9]|RESULT_JSON", re.IGNORECASE
)


@dataclass
class GateOutcome:
    """The bot's decision for one PR."""
    pr: int
    action: str  # see process_pr's docstring for the full set of values
    detail: str = ""
    label: str | None = None


def load_blocked_contributors(path: str = BLOCKED_CONTRIBUTORS_PATH) -> frozenset:
    p = Path(path)
    if not p.exists():
        return frozenset()
    return frozenset(
        line.strip() for line in p.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def already_evaluated(comments: list, head_sha: str) -> bool:
    marker = IDEMPOTENCY_MARKER.format(sha=head_sha)
    return any(marker in c for c in comments)


def has_scorecard(body: str) -> bool:
    return bool(SCORECARD_RE.search(body or ""))


def process_pr(
    pr: PRInfo,
    diff_text: str,
    comments: list,
    blocked: frozenset,
    originals: list,
    run_eval=None,
) -> GateOutcome:
    """Decide the gate-chain outcome for one PR. Pure: takes already-fetched
    data, performs no GitHub I/O, so it's fully unit-testable.

    originals : list of (author, copycat_guard.Fingerprint) for every earlier
                PR, oldest first. The candidate's own earlier PRs are
                filtered out here (self-resubmission is not copying).
    run_eval  : callable(pr) -> dict, or None (Phase 1's stub -- gate chain
                passed but there is no GPU runner wired up yet).

    action values: skip_draft, close_blocked, copycat_block, copycat_warn,
    already_evaluated, needs_scorecard, eval_pending, evaluated.
    """
    if pr.is_draft:
        return GateOutcome(pr.number, "skip_draft")

    if pr.author in blocked:
        return GateOutcome(pr.number, "close_blocked",
                           detail=f"{pr.author} is on the blocked-contributors list")

    fp = copycat_guard.fingerprint(diff_text)
    others = [(a, f) for a, f in originals if a != pr.author]
    matched_author, verdict = copycat_guard.worst_verdict(fp, others)
    if verdict.tier == "block":
        return GateOutcome(pr.number, "copycat_block",
                           detail=f"matches an earlier PR by {matched_author}: {verdict.reason}",
                           label="copycat")
    if verdict.tier == "warn":
        return GateOutcome(pr.number, "copycat_warn",
                           detail=f"matches an earlier PR by {matched_author}: {verdict.reason}",
                           label="copycat-warn")

    if already_evaluated(comments, pr.head_sha):
        return GateOutcome(pr.number, "already_evaluated")

    if not has_scorecard(pr.body):
        return GateOutcome(pr.number, "needs_scorecard",
                           detail="PR body is missing a filled-in scorecard "
                                  "(see CONTRIBUTING.md)",
                           label="status:needs-scorecard")

    if run_eval is None:
        return GateOutcome(pr.number, "eval_pending",
                           detail="gate chain passed; GPU eval runner is not "
                                  "wired up yet in this repo (Phase 2)")

    result = run_eval(pr)
    return GateOutcome(pr.number, "evaluated", detail=json.dumps(result))


def _apply(client: GitHubClient, pr: PRInfo, outcome: GateOutcome) -> None:
    """Perform the write action implied by ``outcome``. Only called from
    :func:`run_once` when ``dry_run=False`` -- not the default in Phase 1."""
    if outcome.action == "close_blocked":
        client.close_pr(pr.number, outcome.detail)
    elif outcome.action == "copycat_block":
        client.add_label(pr.number, "copycat")
        client.post_comment(pr.number, f"Closed as a copycat submission: {outcome.detail}")
        client.close_pr(pr.number, "copycat")
    elif outcome.action == "copycat_warn":
        client.add_label(pr.number, "copycat-warn")
        client.post_comment(pr.number, f"Flagged for maintainer review: {outcome.detail}")
    elif outcome.action == "needs_scorecard":
        client.add_label(pr.number, "status:needs-scorecard")
        client.post_comment(pr.number,
                            "Please add a filled-in scorecard from "
                            "`python -m eval` (see CONTRIBUTING.md).")
    elif outcome.action == "eval_pending":
        client.post_comment(
            pr.number,
            IDEMPOTENCY_MARKER.format(sha=pr.head_sha)
            + "\nGate chain passed. Automated GPU evaluation is not live in "
              "this repo yet (tracked as Phase 2 of the update plan).",
        )
    # skip_draft / already_evaluated / evaluated: nothing to write here.
    # ("evaluated" posts its own scorecard comment from within run_eval /
    # eval.runner once Phase 2 wires that in -- not this function's job.)


def run_once(client: GitHubClient, dry_run: bool = True, run_eval=None) -> list:
    """Fetch state via ``client``, decide via :func:`process_pr` for every
    open PR (oldest first), then apply the outcome unless ``dry_run``."""
    blocked = load_blocked_contributors()
    all_prs = sorted(client.list_prs("all"), key=lambda p: p.number)
    open_prs = sorted(client.list_prs("open"), key=lambda p: p.number)

    diff_by_pr = {}
    fp_by_pr = {}
    for p in all_prs:
        diff_by_pr[p.number] = client.get_diff(p.number)
        fp_by_pr[p.number] = copycat_guard.fingerprint(diff_by_pr[p.number])

    outcomes = []
    for pr in open_prs:
        diff = diff_by_pr.get(pr.number, client.get_diff(pr.number))
        comments = client.get_comments(pr.number)
        # Every earlier PR (any state -- open, closed, or merged) is a valid
        # copycat comparison target; PR number order is creation order.
        originals = [(p.author, fp_by_pr[p.number]) for p in all_prs if p.number < pr.number]
        outcome = process_pr(pr, diff, comments, blocked, originals, run_eval)
        outcomes.append(outcome)

        if not dry_run:
            _apply(client, pr, outcome)
    return outcomes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.pr_bot",
        description="Gate open PRs (draft/blocked/copycat/scorecard checks). "
                    "GPU evaluation itself is stubbed until Phase 2.",
    )
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--write", action="store_true",
                   help="actually write labels/comments/close actions back to "
                        "GitHub. Omit this in Phase 1; dry-run is the safe "
                        "default until Phase 3 wires up a live bot identity.")
    args = p.parse_args(argv)

    client = GitHubClient(args.repo)
    outcomes = run_once(client, dry_run=not args.write)
    for o in outcomes:
        line = f"PR #{o.pr}: {o.action}"
        if o.detail:
            line += f" -- {o.detail}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
