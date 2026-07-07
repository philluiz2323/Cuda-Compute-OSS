# Session Log — CCO / SN74 Strategy & Build Session

*Saved 2026-07-07. A full record of a multi-part working session covering
project review, competitive analysis, a new research direction, an SN74
strategy, and a Phase 1 implementation. Not a polished project doc — this is
a personal working journal, kept outside `docs/` on purpose. Useful for
maintainer planning, but not authoritative: the source of truth is the current
code plus `README.md`, `CONTRIBUTING.md`, and `BENCHMARKS.md`. Historical
status notes below reflect the save-time state of the session and may drift as
the repo changes later.*

---

## 1. What this session covered, in order

1. Deep review of **CCO (Cuda-Compute-OSS)** — what it is, how it works
2. Deep review of **entrius/gittensor** — the SN74 subnet mechanism
3. Competitive comparison: CCO vs **sparkinfer** (SN74's top earner, 34.25%)
4. A new research direction: **Spectral Token Mixing** (FFT-based sub-quadratic
   attention) — feasibility research, an M1–M4 implementation plan, and two
   whitepapers
5. Review of the top 6 SN74 repos by emission share + a targeted strategy for
   CCO's own registry entry
6. A testing-strategy design (local vs. validator tiers) with GPU/model pins
7. A full update plan (plan-mode) covering miner structure, a validation bot,
   scoring, and a dashboard
8. **Phase 1 implementation** — built, tested, and verified (Phases 2–4 not
   started; gated on GPU/hosting access you haven't provided yet)

---

## 2. CCO — what it is (initial review)

CCO is an open-source benchmark arena for **approximate matrix
multiplication**. Core packages:

- **`matmul/`** — the exact `O(N³)` baseline engine (in-core + out-of-core
  tiled GEMM via PyTorch, supports N up to 128k/256k on a single GPU)
- **`strategy/`** — the "smart" subspace strategy: compress `A, B` into an
  `M`-dimensional subspace (`M ≪ N`), multiply the small cores, reconstruct —
  `O(N²M)` instead of `O(N³)`. The pluggable part is a `Transform` supplying
  the basis `Q`. Only one transform existed at session start: `rsvd`
  (randomized SVD range-finder).
- **`eval/`** — the scorer: generates random matrix pairs, computes exact and
  approximate products on identical inputs, reports accuracy (bounded
  Frobenius error), latency, peak VRAM, FLOP ratio, gated by a **dominance
  rule** (admitted only if every cost axis improves *and* accuracy holds).

The repo is whitelisted on **Bittensor SN74 (Gittensor)** for a **1% emission
share** — merged PRs earn TAO. Git history showed a steady stream of small
fix PRs from many different usernames, consistent with active SN74 mining.

Landing page (`index/index.html`, now rewritten — see §9) and a whitepaper
PDF originally pitched a **different, grander vision**: a standalone
commercial venture ("attention_optimization") with 72-minute prize cycles,
winner-take-all payouts, and a commercial API for sub-quadratic attention
kernels — contradicting the actual repo, which only does matmul today. This
mismatch was flagged early and became a recurring theme (see §9).

---

## 3. Gittensor (SN74) — the mechanism

`entrius/gittensor` is the Bittensor subnet codebase. Mechanism:

- Miners register a GitHub PAT with validators (no GPU/neuron needed for
  mining itself — the "mining" is contributing code).
- Validators verify merged PRs on whitelisted repos, score them, and pay TAO.
- **Master registry**: `gittensor/validator/weights/master_repositories.json`
  — each repo has an `emission_share` (bounded % of the pool), optional
  `label_multipliers` (per-label pay tiers), `trusted_label_pipeline` flag
  (delegates scoring to the repo's own deterministic bot), `maintainer_cut`,
  and eligibility gates (`min_credibility`, `min_valid_merged_prs`, etc.).
- Scoring defaults to tree-sitter-based **token scoring** (weighted by
  language, discounting tests/comments) unless a repo opts into
  `fixed_base_score` + `label_multipliers` (the sparkinfer pattern — pay by
  verified outcome, not lines of code).
- 90% of emissions go to the PR/issue-discovery pool (split by repo share);
  10% to an on-chain issues-bounty treasury (an ink! smart contract on
  Subtensor, settled by validator consensus over GitHub state).
- Anti-gaming: credibility ratio gates, PR-spam collateral, review-quality
  multipliers, duplicate-penalty propagation.

**Top repos by emission share** (as reviewed, since revised — check current
`master_repositories.json` before relying on these numbers):

| repo | share | model |
|---|---:|---|
| gittensor-ai-lab/sparkinfer | 34.25% | deterministic eval bot, verified-speedup tiers |
| JSONbored/metagraphed | 23% | maintainer-priced labels, fast decay |
| gittensor-vanguard/vanguarstew | 10% | human review, work-taxonomy multipliers |
| JSONbored/gittensory | 10% | fully automated one-shot auto-merge/close |
| Autovara/kata | 8% | tournament, winner-take-all per round |
| entrius/gittensor | 6% | default token economy |
| zeokin/Cuda-Compute-OSS (CCO) | **1%** | default token economy (flat, no labels) |

Full per-repo scoring-mechanics breakdown is in
[`docs/sn74-emission-strategy.md`](docs/sn74-emission-strategy.md).

---

## 4. CCO vs. sparkinfer

sparkinfer (`gittensor-ai-lab/sparkinfer`) is a Blackwell-native LLM inference
runtime (CUDA kernels for flash-decode, MoE FFN, RMSNorm, etc.) targeting
consumer RTX 50-series GPUs. It's SN74's top earner.

**Shared DNA**: exact-baseline-vs-strategy comparison, dominance/gate logic,
"no LLM-as-a-judge" ethos, numbers-over-narrative, anti-copycat enforcement.

**Key differences found**:
- sparkinfer has a **live, running eval bot** (`eval/pr_eval_bot.py` +
  `vast_eval.py`) that rents a GPU on demand, builds `main` and the PR from
  source, benchmarks same-box, posts a deterministic `eval:XS`–`XL` label,
  and (rarely, gated) auto-merges. CCO had none of this — only the manual
  self-score-and-paste-in-PR flow.
- sparkinfer's **security pattern**: GPU evaluation *never* runs inside
  GitHub Actions — it always runs on an external, maintainer-owned cron job
  over SSH to a rented box. Actions workflows only ever read diff metadata or
  check out trusted `main`, never untrusted PR code. This became the template
  for CCO's own workflow design (§8).
- sparkinfer's threat model differs from CCO's: it benchmarks a **compiled
  C++/CUDA binary**; CCO's scoring path directly **imports and executes a
  PR's own Python** (`Transform.basis()`) in-process with GPU access — a
  materially higher-risk surface, driving the sandboxing design in the
  approved plan's Phase 2.
- sparkinfer's economics: `fixed_base_score` × `eval:*` label multiplier,
  0.5 maintainer cut, lenient entry gates. CCO's target registry entry
  (drafted, not yet filed) mirrors this pattern at a smaller scale.

Production-readiness comparison: neither project is a finished commercial
product. sparkinfer is "production-shaped" (real kernels, weekly releases,
one shipped binary) but pre-1.0 with no serving layer. CCO's commercial
narrative (API, pricing) was pure landing-page fiction with zero backing
code — this mismatch got fixed in the whitepaper/site rewrite (§9).

---

## 5. Spectral Token Mixing — the new research direction

You proposed applying the FFT (time↔frequency round-trip) to break attention's
`O(n²)` bottleneck. This was researched, validated, and written up as a
standalone idea (deliberately *not* tied to CCO/SN74 branding, per your
request).

### The core idea
Self-attention's `O(n²)` cost comes from computing all pairwise interactions
explicitly. For the *convolutional* (position-relative) component of mixing,
the convolution theorem lets you swap an `O(n²)` operation for an `O(n log n)`
FFT round-trip: transform to frequency space, do a cheap pointwise multiply,
transform back. What can't be done this way — content-based lookup — is
handled by a **hybrid**: exact local-window attention (cheap, small w) +
spectral global mixing.

### Feasibility research (web-verified, not just theory)
- **Theory boundary**: exact attention is provably quadratic under SETH
  (Keles et al. 2022); sub-quadratic *approximation* is possible precisely
  when entries are bounded (Alman & Song 2023).
- **Every stage of the plan already has prior art**: Hyena (gated long
  convolutions, matches Transformer perplexity at 20% fewer FLOPs),
  FlashFFTConv (Monarch-decomposed FFT on tensor cores, beats
  FlashAttention-2 past 2k context, up to 7.93× over PyTorch FFT), LoLCATs
  (linearizes Llama 3 8B–405B via distillation + LoRA, closing ~78% of the
  quality gap with 0.2% of the training cost of prior methods).
- **Industry has already converged on the hybrid pattern**: MiniMax-01,
  Qwen3-Next, Nemotron-H, Kimi Linear, GPT-OSS all ship interleaved
  cheap/exact attention layers in production. **Gemma 3 already ships a 5:1
  local:global attention ratio** — confirming the hybrid design isn't
  speculative, it's already how frontier models are built.
- **Known risk, documented**: fixed-state/convolutional mixers measurably
  degrade on needle-in-a-haystack retrieval; the fix (a thin exact-attention
  component) is exactly the hybrid design already planned.

### Math: why spectral mixing can beat matmul (by category)
- **Speed**: O(n log n) vs O(n²) — asymptotic, provable.
- **Memory**: no n² object ever materializes; O(1) decode state once
  distilled to recurrent form (vs. an O(n) KV cache).
- **Accuracy on arbitrary inputs**: matmul/exact attention wins — it *is* the
  ground truth; content-based lookup is what spectral mixing can't do alone.
- **Accuracy in finite precision, for the convolutional class**: spectral
  actually wins — the FFT is a unitary (norm-preserving) transform, so
  roundoff grows as O(√log n) vs. O(√n) for direct dot-product evaluation.
- **Quality per FLOP at long context**: spectral wins (Hyena's 20% FLOP
  saving at equal perplexity).
- **Reachable regime**: at 1M+ tokens, an approximate answer beats no answer
  at all — quadratic attention becomes computationally infeasible first.

Full two-level explanation (rigorous + a "teach a high-schooler" version using
the slide-rule/logarithm and music-equalizer analogies) is saved in
[`docs/spectral-vs-matmul-explained.md`](docs/spectral-vs-matmul-explained.md).

### The accounting (verified arithmetic, 70B-class model, d=8192, 80 layers)

| context | attention's share of prefill compute | spectral speedup |
|---:|---:|---:|
| 8k | 12% | 1.1× |
| 128k | 69% | 3.2× |
| 1M | 95% | **18.6×** |

At 128k, the KV cache (~43 GB at fp16, 8× GQA) is what a recurrent-form
distillation would eliminate — the decode-side payoff, distinct from the
prefill-side FLOP savings above.

### The M1–M4 plan (explained carefully, milestone by milestone)
- **M1** (weeks): reference `SpectralMixer` operator in plain PyTorch + a
  fixed-shape eval harness. **Expected to fail its accuracy budget** — that's
  the honest baseline, not a bug.
- **M2** (1–2 months): hybrid (local-exact window + spectral-global path) +
  calibration/distillation against the exact layer as teacher. Gate: MSE <
  0.01 @ 32k, latency/VRAM strictly below exact. This is the real
  go/no-go milestone — placed early, before expensive kernel work.
- **M3** (2–4 months): fused Monarch-FFT tensor-core kernel. Gate: measured
  scaling proof (2× length → ≤2.3× latency, not 4×) — certifies
  "sub-quadratic" as measured, not claimed.
- **M4** (3–6 months): distill to a recurrent/state-space decode form (O(1)
  state, replacing the KV cache). Gate: full-model spot check + needle
  retrieval (the documented weak spot of this whole model family).

Each milestone retires exactly one risk in order: accuracy → speed →
deployability. Sequenced so the cheapest, most decisive test (M2) comes
before any expensive kernel engineering.

### Deliverables
- [`docs/spectral-mixing.md`](docs/spectral-mixing.md) — the full standalone
  whitepaper (math, implementation stages, accounting, milestones,
  references). Includes 6 figures: Mermaid flowcharts (GitHub-native) +
  ASCII diagrams (portable to any renderer/PDF).
- [`docs/spectral-mixing.pdf`](docs/spectral-mixing.pdf) +
  [`docs/spectral-mixing-print.html`](docs/spectral-mixing-print.html) — an
  investor-grade print-designed PDF (hand-built SVG figures, A4, 7 pages),
  rendered via headless Chrome from the print HTML. Regenerate with:
  ```bash
  google-chrome --headless --no-pdf-header-footer \
    --print-to-pdf=docs/spectral-mixing.pdf docs/spectral-mixing-print.html
  ```
- Title evolution (per your feedback): "Under the Quadratic Wall" →
  "Spectral Token Mixing" → final: **"Spectral Token Mixing:
  Frequency-Domain Speed Optimization for Long-Context LLMs"** (names the
  idea, its role as speed optimization, and the LLM target, backed by the
  95%-compute-share and 18.6×-speedup numbers in the subtitle).
- [`docs/spectral-vs-matmul-explained.md`](docs/spectral-vs-matmul-explained.md)
  — the category-by-category math comparison + the plain-language
  explanation, exported per your request.

You also asked me to draft a casual message about applying this to Gemma 4
and pointed out this is intended for **Gemma 4** (confirmed real: released
April 2, 2026, E2B/E4B/26B-MoE/31B variants, hybrid sliding-window + global
attention, Apache 2.0, built from Gemini 3 research) with **sparkinfer's
approach** as an engineering reference.

---

## 6. GPU/model pins for the future validating system

After discussion, you chose (over my initial A100 recommendation):

- **GPU: RTX 5090** (consumer-tier, matches the "runs on every GPU" framing
  and sparkinfer's own target hardware). Noted risk: RTX 5090 cloud spot
  availability was inconsistent as of research time (~$0.50–2.00/hr,
  limited providers) — recommended a reserved instance or self-hosted
  physical box for the *validator* specifically, not ad-hoc spot rental,
  with A100 as fallback.
- **Reference model: Gemma 4 E4B** (~4.5B effective params, Apache 2.0).
  Chosen because its real architecture (5 local sliding-window layers + 1
  global layer, repeated 7×, 512-token window) is structurally the same
  shape as the hybrid spectral operator's design — not an arbitrary pick.

Saved in [`docs/testing-strategy.md`](docs/testing-strategy.md): the two-tier
testing framework —

- **Tier 1 (local, informational)**: CPU or GPU per the miner's own
  hardware, always includes a fast smoke test.
- **Tier 2 (online/validator, authoritative)**: pinned RTX 5090, pinned
  Gemma-4-E4B-derived shapes, exact-vs-candidate on the *same freshly
  randomized* input every run, multiple regimes (not just one fixed
  dataset) — this exactly matches how `eval/evaluator.py` already worked
  structurally; it just needed the seed fixed (done in Phase 1, see §8).

---

## 7. SN74 emission strategy (drafted, not yet filed)

[`docs/sn74-emission-strategy.md`](docs/sn74-emission-strategy.md) contains:
a full per-repo scoring-mechanics table for the six repos above 5% share, the
patterns worth copying (deterministic bots as pricer, `fixed_base_score` +
tiered labels, lenient entry gates) and patterns to avoid (kata's
winner-take-all — suits tournaments not a slow research frontier;
metagraphed's 3-day decay — punishes careful work), a target
`master_repositories.json` entry for CCO, and the execution order: eval bot
→ anti-gaming → winnable tracks → public track record → file a
`weight_adjustment` PR to `entrius/gittensor`.

---

## 8. The "update all" plan-mode session

You asked for a detailed plan to update everything discussed (miner
structure, validating system, scoring, dashboard, bot). I entered plan mode:

- Two parallel **Explore** agents: (1) verified current repo state — `docs/`
  still uncommitted, `dashboard/` confirmed gone, no CI, seed=0 and
  single-transform issues confirmed as expected, `index/` confirmed
  contradictory; (2) extracted sparkinfer's exact bot mechanics — gate
  chain order, `vast_eval.py`'s GPU lifecycle, `label.py`'s exact thresholds,
  copycat detection algorithm, and — the key finding — **all four of
  sparkinfer's GitHub Actions workflows are read-only or trusted-`main`-only;
  the actual GPU eval always runs outside Actions**, on a maintainer cron job
  over SSH.
- One **Plan** agent: designed a concrete phased plan (file layout, 4
  phases with explicit external-dependency gates, the `index/` reconciliation
  decision, the GitHub Actions recommendation, the Next.js dashboard
  architecture, and — critically — flagged that **CCO's threat model is
  harder than sparkinfer's** since it executes raw Python, not a compiled
  binary, driving the Phase 2 sandboxing design).
- I wrote the final plan and got your approval via `ExitPlanMode`. Plan file:
  `~/.claude/plans/delegated-toasting-nebula.md` (local to this machine, not
  in the repo).

### The 4 phases (as approved)
1. **Phase 1** — harness hardening + governance scaffolding. Buildable now,
   zero external accounts. **✅ Done this session — see §9.**
2. **Phase 2** — sandboxed runner. Needs any CUDA/MPS box to validate
   end-to-end. Not started.
3. **Phase 3** — live bot operation. Needs a persistent GPU (your RTX 5090 or
   a funded cloud account) + a dedicated bot GitHub PAT. Not started —
   explicitly your call when ready.
4. **Phase 4** — Next.js dashboard. Data contract already fixed by Phase 1's
   `eval/ledger.py`; can be built in parallel with Phase 3 once a hosting
   decision is made (recommended: GitHub Pages static export). Not started.

---

## 9. Phase 1 — what was actually built (this session, verified)

**Committed** (its own clean commit, before any code changes): the entire
`docs/` directory (whitepaper.md v0.2, spectral-mixing.md + pdf + print-html,
spectral-vs-matmul-explained.md, sn74-emission-strategy.md,
testing-strategy.md).

**Everything below is in the working tree, uncommitted, awaiting your
review/commit decision:**

### Harness fix (the known exploit)
- `eval/evaluator.py`: `EvalConfig.seed` changed from a fixed `0` to
  `int | None = None` — draws a fresh unpredictable seed per run via
  `secrets.randbelow()` unless an explicit seed is passed (for reproducing a
  specific prior run only). The resolved seed is now recorded in the output
  config for auditability.
- Added `decaying-spectrum` as a new fill regime (`strategy/storage.py`'s
  `_fill_decaying_spectrum`): a rank-truncated matrix with **smoothly
  decaying** component weights (`k^-alpha`), distinct from `lowrank`'s
  uniform weighting — tests whether a transform prioritizes the strongest
  structure rather than needing full rank. Verified empirically: top
  singular values genuinely decay (39.7 → 25.1 → 14.4 → ...) with a sharp
  drop past the truncation rank.
- Wired into `eval/cli.py` and `strategy/cli.py`'s `--fill` choices too, for
  consistency across both entry points.

### New scoring/bot library (`eval/`)
- **`eval/label.py`** — deterministic verdict tiering
  (`REJECT`/`none`/`BASELINE`/`XS`–`XL`), reusing `eval/metrics.py`'s
  existing dominance gate rather than reimplementing it. Tiers sized against
  a **fixed reference anchor**, not the moving frontier, so tier meaning
  doesn't drift (mirrors sparkinfer's anti-drift design). 7/7 tests passing.
- **`eval/ledger.py`** — append-only `ledger.jsonl` writer/reader plus a
  **pure-projection** `dashboard/data.json` builder (always fully rebuilt
  from the ledger, never hand-patched, so the two can't drift apart). 6/6
  tests passing.
- **`eval/copycat_guard.py`** — pure diff-comparison algorithm (fingerprint →
  containment ratio → structural Levenshtein+bigram fallback), adapted from
  sparkinfer. Three tiers: ≥80% containment = block, 70–79% = warn, ≥40%
  structural match = warn. Also has a `main()` real-time single-PR
  entrypoint for the GitHub Actions hook. 10/10 tests passing, including a
  carefully experimentally-tuned structural-fallback fixture (empirically
  found the exact containment/Levenshtein/bigram numbers rather than
  hand-deriving them).
- **`eval/github_client.py`** — extracted shared `GitHubClient`/`PRInfo` (was
  originally inline in `pr_bot.py`; pulled out to avoid a circular import
  once `copycat_guard.py` needed the same client for its real-time hook).
- **`eval/pr_bot.py`** — the full gate chain, oldest-PR-first: draft skip →
  blocked-contributor check → copycat check → idempotency marker
  (`<!-- cco-eval:<sha> -->`) → scorecard greenlight (aligned to reuse the
  **existing** `labeler.yml` JS detector's exact regex — found and fixed a
  near-duplicate-but-inconsistent detector I'd almost introduced) → GPU eval
  (stubbed in Phase 1, wired to a real runner in Phase 2). `process_pr()` is
  a pure function of already-fetched data — fully unit-testable without any
  GitHub I/O. 13/13 tests passing, including a fake-client-based
  `run_once()` test proving dry-run mode **never** calls any write method,
  and a test confirming closed/merged PRs *between* two open PRs are still
  included as valid copycat-comparison targets (a real bug I found and fixed
  in my own first draft).

### Governance & CI
- `.github/CODEOWNERS` (protects `/eval/`, `/docs/`, `/.github/`,
  `/dashboard/data.json`), `blocked-contributors.txt`, `copycats.json`
  (seeded `[]`), `COPYCATS.md`, `FLAGGED.md` (both seeded empty, ready for
  the bot to append to).
- Three new GitHub Actions workflows. Two are `main`/metadata-only guards,
  while the CPU CI intentionally runs PR code on a read-only, no-secrets
  runner:
  - `sensitive-paths-guard.yml` — `pull_request_target`, reads changed-file
    names only, hard-fails non-maintainer edits to protected paths.
  - `copycat-guard.yml` — checks out `main` (never the PR's own branch) even
    though triggered by a PR event; runs `eval/copycat_guard.py`'s real-time
    single-PR check.
  - `eval-policy.yml` — plain CPU CI (`py_compile` + `pytest`) on the PR's own
    code, deliberately with no secrets and no torch so GPU-only tests skip
    cleanly instead of needing special-casing.
- `.github/labels.yml` — added the full `eval:*` tier family plus
  `copycat`/`copycat-warn`/`flagged:gaming`/`mult:harness`/`mult:docs`,
  matching the target registry entry in `docs/sn74-emission-strategy.md`
  exactly.

### Miner-facing tooling (`strategy/`)
- `strategy/cpu_backend.py` — a NumPy-only backend shim (smoke-test only,
  never used for real scoring) satisfying the same `xp`/`to_device`/`matmul`
  surface the real GPU backend does.
- `strategy/smoke.py` — `python -m strategy.smoke`: tiny-N, CPU-friendly
  check of every registered transform's basis (shape, orthonormality, no
  NaN/Inf). Verified working: `rsvd` orthonormal to `7.9e-8`.
- `strategy/examples/transform_template.py` — copy-paste `Transform`
  skeleton with a runnable `__main__` self-check.

### Packaging & docs
- `pyproject.toml` — `pip install -e .` verified working in a clean venv
  (three top-level packages: `matmul`, `strategy`, `eval`).
- `CONTRIBUTING.md` — added an open/protected path table, the smoke test as
  step 0, a copycat-detection note, and a note that scoring is moving to
  automated labels (honestly described as not-yet-live).
- `README.md` — light pass: added the whitepaper link, the smoke-test step,
  fixed the phantom `CHANGELOG.md` reference, added `docs/` to the layout
  tree.
- **`index/index.html`** — full content rewrite (kept the same CSS/visual
  shell). Removed: "Get API access", the entire Pricing section, the
  "Commercial API" product card, winner-take-all language, the placeholder
  `hello@example.com` mailto, the stale PDF link. Replaced with: the real
  Gittensor payment mechanism (submit → score → merge → SN74 pays), the real
  subspace math (compress/compute/reconstruct with the actual code from
  `strategy/examples/run_example.py`), the real 3-tier dominance gate, and a
  roadmap that now shows **Phase 0 (matmul, live today)** before the
  attention Phases 1–4 (clearly marked as roadmap, not current). Removed
  `index/attention_optimization_whitepaper.pdf` (`git rm`), confirmed no
  other file references it.

### A genuine pre-existing bug found and fixed
`tests/test_host_memory.py::test_windows_global_memory_status_ex` was
silently broken on Linux — it tried to `unittest.mock.patch()` the
`ctypes.windll` attribute, which doesn't exist at all outside Windows,
raising `AttributeError` before the test body ever ran. Invisible until now
because **this repo had no CI before this session** — the new
`eval-policy.yml` would have been permanently red from day one otherwise.
Fixed by patching the whole `ctypes.windll` attribute with `create=True` via
a `MagicMock`, rather than a nested path that doesn't exist.

### Final verification results
- `pytest tests/ strategy/tests/ eval/tests/ -v` → **55 passed, 23 skipped
  (GPU-only, expected), 0 failed**
- `python -m strategy.smoke` → PASS
- `pip install -e .` → verified in a clean venv
- Real (read-only) integration check against the live repo — `list_prs`,
  `get_diff`, `get_comments`, `has_scorecard` — all confirmed working
  against real data (19 real open PRs at check time), **no write calls
  attempted**
- `index/index.html` — no stale patterns, balanced tags, dead CSS cleaned up

---

## 10. Current state / what's next

- **Dashboard**: does not exist. Two static HTML versions were built earlier
  in the session (Artifact previews only) and deleted before commit. Rebuild
  is Phase 4 — Next.js, not static HTML this time, data contract already
  defined by `eval/ledger.py`.
- **Uncommitted work**: everything in §9 is sitting in the working tree,
  not yet committed (only the `docs/` commit happened). Awaiting your
  go-ahead.
- **Phases 2–4**: not started. Phase 2 (sandboxed runner) needs any GPU box
  to validate against; Phase 3 (live bot) needs a persistent GPU + a
  dedicated bot GitHub PAT — explicitly your call, not something to build
  speculatively; Phase 4 (dashboard) is unblocked and could start anytime.
- **SN74 registry**: the target `master_repositories.json` entry is drafted
  in `docs/sn74-emission-strategy.md` but not filed — per the plan, that PR
  should wait until there's a real public track record (a running bot, a
  moving ledger) to point to.

Full plan reference (if resuming Phase 2+ later): the approved plan is saved
locally at `~/.claude/plans/delegated-toasting-nebula.md` (not in the repo).
