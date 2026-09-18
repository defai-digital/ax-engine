# Qwen 3.8 27B AXQ Certification

Status: **Candidate**

Primary optimization target: **AXQ 6-bit MTP** (`qwen3.8-27b:axq`)

Compact sibling: **AXQ 4-bit MTP** (`qwen3.8-27b:axq-4bit`)

Last reviewed: **2026-09-18**

Primary optimization target. Checkpoint Tier 1. MTP Tier 2 pending. AX certification record: Candidate (gates open).

This is the promotion record for the production-size Qwen 3.8 27B pack. It is
the general-purpose default serve target. It is **not** MTP Tier 2 certified
and it is **not** a 72-hour endurance pass. Super-class Qwen 3.8 (2.4T) is a
different, experimental path and is out of this record.

## Pinned Checkpoints

| Selector | Repository | Revision |
| --- | --- | --- |
| `qwen3.8-27b:axq`, `qwen3.8-27b:axq-6bit`, `ax-qwen3.8-27b` | `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` | `3e290738e96972307c6aeb9934ab170ca0eae1c1` |
| `qwen3.8-27b:axq-4bit` | `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP` | `7e865596cb32bd41b29c7a25c5b66b9c3ea25e5e` |
| `qwen3.8-27b:axq-8bit` | `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP` | `4037b7242a4de8deaf71247a685538591cad160a` |
| `qwen3.8-27b:axq-mxfp4` | `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP` | `b2c5354f779e430d0c1733143db848a72b71c16e` |

CLI aliases pin the exact checkpoint revisions above. Moving a Hub branch
cannot silently change what the selector loads.

## Axes (do not collapse)

| Axis | This pack |
| --- | --- |
| Product focus | Primary optimization target (unique general-purpose default) |
| Hub checkpoint | Tier 1 for 6-bit / 8-bit / MXFP4; 4-bit remains a compact candidate |
| AX certification record | Candidate — gates open |
| MTP | Sidecar present; Tier 2 performance certification pending |
| Support tier | Family follows the Qwen 3.x Certified graph path; this *checkpoint* is not `release_ready` |

## Current Evidence

Landed, labeled:

- 2026-09-18 actual first-split observation binds compsec-079 output index 116
  (input position 524) to the live MTP window. All 192 output IDs are preserved;
  a faithful replay matches all 993,280 logits, and three cache witnesses of
  128 logical arrays remain unchanged. From that same MTP state, live and
  ordinary-batch paths rank token 6397 at 21.875, while ordinary singleton
  scores 279, 2849 and 6397 equally at 21.75. Selecting 279 uses the recorded
  first-index tie rule; that control did not independently materialize MLX
  argmax. This establishes a decision-relevant execution-path difference,
  without identifying one kernel defect or proving actual direct-cache identity.
  The standalone oracle now uses the direct prefill entry: the old/new/old
  control changes a prefix rejection at 109 to validation through 116 and back.
  Its subsequent prediction still differs from live direct, so it remains a
  bounded diagnostic.
  A separate forced-replay control-flow defect is repaired: explicitly forced,
  unprocessed greedy replay revalidates drafts before consumption, updates the
  committed count consistently, and discards stale experimental skip-state.
  The injected false-accept and stale-state regressions fail before correction
  and pass afterward with real input-dependent KV. Reference contract comparison
  precedes this AX-owned change. Default relaxed acceptance and sampled behavior
  remain unchanged; optimistic acceptance and whole-request direct equivalence
  remain outside the guarantee. Final-wheel qualification is recorded separately
  when completed. Quality, numerical consistency and Tier 2 gates remain open;
  **not ship-ready**.
  [First-split observation and forced-replay regression evidence](../../benchmarks/results/qualification/2026-09-18-qwen38-27b-first-split-replay/).

- 2026-09-18 reference comparison and actual-input FFN attribution locate the
  first captured layer-zero difference at gate/up QMM, with BF16 activation
  and metadata and no dense bias. The observation-only build preserves all
  192 tokens, full live logits and 288 compiled/eager leaves. Nested SwiGLU
  compilation is exact in this control. Disabling custom QMM does not restore
  direct/reference identity; this earlier FFN probe does not itself locate the
  later first-token split examined in the observation above.
  Separate reproduced integration defects are repaired in `d121f107`: dense
  Linear bias is applied once by each caller, and mixed affine metadata retains
  stock dtype promotion. These repairs do not explain the captured bias-free,
  same-dtype FFN difference. Defaults and numerical tolerances are unchanged.
  The clean bundled `d121f107` wheel passes executable qualification on the
  selected mini: doctor and installed-package identity, 32/32 hard QA and
  7/7 surfaces per route, active MTP counters, and the paired 64-token probe.
  This is version-bound qualification, not broad quality or Tier 2 acceptance.
  [Reference comparison, regressions and version-bound validation](../../benchmarks/results/qualification/2026-09-18-qwen38-27b-ffn-qmm-contract/).

- **LINE_SET interpretation correction:** the imported source asks for a primary
  bug location and its scorer accepts a nonempty subset of audited locations.
  The historical source-file hashes confirm that contract was present when the
  questions were imported. AX's exact-set score measures an additional,
  stricter enumeration requirement. Its raw scores and gold sets remain
  unchanged, but **0/24 full recall is not evidence that the original tasks
  required every gold location**. On the retained `aa38f18b` twelve-case replay,
  each route has seven accepted-location subset detections and five truncations,
  while its exact-set score remains 0/12. Do not report that as twelve semantic
  failures or population accuracy. An aligned quality protocol, remaining
  truncation/recovery failures and broader qualification are still open.
  A separate two-case complete-set prompt diagnostic returns `0` for 079 and
  `3` for 086 on both AX routes and the pinned independent direct reference,
  with full rendered prompt/input-token parity. The reused location keys are
  not validated exhaustive causal annotations. This selected result does not
  establish correctness, a quality repair or broader numerical equivalence.
  [Source contract and diagnostic evidence](../../benchmarks/results/qualification/2026-09-18-qwen38-27b-ffn-qmm-contract/quality-contract.md).

- 2026-09-18 GDN prework rounding repair (`aa38f18b`) restores intermediate
  activation-dtype boundaries in fused convolution, SiLU, normalization, and
  scaling. Exact BF16/FP16 prework and BF16 fused-verifier regressions pass.
  Earlier instrumented observations, before the final FP16 refinement,
  eliminate the captured post-input and same-input recurrent-state differences;
  full-layer hidden differences remain. Clean final native-server controls on
  the selected mini improve 192-token direct/MTP agreement from **0/4 to 2/4**;
  compsec-079 and 087 still split at indices 116 and 155. Paired agreement is
  not reference fidelity: final compsec-092 matches between AX routes but
  differs from the independent reference, which old MTP matched.
  The clean final bundled wheel passes executable qualification on the
  selected SKU: doctor ready, installed-package identity, 32/32 hard QA and
  7/7 surfaces per route, active MTP without silent fallback, and the paired
  64-token greedy probe. This closes that version-bound scope only.
  The final bundled wheel's unchanged twelve-case 512-token diagnostic remains
  **0/12 exact-set passes per route**, with seven strict-wrong and five truncated responses
  per route and **6/12** paired content matches. Truncation counts increase
  from four direct / two MTP on the previous wheel. This failure subset is not
  overall model accuracy, and the repair does not establish a quality gain.
  Original long-thinking, saved recovery, broad route consistency, and MTP
  Tier 2 remain open; **not ship-ready**.
  [Rounding attribution, final controls and validation](../../benchmarks/results/qualification/2026-09-18-qwen38-27b-gdn-rounding/).

- 2026-09-18 residual-state investigation reproduces all four remaining route
  differences in fresh processes on the unchanged `4cbda8b5` wheel. LA fusion
  off, full checkpoint capture, and FA storage rebind off do not remove the
  earliest split. The validated singleton oracle ties two target scores at
  23.125; the live MTP verifier instead scores them 23.0 and 23.125 and follows
  its actual argmax. An observation-only build preserves all 192 production
  tokens. This establishes differing target scores, not an acceptance-rule
  defect or a complete attribution of numerical drift. From the same live
  MTP cache and identical four-row input, ordinary batched forwarding and
  singleton replay both recover the tie, localizing this split to the
  MTP-specific target path. Both controls preserve all 192 production tokens.
  `c9a7ff97` adds a passing exact BF16 affine FFN regression; its small fixture
  does not prove whole-layer or production-shape equivalence. The unchanged pinned
  reference also differs from AX direct on these long prompts, and fails two
  original strict line-set cases. An altered wording diagnostic retains two
  failing grades, but its contiguous-span instruction conflicts with one
  non-contiguous gold set and does not resolve the prompt/scorer mismatch.
  `276bd1d4` improves the replay oracle without changing
  production arithmetic. Quality and numerical-route gates remain open;
  **not ship-ready**.
  [Controls, state evidence and reference](../../benchmarks/results/qualification/2026-09-18-qwen38-27b-residual-state/).

- 2026-09-18 scheduler-prefix repair (`4cbda8b5`) closes a separate context
  omission: cold-grid trimming could discard prefix tokens already removed
  from the scheduler input without requesting their recomputation. The
  identical selected-SKU cold/warm request changes from **4 versus 128 tokens**
  to **4 versus 4 identical tokens**, while retaining the scheduler cache hit
  and explicitly replaying the missing 96-token prefix. Cache/MTP defaults
  remain unchanged. The clean bundled wheel again passes executable
  qualification: 32/32 hard QA and all surfaces per route, doctor, package
  identity and the paired 64-token probe, with active MTP verification.
  Earlier warm-request quality failures cannot be attributed solely to model
  capability because this context defect was present. The unchanged twelve-case
  diagnostic now has **8/12** paired response matches, with direct four and MTP
  two truncations, but strict complete-span grading remains **0/12 per route**.
  This is a selected failure subset, not overall model accuracy. Four response
  differences remain; original long-thinking/recovery and endurance were not
  rerun on this build. Quality, broad route consistency and MTP Tier 2 remain
  open; **not ship-ready**.
  [Prefix replay evidence](../../benchmarks/results/qualification/2026-09-18-qwen38-27b-prefix-replay/).

- 2026-09-18 linear-attention output correction (`21687d71`, completed by
  `892c2fc9`) restores float32 gated normalization and shares the ordinary
  target's layer-specific Metal gate policy. The dtype regression failed
  before repair (BF16 S=2 maximum absolute error 0.015625); a second regression
  showed why a portable-only repair was insufficient. Final BF16/FP16 exact
  and relaxed gate-policy tests pass. On the selected mini, the fixed-token
  compsec-092 split at output index 5 is closed; both predeclared raw controls
  match direct/MTP under the 128-token cap. Clean final bundled-wheel
  qualification passes 32/32 hard QA and 7/7 surfaces per route, doctor,
  installed-package checks, active MTP telemetry and the paired 64-token probe.
  However, the unchanged 512-token LINE_SET diagnostic still passes only
  **1/12 per route** (direct nine truncations; MTP eight), and just **3/12**
  response texts match. All twelve chat templates/token arrays match the
  pinned tokenizer, ruling out a template mismatch for those inputs.
  Broader sequence consistency and quality remain open; **not ship-ready**.
  [Precision controls and final validation](../../benchmarks/results/qualification/2026-09-18-qwen38-27b-la-precision/).

- 2026-09-17 peer throughput campaign on the selected **Mac mini M4 Pro
  64 GB** SKU with the clean `ad999f3f` bundled wheel: `flappy` contract,
  20-run medians. AX Engine **31.05 tok/s** decode / **120.3 tok/s** prefill;
  MTPLX 2.11.3 28.16 / 114.0; OMLX 0.6.4 (imported sidecar, Lightning depth 1)
  15.02 decode; mlx-lm 0.31.3 direct AR 12.78 decode. Harness stability and
  MTP-correctness publication gates passed; two host daemons held about 1.2
  CPU cores throughout and the AX lane was repeated with agreement within
  0.4%. Throughput only; status remains **Candidate; not ship-ready**.
  [Evidence](../../benchmarks/results/mtp-axq-peer/2026-09-17-mac-mini-m4-pro-64gb/).

- 2026-09-17 low-precision SwiGLU correction (`891385f8`, dimension guard
  `ad999f3f`) preserves MLX BF16/FP16 tensor activation semantics in singleton
  gate/up and packed paths. On this 6-bit checkpoint the default runtime
  reaches only the packed dense path; the singleton matvec kernel admits
  4-bit weights unless an opt-in flag is set. At that revision, two opt-in
  kernels (prefill dual-QMM MMA, fused MoE expert block) still used float-only
  activation; `483bc92a` subsequently corrected them with both flags still off. A same-session A/B on the campaign laptop (M5 Max, peer-table
  contract) shows decode within 0.3% of the 2026-09-15 binary and prefill
  0.5-1.4% lower; the README peer table is not refreshed by that check. An isolated exact-projection regression failed
  before repair (BF16 maximum absolute error 0.001953125); all 22 SwiGLU tests
  now pass. On the selected mini, clean installed `891385f8` gives **9/9**
  paired direct/MTP token matches on the original diagnostics and **12/12**
  on a predeclared holdout. Both AX routes match the same-pack reference on
  8/9 original cases; retrieval wording differs. These finite token checks
  do not establish task accuracy or general equivalence. The separate
  512-token LINE_SET replay still has direct/MTP content divergences, so
  broader route consistency remains open. No fastpath default changed and
  no post-fix speed claim is made.
  The unchanged twelve-case LINE_SET diagnostic passes only **1/12 direct**
  and **2/12 MTP**, with nine and six truncations respectively; only **4/12**
  paired response texts match. Both 32851-token recovery replays complete but
  still answer B against gold C. This is a 512-token answer-only failure
  diagnostic, not a replacement for the original 32k thinking campaign.
  A direct-only replay at a 2048-token cap on the final wheel removes eight
  of nine truncations (3/12 correct, 8/12 wrong, 1/12 truncated); six of
  the wrong answers carry the grader's detection flag under strict grading. The
  truncations are answer-format non-compliance, not empty output.
  Final clean `ad999f3f` bundled-wheel qualification passes: each route has
  **32/32 hard QA**, zero soft failures and **7/7 surface probes**; doctor is
  ready, packaged libraries load in the isolated environment, and the paired
  64-token probe now matches with active MTP counters. This closes that scoped
  gate, not the broader failures above. Status remains **Candidate; not ship-ready**.
  [Activation correction and scoped evidence](../../benchmarks/results/qualification/2026-09-17-qwen38-27b-swiglu-consistency/).

- 2026-09-17 target-head correction (`1b15fdf0`) preserves the checkpoint's
  dense BF16 output head instead of automatically substituting a singleton-only
  2-bit cache. On nine fixed inputs, direct/reference token agreement improved
  from 3/9 to 7/9; MTP stayed 8/9 and direct/MTP agreed on 8/9. These are token
  diagnostics, not model accuracy. That revision still split at output index
  14 (closed by the later activation correction above). The live qualifier
  now requires paired greedy equality and
  records schema-2 request/response hashes; the previous health-only pass does
  not satisfy that gate. Fastpath overrides did not establish general route
  equivalence. The original full 32k thinking LINE_SET campaign has not
  been rerun; later bounded diagnostics are recorded above. Clean `875aa5da` with the final bundled wheel passed
  direct/MTP 32/32 hard QA and 7/7 surface probes per route, but the schema-2
  qualifier exited 1 at paired token index 14. That revision's gate is **failed**.
  Status remains **Candidate; not ship-ready**.
  [Earlier precision comparison and qualification evidence](../../benchmarks/results/qualification/2026-09-17-qwen38-27b-target-precision/).

- 2026-09-17 product-surface qualification on the selected **Mac mini M4 Pro
  64 GB** SKU, macOS 26.6.2, clean source `e2b3e354`, installed bundled wheel:
  direct **32/32** and MTP **32/32** hard QA, zero soft failures, each **7/7**
  product-surface probes with no skips. Each route runs the same 16 stratified
  questions in streaming and non-streaming modes. Doctor reported ready;
  wheel dependencies loaded from the installed package. Actual server counters
  prove direct without MTP and an active MTP route; terminal API MTP
  reports match those counters.
  [Scoped evidence and reproduction](../../benchmarks/results/qualification/2026-09-17-qwen38-27b-m4-pro-64gb/).
  These are product health gates, not representative benchmark accuracy or a
  Tier 2 promotion. The paired raw-token greedy probe diverged at output index
  25; exact direct/MTP token equivalence remains unqualified.

- Default serve alias and revision pin in the CLI.
- 2026-08-30 direct + MTP refresh on Apple M5 Max, 128 GB, from the v7.2.0
  binary at commit `3cea9def`. The host recorded tracked runtime changes;
  treat those numbers as refresh evidence pending a clean-build rerun.
  Artifacts:
  [`mlx-lm` reference](../../benchmarks/results/inference/mlx-lm-reference/2026-08-30-qwen38-27b-axq-6bit-m5-readme-refresh/),
  [AX direct](../../benchmarks/results/inference/ax-direct/2026-08-30-v7.2.0-qwen38-27b-axq-6bit-m5-readme-refresh/),
  [AX MTP](../../benchmarks/results/speculative/mtp-6bit/2026-08-30-v7.2.0-qwen38-27b-axq-6bit-m5-readme-refresh/).
- 2026-08-31 AXQ MTP peer campaign row for this pack (Apple M5 Max, 128 GB):
  [campaign](../../benchmarks/results/mtp-axq-peer/2026-08-31-df-macbookpro-m5/).
- 2026-09-15 same-pack latest-runtime campaign (Apple M5 Max, 128 GB; product-path
  MTP depth 3, 20-run median): AX Engine **76.90 tok/s**, MTPLX 2.11.2
  **70.62 tok/s**, mlx-lm 0.31.3 direct AR baseline **27.90 tok/s** (same
  `flappy` greedy 256-token prompts; no MTP head). Other latest runtimes failed
  to load this snapshot.
  [campaign](../../benchmarks/results/mtp-axq-peer/2026-09-15-apple-m5-max-128gb/).
- 2026-09-16/17 failed-pair retest (Apple M5 Max, 128 GB; both packs, 34
  questions selected from the prior failed-pair union, greedy, 32k output
  budget, answer-recovery pass enabled; one stalled case excluded). Result:
  AXQ 6-bit **10/34 (29.4%)**, MXFP4 **7/34 (20.6%)** correct. Raw per-question
  records (authoritative) live on the campaign host under
  `artifacts/qwen38-retest-failed-pair-20260916/` and are not committed here.
  The saved provenance has a binary hash but no source commit; these records
  do not qualify the current source tree or the target mini SKU. Recompute
  final grades and evidence hashes with
  `python scripts/audit_qa_retest.py /path/to/retest` (requires the saved
  artifact directory; primary grades are preserved records, not redecoded).
  The selected
  failure subset is not an estimate of overall model accuracy.
  Two failure modes dominate:
  - **Output-budget exhaustion.** 11/34 (AXQ) and 12/34 (MXFP4) rows reached the
    full 32000-token cap (`finish_reason=max_output_tokens`). The continuation
    recovery pass recovered only 4/11 and 5/12 of those rows, so most
    budget-exhausted questions still fail after extension.
  - **LINE_SET scoring-contract mismatch.** All 24 `LINE_SET` rows across both packs were
    graded (none truncated), reported exactly **one** line each, and every
    reported line fell inside the gold span (detection 24/24, precision 24/24).
    Gold spans were 2-6 lines (median 3); full recall was **0/24** (median
    recall 0.33). The saved replies contain single-line answers; the grader
    accepts comma/range values. This rules out the proposed first-line
    truncation explanation for these replies, not every harness defect.
    The source prompt and scorer permit accepted-location subsets; complete
    enumeration is an additional AX metric, not the original task contract.
    These counts therefore do not establish a model completeness defect.
    A later controlled diagnostic on the selected mini used 12 of these
    questions, a fixed answer-only system message, thinking disabled and a
    512-token cap. Original wording passed 0/12 direct and 1/12 MTP; a generic
    complete-span instruction passed 2/12 direct and 5/12 MTP, with failures
    and truncations remaining. This demonstrates prompt sensitivity but does
    not replace the original 32k thinking scores or establish broad accuracy.
    [Diagnostic settings and counts](../../benchmarks/results/qualification/2026-09-17-qwen38-27b-m4-pro-64gb/quality-diagnostics.json).
  One AXQ row was flagged as a degenerate repetition loop
  (`repetition_max_run=1644`, `loop_suspect=true`); repetition was otherwise
  absent (MXFP4 `repetition_max_run` max 2).

- Target-mini diagnostics on clean `8c8217b2`: both AX routes completed the
  saved 32851-token recovery, but returned `Answer: B` against gold C. An
  isolated same-pack mlx-lm 0.31.3 / MLX 0.32.2 replay returned the identical
  token sequence. This reproduces completion and the wrong answer on the
  saved continuation; it does not qualify fresh-question accuracy. Each AX
  greedy route was repeatable across two requests. The independent 64-token
  mlx-lm probe matched AX MTP exactly and differed from AX direct at index 25,
  so the route split does not establish that MTP caused an error. The later
  activation correction closes the reproduced split on the scoped diagnostics;
  universal equivalence is not claimed.

Not claimed:

- MTP Tier 2 / `release_ready`
- 8-hour or 72-hour endurance (the published 8.87h soak is
  [Qwen 3.6 27B AXQ](qwen3.6-27b-axq-6bit-8h-endurance-2026-08-08.md))
- Multi-model `load_mode=add`
- P0 multimodal quality for this pack
- Long-context decode-at-depth
- Clean-worktree replacement of the 2026-08-30 refresh
- Any release-quality accuracy bar on the 2026-09-16/17 failed-pair retest;
  29.4% / 20.6% are strict selected-subset scores with the known LINE_SET
  source-contract mismatch, not an aligned release-quality accuracy measure
- Qualification under an aligned `LINE_SET` protocol: original full recall
  remains 0/24 under the additional exact-set metric; stricter enumeration
  diagnostics are separate tasks and do not replace the original contract
- General immunity to generation stalls. The diagnosed oversized recovery
  prefill now reaches the existing KV starvation failure bound and delivers
  its terminal response. Target-mini recovery also completes with adequate
  KV capacity; finite replays do not establish general stall immunity.
  The provisional 3600s collection backstop still excludes queue/startup
  waits and cannot interrupt a synchronous engine step

## Qualification

Operator procedure: [Testing](../TESTING.md) and
`python3 scripts/qualify_qwen38_27b.py --dry-run`. Live 27B runs belong on a
Mac mini M4 Pro 64 GB host with a clean checkout of the engine under test.

## Related

- [Qwen 3.6 27B AXQ certification](qwen3.6-27b-axq.md) — secondary, evidence-rich candidate
- [Supported Models](../SUPPORTED-MODELS.md)
- [Model Support Policy](../MODEL-SUPPORT-POLICY.md)
