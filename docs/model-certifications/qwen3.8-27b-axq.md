# Qwen 3.8 27B AXQ Certification

Status: **Candidate**

Primary optimization target: **AXQ 6-bit MTP** (`qwen3.8-27b:axq`)

Compact sibling: **AXQ 4-bit MTP** (`qwen3.8-27b:axq-4bit`)

Last reviewed: **2026-09-17**

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

- 2026-09-17 product-surface qualification on the selected **Mac mini M4 Pro
  64 GB** SKU, macOS 26.6.2, clean source `8c8217b2`, installed bundled wheel:
  direct **32/32** and MTP **32/32** hard QA, zero soft failures, each **7/7**
  product-surface probes with no skips. Each route runs the same 16 stratified
  questions in streaming and non-streaming modes. Doctor reported ready;
  wheel dependencies loaded from the installed package. Actual server counters
  prove direct without MTP and an active MTP route.
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
  - **LINE_SET under-reporting.** All 24 `LINE_SET` rows across both packs were
    graded (none truncated), reported exactly **one** line each, and every
    reported line fell inside the gold span (detection 24/24, precision 24/24).
    Gold spans were 2-6 lines (median 3); full recall was **0/24** (median
    recall 0.33). The saved replies contain single-line answers; the grader
    accepts comma/range values. This rules out the proposed first-line
    truncation explanation for these replies, not every harness defect.
    Whether the source wording ("smallest comma-separated set") suppresses
    enumeration remains open until a controlled prompt ablation is run.
  One AXQ row was flagged as a degenerate repetition loop
  (`repetition_max_run=1644`, `loop_suspect=true`); repetition was otherwise
  absent (MXFP4 `repetition_max_run` max 2).

Not claimed:

- MTP Tier 2 / `release_ready`
- 8-hour or 72-hour endurance (the published 8.87h soak is
  [Qwen 3.6 27B AXQ](qwen3.6-27b-axq-6bit-8h-endurance-2026-08-08.md))
- Multi-model `load_mode=add`
- P0 multimodal quality for this pack
- Long-context decode-at-depth
- Clean-worktree replacement of the 2026-08-30 refresh
- Any release-quality accuracy bar on the 2026-09-16/17 failed-pair retest; the
  29.4% / 20.6% strict accuracy above is recorded as **evidence of an open
  gap**, not as a passing qualification
- `LINE_SET` enumeration completeness: full recall is 0/24 across both packs
- General immunity to generation stalls. The diagnosed oversized recovery
  prefill now reaches the existing KV starvation failure bound and delivers
  its terminal response; campaign replay is not target-SKU qualification.
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
