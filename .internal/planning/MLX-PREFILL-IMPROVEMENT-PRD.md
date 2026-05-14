# MLX Prefill Improvement PRD

Status: Closed for W0-W4 current evidence cycle
Date: 2026-05-13
Owner: AX Engine
Depends on: MLX-RUNTIME-PERFORMANCE-PRD.md, GEMMA-QWEN-MLX-PERFORMANCE-PRD.md, KV-SCHEDULER-REVAMP-PRD.md, ../benchmark/BENCHMARK-LAB-PRD.md
Related ADRs: ../adr/0017-mlx-runtime-optimization-governance.md, ../adr/0018-kv-cache-and-scheduler-revamp-strategy.md, ../adr/0021-continuous-batching-contract-v2.md, ../adr/0024-gemma-qwen-mlx-performance-strategy.md
Latest evidence: 2026-05-13 W4 forward-path trigger

Closure report: ../reports/MLX-PREFILL-PRD-CLOSURE-2026-05-14.md

## 1. Summary

This PRD defines a practical, evidence-first prefill improvement program for the
repo-owned MLX runtime. The purpose is not to chase every possible kernel or
batching idea. The purpose is to improve the prefill surfaces that matter most
to users:

- short and mid-prompt README rows that shape public confidence;
- long-context TTFT and prefill scaling;
- repeated hot-prefix workloads such as system prompts, tool schemas, and RAG
  headers;
- multi-request server paths where queueing and serialized prefill dominate
  perceived latency.

The best-practice strategy is to separate these surfaces. A win in one surface
must not be presented as a win in another without the matching artifact gate.

As of the 2026-05-14 closure report, W0-W4 are closed for the current evidence
cycle:

- W0 has a validator-backed public README refresh.
- W1 is closed as bounded Qwen long-context evidence, not a broad
  Gemma/Qwen/GLM win.
- W2 has a positive Qwen warm-repeat physical prefix snapshot claim.
- W3 is closed as a serialized/partial-overlap concurrency boundary, not a
  continuous-batching claim.
- W4 has diagnostic packed-projection plumbing and remains opt-in until a
  matched artifact proves a repeatable win.

## 2. Problem Statement

AX Engine now has enough MLX benchmark evidence that "make prefill faster" is
too broad to be a useful implementation plan.

Current evidence says:

- Public 128/512-token README rows are already strong for many supported models.
  AX prefill spans roughly from a small regression to a large win versus
  `mlx_lm`, depending on model family and prompt shape.
- Long-context evidence is weaker: the checked-in Qwen3-4B P1 artifact shows AX
  ahead at 1k, near parity at 2k, and behind `mlx_lm` at 4k/8k.
- Prefix reuse is a separate product value from raw cold-prefill throughput. A
  physical prefix snapshot hit can improve hot-prompt TTFT even if cold prefill
  tok/s is unchanged.
- Concurrent prefill evidence is not yet a positive continuous-batching claim.
  The latest checked-in P2 long-prompt artifact still includes serialized
  behavior at 4 concurrent requests.
- GatedDelta, sliding-window, MLA, and standard attention have different safety
  constraints. A change that is safe for one family may be incorrect or
  unmeasured for another.

The risk is that prefill work becomes a large rewrite that mixes benchmark
refreshes, prefix-cache semantics, scheduling changes, and model-forward kernels.
That would make wins hard to attribute and correctness regressions hard to
contain.

## 3. Best-Practice Principles

### BP-1. Split claim surfaces

Every prefill claim must name its surface:

- README short/mid prompt throughput;
- long-context scaling;
- hot-prefix physical reuse;
- startup/cold-vs-warm TTFT;
- concurrent server-path prefill;
- model-family-specific forward-path profiling.

Do not use one artifact family to justify another.

### BP-2. Measure before changing architecture

Kernel, eval-boundary, scheduler, and cache-ownership changes require a
before/after artifact that identifies the phase and route. Narrow graph-level
fixes may proceed when code inspection proves a clear redundant operation, but
they still need a post-change artifact before public claims change.

### BP-3. Preserve direct-policy comparability

Prefill rows must use direct AX policy. N-gram acceleration is a decode policy
and must not be credited as a prefill optimization.

### BP-4. Treat logical and physical prefix reuse separately

Logical prefix reuse from the engine is not the same as a physical MLX snapshot
hit. Hot-prefix PRD requirements must track:

- logical retained-prefix hits;
- physical snapshot hits;
- eligible physical misses;
- warmup tokens;
- blocked reasons;
- snapshot stores and evictions.

### BP-5. Keep correctness gates close to the risky code

Any change to physical prefix snapshot restore/store, prompt warmup fallback, or
architecture support must pass the prefix-reuse equivalence gate before any
throughput claim is trusted.

### BP-6. Optimize for product-visible latency, not only tok/s

Prefill tok/s matters, but users experience TTFT, queue delay, and tail latency.
The program should prioritize improvements that reduce real request latency under
long context or repeated prompt prefixes.

## 4. Goals

G1. Keep public short/mid prompt prefill rows accurate and refreshed after known
prefill fast-path fixes.

G2. Build a representative long-context prefill campaign for Gemma, Qwen, and
GLM with validated `ax.mlx_prefill_scaling.v1` artifacts.

G3. Prove and improve hot-prefix physical snapshot reuse for practical repeated
prompt workloads.

G4. Make concurrent prefill behavior measurable before promoting any continuous
batching or multi-request TTFT claim.

G5. Identify model-family-specific prefill bottlenecks only when family-level
evidence shows a real gap.

G6. Keep each implemented slice small enough to validate, revert, and explain.

## 5. Non-Goals

- Do not rewrite the Qwen GatedDelta path unless a fresh profile identifies a
  dominant stage or a supported Qwen row falls below the reference in a relevant
  prefill surface.
- Do not promote MLA warm-extend physical snapshot restore until token-exact
  equivalence is proven for GLM-style compressed-latent KV.
- Do not claim continuous batching from serialized or single-request artifacts.
- Do not mix delegated `mlx_lm.server`, `llama_cpp`, or app-server Swift rows into
  repo-owned MLX prefill claims.
- Do not optimize every model family at once. Campaign evidence chooses the next
  family-specific implementation slice.
- Do not make disk KV cache the first optimization layer. It may be evaluated
  later after RAM-first physical snapshot behavior is understood.

## 6. Baseline Evidence

Primary public baseline:

```text
benchmarks/results/mlx-inference/2026-05-12-full-fresh-readme-refresh/
```

Current interpretation:

| Surface | Evidence | Current conclusion |
|---|---|---|
| README 128/512 prefill | Root README table | AX is often ahead; weakest rows are Gemma sliding-window 512-token rows captured before a later mask fix |
| Long-context prefill | `2026-05-07-real-p1/qwen3-4b-4bit-prefill-scaling/` | AX wins at 1k, near parity at 2k, below `mlx_lm` at 4k/8k |
| Concurrent prefill | `2026-05-07-real-p2/qwen3-4b-4bit-p2-latency/` | 4-request row is serialized; not a positive continuous-batching claim |
| Prefix reuse | KV and MLX runner route metadata | Logical and physical reuse are observable separately; physical snapshot hits are required for a public prefix-reuse claim |
| GatedDelta profiling | `docs/BENCHMARKS.md` profile gate | Valid diagnostic route exists; use only when Qwen evidence justifies it |

### 2026-05-13 W4 trigger evidence

`benchmarks/results/mlx-inference/2026-05-13-ttft-breakdown/` plus the matched
`llama.cpp Metal` sweep shows short-prompt rows where AX MLX is faster than
`mlx_lm` but slower than `llama.cpp Metal`:

| Model | Prompt tok | AX/MLX | AX/llama.cpp | Prefill wall ms | Forward share |
|---|---:|---:|---:|---:|---:|
| `qwen3_6_35b_a3b_8bit` | 128 | 2.373x | 0.580x | 406.0 | 100.0% |
| `qwen3_coder_next_4bit` | 128 | 2.357x | 0.617x | 507.4 | 100.0% |
| `glm_4_7_flash_4bit` | 128 | 1.760x | 0.624x | 451.0 | 100.0% |

This evidence changes W4 from "not triggered" to "triggered for diagnostic
forward-path profiling." The new implementation target is not serving bypass,
prefix-cache policy, or scheduler rewrite: the measured prefill time is almost
entirely model forward. The first W4 slice must therefore produce a diagnostic
forward-stage report before proposing projection, recurrent-scan, MLA, or
layout/fusion changes.

## 7. Requirements

### REQ-P0. Refresh known fixed short/mid prompt rows

Run a targeted AX-only refresh for rows affected by prefill fast-path changes,
starting with Gemma sliding-window 512-token rows after the sliding-window mask
fix.

Acceptance criteria:

- Uses the same prompt-token contract as the README table.
- Preserves direct AX policy for prefill.
- Reuses reference rows only when shape and prompt hash parity are enforced.
- Updates public tables only after `scripts/check_readme_performance_artifacts.py`
  passes.
- Records whether the weaker Gemma 512-token rows improved, stayed neutral, or
  regressed.

### REQ-P1. Long-context prefill campaign

Capture representative Gemma, Qwen, and GLM prefill scaling artifacts.

Required prompt shapes:

- 1k, 2k, 4k, 8k;
- 16k where the model and host make it practical;
- largest supported context only when it can run without thermal or memory noise
  dominating the result.

Acceptance criteria:

- Each model has a validated `ax.mlx_prefill_scaling.v1` artifact.
- Each artifact includes matched `mlx_lm` reference rows, prompt hashes, median
  prefill tok/s, median TTFT, peak memory, direct AX policy, and ratios.
- Campaign summary marks the first context where AX prefill bends below the
  configured threshold.
- Results are interpreted by family, not averaged into one headline.

### REQ-P2. Hot-prefix physical reuse workload

Create or reuse replay/scenario manifests that mimic practical repeated-prefix
traffic:

- shared system prompt;
- shared tool schema;
- branch decode after a common prefix;
- retained prefix after cleanup;
- memory-blocked recovery when retained cache can be evicted or reused.

Acceptance criteria:

- Artifacts show logical prefix hits and physical snapshot path counters.
- A positive hot-prefix claim requires `ax_mlx_prefix_cache_hits > 0`,
  `ax_mlx_prefix_cache_reused_tokens > 0`, and no warmup-token substitution for
  the claimed hit path.
- Miss/warmup-only artifacts remain diagnostic and cannot be marketed as
  physical prefix reuse.
- Any store/restore gate change passes `scripts/check-prefix-reuse-equivalence.sh`.

### REQ-P3. Prefix snapshot coverage expansion

Improve physical snapshot reach only where the architecture-specific safety
contract is clear.

Priority order:

1. Qwen-style linear attention exact full-prefix, block-aligned reuse.
2. Gemma sliding-window exact full-prefix, block-aligned reuse.
3. Standard full-attention intermediate block snapshots if/when a supported tier
   model needs them.
4. GLM/MLA warm-extend only after equivalence proves no drift.

Acceptance criteria:

- Coverage expansion includes before/after counters for hits, misses, warmup
  tokens, blocked reasons, stores, evictions, and bytes.
- Token-exact equivalence passes in both warm-repeat and warm-extend modes for
  the affected family.
- The implementation keeps miss fallback behavior correct and observable.

### REQ-P4. Concurrent prefill evidence before contract promotion

Capture server-path concurrent prefill evidence for at least one representative
long-context model.

Acceptance criteria:

- Uses `ax.mlx_concurrent_prefill.v1`.
- Includes 1/2/4 request levels and 8 requests only when host memory permits.
- Reports per-request TTFT, total wall time, queue delay, failures, peak memory,
  prompt hashes, direct AX policy, route identity, and overlap classification.
- Any positive continuous-batching claim requires positive overlap evidence and
  must be rejected by validation when evidence is serialized or single-request.

### REQ-P5. Model-family prefill profiling trigger

Run family-specific profiling only when campaign evidence shows a meaningful
prefill gap.

Triggers:

- Qwen/GatedDelta: AX is below `mlx_lm` or `mlx_swift_lm` on a relevant long or
  short prompt shape, or profile counters show a dominant recurrent stage.
- Gemma/sliding-window: a post-mask-fix row still underperforms on prefill or
  TTFT.
- GLM/MLA: warm-extend replay shows material recompute cost and equivalence
  blocks physical restore.
- External reference gap: AX is below `llama.cpp Metal` on a shape-compatible
  public short-prompt row and the prefill breakdown attributes roughly all wall
  time to model forward.

Acceptance criteria:

- Profiling artifact names the dominant stage before a kernel or eval-boundary
  patch.
- The profile is diagnostic and not mixed into normal throughput rows.
- Any follow-up implementation has a matching before/after artifact.
- Profile renderers reject known-invalid token counters such as the historical
  `4294967295` sentinel, so stale artifacts cannot guide implementation.

### REQ-P6. Reporting and claim hygiene

Every PR or release note that references prefill performance must include:

- the exact artifact path;
- model family and concrete model artifact;
- prompt tokens and generation tokens;
- direct AX policy;
- reference runtime;
- whether TTFT is measured directly or derived;
- whether prefix reuse was cold, miss/warmup, or physical hit.

## 8. Workstream Sequence

### W0. Triage and refresh

Goal: determine whether known prefill fixes already changed the public weak
rows.

Deliverables:

- targeted README-row refresh for affected Gemma rows;
- short report classifying each row as improved, neutral, or regressed.

### W1. Long-context campaign

Goal: choose the next implementation target from real scaling curves.

Deliverables:

- Gemma/Qwen/GLM `ax.mlx_prefill_scaling.v1` artifacts;
- campaign summary with bend points and family ranking.

### W2. Hot-prefix physical reuse

Goal: turn repeated-prefix behavior into a product-visible TTFT win.

Deliverables:

- replay/scenario artifacts with physical snapshot hit evidence;
- any required prefix-cache policy or capacity tuning;
- equivalence-gated changes only.

### W3. Concurrent prefill boundary

Goal: determine whether the current server path can support a positive
multi-request prefill claim.

Deliverables:

- P2 latency artifacts;
- clear decision: document current serialized boundary, or open a scoped
  continuous-batching implementation plan.

### W4. Family-specific profiling and fixes

Goal: touch model-forward kernels only after evidence chooses the family and
stage.

Deliverables:

- GatedDelta, sliding-window, or MLA profile artifact;
- forward-profile report for any short-prompt `llama.cpp Metal` gap where
  prefill breakdown shows forward-path dominance;
- one narrow implementation slice with before/after evidence.

## 9. Success Metrics

Public short/mid prompt:

- no unvalidated README prefill row;
- known affected rows refreshed after fast-path fixes;
- no regression greater than 5% without a documented reason.

Long context:

- representative campaign covers Gemma, Qwen, and GLM;
- at least one actionable bottleneck or explicit no-action decision is produced;
- 8k and 16k claims are never inferred from 128/512-token rows.

Hot prefix:

- at least one practical repeated-prefix workload shows physical snapshot hits;
- TTFT improvement is reported with reused-token evidence;
- miss/warmup paths remain correctly labeled.

Concurrency:

- serialized, partial-overlap, and positive-overlap outcomes are classified
  consistently;
- no public continuous-batching claim is possible without validator-approved
  evidence.

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Benchmark noise drives the wrong implementation | Use repeated runs, cooldown, prompt hashes, and direct-policy labeling |
| Prefix-cache hit counters are mistaken for logical reuse | Require physical snapshot counters and reused-token evidence |
| MLA warm-extend restore causes token drift | Keep restore blocked until equivalence passes |
| Concurrent prefill changes destabilize scheduler fairness | Treat concurrency as a separate contract with explicit route and queue metrics |
| Qwen/GatedDelta work consumes time while public rows already win | Require a profiling trigger before model-forward rewrites |
| Public docs overclaim internal diagnostics | Keep PRD internal; publish only validated artifact-backed claims |

## 11. Open Questions

- Which model should be the first canonical hot-prefix serving workload:
  Qwen3.5 9B, Qwen3.6 35B A3B, or Gemma 4 E2B?
- Should the prefix snapshot cache policy expose a user-visible capacity knob, or
  remain an internal runtime policy until miss/eviction artifacts show need?
- Is block-aligned full-prefix reuse sufficient for practical agent/tool-schema
  workloads, or do we need safe intermediate snapshots for a standard
  full-attention model first?
- What is the minimum positive-overlap threshold before a concurrent prefill row
  can support a product claim?

## 12. First Implementation Slice

The original first slice remains valid for W0/W2, but the 2026-05-13 evidence
adds an immediate W4 diagnostic slice. The current next slice is:

1. Keep the prefill breakdown report as the public-vs-diagnostic separator:
   normal README artifacts exclude `*-linear-profile*.json` by default.
2. Add a diagnostic MLX forward-profile report that reads
   `AX_MLX_LINEAR_ATTENTION_PROFILE=1` artifacts, rejects stale token sentinels,
   and ranks projection/conv/qk-norm/recurrent/output stages.
3. Add projection-substage rendering for Qwen linear-attention rows so split
   `qkv/z/a/b` layouts are not mistaken for a generic projection bottleneck.
4. Use the report to choose the first code patch only after the dominant stage is
   visible for the slow p128 rows.
5. Do not update README performance claims from barrier-profile artifacts.

This sequence keeps the implementation small and prevents a speculative
forward-kernel rewrite from being justified by a serving-overhead hypothesis
that the current evidence does not support.

## 13. W4 Projection-Pack Gate

The current Qwen p128 profile identifies `split_qkv_z_a_b` as an offline pack
candidate, but the implementation gate is stricter than "concatenate rows":

- `qkvz` must be laid out per key head as `q,k,v,z`.
- `ba` must be laid out per key head as `b,a`.
- Quantized weight sidecars (`scales` and `biases`) must be reordered in the
  same row order as the packed weight.
- Group size, bit width, and bias presence must match across fused tensors.
- A row-order equivalence test must compare packed-path outputs against the
  split path before any prefill speed claim.

2026-05-13/14 update:

- The manifest-level packed role validator and row-order oracle now exist.
- The loader has an opt-in diagnostic gate,
  `AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS=1`, for split-to-packed projection
  packing.
- The inference-stack benchmark can emit matched split and packed AX rows via
  `--ax-compare-linear-attention-projection-pack`.
- The forward-profile renderer now includes a `Pack Comparison` section with
  packed/split prefill and projection deltas.

The remaining W4 gate is empirical, not structural: do not promote the packed
path or update public performance claims until a matched artifact shows a
repeatable packed/split win for the same model, prompt shape, direct AX policy,
and profile-counter contract.
