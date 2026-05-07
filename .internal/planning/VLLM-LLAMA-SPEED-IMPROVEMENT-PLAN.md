# vLLM and llama.cpp Speed Improvement Plan

Status: In Progress
Date: 2026-05-07
Owner: AX Engine

## 1. Summary

AX Engine should absorb vLLM and llama.cpp lessons in layers instead of trying
to clone either runtime wholesale.

The recommended sequence is:

1. Benchmark and telemetry gates for serving-shaped workloads.
2. MLX exact-prefix KV reuse for repeated agent prompts.
3. Token-budget scheduling for chunked prefill and decode fairness.
4. N-gram decode policy hardening.
5. TurboQuant fused compressed decode promotion.
6. llama.cpp delegated backend presets and metrics pass-through.

This keeps repo-owned MLX work separate from the delegated llama.cpp route
defined by ADR 0012.

## 2. Reference Lessons

### vLLM

vLLM's speed comes primarily from serving architecture:

- PagedAttention manages KV cache in reusable blocks.
- Continuous batching schedules active requests into each engine step.
- Chunked prefill prevents long prompts from monopolizing the engine.
- Prefix caching skips repeated prefill work.
- Speculative decoding improves effective decode throughput.
- Disaggregated prefill/decode separates compute-bound and bandwidth-bound
  stages at larger deployment scale.

For AX Engine, the portable lessons are the contracts: block-level KV ownership,
token-budget scheduling, prefix reuse accounting, and decode telemetry. CUDA
graph and Triton-specific implementation details are not directly portable to
the current MLX runtime.

### llama.cpp

llama.cpp is the right reference for local delegated inference:

- `llama-server` exposes parallel slots and continuous batching.
- `cache_prompt` skips repeated prompt suffix processing.
- slot save/restore persists prompt cache state.
- `-b` and `-ub` separate logical and physical batch sizing.
- speculative decode includes draft-model and draftless n-gram variants.
- n-gram variants include simple recency lookup, map-based support counts, and
  shared hash-pool modes.

For AX Engine, the portable lessons are practical knobs and observability:
parallelism presets, cache hit metrics, request deferral metrics, and n-gram
policy variants.

## 3. Current AX Baseline

MLX mode already has:

- chunked prefill
- chunked KV backing buffers with `slice_update`
- direct double-buffer decode for deterministic direct rows
- n-gram self-speculative decode with telemetry
- Qwen linear-attention safe fallback behavior
- TurboQuant compressed-KV shadow storage and fused-decode telemetry

Known gaps:

- MLX runner execution is still effectively item-serial for batch throughput.
- MLX KV append and cache state are batch=1.
- Prefix reuse is not yet a repo-owned MLX runtime feature.
- Concurrent 8k prefill evidence currently classifies 4-request behavior as
  serialized.
- TurboQuant fused compressed decode is not yet a default production path.

## 4. Implementation Phases

### Phase 0: Claim Gate

Status: completed for the public MLX inference benchmark path. New
`ax.mlx_inference_stack.v2` artifacts emitted by
`scripts/bench_mlx_inference_stack.py` carry `ax.phase0_claim_gate.v1`, runtime
identity, and overlap classification metadata. The README artifact checker
enforces the gated contract while preserving compatibility with older artifacts
that predate the schema.

Add or preserve benchmark gates that fail closed for:

- same prompt hash across AX and reference rows
- selected backend and route identity
- direct-vs-n-gram decode policy
- prefill/decode split
- concurrent prefill overlap classification
- prefix reuse hit/miss counters once implemented
- llama.cpp delegated metrics when delegated mode is used

Acceptance:

- Public tables cannot claim continuous batching, prefix reuse, or long-context
  prefill improvement without a matching artifact.

### Phase 1: Exact Prefix KV Reuse

Status: completed for safe repo-owned MLX full-attention routes. The MLX
runner now restores block-aligned exact prompt-prefix KV snapshots when core
prefix metadata reports a reusable prefix. Linear-attention, MLA, and
sliding-window routes fail closed by rewarming the prefix without claiming a
cache hit. New benchmark artifacts expose `prefix_reuse_evidence` and route
counters for hits, misses, blocked fallbacks, stored prefixes, reused tokens,
warmup tokens, entries, bytes, and evictions.

Implement an MLX prefix cache for safe full-attention models:

- cache key: model id, tokenizer id/hash, route policy, token prefix hash,
  layer layout, dtype, sliding-window support marker
- cache value: immutable per-layer KV snapshot plus logical token count
- lookup: longest exact block-aligned prefix
- policy: byte cap, sequence cap, refcount, LRU eviction
- fail-closed for hybrid recurrent state until state snapshot/restore is proven

Acceptance:

- repeated-prefix replay shows reused tokens and lower TTFT
- deterministic replay matches direct baseline
- memory cap eviction is visible in route metadata

### Phase 2: Token-Budget Scheduling

Status: completed for the core scheduler contract and P2 artifact evidence
path. The scheduler now prioritizes decode work for the selected model, then
uses remaining token budget for compatible bounded prefill chunks. Mixed
prefill/decode batches stay conservative: same-mode routes must still match
exactly, while prefill/decode mixing is allowed only when route metadata is
compatible. Route metadata records scheduled prefill tokens, scheduled decode
tokens, skipped prefill/decode tokens, and whether the batch mixed prefill with
decode. The P2 concurrent artifact runner/checker/report path now carries this
scheduler evidence fail-closed.

Move from item-serial MLX execution toward a scheduler contract that can mix:

- decode tokens from running requests
- bounded prefill chunks from waiting requests
- skipped long prefill chunks when token budget is exhausted

The first version does not need full vLLM-style paged attention. It should
prove that long prefill no longer blocks all decode progress.

Acceptance:

- 2/4 concurrent long-prompt artifact moves from serialized toward partial or
  full overlap without correctness regressions
- route metadata records scheduled prefill tokens and scheduled decode tokens

### Phase 3: N-Gram Decode Policy Hardening

Status: completed for the repo-owned MLX n-gram policy path. The proposer now
has an explicit draft policy object with majority+recency default selection,
llama-map/latest and shared-pool policy variants behind `AX_MLX_NGRAM_POLICY`,
confidence-filtered vs no-candidate draft outcome labels, request-local fallback
labels, and adaptive draft length based on the observed acceptance posterior.
Benchmark and README artifact telemetry now carry the policy, fallback, and
adaptive draft counters.

Keep AX's current verifier semantics, but improve proposal quality:

- preserve majority continuation against one-off outliers
- break equal-support ties by recency
- expose per-request no-draft fallback labels
- evaluate llama.cpp-style map and shared-pool variants behind policy flags
- auto-tune draft length from observed acceptance and model family

Acceptance:

- unit tests cover majority, recency tie-break, confidence filtering, and
  no-draft fallback labels
- coding-shaped benchmark rows retain or improve effective throughput

### Phase 4: TurboQuant Fused Compressed Decode

Promote fused compressed decode only after it is integrated into real runner
decode, not only microbenchmarks.

Acceptance:

- long-context quality gate passes
- route metadata records attempts, successes, fallbacks, and byte savings
- fallback path is deterministic and observable

### Phase 5: Delegated llama.cpp Presets

For non-MLX delegated routes, expose safe presets rather than hiding llama.cpp
performance controls:

- parallel slots
- continuous batching toggle
- logical/physical batch sizing
- cache prompt
- slot save/restore path
- speculative decode mode
- metrics endpoint capture

Acceptance:

- delegated artifacts record llama.cpp prompt/decode throughput, KV usage,
  requests processing/deferred, and cache reuse where available

## 5. Implemented Slices

### Slice 1: Recency-Aware N-Gram Tie-Breaking

The first code slice is Phase 3: recency-aware n-gram tie-breaking. It is small,
local to the draft proposer, and does not change verifier semantics. It improves
proposal choice for repeated code/tool patterns where two continuations have
equal support but the latest local pattern is more likely to continue.

### Slice 2: Four-Token N-Gram Context

The second code slice extends the local n-gram proposer from 2/3-token contexts
to 2/3/4-token contexts. Prediction now tries the most specific 4-token context
first, then falls back to trigram and bigram suffixes when the longer context is
missing, sparse, or below the confidence threshold.

This pulls in the same practical lesson as vLLM and llama.cpp's longer prompt
lookup windows without changing AX's verifier semantics, scheduler, or KV
layout.

### Slice 3: Bounded N-Gram Context Tables

The third code slice keeps each n-gram order bounded with a recency-based
eviction policy. AX's proposer stores richer per-context continuation counts
than llama.cpp's constant-memory `ngram-mod`, so long generations should not
allow bigram, trigram, or fourgram maps to grow without a cap.

The cap is per order, and eviction removes the least recently observed context
key. This preserves recent code/tool-output patterns while making proposer
memory predictable.

### Slice 4: Bounded Continuations Per Context

The fourth code slice bounds the continuation candidates stored under each
context key. Each context now keeps only a small recent set of candidate
continuations, then recomputes the selected champion from the retained
statistics. This follows the practical shape of llama.cpp's bounded n-gram map
variants while preserving AX's support, confidence, and recency tie-break
semantics.

This prevents a single high-entropy context from accumulating an unbounded
continuation histogram during long outputs.

### Slice 5: Small-Vector Continuation Storage

The fifth code slice changes each context's continuation storage from a nested
hash map to a small bounded vector. Since Slice 4 caps each context at four
candidate continuations, linear scans are cheaper and more predictable than
allocating and hashing inside every context record.

This keeps the same majority, confidence, and recency semantics while reducing
per-context metadata overhead.

### Slice 6: Acceptance-Aware N-Gram Confidence

The sixth code slice records verifier feedback on accepted draft tokens and the
first rejected draft token. Confidence-gated prediction now discounts a
candidate after real verifier rejection while leaving raw no-threshold
prediction available for callers that explicitly want it.

This follows the same practical lesson as llama.cpp's n-gram map variants,
where used n-grams track accepted-token outcomes, but keeps AX's existing
single verifier path and KV rollback semantics unchanged.

### Slice 7: Source-Aware N-Gram Feedback

The seventh code slice makes verifier feedback use the same source-selection
logic as draft prediction. If a longer context fails the confidence/support gate
and a shorter suffix context creates the draft, acceptance or rejection is
charged to the fallback source instead of whichever longer context happens to
contain the same token.

This keeps the acceptance-aware confidence signal precise and avoids poisoning
or rewarding context orders that did not actually propose the verified token.

### Slice 8: N-Gram Table Stats Snapshot

The eighth code slice adds an aggregate stats snapshot for the n-gram proposer.
It reports context counts, continuation pressure, configured caps, and verifier
feedback totals without exposing token content.

This gives future benchmark artifacts and route metadata a safe, stable source
for distinguishing no-draft fallback, table-growth pressure, and verifier
accept/reject behavior.

### Slice 9: Batched Context Pruning

The ninth code slice reduces long-generation maintenance overhead once n-gram
context maps reach their cap. Instead of scanning a full map every time one new
context crosses the cap, the proposer evicts a small batch of the oldest
contexts in one pass and then has headroom before the next prune.

The table remains under the configured cap after pruning, while steady-state
long outputs perform fewer full-map pruning scans.

### Slice 10: Bounded Oldest-Set Pruning

The tenth code slice keeps the batched pruning behavior but avoids cloning and
sorting every context key on each prune. Pruning now maintains only the bounded
set of oldest candidates needed for the eviction batch, then removes that set.

This lowers pruning allocation and sort overhead while preserving the same
oldest-context eviction contract.

### Slice 11: Insert-Only Context Prune Gate

The eleventh code slice avoids entering the context-prune path when an observed
n-gram only updates an existing context key. Since map size cannot grow on an
existing-key update, pruning is now checked only after inserting a new context.

This removes avoidable cap checks and function calls for repetitive prompts or
generated loops where most n-gram observations hit existing contexts.

These slices intentionally do not touch `runner.rs`, scheduler, or KV ownership
because there are active local changes on those surfaces and the larger Phase 1
and Phase 2 work need separate tests and artifacts.

### Slice 12: Phase 0 Public Claim Gate

The twelfth slice completes Phase 0 for the public MLX inference benchmark
path. New artifacts carry a claim-gate schema, explicit repo-owned MLX runtime
identity, and a single-request/no-overlap classification. The README artifact
checker now fail-closes gated artifacts if AX rows lose their runtime identity,
direct-vs-n-gram policy identity, prefill/decode split, n-gram telemetry, or
unproven public claims for continuous batching, prefix reuse, or long-context
prefill improvement.

The checker remains compatible with older README artifacts that were generated
before the Phase 0 schema existed, so the historical public table can still be
validated while all newly generated claim-gated artifacts use the stricter
contract.

### Slice 13: MLX Exact Prefix KV Snapshot Cache

The thirteenth slice completes Phase 1 for safe full-attention MLX routes. The
runner owns an LRU-bounded prefix snapshot cache keyed by model id, route
policy, layer layout, block size, prefix token count, and token hash. When core
prefix metadata marks a request as reusable, the MLX runner restores the exact
block-aligned KV snapshot instead of rebuilding the reused prefix. Full-prompt
greedy hits also carry the cached prefill output token so deterministic replay
can emit the same first token without duplicating the last prompt token.

Unsafe hybrid paths do not use the snapshot cache: linear attention, GLM MLA,
and sliding-window layouts rewarm the reused prefix and record blocked/warmup
telemetry. The route metadata and benchmark artifact now expose enough
hit/miss/eviction evidence for public prefix-reuse claims to stay gated by
Phase 0.

### Slice 14: Decode-First Token-Budget Scheduler

The fourteenth slice completes Phase 2's first scheduler contract. Core
scheduling now selects decode items before prefill items for the oldest runnable
model family, so a request that has already finished prefill can make decode
progress even while other long prompts are still being chunked. Remaining token
budget is assigned to bounded prefill chunks; unscheduled prefill/decode work is
counted as skipped token-budget work rather than disappearing from telemetry.

The route contract remains conservative. Same-mode items still require exact
execution-plan and route-metadata matches. Mixed prefill/decode batches are
allowed only when their metadata is compatible, and they are labeled as
`phase2.token_budget` / `mixed_prefill_decode`. Allocation rebuild now preserves
the scheduler token-budget counters while appending prefix-reuse metadata.

The P2 artifact path extracts these counters from route metadata, writes
`scheduler_evidence` into concurrent-prefill artifacts, validates the evidence
fail-closed, and renders the scheduled prefill/decode/mixed-batch counts in the
Markdown report.

### Slice 15: N-Gram Policy Hardening

The fifteenth slice completes Phase 3 for the MLX runner's n-gram decode policy.
The table keeps the current verifier semantics but makes draft selection
explicit: majority+recency remains the default, late one-off outliers no longer
replace a supported continuation, and equal-support ties still prefer the most
recent continuation. Empty drafts now carry a reason (`no_candidate` or
`confidence_filtered`) so no-draft fallback claims can be audited from route
metadata.

The runner records request-local fallback labels for short-output direct
fallback and linear-attention no-draft disablement. It also chooses draft length
from the request-local acceptance posterior: dense models can extend to the
high-confidence draft ceiling, middling confidence uses the default draft size,
and low confidence shrinks probes before cooldown. Linear-attention routes stay
capped to the bounded recompute cost profile.

`AX_MLX_NGRAM_POLICY` now exposes policy variants for evaluation:
`majority-recency` (default), `llama-map` / `latest`, and `shared-pool`.
Artifacts include the selected policy code, adaptive draft length counters, and
fallback reason counters so coding-shaped benchmark rows can distinguish real
throughput gains from direct fallback.

Follow-up validation on 2026-05-07 used the rebuilt release server and the
existing Qwen3.5 9B 4-bit reference rows:
`benchmarks/results/mlx-inference/2026-05-07-phase3/qwen3_5-9b-mlx-4bit-phase3.json`.
The 128-token decode row retained effective n-gram acceleration at 196.8 tok/s
(2.067x `mlx_lm`) with 69 draft attempts and 276 accepted draft tokens. The
512-token random-prompt row produced no draft candidates and is now labeled
`ngram_no_draft_direct_fallback` instead of being counted as effective
acceleration, while the matching direct row stayed near reference parity
(0.991x `mlx_lm`). This keeps the Phase 3 throughput claim scoped to rows with
observed draft acceptance.

### Slice 16: TurboQuant Fused Path Quality Gate

The sixteenth slice completes Phase 4's experimental quality/path gate without
turning it into a premature public performance claim. The real MLX runner can
exercise `turboquant-fused-experimental` on eligible K8/V4 single-token decode
layers, and route metadata distinguishes candidate snapshots, attempts,
successes, fallbacks, fallback reasons, runtime compressed-slot writes, and
estimated byte savings.

The artifact contract now separates two decisions:

- the long-context quality/path gate, which requires fused_compressed_decode
  attempts and successes, zero fused fallbacks, K8/V4 metadata, runtime storage
  writes, byte savings, and reference-quality logits/output evidence
- the public-support performance gate, which remains blocked until fused decode
  throughput reaches the configured decode-ratio threshold

This split lets Phase 4 close honestly: real-runner fused compressed decode can
produce auditable long-context quality evidence, while public TurboQuant support
wording remains experimental until the performance gate is also satisfied.
