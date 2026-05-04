# MLX Mode Performance Hardening PRD

Status: Draft
Date: 2026-05-03
Owner: AX Engine

## 1. Summary

AX Engine v4 MLX mode is the repo-owned Mac-local inference path. It is selected
explicitly with `--mlx` or Python `mlx=True`, uses `selected_backend=mlx`, and
is implemented through `ax-engine-mlx`, `mlx-sys`, and `MlxRunner`.

The current direction is sound: MLX mode is direct Rust to `mlx-c`, not a
subprocess wrapper. It already has dedicated GPU stream setup, model warmup,
chunked prefill, MLX fast ops, quantized matmul, per-request KV state, and
n-gram speculative decode.

The remaining gap is not strategy. The remaining gap is performance credibility:
benchmark controls are incomplete, KV growth may be expensive, execution
batches are effectively serial, and some documentation claims custom-kernel
integration that is not yet wired into the model path.

This PRD defines the improvement plan required before AX Engine can call MLX
mode performance hardened.

## 2. Background

Earlier planning separated several possible inference routes:

- direct MLX compatibility through `mlx_lm` or an OpenAI-compatible MLX server
- repo-owned MLX mode through `ax-engine-mlx`, `mlx-sys`, and `MlxRunner`
- retired AX native Metal execution
- non-MLX delegated compatibility through `llama.cpp`, vLLM, or mistral.rs

That taxonomy is obsolete for the shipping surface. Current routing is:

- explicit MLX requests route to repo-owned AX MLX mode
- non-MLX local/server inference routes to `llama.cpp`
- AX native mode is retired
- direct MLX adapters, vLLM, and mistral.rs are reference or research inputs,
  not peer shipping inference routes

The performance hardening work should therefore optimize and measure the one
supported MLX path instead of reviving multiple runtime identities.

### 2.1 Reference Runtime Lessons

The reference runtimes are useful as implementation evidence, not as alternate
shipping routes.

High-value lessons from `mlx_lm`:

- its normal `KVCache` grows in fixed-size chunks instead of concatenating on
  every token, while keeping a logical offset and slice assignment for appends.
- `ConcatenateKVCache` is explicitly treated as a simple/mock cache, which
  reinforces that AX should not keep concat-every-append as the long-context
  decode layout.
- `RotatingKVCache`, `QuantizedKVCache`, and `BatchKVCache` provide concrete
  future layouts for sliding-window attention, long-context memory reduction,
  and left-padded batch execution.
- `LRUPromptCache` separates active request KV state from retained prompt-prefix
  reuse, using nearest-prefix lookup plus sequence and byte limits.
- speculative generation is guarded by trimmable caches, explicit cache rewind,
  and acceptance accounting.

High-value lessons from `mlx-swift-lm`:

- the cache protocol makes `offset`, `maxSize`, `isTrimmable`, `trim`, masks,
  state, and meta-state explicit, which is a better contract than shape-only
  cache handling.
- its simple KV cache also uses chunked growth and slice updates, matching
  `mlx_lm` and giving AX two independent references for the same near-term
  fix.
- `GenerateParameters` centralizes prefill step size, max KV size, KV
  quantization settings, and sampler/logit-processor choices.
- GPU-resident token-ring logit processors avoid repeated CPU/GPU
  synchronization for repetition, presence, and frequency penalties.
- wired-memory utilities measure weights, KV bytes, prefill workspace, and peak
  active memory before choosing policy caps.

High-value lessons from SwiftLM:

- DFlash rollback separates full-attention cache trimming from recurrent or
  hybrid-model rollback, which is valuable for future hybrid/linear attention
  models but not needed for the first dense-Qwen MLX hardening PR.
- server-side prefill progress and memory snapshots are useful user-facing
  liveness features for long prompts, provided AX does not present heuristic
  memory accounting as authoritative runtime telemetry.
- kernel microbenchmarks are a good methodology input, but custom-kernel work
  should remain behind profiling evidence after benchmark, KV, and runner
  contracts are fixed.

## 3. Problem Statement

AX Engine cannot make durable MLX performance claims until the supported MLX path
has trustworthy evidence and a small set of high-value runtime bottlenecks are
addressed.

Current review findings:

1. `scripts/bench_mlx_inference_stack.py` is the current MLX inference-stack
   harness. It uses upstream `mlx_lm.benchmark` as the primary reference,
   admits `mlx-swift-lm` only through an explicit JSON adapter, and separates
   greedy versus speculative AX Engine MLX runs with stable result labels.
2. `MlxKVCache::append` grows K/V arrays with `mlx_concatenate_axis` on every
   append. Decode and long-context paths can pay repeated allocation/copy cost.
3. `MlxRunner::run` processes execution batch items one by one, and `run_item`
   holds the request-state mutex across GPU work. This is safe but limits
   batching and prefix-reuse throughput.
4. `docs/MLX-BACKEND.md` claims phase1 custom Metal kernels are called inside
   the MLX graph, but the current `ax-engine-mlx` model path does not call
   `MlxMetalKernel`.

## 4. Goals

- Make MLX benchmark output decision-grade before publishing performance claims.
- Provide explicit runtime controls for greedy versus speculative decode.
- Measure and then reduce KV cache append/growth overhead.
- Remove lock-held GPU work from the hot path where it blocks batch throughput.
- Make batching and prefix reuse either real in MLX mode or explicitly out of
  scope for the current milestone.
- Align docs with implementation truth for custom kernels and performance
  claims.
- Keep all artifacts explicit about `selected_backend`, `support_tier`, and
  `resolution_policy`.

## 5. Non-Goals

This PRD does not attempt to:

- reintroduce AX native mode
- make `mlx_lm`, direct MLX compatibility, vLLM, or mistral.rs shipping routes
- support distributed serving, remote KV, or multi-node scheduling
- claim broad model-family coverage beyond the current MLX preview target
- implement speculative decoding with a second draft model
- optimize custom Metal kernels before profiling proves the hot path
- implement recurrent rollback, TurboKV-style hybrid KV compression, or
  broad hybrid-attention model support in the first performance milestone

## 6. Success Metrics

Performance claims require release-mode evidence on a named Apple Silicon host,
model, quantization, prompt length, decode length, concurrency level, and commit
or diff range.

Minimum metrics:

| Metric | Required evidence |
|---|---|
| TTFT | median/min/max plus raw runs |
| Prefill throughput | tokens/sec by prompt length |
| Decode throughput | tokens/sec with greedy and speculative modes separated |
| Speculation quality | acceptance rate, disabled-step count, effective tokens/model-run |
| KV growth cost | append time or allocated bytes versus sequence length |
| Batch throughput | single request versus concurrent requests under the same prompt shape |
| Correctness | deterministic output and request-state invariants still pass |
| Route identity | `selected_backend=mlx`, `support_tier`, and `resolution_policy` recorded |

Publication gate:

- no performance table may mix MLX mode, `mlx_lm`, and llama.cpp without labels
- reference numbers must be marked reference-only
- benchmark commands must fail closed on subprocess/server errors
- raw JSON artifacts must be preserved for every summarized table

## 7. Product Requirements

### PR-1: Benchmark Controls Must Be Real

Required behavior:

- `--axengine-no-speculative` must map to a real server/runtime option or be
  removed from the script.
- MLX runner configuration must expose whether speculative decode is enabled.
- benchmark artifacts must record speculative mode, draft length, acceptance
  rate, and effective tokens per model invocation.
- failed subprocesses or unsupported flags must fail the benchmark, not produce
  a partial success summary.

Acceptance criteria:

- `cargo run -q -p ax-engine-server -- --help` lists the supported speculative
  control if the benchmark uses it.
- a server benchmark can produce both greedy and speculative MLX runs on the
  same model, prompt, and decode length.
- benchmark JSON distinguishes `ax_engine_mlx_greedy` from
  `ax_engine_mlx_speculative`.

### PR-2: KV Cache Growth Must Be Measured And Improved

Required behavior:

- add instrumentation for KV append/growth cost by layer or aggregate step
  bucket.
- benchmark short, medium, and long contexts against the same model artifact.
- replace per-token full-array concatenation if measurement confirms it is a
  material decode or long-context cost.

Candidate implementation paths:

- chunked growth with reserved capacity and append windows
- per-layer slab arrays with logical sequence length
- paged MLX KV layout aligned with existing block-size contracts
- hybrid approach that keeps the current simple layout for short contexts and
  switches to chunked growth after a threshold

Acceptance criteria:

- long-context decode does not show superlinear KV append cost across the
  canonical MLX scenario matrix.
- `MlxKVCache::trim_to` still preserves speculative rollback correctness.
- prefix and terminal cleanup tests continue to pass.

### PR-3: Runner State Must Not Serialize GPU Work Unnecessarily

Required behavior:

- separate request-state lookup/mutation from GPU forward execution.
- avoid holding the runner state mutex while running prefill, decode, or
  speculative verification.
- document and test the concurrency contract for per-request state ownership.

Acceptance criteria:

- multi-request execution cannot poison or block unrelated request state during
  a long prefill.
- runner tests cover at least one concurrent or interleaved request scenario.
- benchmark artifacts show whether MLX mode is single-request optimized or
  batch-throughput capable.

### PR-4: Batch Execution Must Be Explicit

Required behavior:

- decide whether the near-term MLX runner supports true multi-item batching.
- if batching is supported, batch compatible decode items through shared MLX
  calls where model shapes allow it.
- if batching is deferred, docs and benchmark labels must describe MLX mode as
  single-request optimized for the milestone.

Acceptance criteria:

- `MlxRunner::run` no longer silently implies batch optimization while looping
  through all items serially.
- concurrent benchmark rows include enough route metadata to explain whether
  throughput is from batching, speculation, or request interleaving.

### PR-5: Custom Kernel Claims Must Match Runtime Reality

Required behavior:

- either wire `MlxMetalKernel` into a measured MLX model hot path or update docs
  to say the wrapper exists but is not active in the current model path.
- custom-kernel work must start from profiling evidence and include a rollback
  path.
- docs must separate MLX fast ops from AX custom Metal kernels.

Acceptance criteria:

- `docs/MLX-BACKEND.md` has no claim that active model execution uses custom
  phase1 kernels unless code and benchmark evidence prove it.
- any custom-kernel PR includes before/after measurements on the same benchmark
  row.

### PR-6: Runtime Semantics Must Stay Honest

Required behavior:

- deterministic argmax behavior must be explicit.
- unsupported sampling or logprob behavior must fail closed or be reported as
  unavailable.
- route reports must not imply feature parity with llama.cpp or `mlx_lm`.

Acceptance criteria:

- MLX mode does not silently ignore requested sampling semantics.
- runtime reports expose enough capability metadata to distinguish preview,
  certified, and unsupported behavior.

### PR-7: Reference-Learned Features Must Be Sequenced By Risk

Required behavior:

- adopt reference-runtime lessons in an order that improves the current AX MLX
  path before expanding model or kernel scope.
- treat chunked KV growth as the first reference-learned implementation target.
- treat prompt-prefix reuse, batch cache primitives, KV quantization, and
  sliding-window cache layouts as follow-on work after the base cache contract
  is measured and stable.
- treat DFlash/recurrent rollback, TurboKV-style hybrid compression, and custom
  kernels as research items until AX has a model target and profiling evidence
  that justify them.

Acceptance criteria:

- each reference-inspired feature has a named trigger metric or model-support
  reason before implementation starts.
- no PR introduces a reference feature that expands runtime semantics without a
  capability report and correctness test.
- docs describe the feature as implemented, experimental, or reference-only.

## 8. Phased Delivery Plan

### Phase 0: Evidence Repair

Purpose: make performance measurement trustworthy before changing the hot path.

Tasks:

- add or remove the server speculative-disable flag so benchmark controls are
  real.
- record speculative mode and acceptance metrics.
- require benchmark subprocess failures to fail the run.
- label reference-only `mlx_lm` and SwiftLM rows clearly.
- preserve raw JSON artifacts for every summary.

Exit criteria:

- greedy and speculative MLX server benchmarks can run against the same model.
- summaries include raw run records and route metadata.
- no benchmark script advertises an unsupported server flag.

### Phase 1: KV Cache Growth Hardening

Purpose: target the highest-value likely bottleneck with measurement first.

Tasks:

- instrument K/V append time and sequence-length buckets.
- run short, medium, long-context MLX benchmarks.
- prototype chunked growth first, using the `mlx_lm` and `mlx-swift-lm`
  fixed-step cache pattern as the lowest-risk reference.
- keep speculative `trim_to` rollback correct.

Exit criteria:

- measured KV growth cost is reduced or proven not to be material.
- long-context rows no longer regress from repeated append allocation.
- correctness and deterministic tests still pass.

### Phase 2: Runner Concurrency And Batch Contract

Purpose: remove avoidable serialization and make batch behavior explicit.

Tasks:

- avoid holding `states` mutex during GPU work.
- define per-request state ownership during interleaved execution.
- add concurrent/interleaved request tests.
- decide whether true multi-item MLX batching is in scope for this milestone.
- if batching is in scope, model the contract after explicit batch-cache
  primitives such as prepare, merge, filter, extend, and extract.

Exit criteria:

- long prefill does not block unrelated request-state access.
- concurrent benchmark rows can be interpreted without guessing.
- docs accurately describe the supported batch behavior.

### Phase 3: Targeted Kernel/Fusion Work

Purpose: only optimize kernels after higher-level overhead is measured.

Tasks:

- profile MLX model hot ops after Phases 0-2.
- choose one kernel/fusion target with clear before/after criteria.
- wire `MlxMetalKernel` only if it beats MLX fast ops on the measured path.
- update docs to distinguish active runtime from future hooks.

Exit criteria:

- custom-kernel code is either active with proof or documented as inactive.
- before/after benchmark artifacts show the benefit and risk.

### Phase 4: Publication Gate

Purpose: make MLX performance claims suitable for users and release notes.

Tasks:

- run release-mode benchmark matrix.
- record host/model/quantization/prompt/decode/concurrency/commit metadata.
- update public docs only with evidence-backed claims.
- keep internal raw artifacts linked from planning notes.

Exit criteria:

- public claims match current implementation.
- raw artifacts exist for every summarized performance claim.
- MLX, llama.cpp, and reference-only paths are never blended.

## 9. Implementation Notes

Recommended first PR:

- Add a real `no_speculative_decode` runtime option across server args,
  session config, and `MlxRunner`.
- Keep `scripts/bench_mlx_inference_stack.py` as the canonical MLX
  model-inference harness, with separate greedy and speculative result labels.
- Add a small unit or CLI test that catches future flag drift.

Recommended second PR:

- Add KV append instrumentation without changing layout.
- Run the canonical MLX scenario matrix and inspect whether concatenate growth
  is the actual bottleneck.

Recommended third PR:

- Implement the lowest-risk KV growth improvement supported by the measurements.

Recommended reference-followup PRs:

- Add prompt-cache and prefix-reuse design notes after chunked KV lands,
  separating active request cache from retained LRU prompt cache.
- Add a runner batch contract using explicit cache merge/extract/filter
  primitives before optimizing shared GPU calls.
- Add sampler/logit-processor capability reporting before implementing
  GPU-resident penalties or logprob surfaces.
- Add measured wired-memory budgeting after KV byte accounting exists.
- Defer KV quantization, sliding-window cache layout, DFlash rollback, and
  custom kernels until their trigger metrics are met.

Risk control:

- do not combine benchmark semantics, KV layout, runner lock changes, and custom
  kernel work in one PR.
- every performance PR must include before/after numbers and a correctness gate.

## 10. Open Questions

- Should `no_speculative_decode` be a server-only benchmark/debug flag, or part
  of public SDK config?
- What is the minimum host baseline for publication: M4, M4 Pro, M4 Max, or M5?
- Should MLX mode advertise single-request latency first, or batch throughput?
- Does the current model artifact format have enough metadata to choose KV
  layout thresholds safely?
- Which public doc should carry the first evidence-backed MLX performance
  table after this PRD exits?
- Which model family should trigger the first non-dense cache layout:
  sliding-window Gemma, Qwen3.5 hybrid attention, or long-context Qwen?
