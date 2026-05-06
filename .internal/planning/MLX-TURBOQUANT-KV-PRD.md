# MLX TurboQuant KV Cache PRD

Status: Draft
Date: 2026-05-06
Owner: AX Engine

## 1. Summary

AX Engine will evaluate and implement TurboQuant-class KV cache compression as an
experimental capability inside the repo-owned MLX runtime.

TurboQuant is not a new backend and not a delegated runtime. It is an MLX KV
cache storage and attention-kernel policy under `selected_backend=mlx`.

The first product goal is:

> Prove whether AX Engine can reduce full-attention KV memory for long-context
> MLX workloads without weakening correctness, decode throughput, or the ADR
> 0012 routing contract.

The implementation must be staged:

1. Reference review and scalar codec validation.
2. Internal cache policy and benchmark metadata.
3. Store-side compressed KV prototype.
4. Fused decode attention over compressed KV.
5. Long-context quality and promotion gates.

Implemented safety slice on 2026-05-06:

- Added `MlxKvCompressionConfig`, defaulting to `Disabled`.
- Added `TurboQuantShadow` as an opt-in accounting-only policy.
- Added server CLI switch:
  `--experimental-mlx-kv-compression turboquant-shadow`.
- Added hot-window and minimum-context knobs for the experimental policy.
- Added MLX KV cache eligibility and byte-estimation counters.
- Kept K/V storage, SDPA inputs, logits, sampling, and cache mutation unchanged.
- Kept compression route counters absent when the policy is disabled.

Implemented Phase 2 reference slice on 2026-05-06:

- Added a CPU-only scalar TurboQuant reference codec in `ax-engine-mlx`.
- Added normalized Hadamard rotation for power-of-two head dimensions.
- Added pluggable centroid lookup and default symmetric K3/K4/K8 centroid
  tables for reference testing.
- Added 3-bit, 4-bit, and 8-bit index packing helpers.
- Added group-wise uniform 4-bit value quantization helpers.
- Added packed-byte accounting helpers for K/V slot estimates.
- Kept the codec disconnected from runtime KV storage and attention.

Implemented Phase 4 prototype slice on 2026-05-06:

- Added a CPU-only `TurboQuantKvPrototypeStore`.
- Kept recent tokens in a full-precision hot window.
- Evicted older tokens into compressed cold K/V records using the scalar codec.
- Added debug reconstruction for cold compressed plus hot full-precision tokens.
- Added prototype stats for cold/hot tokens, full-precision bytes, compressed
  bytes, saved bytes, and compression ratio.
- Kept the prototype disconnected from `MlxKVCache`, MLX arrays, and SDPA.

Implemented runtime guardrail slice on 2026-05-06:

- Added a per-layer TurboQuant support report derived from `ModelConfig`.
- Classifies eligible full-attention layers separately from linear-attention,
  sliding-window, KV-shared, and unsupported-head-dimension layers.
- Feeds the eligibility mask into shadow byte-estimation metadata only.
- Keeps compression disabled by default, and keeps runtime KV storage, SDPA
  inputs, logits, sampling, and cache mutation unchanged.

Implemented reference decode oracle slice on 2026-05-06:

- Added CPU-only scaled dot-product decode attention for reconstructed K/V
  tokens.
- Added `TurboQuantKvPrototypeStore::debug_decode_attention` for deterministic
  compressed-store versus full-precision comparisons.
- Added fail-closed validation for empty KV history and dimension mismatches.
- Kept the oracle disconnected from MLX arrays, runtime SDPA, and generation.

Implemented compressed layout contract slice on 2026-05-06:

- Added a CPU-only `TurboQuantBlockLayout` contract for future compressed decode
  kernels.
- Defines per-token, per-KV-head compressed slots inside fixed token blocks.
- Accounts for packed K payload, K norm metadata, packed V payload, V group
  min/scale metadata, and 16-byte slot alignment.
- Added deterministic block/token/head address mapping and buffer sizing.
- Added fail-closed validation for invalid block/head/group dimensions and
  out-of-range slot addresses.
- Kept the layout disconnected from MLX allocation, runtime KV storage, and
  generation.

Implemented compressed block writer prototype slice on 2026-05-06:

- Added a CPU-only `TurboQuantCompressedBlockBuffer` backed by `Vec<u8>`.
- Writes one compressed token/head slot into the `TurboQuantBlockLayout`
  contract.
- Reads back quantized K/V slots and reconstructs full-precision debug vectors.
- Tracks written slots and fails closed for unwritten reads, wrong vector
  dimensions, and out-of-range head indices.
- Keeps the writer disconnected from MLX allocation, runtime KV storage, SDPA,
  and generation.

Implemented compressed head decode oracle slice on 2026-05-06:

- Added CPU-only per-head decode attention over `TurboQuantCompressedBlockBuffer`.
- Reconstructs the requested head history for a caller-specified token count.
- Reuses the scalar reference attention oracle for deterministic output.
- Fails closed for invalid heads, missing slots inside the requested history,
  empty histories, and query shape errors.
- Keeps the oracle disconnected from MLX allocation, runtime KV storage, SDPA,
  and generation.

Implemented compressed all-head decode oracle slice on 2026-05-06:

- Added CPU-only all-head decode attention over `TurboQuantCompressedBlockBuffer`.
- Accepts one query vector per KV head and returns one decoded output per head.
- Reuses the per-head compressed decode oracle, preserving deterministic
  comparison behavior for future fused kernels.
- Fails closed when the query head count does not match the compressed layout.
- Keeps the oracle disconnected from MLX allocation, runtime KV storage, SDPA,
  and generation.

Implemented decode comparison report slice on 2026-05-06:

- Added CPU-only comparison metrics for expected versus actual all-head decode
  outputs.
- Reports per-head max absolute error, mean absolute error, and cosine
  similarity.
- Reports aggregate max absolute error, mean absolute error, and minimum cosine
  similarity across heads.
- Added a compressed-buffer comparison helper that compares compressed all-head
  decode output against caller-provided full-precision oracle output.
- Fails closed for output head-count and vector-dimension mismatches.
- Keeps the comparison contract disconnected from MLX allocation, runtime KV
  storage, SDPA, and generation.

Implemented decode quality gate slice on 2026-05-06:

- Added `TurboQuantDecodeQualityGate` for deterministic pass/fail evaluation of
  decode comparison reports.
- Requires aggregate max absolute error, mean absolute error, and minimum cosine
  similarity to pass together.
- Returns `TurboQuantDecodeQualityDecision` with individual condition results
  and observed-versus-limit values.
- Keeps the quality gate disconnected from MLX allocation, runtime KV storage,
  SDPA, and generation.

Implemented decode quality preset slice on 2026-05-06:

- Added named `TurboQuantDecodeQualityGate` presets for strict debug checks,
  K8V4 reference promotion checks, and loose research exploration.
- Replaced the compressed-buffer oracle test's ad hoc K8V4 threshold with the
  reference preset.
- Added an ordering test so stricter presets cannot accidentally become more
  permissive than promotion or research gates.
- Keeps the presets disconnected from MLX allocation, runtime KV storage, SDPA,
  generation, and user-visible defaults.

Implemented decode quality profile mapping slice on 2026-05-06:

- Added `TurboQuantDecodeQualityProfile` as the named layer above gate
  thresholds.
- Mapped `K8V4` to the reference profile and kept `K4V4` / `K3V4Research` on
  the loose research profile until model-specific quality evidence promotes
  them.
- Added tests for profile-to-gate and quantization-preset-to-profile mapping.
- Keeps the mapping disconnected from MLX allocation, runtime KV storage, SDPA,
  generation, and user-visible defaults.

Implemented decode quality evaluation slice on 2026-05-06:

- Added `TurboQuantDecodeQualityEvaluation` as the typed output for applying a
  quantization preset to a decode comparison report.
- Added `evaluate_decode_quality_for_preset` so benchmark and promotion harnesses
  receive the preset, mapped profile, gate thresholds, and pass/fail decision
  from one CPU-only contract.
- Added tests proving K8V4 uses the reference gate while aggressive presets stay
  on the research gate until separate quality evidence promotes them.
- Keeps the evaluation contract disconnected from MLX allocation, runtime KV
  storage, SDPA, generation, and user-visible defaults.

Implemented compressed-buffer quality check slice on 2026-05-06:

- Added `TurboQuantDecodeQualityCheck` to carry the decode comparison report and
  preset quality evaluation together.
- Added `debug_evaluate_attention_quality_for_all_heads` on the compressed block
  buffer so benchmark/debug callers can compare reconstructed attention output
  and evaluate it through the layout's quantization preset in one step.
- Added tests covering direct quality-check construction and compressed-buffer
  quality evaluation.
- Keeps the helper disconnected from MLX allocation, runtime KV storage, SDPA,
  generation, and user-visible defaults.

Implemented compressed decode planning slice on 2026-05-06:

- Added `TurboQuantCompressedDecodePlan` as the CPU-only launch/coverage
  contract for future fused compressed decode kernels.
- Splits a decode history into compressed cold tokens and full-precision hot
  tokens using the configured hot window.
- Reports compressed block count, buffer bytes, required compressed slot writes,
  decode path, and quality profile derived from the layout preset.
- Fails closed for empty decode histories and falls back to full-precision-only
  when the full history remains inside the hot window.
- Keeps the plan disconnected from MLX allocation, runtime KV storage, SDPA,
  generation, and user-visible defaults.

Implemented compressed decode coverage guard slice on 2026-05-06:

- Added `TurboQuantCompressedDecodePlan::validate_compressed_buffer` to verify
  that a compressed buffer contains every cold token/head slot required by the
  plan before a future fused decode path can consume it.
- Fails closed for mismatched compressed-buffer layouts and incomplete cold-slot
  coverage.
- Skips compressed coverage validation for hot-window-only decode plans because
  they do not require compressed decode.
- Keeps the guard disconnected from MLX allocation, runtime KV storage, SDPA,
  generation, and user-visible defaults.

Implemented compressed decode input validation slice on 2026-05-06:

- Added `TurboQuantCompressedDecodePlan::validate_queries` for per-head query
  count and head-dimension validation before future compressed decode launches.
- Added `validate_decode_inputs` to compose query validation with compressed
  cold-slot coverage validation.
- Fails closed for query head-count mismatch and query-dimension mismatch before
  any compressed-buffer read.
- Keeps the validation disconnected from MLX allocation, runtime KV storage,
  SDPA, generation, and user-visible defaults.

Implemented compressed decode readiness report slice on 2026-05-06:

- Added `TurboQuantCompressedDecodeReadiness` as the CPU-only launch metadata
  report above future fused compressed decode kernels.
- Added `TurboQuantCompressedDecodePlan::decode_readiness` to validate query
  shape, compressed cold-slot coverage, decode path, buffer bytes, slot counts,
  and quality profile in one deterministic contract.
- Keeps hot-window-only plans independent from compressed buffer layout checks,
  matching the existing fallback semantics.
- Keeps the report disconnected from MLX allocation, runtime KV storage, SDPA,
  generation, and user-visible defaults.

Implemented fused decode candidate gate slice on 2026-05-06:

- Added preset, key-bit, and value-bit metadata to compressed decode readiness.
- Added `TurboQuantFusedDecodeCandidate` and status values for candidate,
  full-precision-only, unsupported head dimension, and unsupported preset.
- Gates the first fused decode candidate to compressed-path `K8V4` with
  `head_dim=128`, matching the initial promotion target while keeping lower-bit
  presets behind future quality work.
- Keeps the gate as CPU-only metadata; it does not launch a kernel, allocate MLX
  storage, change SDPA, change generation, or expose a user-facing switch.

Implemented fused decode launch descriptor slice on 2026-05-06:

- Added `TurboQuantFusedDecodeLaunchDescriptor` as the deterministic kernel
  geometry contract for the first fused decode candidate.
- Added `TurboQuantCompressedDecodePlan::fused_decode_launch_descriptor` to
  compose input readiness, candidate gating, and slot layout offsets/strides.
- Reports block tokens, KV heads, head dimension, compressed block bytes, value
  grouping, per-head slot size, token stride, block size, and payload offsets.
- Fails closed for non-candidates before producing a descriptor.
- Keeps the descriptor CPU-only; it does not launch a kernel, allocate MLX
  storage, change SDPA, change generation, or expose a user-facing switch.

Implemented fused decode workload accounting slice on 2026-05-06:

- Added per-slot payload byte counts to the fused decode launch descriptor.
- Added `TurboQuantFusedDecodeLaunchWorkload` and descriptor `workload()` for
  deterministic cold-score, hot-score, output-element, compressed-byte, and
  hot full-precision byte accounting.
- Reports raw compressed slot bytes separately from aligned slot bytes so future
  benchmark rows can distinguish logical read volume from layout allocation.
- Keeps the accounting CPU-only; it does not claim throughput, launch a kernel,
  allocate MLX storage, change SDPA, change generation, or expose a user-facing
  switch.

Implemented fused decode savings accounting slice on 2026-05-06:

- Added full-precision cold and total K/V byte accounting to fused decode
  workload estimates.
- Added estimated cold saved bytes, estimated total saved read bytes, and cold
  compression ratio in milli-units.
- Added shared full-precision K/V byte and compression-ratio helpers so future
  benchmark rows use the same accounting contract as the launch descriptor.
- Keeps the savings accounting CPU-only; it does not claim throughput, launch a
  kernel, allocate MLX storage, change SDPA, change generation, or expose a
  user-facing switch.

Implemented fused decode benchmark estimate slice on 2026-05-06:

- Added `TurboQuantFusedDecodeBenchmarkEstimate` as the benchmark-facing view of
  the fused decode launch descriptor and workload accounting.
- Added descriptor `benchmark_estimate()` to report preset, key/value bits,
  hot/cold token split, KV-head shape, compressed blocks, score/output element
  counts, KiB-rounded full-precision/compressed/read/saved bytes, and cold
  compression ratio.
- Keeps KiB rounding inside the CPU-only contract so future benchmark metadata
  can avoid re-deriving these fields differently.
- Keeps the estimate disconnected from runner route metadata for now; it does
  not launch a kernel, allocate MLX storage, change SDPA, change generation, or
  expose a user-facing switch.

Implemented fused decode promotion readiness slice on 2026-05-06:

- Added `TurboQuantFusedDecodePromotionReadiness` as the internal evidence gate
  above launch descriptor and benchmark estimate accounting.
- Added descriptor `promotion_readiness()` to require matching quality preset,
  passing decode quality gate, and positive cold-history savings before marking
  a fused decode candidate as ready for promotion evidence.
- Added fail-closed status values for quality-preset mismatch, quality-gate
  failure, and no cold savings.
- Keeps the readiness contract CPU-only and internal; it does not launch a
  kernel, allocate MLX storage, change SDPA, change generation, publish route
  metadata, or expose a user-facing switch.

Implemented fused decode promotion evidence slice on 2026-05-06:

- Added `TurboQuantFusedDecodePromotionEvidence` as the promotion artifact view
  above readiness, benchmark estimate, and decode quality gate decisions.
- Added descriptor `promotion_evidence()` to report readiness plus quality pass
  status, max/mean absolute error, cosine similarity, and their gate limits.
- Keeps evidence generation CPU-only and internal; it does not launch a kernel,
  allocate MLX storage, change SDPA, change generation, publish route metadata,
  or expose a user-facing switch.

Implemented fused decode promotion evidence summary slice on 2026-05-06:

- Added stable numeric codes for decode quality profiles and fused decode
  promotion statuses.
- Added `TurboQuantFusedDecodePromotionEvidenceSummary` and evidence `summary()`
  for artifact-friendly ready/status/preset/profile codes, quality pass status,
  micro-unit quality metrics, saved KiB, and compression ratio.
- Keeps summary generation CPU-only and internal; it does not launch a kernel,
  allocate MLX storage, change SDPA, change generation, publish route metadata,
  or expose a user-facing switch.

Implemented production readiness cutoff slice on 2026-05-06:

- Added `TurboQuantProductionRequirements` and `TurboQuantProductionReadiness`
  as the fail-closed production gate above the CPU/reference foundation.
- Requires all five production gates before TurboQuant can be called runtime
  complete:
  - real fused decode kernel
  - runtime KV storage integration
  - runner route metadata integration
  - long-context benchmark and model-quality artifact
  - public switch plus public support docs
- Defaults to blocked until every gate is explicitly satisfied.
- Keeps the current implementation optional and non-invasive; no kernel is
  launched, no MLX storage changes, no SDPA/generation change, no route metadata
  publication, and no public support claim is introduced by this cutoff.

Implemented runner route metadata gate slice on 2026-05-06:

- Added route metadata schema/version, production-ready, and production-blocker
  counters for the internal `turboquant-shadow` path.
- Marks only the runner route metadata production gate present; fused kernel,
  runtime KV storage, long-context benchmark/model-quality artifact, and public
  switch/docs gates remain blocked.
- Keeps the default disabled path silent: TurboQuant compression metadata is
  emitted only when the experimental shadow policy is explicitly selected.
- Still does not allocate compressed KV storage, change logits, alter SDPA, or
  expose a public support claim.

Implemented runtime shadow storage gate slice on 2026-05-06:

- Added `MlxKVCache`-owned TurboQuant shadow compressed storage for eligible
  full-attention cold tokens.
- The storage path writes cold K/V token heads into the existing
  `TurboQuantCompressedBlockBuffer` and tracks runtime storage layers,
  token-layers, bytes, and written slots in route metadata.
- Marks the runtime KV storage and runner route metadata gates present for the
  internal shadow path.
- Keeps decode on the full-precision cache; the compressed storage is prepared
  for future fused decode integration but is not consumed by SDPA yet.
- Keeps the default disabled path silent and allocation-free.
- Remaining production blockers: real fused MLX/Metal decode kernel,
  long-context benchmark/model-quality artifact, and public switch/docs.

Implemented fused Metal compressed cold decode kernel slice on 2026-05-06:

- Added an MLX fast Metal kernel for K8/V4 compressed cold-token decode.
- The kernel reads `TurboQuantCompressedBlockBuffer` bytes, decodes K8 key
  centroids plus norms and V4 value groups, computes score/softmax/value
  accumulation in Metal, and returns per-head Float32 decode outputs.
- Added a launch test comparing the Metal output against the existing CPU
  reference decode for a 128-dim K8/V4 descriptor.
- Marks the fused kernel, runtime KV storage, and runner route metadata gates
  present for the internal shadow/fused path.
- Keeps the production decode path on full precision until long-context
  model-quality artifacts and public switch/docs are approved.
- Remaining production blockers: long-context benchmark/model-quality artifact
  and public switch/docs.

Implemented quality artifact contract slice on 2026-05-06:

- Added `scripts/check_turboquant_quality_artifact.py` as the fail-closed
  validator for long-context, model-level TurboQuant promotion artifacts.
- Added unit tests covering passing artifacts, short-context rejection, quality
  metric regression rejection, missing runtime storage metadata rejection, and
  separation from public-doc approval.
- Added `.internal/turboquant/QUALITY-GATE-ARTIFACT.md` as the internal artifact
  contract.
- Requires the initial promoted artifact to use an MLX full-precision baseline,
  a fused compressed K8/V4 candidate, at least 8192 context tokens, at least
  128 generation tokens, positive KV savings, positive runtime compressed slot
  writes, decode throughput ratio >= 0.85, and `reference_k8v4` quality limits.
- Keeps the production readiness gate blocked until a real model artifact passes
  this validator and public support docs are approved.

Implemented benchmark harness metadata capture slice on 2026-05-06:

- Extended `scripts/bench_mlx_inference_stack.py` with pass-through support for
  `--experimental-mlx-kv-compression turboquant-shadow` and the matching
  hot-window/min-context knobs.
- Added TurboQuant KV compression route telemetry extraction and per-row
  summaries for the MLX inference-stack benchmark artifact.
- Keeps compression metadata silent when the policy is disabled.
- Keeps the switch experimental; this is benchmark evidence capture, not public
  production enablement.

Implemented experimental switch documentation slice on 2026-05-06:

- Added public docs for the existing
  `--experimental-mlx-kv-compression turboquant-shadow` server switch and the
  benchmark harness pass-through.
- Documents that the mode is disabled by default, keeps generation on the
  full-precision MLX KV path, and is benchmark/telemetry evidence only.
- Keeps production support blocked until a real quality artifact passes and the
  public support claim is explicitly approved.

Implemented quality artifact builder slice on 2026-05-06:

- Added `scripts/build_turboquant_quality_artifact.py` to compile a promotion
  artifact from a full-precision benchmark artifact, a TurboQuant candidate
  benchmark artifact, and a quality-metrics JSON file.
- The builder computes input artifact SHA-256 values, checks matching prompt
  hashes, carries route metadata forward, derives decode throughput ratio, and
  runs the fail-closed validator before writing the artifact.
- Candidate rows labelled `full_precision_shadow` are rejected and cannot be
  promoted as fused compressed decode evidence.

Implemented quality metrics builder slice on 2026-05-06:

- Added `scripts/build_turboquant_quality_metrics.py` to compare same-shaped
  baseline and candidate decode-output vectors.
- Emits `max_abs_diff`, `mean_abs_diff`, and `min_cosine_similarity` in the JSON
  shape consumed by the quality artifact builder.
- Fails closed for empty vectors, non-numeric/non-finite values, output-count
  mismatches, and vector-dimension mismatches.

Implemented quality gate CLI smoke slice on 2026-05-06:

- Added `scripts/check-turboquant-quality-gate.sh` as a lightweight end-to-end
  check for the TurboQuant artifact pipeline.
- The smoke builds synthetic decode quality metrics, compiles a quality
  artifact, validates it, and proves `full_precision_shadow` candidates fail
  promotion.
- Added the smoke to `scripts/check-scripts.sh`.

Implemented promotion readiness report slice on 2026-05-06:

- Added `scripts/check_turboquant_promotion_readiness.py` to scan local model
  manifests and saved quality-gate artifacts before any public support claim.
- The report currently blocks promotion because the local model set uses
  `attention_head_dim=256` and/or grouped-query, linear-attention, or MLA
  layouts, while the first fused K8/V4 promotion gate accepts only
  `head_dim=128`, non-GQA full-attention decode with `fused_compressed_decode`
  successes and zero fallbacks.
- This makes the correct current decision explicit: public docs remain
  experimental until either a qualifying `head_dim=128` model artifact passes,
  or the fused kernel, runtime gate, and quality artifact validator are extended
  and revalidated for 256-dim/GQA model families.

## 2. Reference Lessons

The local reference implementations point in the same architectural direction
but disagree on aggressive details.

### 2.1 `.internal/reference/turboquant`

Useful lessons:

- The reference implements the paper shape directly: key vectors use MSE
  TurboQuant plus optional QJL residual correction; value vectors use cheaper
  group quantization.
- It keeps a recent unquantized buffer and compresses older history.
- It explicitly warns that hybrid decode can save storage while still paying
  compute cost if it dequantizes all history.
- It shows that linear-attention layers are not compressible by the same full
  attention KV path, so total-model savings depend on the fraction of SDPA
  layers.

AX interpretation:

- Adopt the split between full-attention KV compression and linear/recurrent
  state exclusion.
- Do not count storage savings as throughput wins until the decode kernel reads
  compressed KV directly.
- Treat value precision as a quality-sensitive knob; 2-bit V is not the safe
  default.

### 2.2 `.internal/reference/vllm`

Useful lessons:

- TurboQuant is integrated as an attention backend, not as a generic tensor
  cache wrapper.
- The cache layout is block/page based:
  `(num_blocks, block_size, num_kv_heads, packed_slot_size)`.
- Store-side quantization is fused: rotate/normalize/bucketize/pack keys and
  quantize/pack values before writing the cache slot.
- Decode attention directly reads packed K/V, computes online softmax, and
  accumulates values without reconstructing the entire cache first.
- Prefill keeps the fast raw-QKV attention path where possible, then stores
  compressed K/V.
- QJL is intentionally omitted in this variant because multiple implementations
  reported softmax-quality problems from residual variance.
- The safer presets start with FP8/8-bit keys and 4-bit values before lower-bit
  key modes.

AX interpretation:

- The production-grade target must be fused compressed decode attention.
- Store and decode kernels are separate deliverables.
- The first kernelable mode should be non-QJL MSE/FP8-style keys plus 4-bit
  values, not the full paper QJL path.
- Block/page metadata is not optional for a scalable version.

### 2.3 `.internal/reference/SwiftLM`

Useful lessons:

- The SwiftLM design explicitly separates a hot fp16 window from compressed cold
  history.
- It argues that short 100-400 token tool calls should not pay compression
  latency.
- It treats QJL as K-cache-only additive correction rather than replacing useful
  MSE payload bits.
- It records compression ratio and saved bytes as telemetry, not just a hidden
  internal optimization.

AX interpretation:

- Add a hot-window policy before making compression always-on.
- Keep QJL behind a separate research gate even if the base MSE path succeeds.
- Expose memory-saved counters in benchmark artifacts.

### 2.4 `.internal/reference/llama-cpp-turboquant`

Useful lessons:

- Turbo cache types are cache dtypes, not model formats or backend routes.
- Turbo cache types require flash attention and fail closed for unsupported
  tensor-split or block-size cases.
- Head dimensions are padded/validated against fixed block constraints.
- The implementation adds CPU reference quantizers plus GPU/backend-specific
  fast paths.

AX interpretation:

- AX must fail closed for unsupported head dimensions, mixed layouts, and
  non-fused paths.
- A CPU/Rust reference codec is useful for tests, but not sufficient for runtime
  promotion.
- Public route metadata must make the cache dtype/policy observable.

## 3. Problem Statement

AX Engine currently stores full-attention K/V in MLX arrays and calls MLX
scaled-dot-product attention with full logical K/V views. This is simple and has
recently been improved with chunked backing buffers, but it cannot reduce KV
memory below the dtype size of those arrays.

TurboQuant-class compression can reduce full-attention KV memory, but only if AX
changes the cache and attention contract. A naive implementation that compresses
K/V and then dequantizes the whole history before every decode step would:

- save storage only transiently or partially,
- add decode overhead,
- miss the main long-context hot path,
- make benchmark claims misleading.

Therefore the product problem is not "add TurboQuant." The product problem is:

> Add an evidence-gated compressed-KV attention path that preserves AX MLX
> correctness and proves memory, throughput, and quality tradeoffs.

## 4. Goals

- Keep TurboQuant under `selected_backend=mlx`.
- Start as internal/experimental only.
- Compress only full-attention SDPA K/V layers in the first version.
- Preserve full precision for linear-attention recurrent state.
- Preserve a hot fp16/bf16 window for recent tokens.
- Add benchmark-visible KV compression policy, bytes, and status fields.
- Provide a CPU/Rust scalar codec for deterministic tests.
- Provide fused MLX/Metal store and decode kernels before runtime promotion.
- Require model-specific quality gates before public documentation or release
  claims.

## 5. Non-Goals

- No new backend route.
- No public `--turboquant` switch in the first implementation slice.
- No delegated vLLM, SwiftLM, or llama.cpp TurboQuant behavior as AX-owned MLX
  evidence.
- No compression of Qwen3.5 gated-delta recurrent state.
- No claim that storage savings imply decode speedup.
- No QJL in the first promoted runtime path.
- No broad model-family support without head-dim and attention-layout gates.

## 6. User And Product Value

### Long-context local inference

Users running long prompts or long conversations on Apple Silicon can fit more
context into the same memory budget if older full-attention K/V is compressed.

### Agent workloads

Agent sessions often combine repeated prefixes with long rolling context. A
hot-window plus cold-history compression policy can avoid penalizing short tool
turns while helping long sessions.

### Benchmark credibility

AX can report memory, throughput, and quality as separate outcomes instead of
collapsing them into a single "TurboQuant is faster" claim.

## 7. Requirements

### Phase 1: Reference-Grounded Design

- Document the selected AX interpretation of the local reference projects.
- Define internal enum names for compression policy and codec status.
- Define fail-closed support rules for model family, head dimension, attention
  kind, and KV sharing.
- Keep ADR 0013 intact but refine Phase 5 into a concrete TurboQuant track.

Acceptance:

- PRD and ADR exist under `.internal/`.
- No public docs claim TurboQuant support.

### Phase 2: Scalar Codec And Accounting

- Add a Rust scalar reference codec for:
  - Hadamard or orthogonal rotation.
  - Lloyd-Max centroid lookup for supported head dimensions.
  - 3-bit and 4-bit index packing.
  - optional norm correction.
  - uniform 4-bit value packing.
- Keep QJL out of the default codec; add only isolated test scaffolding if
  needed for comparison.
- Add byte accounting helpers for packed-slot size and hot/cold windows.

Acceptance:

- Unit tests cover pack/unpack round-trip.
- Unit tests cover centroid lookup boundaries.
- Unit tests compare MSE/cosine error against fixed vectors.
- No MLX runtime path uses the codec yet.

Status: implemented for CPU/reference use. The default K tables are symmetric
reference tables; paper-specific Lloyd-Max tables can replace them behind the
same centroid lookup contract when promoted.

### Phase 3: Internal Cache Policy

- Add an internal `MlxKvCompressionPolicy` with at least:
  - `Disabled`
  - `TurboQuantShadow`
  - future `TurboQuantK8V4`
  - future `TurboQuantK4V4`
  - future `TurboQuantK3V4Research`
- Add `compression_status` values:
  - `disabled`
  - `shadow_estimate_active`
  - `short_context`
  - `no_eligible_layer`
  - future `unsupported_model`
  - future `unsupported_head_dim`
  - future `unsupported_attention_kind`
  - future `fallback_full_precision`
  - future `compressed_storage_only`
  - future `fused_decode`
- Add hot-window configuration, defaulting to no compression for short contexts.
- Record metadata:
  - `ax_mlx_kv_compression_status`
  - `ax_mlx_kv_compression_preset`
  - `ax_mlx_kv_compression_key_bits`
  - `ax_mlx_kv_compression_value_bits`
  - `ax_mlx_kv_compression_eligible_layers`
  - `ax_mlx_kv_compression_candidate_token_layers`
  - `ax_mlx_kv_compression_hot_token_layers`
  - `ax_mlx_kv_compression_full_precision_kib`
  - `ax_mlx_kv_compression_estimated_compressed_kib`
  - `ax_mlx_kv_compression_estimated_saved_kib`
  - `ax_mlx_kv_compression_ratio_milli`

Acceptance:

- Benchmark artifacts expose the fields when the policy is enabled.
- Disabled default emits no compression fields and keeps the existing KV route
  metadata shape.
- Unsupported layouts fail closed and continue with full precision only when the
  policy explicitly allows fallback.

### Phase 4: Store-Side Compressed KV Prototype

- Store compressed cold K/V for full-attention layers.
- Keep the hot window in full precision.
- Keep KV-shared layers tied to their source-layer compression decision.
- Keep linear-attention state uncompressed.
- Materialize the cache through deterministic tests before attempting speed
  claims.

Acceptance:

- Memory accounting shows lower compressed cold-history bytes.
- A debug/dequant path reconstructs K/V for correctness comparison.
- Existing non-compression behavior remains unchanged by default.

Status: implemented as a CPU/reference prototype store. It is not a runtime
`MlxKVCache` storage mode yet, so it does not change generation behavior.

### Phase 5: Fused Decode Attention

- Add a Metal/MLX custom kernel path that reads packed cold K/V directly.
- Combine packed cold history with hot full-precision K/V.
- Implement online softmax over cold and hot regions without full-cache
  dequantization.
- Support only validated head dimensions initially, expected first target:
  `head_dim=128`.

Acceptance:

- Decode does not dequantize the full cold cache into a temporary full-precision
  K/V tensor.
- Benchmark rows include before/after memory and decode throughput.
- Kernel path has a CPU/reference comparison for a small deterministic shape.

### Phase 6: Quality And Promotion Gates

- Run long-context quality checks for each candidate model family.
- Compare against:
  - AX MLX full-precision KV baseline.
  - upstream `mlx_lm.benchmark` where applicable for throughput.
  - AX greedy/no-ngram mode for direct runtime comparisons.
- Promote only if the result satisfies policy-specific thresholds.

Acceptance:

- Each promoted mode has model-specific benchmark artifacts.
- Public docs describe only modes that passed gates.

Status: artifact validator implemented. No real long-context model artifact has
been accepted yet, so the production gate remains blocked. The MLX
inference-stack benchmark harness can now capture TurboQuant route metadata for
future candidate artifacts. Public docs expose only the experimental shadow
switch and do not claim production TurboQuant support. The quality artifact
builder can now compile a future fused candidate artifact while rejecting shadow
telemetry as promotion evidence. The metrics builder can now produce the
quality-metrics input from decode-output vectors without hand-written numbers.
The CLI pipeline smoke now exercises that flow without loading a model.

## 8. Metrics

- KV full-precision estimated KiB.
- KV compressed estimated KiB.
- KV saved KiB and compression ratio.
- Hot-window token count.
- Compressed cold-history token count.
- Prefill tok/s.
- Decode tok/s.
- TTFT.
- Peak process memory or MLX memory proxy.
- Attention-logit error against full precision.
- Output correctness for deterministic prompts.
- Long-context quality score or task pass rate.

## 9. Risks

- Low-bit K error is amplified by softmax.
- Low-bit V error may dominate perceptual quality even when K is near-lossless.
- QJL may improve unbiased inner products in isolation while hurting end-to-end
  attention quality.
- Storage-only compression can regress decode throughput.
- KV-shared Gemma layers can double-count or misread source-layer cache unless
  the source policy is explicit.
- Sliding-window and full-attention layers need different physical policies.
- MLX custom kernels may be harder to validate and package than pure MLX ops.

## 10. Open Questions

- Should the first fused kernel use pure Hadamard rotation or a fixed QR matrix?
- Should `TurboQuantK8V4` be considered TurboQuant or a conservative FP8/4-bit
  bridge mode?
- Which long-context quality harness should become canonical for AX MLX preview
  models?
- Should compression metadata remain in route `crossover_decisions`, or should
  AX add typed KV compression fields?
- What is the smallest useful hot window: 128, 256, or model-specific?

## 11. Related Documents

- ADR 0003: Paged KV and Prefix Caching
- ADR 0012: Retire AX Native Mode and Route Inference to MLX or llama.cpp
- ADR 0013: MLX KV Cache Improvement Strategy
- ADR 0016: Experimental TurboQuant KV Compression for MLX Runtime
- `.internal/planning/MLX-KV-CACHE-IMPROVEMENT-PRD.md`
- `.internal/planning/MLX-MODE-PERFORMANCE-PRD.md`
