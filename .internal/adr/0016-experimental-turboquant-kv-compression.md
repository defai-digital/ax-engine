# ADR 0016: Experimental TurboQuant KV Compression for MLX Runtime

Status: Accepted
Date: 2026-05-06

## Context

ADR 0012 established the current shipping runtime contract:

- explicit MLX mode routes to the repo-owned `ax-engine-mlx` runtime
- explicit `mlx_lm_delegated` routes to upstream `mlx-lm` text compatibility
- non-MLX inference routes to delegated `llama.cpp`
- AX native Metal mode is no longer a user-facing runtime

ADR 0013 then staged MLX KV cache improvement work. It placed
INT4/Hadamard/TurboQuant-class methods behind observability, prefix reuse,
sliding-window trimming, and conservative compression gates.

The local reference implementations now give enough implementation evidence to
start a concrete TurboQuant research track:

- `.internal/reference/turboquant` implements the paper-oriented split:
  TurboQuant-style keys, group-quantized values, and a recent unquantized buffer.
- `.internal/reference/vllm` implements TurboQuant as an attention backend with
  fused store and decode kernels over a block/page cache layout.
- `.internal/reference/SwiftLM` argues for a hot fp16 window plus compressed cold
  history, and treats QJL as optional K-only correction rather than the default
  path.
- `.internal/reference/llama-cpp-turboquant` treats TurboQuant as cache dtype
  policy, requires flash attention, and fails closed for unsupported dimensions
  or incompatible runtime modes.

The references also show a repeated failure mode: compressing KV but
dequantizing the whole history before attention can save storage while failing
to improve, or even regressing, decode throughput.

## Decision

AX Engine will implement TurboQuant-class KV compression only as an experimental
MLX runtime capability, not as a backend route or public product switch.

The accepted implementation direction is:

1. Keep `selected_backend=mlx`.
2. Add internal compression policy and benchmark metadata before any user-facing
   docs.
3. Start with non-QJL MSE/FP8-style keys plus 4-bit values.
4. Keep QJL behind a separate research gate.
5. Compress only full-attention SDPA K/V layers in the first version.
6. Keep linear-attention recurrent state full precision and separate.
7. Preserve a hot full-precision KV window for recent tokens.
8. Require fused compressed decode attention before any performance promotion.

The first candidate presets are:

- `TurboQuantK8V4`: conservative bridge mode; FP8/8-bit keys and 4-bit values.
- `TurboQuantK4V4`: MSE/Lloyd-Max 4-bit keys with 4-bit values.
- `TurboQuantK3V4Research`: lower-bit key research mode, not a default.

The initial public product surface remains unchanged. The first implemented
switch is explicitly experimental and accounting-only:
`--experimental-mlx-kv-compression turboquant-shadow`. It does not change K/V
storage, SDPA inputs, logits, sampling, or cache mutation. It only reports
eligibility and estimated cold-cache savings so later compressed storage and
fused decode work can be validated without changing default behavior.

The next implemented slice is a CPU-only scalar reference codec. It provides the
correctness oracle for future store/decode kernels, but it remains disconnected
from runtime KV storage and attention until a compressed storage path is
explicitly enabled.

The store-side prototype is also CPU-only at this stage:
`TurboQuantKvPrototypeStore` keeps recent tokens full precision, evicts older
tokens into compressed cold records, and can reconstruct K/V for deterministic
debug comparison. This is accepted as a storage-contract prototype, not as the
runtime `MlxKVCache` storage mode.

The runtime guardrail is accepted before runtime storage activation. AX computes
a per-layer TurboQuant support report from `ModelConfig` and classifies
full-attention, linear-attention, sliding-window, KV-shared, and unsupported
head-dimension layers. The runner feeds that eligibility mask only into
experimental shadow metadata. Unsupported layers are fail-closed in estimates,
and the default disabled path still emits no compression metadata and changes no
generation behavior.

The decode-attention oracle is accepted as test infrastructure before any fused
kernel work. `TurboQuantKvPrototypeStore::debug_decode_attention` reconstructs
prototype K/V and runs CPU-only scaled dot-product decode attention, allowing
future compressed decode kernels to compare against a deterministic contract.
This oracle is intentionally not wired into MLX arrays, runtime SDPA, or
generation.

The compressed layout contract is accepted before MLX allocation or fused kernel
work. `TurboQuantBlockLayout` defines fixed-size token blocks with per-token,
per-KV-head slots, including packed K payload, K norm metadata, packed V payload,
V group min/scale metadata, and 16-byte slot alignment. This contract gives
future store/decode kernels deterministic address mapping without changing
runtime KV storage.

The compressed block writer prototype is accepted as CPU-only contract evidence.
`TurboQuantCompressedBlockBuffer` writes compressed token/head slots into a
`Vec<u8>` according to `TurboQuantBlockLayout`, reads the quantized slot back,
and can reconstruct debug vectors. It tracks written slots and fails closed for
unwritten reads or invalid dimensions, but remains disconnected from MLX
allocation, runtime KV storage, SDPA, and generation.

The compressed head decode oracle is accepted as the next contract layer.
`TurboQuantCompressedBlockBuffer::debug_decode_attention_for_head` reconstructs
one KV head over a caller-specified token history and runs the scalar reference
attention oracle. This gives future fused compressed decode kernels a
per-head comparison target while still avoiding MLX allocation, runtime KV
storage changes, SDPA integration, or generation behavior changes.

The compressed all-head decode oracle extends that contract without changing the
runtime boundary. `debug_decode_attention_for_all_heads` accepts one query vector
per KV head and returns one decoded output per head by reusing the per-head
oracle. It fails closed on query/head-count mismatch and remains disconnected
from MLX allocation, runtime KV storage, SDPA, and generation.

The decode comparison report is accepted as the first quality-gate contract for
future kernels. `compare_decode_outputs` reports per-head and aggregate error
metrics for expected versus actual all-head decode outputs, and
`debug_compare_attention_for_all_heads` compares compressed-buffer decode output
against a caller-provided full-precision oracle. The comparison remains CPU-only
and does not change runtime KV storage, SDPA, or generation behavior.

The decode quality gate is accepted as the pass/fail layer above comparison
metrics. `TurboQuantDecodeQualityGate` evaluates aggregate max absolute error,
mean absolute error, and minimum cosine similarity together, returning
`TurboQuantDecodeQualityDecision` with individual condition results. This keeps
future benchmark and fused-kernel promotion criteria explicit while remaining
CPU-only and disconnected from generation behavior.

Named decode quality gate presets are accepted as benchmark and promotion
contracts, not as public quality claims. The initial set separates strict debug,
K8V4 reference, and loose research thresholds so tests and future benchmark
harnesses share one vocabulary without enabling TurboQuant in runtime KV
storage, SDPA, or generation.

## Rationale

TurboQuant is a KV cache storage and attention-kernel policy. Treating it as a
backend would conflict with ADR 0012 and blur AX-owned MLX evidence with
delegated runtime evidence.

The vLLM reference is the strongest implementation guide because it models the
actual hot path:

- prefill can use raw Q/K/V attention
- store writes compressed K/V slots
- decode reads compressed K/V directly
- continuation-prefill avoids full-cache dequantization for small chunks

The SwiftLM and Python references add two product guardrails:

- short contexts should stay full precision
- compression telemetry must report saved bytes and active policy

The llama.cpp branch adds fail-closed discipline:

- cache quantization requires a compatible attention path
- unsupported dimensions and layouts are rejected rather than silently degraded

QJL is deliberately not accepted as the first runtime default. The paper and
some references include QJL, but the vLLM reference omits it after quality
experience, and SwiftLM treats it as K-only additive correction. AX should first
establish a stable non-QJL compressed attention baseline, then compare QJL as an
isolated experiment.

## Architecture Rules

- TurboQuant work belongs in `ax-engine-mlx` and `mlx-sys`, not
  `ax-engine-sdk` backend resolution.
- `mlx_lm_delegated` and `llama_cpp` may expose their own delegated behavior,
  but that is not AX MLX TurboQuant evidence.
- Compression support is per model family, per attention kind, and per head
  dimension.
- Full-attention K/V, sliding-window K/V, KV-shared layers, and linear recurrent
  state must have explicit storage policies.
- The default path remains full precision until an internal compression policy is
  explicitly enabled.
- Public claims require model-specific benchmark artifacts.

## Required Metadata

Benchmark and route artifacts must expose:

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
- selected backend and support tier

The first implementation may encode numeric values through route
`crossover_decisions`; string policy/status fields may require a typed metadata
extension.

When compression is disabled, AX must not emit these compression fields. This
keeps the existing default route metadata shape and avoids surprising consumers
that compare metadata keys.

## Consequences

- ADR 0013 is refined: TurboQuant-class work can begin now, but only on the
  experimental/fused-kernel-gated track described here.
- AX avoids a misleading storage-only "TurboQuant support" claim.
- The first useful runtime target is fused decode attention, not only a codec.
- The scalar codec is accepted as test infrastructure, not runtime support.
- The prototype compressed store is accepted as storage-contract evidence, not
  generation behavior.
- Quality gates become part of the feature, not a follow-up.
- The public API surface remains stable while internal evidence accumulates.

## Alternatives Considered

### Add a public TurboQuant switch immediately

Rejected. The current MLX cache path still calls normal SDPA with full K/V
arrays. A public switch before a fused compressed-attention path would imply a
maturity level that does not exist.

### Implement only a compress/dequant wrapper around `MlxKVCache`

Rejected as a promoted runtime path. It is acceptable for scalar tests and
debugging, but storage-only compression does not prove decode performance and can
increase overhead.

### Follow the paper exactly with QJL first

Rejected as the first default. QJL remains available for research, but the
reference set is mixed and attention softmax can amplify residual variance.

### Delegate TurboQuant to llama.cpp or vLLM

Rejected for AX-owned MLX mode. Delegated runtimes can be studied, but AX MLX
performance claims must come from `ax-engine-mlx`.

### Start with value-only quantization

Deferred. Value-only compression may be useful, but the reference evidence
suggests attention-score correctness depends most heavily on K. AX should design
the K/V slot layout together even if presets vary precision.

## Implementation Plan

Phase 1: Design and metadata

- add `.internal/planning/MLX-TURBOQUANT-KV-PRD.md`
- define internal policy/status names
- add benchmark metadata requirements

Phase 2: Scalar reference codec

- implement Rust reference codec and byte accounting
- add deterministic unit tests
- keep it disconnected from default MLX runtime behavior

Phase 3: Compressed storage prototype

- add full-attention cold-history compressed storage behind an internal policy
- keep hot recent tokens full precision
- keep linear/recurrent state and unsupported layers full precision

Phase 4: Fused decode attention

- add Metal/MLX custom kernel path for packed cold K/V plus hot full-precision
  K/V
- validate against the scalar reference on small shapes
- benchmark before any promotion

Phase 5: Promotion

- run long-context model-specific quality gates
- update public docs only for modes that pass
- keep experimental modes internal

## Related Documents

- ADR 0003: Paged KV and Prefix Caching
- ADR 0012: Retire AX Native Mode and Route Inference to MLX or llama.cpp
- ADR 0013: MLX KV Cache Improvement Strategy
- `.internal/planning/MLX-TURBOQUANT-KV-PRD.md`
