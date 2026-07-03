# ADR-034: Evidence-Gated Decode Hot-Path Kernel Strategy

**Date:** 2026-07-03
**Status:** Accepted
**Deciders:** AX Engine MLX
**Related:** [PRD-2026-07-03-decode-hot-path-kernel-strategy.md](../prd/PRD-2026-07-03-decode-hot-path-kernel-strategy.md), [TECH-SPEC-2026-07-03-decode-hot-path-kernel-strategy.md](../tech-spec/TECH-SPEC-2026-07-03-decode-hot-path-kernel-strategy.md)

## Context

AX Engine's repo-owned MLX runtime currently delegates broad tensor operations to
MLX while owning runtime behavior above that graph: scheduling, speculation, KV
state, benchmark provenance, and selected custom kernels. This is the right
default. MLX already exposes custom Metal kernels, and the repo already wraps
that capability through `MlxMetalKernel`.

The question is not whether AX may ever own Metal kernels. It already does. The
question is how to decide which decode hot-path kernels are worth owning without
repeating low-leverage or negative fusion work.

Recent evidence matters:

- The fused residual-add + RMSNorm sidecar regressed decode and prefill against
  MLX's split path.
- The fused MoE gather-GEMV regressed real decode after an encouraging
  microbench.
- Existing phase1 Metal kernels include paged decode attention, Q4_K_M
  projection, RMSNorm, RoPE, logits projection, and argmax/sample surfaces, but
  that does not mean each is production-positive in the MLX model path.
- Direct decode gaps have repeatedly looked like dispatch/graph-bound behavior,
  not a single missing primitive.

## Decision Drivers

- Keep the MLX boundary clean and avoid building a replacement tensor runtime.
- Improve real decode throughput on supported local-agent workloads.
- Preserve greedy parity and sampling semantics.
- Reduce bytes/token, dispatches/token, or CPU/GPU synchronization in a measurable
  way.
- Make failed experiments durable so they become future gates rather than lost
  context.
- Keep every experimental path rollback-safe.

## Options Considered

### Option A: Keep MLX-only forever

**Pros**

- Minimal kernel maintenance.
- Lowest correctness risk.
- Aligns with Apple-maintained operation kernels.

**Cons**

- Leaves AX unable to specialize decode token = 1, long-context KV behavior, or
  speculative-runtime ownership.
- Does not address dispatch-bound shapes where AX can reduce graph overhead.
- Weakens AX's differentiation for local-agent workloads.

### Option B: Rewrite broad MLX primitives in AX-owned Metal

**Pros**

- Maximum theoretical control.
- Could eventually fuse full decode blocks.

**Cons**

- Too expensive and high-risk.
- Competes with MLX on generic kernels that Apple already tunes.
- Creates correctness, portability, and maintenance burdens across every model
  family.

### Option C: Implement the intuitive high-value kernel list directly

Examples: quantized matvec first, then decode attention, then KV rollback,
RMSNorm/residual, RoPE, and sampling.

**Pros**

- Easy to communicate.
- Contains several plausible targets.

**Cons**

- Misstates live repo reality: several kernels already exist, some are wired,
  some are optional, and some similar attempts already failed.
- Treats microbench or generic reasoning as promotion evidence.
- Over-prioritizes standalone RMSNorm/residual despite negative A/B evidence.
- Does not separate MLX affine quantized matmul from Q4_K_M or other layouts.

### Option D: Evidence-gated hybrid strategy

Keep MLX as the general runtime. Admit AX-owned kernels only through a staged
workflow: profile, contract, oracle, microbench, real-graph A/B, promotion gate,
and rollback. Prefer graph-level dispatch reduction before single-op rewrites.

**Pros**

- Matches the repo's best evidence.
- Preserves MLX where it is already strong.
- Lets AX specialize on decode, KV, speculation, and shape-specific gaps.
- Prevents failed kernel ideas from re-entering production without new evidence.

**Cons**

- Slower to start than intuition-led kernel work.
- Requires artifact discipline for every candidate.
- Some valid ideas will wait until profiling justifies them.

## Decision

Adopt **Option D: evidence-gated hybrid strategy**.

AX will keep MLX as the general tensor/runtime layer and own only the decode
hot-path kernels or graph-fusion routes that pass staged evidence gates. The
default optimization order is:

1. Profile and classify the bottleneck.
2. Prefer MLX graph compilation / per-layer closure approaches when the gap is
   dispatch-bound.
3. Wire or prototype existing AX Metal kernels only when the profile shows a
   specific high-share stage and the integration does not fragment lazy eval.
4. Add new Metal kernels only when they remove bytes/token, materialization,
   readback, or dispatches that MLX cannot remove from the current graph.
5. Promote only with real-model E2E artifacts and rollback telemetry.

## Consequences

### Accepted defaults

- `MlxMetalKernel` remains an allowed tool in `ax-engine-mlx` and `mlx-sys`.
- `core/metal` kernels remain valid for the native Metal runner surface.
- Production MLX model paths must not route through a new kernel by default until
  an E2E promotion artifact exists.
- Every runtime-visible kernel path needs a kill switch.
- Candidate promotion evidence must pass
  `scripts/check_decode_hot_path_kernel_admission.py` before default routing.

### Candidate priority

| Track | Status under this ADR |
|---|---|
| Graph-level decode compile / MLX compile exposure | P0, because dispatch overhead is a repeated root cause |
| `paged_decode_attention` production validation | P1, focused on single-token long-context decode |
| Quantized projection kernels | P1, layout-specific only; do not assume generic `quantized_matmul` replacement |
| KV/TurboQuant on-GPU merge | P2, because runtime-owned KV behavior can differentiate AX |
| Sampling/top-k/top-p | P2, after CPU sync evidence |
| Standalone residual + RMSNorm | NO-GO until new evidence changes the prior regression |

### Explicitly rejected

- No full MLX replacement runtime.
- No generic matmul rewrite.
- No training-kernel scope.
- No default-on kernel from microbench evidence alone.
- No repeated RMSNorm/residual sidecar attempt without a new algorithmic reason.

## References

- MLX custom Metal kernel documentation: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- Hugging Face Metal quantization documentation: https://huggingface.co/docs/transformers/en/quantization/metal
- `docs/MLX-BACKEND.md`
- `docs/performance/decode-gap.md`
- `docs/performance/moe-fused-downproj.md`
- `benchmarks/results/inference/mlx-inference/ab-rmsnorm-add/README.md`
