# Tech Spec: Decode Hot-Path Kernel Admission and Rollout

**Date:** 2026-07-03
**Status:** Draft
**PRD:** [PRD-2026-07-03-decode-hot-path-kernel-strategy.md](../prd/PRD-2026-07-03-decode-hot-path-kernel-strategy.md)
**ADR:** [ADR-034-decode-hot-path-kernel-strategy.md](../adr/ADR-034-decode-hot-path-kernel-strategy.md)

## Summary

This spec defines the engineering workflow for admitting new decode hot-path
kernels or graph-fusion routes into AX Engine. It is intentionally process-heavy:
the repo has already seen custom-kernel microbench wins fail in real decode. The
workflow is designed to make that impossible to promote accidentally.

## Candidate Admission Checklist

A candidate must have all of the following before production routing:

1. **Profile source**
   - `AX_MLX_DECODE_PROFILE=1` or model-family-specific telemetry identifies the
     target stage.
   - The profile names the affected model, prompt length, generation length,
     sampling policy, build commit, and host.
2. **Mechanism statement**
   - State the removed cost: bytes/token, materialization, dispatch count,
     readback, eval barrier, or duplicated computation.
   - State why MLX cannot already remove that cost in the current graph.
3. **Correctness oracle**
   - CPU reference or current-MLX reference.
   - Greedy parity expectations.
   - Numeric tolerance for logits/logprobs if exact parity is impossible.
4. **Microbench**
   - Lives in `crates/ax-engine-microbench/src/bin/` or `core/metal` GPU tests.
   - Reports warmup, measurement count, host, and variance.
   - Cannot be used as the promotion gate.
5. **Real-graph A/B**
   - Uses the actual model path and real checkpoint.
   - Includes baseline and candidate rows in one artifact family.
   - Reports decode tok/s, prefill tok/s, TTFT, accept rate when applicable, and
     route counters.
6. **Rollback**
   - Feature flag defaults OFF for prototype.
   - Kill switch is documented.
   - Fallback route keeps current behavior.
7. **Promotion decision**
   - Promotion requires >= 5% median decode win on the targeted row, or >= 3%
     with lower TTFT/variance and no quality risk.
   - Default-ON requires a follow-up ADR status update or explicit promotion PRD
     note.

## Artifact Layout

Use this layout for each candidate:

```text
.internal/analysis/decode-hot-path-kernels/
  YYYY-MM-DD-<candidate>/
    candidate.json
    profile.md
    mechanism.md
    correctness.md
    microbench.json
    e2e-summary.md
    promotion-decision.md
```

Public benchmark artifacts, when appropriate, should stay under
`benchmarks/results/...` and be referenced from the internal decision file.

`candidate.json` uses schema `ax.decode_hot_path_kernel_candidate.v1` and is
validated by `scripts/check_decode_hot_path_kernel_admission.py`. Empty
candidate roots pass so the checker can stay in CI before the next candidate is
opened. Any discovered manifest fails closed when it declares a
`promotion_candidate`, `promoted`, or `production_default=true` state without the
complete checklist.

## Candidate Classes

### Class A: Graph-Level Decode Compile

**Use when:** the profile shows many small MLX ops per layer or per decode step.

**Surfaces:**

- `crates/ax-engine-mlx/src/per_layer_compile.rs`
- MLX compile exposure through `mlx-sys` if the current wrapper is insufficient.
- Existing route counters in `runner/mod.rs`.

**Promotion evidence:**

- E2E decode improvement on the target model family.
- No greedy drift because the MLX graph semantics are unchanged.
- Fallback on compile failure.

**Best-practice default:** prefer this before a new sidecar Metal kernel when
the target is dispatch-bound rather than memory-bandwidth-bound.

### Class B: Existing Phase1 Metal Kernel Wiring

**Use when:** a kernel already exists in `metal/kernels/phase1_dense_path.metal`
and the problem is production integration, not authoring.

**Candidate examples:**

- `paged_decode_attention`
- `decode_projection_q4km`
- `sample_argmax_logprob_f32`
- `apply_rope_f32`
- `rms_norm_f32` only for native Metal runner surfaces, not as a repeat of the
  failed MLX residual+RMSNorm sidecar.

**Required steps:**

1. Confirm manifest entry in `metal/phase1-kernels.json`.
2. Confirm GPU correctness test or add one under `crates/ax-engine-core/src/metal/tests.rs`.
3. Add route counters before any default path.
4. Run `bash scripts/check-metal-kernel-contract.sh`.
5. Run the relevant model benchmark outside sandbox when Metal access is needed.

### Class C: New MLX-Sidecar `MlxMetalKernel`

**Use when:** the target belongs inside the MLX lazy graph and can be expressed as
a shape-specialized sidecar kernel.

**Surfaces:**

- `crates/mlx-sys/src/metal.rs`
- `crates/ax-engine-mlx/src/model/shared/*`
- `crates/ax-engine-mlx/src/linear_attention_ops.rs`
- `crates/ax-engine-mlx/src/turboquant_metal.rs`

**Required gates:**

- `OnceLock<MlxMetalKernel>` construction.
- Default-OFF fastpath flag.
- Fallback to current MLX composition.
- `try_eval`/error handling where the route can fail.
- Shape and dtype guards before dispatch.

**Do not use for:** replacing MLX `rms_norm` or `quantized_matmul` generically
without E2E proof. MLX's tuned kernels are the baseline to beat.

### Class D: KV/TurboQuant Runtime Kernels

**Use when:** the target is runtime-owned KV behavior that MLX cannot infer.

**Candidate examples:**

- TurboQuant hot-tail merge on GPU.
- Batched per-layer fused cold decode dispatch.
- Paging/compaction primitives tied to AX block layout.

**Required gates:**

- Explicit candidate/fallback status in `MlxKvCompressionUsage`.
- Greedy parity against full-precision KV.
- Long-context artifact with eligible layers, attempts, successes, fallbacks,
  and sync overhead.
- No claim of memory reduction unless full-precision backing storage is actually
  removed.

## First Candidate Queue

| Order | Candidate | Class | Gate |
|---|---|---|---|
| 1 | Per-layer decode compile expansion / MLX compile exposure | A | Profile shows dispatch-bound decode and E2E improves target rows |
| 2 | `paged_decode_attention` production validation | B | Long-context single-token decode improves without lazy-graph fragmentation |
| 3 | Quantized projection feasibility for layout-specific weights | B/C | Beats MLX `quantized_matmul` or `gather_qmm` in real graph |
| 4 | TurboQuant hot-tail merge on GPU | D | Removes CPU merge/readback and restores decode benefit |
| 5 | Top-k/top-p sampling GPU path | C | CPU sync/logits processor telemetry is material |

## Validation Commands

Run the narrowest relevant commands first:

```bash
cargo fmt --check
python3 scripts/check_decode_hot_path_kernel_admission.py
cargo test -p ax-engine-core metal::
cargo test -p ax-engine-mlx <candidate_test_name>
bash scripts/check-metal-kernel-contract.sh
bash scripts/check-mlx-telemetry.sh
```

For real model evidence, use the relevant benchmark harness and run outside the
sandbox if Metal access is unavailable:

```bash
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 128,512,2048 \
  --generation-tokens 128 \
  --ax-compare-policies
```

Candidate-specific benchmark scripts may be added when the comparison contract is
not covered by `bench_mlx_inference_stack.py`, but they must emit host/build
provenance and matching baseline/candidate rows.

## Telemetry Requirements

Every candidate route must expose route counters before promotion:

- attempts
- hits
- fallbacks
- profile-blocked or shape-blocked count
- elapsed wall time if the route is intended to replace a measured stage
- correctness fallback reason where applicable

Naming should follow existing `ax_mlx_*` route-decision keys.

## Promotion / Revert Policy

Promote to default-ON only when:

- E2E benchmark meets the PRD threshold.
- Greedy parity or accepted numeric drift is documented.
- The fallback route remains tested.
- The kill switch is named in docs or route metadata.
- README/public docs are updated only after benchmark artifacts are stable.

Revert or keep default-OFF when:

- E2E regresses even if the microbench wins.
- The candidate fragments MLX lazy eval and increases command-buffer overhead.
- Greedy output diverges without an explicit quality ADR.
- The candidate only wins on synthetic random-token rows and loses on realistic
  prompt suites.

## Known NO-GO Records

- **Residual + RMSNorm sidecar:** regressed decode and prefill in
  `benchmarks/results/inference/mlx-inference/ab-rmsnorm-add/README.md`.
- **MoE fused gather-GEMV:** regressed real Qwen3.6-35B-A3B decode by roughly
  2-4% and changed greedy output; see `docs/performance/moe-fused-downproj.md`.
- **TurboQuant fused decode as production default:** blocked while CPU hot-tail
  merge and per-layer dispatch fragmentation remain; see `docs/KV-CACHE.md`.
