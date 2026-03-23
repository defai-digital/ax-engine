# PRD-0008: Prefill Performance Decision and Profiling Plan

## Status

Done

## Problem

AX still shows a prefill gap versus local `llama.cpp` across key families. The latest
benchmark cycle showed that enabling `AX_METAL_BATCH_Q4K_V2` regresses prefill,
while default behavior and `AX_METAL_BATCH_Q4K_V2=0` are equivalent.

## Experiment Scope

- Compare prefill throughput for `256` prompt tokens, `0` decode tokens, one warmup, three measure iterations, `--intent latency`.
- Models:
  - `Qwen3-0.6B-Q4_K_M.gguf`
  - `Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf`
  - `gemma-3-4b-it-Q4_K_M.gguf`
- Env states:
  - default
  - `AX_METAL_BATCH_Q4K_V2=0`
  - `AX_METAL_BATCH_Q4K_V2=1`

Command:

```bash
./target/release/ax-bench bench \
  --model ./models/<model>.gguf \
  --prompt-tokens 256 --decode-tokens 0 \
  --warmup-iters 1 --measure-iters 3 --intent latency
```

## Observed Results (M3 Max, release build)

| Model | Default | `Q4K_V2=0` | `Q4K_V2=1` |
|---|---:|---:|---:|
| `Qwen3-0.6B-Q4_K_M` | 4189.5 tok/s | 4192.8 tok/s | 3479.3 tok/s |
| `Llama-3-8B-Instruct-GGUF-Q4_K_M` | 385.9 tok/s | 384.0 tok/s | 269.6 tok/s |
| `gemma-3-4b-it-Q4_K_M` | 811.5 tok/s | 813.4 tok/s | 614.6 tok/s |

Implication:

- `v2=1` is materially slower in all tested short shapes.
- `v2=0` tracks default closely.
- The shipped default state should remain without opt-in v2 enabled.

## What We Learned

1. Barrier overhead is not the prefill blocker: 310 barriers are negligible
   versus measured ~60ms prefill runtime.
2. Attention is not dominant in this regime; FA2 path changes do not shift the
   overall prefill baseline.
3. The `f16`-in matmul path itself is slower in tested settings; the v2 regression
   is a load/dequant path issue rather than routing alone.
4. RMSNorm optimization is low priority (~3% of total prefill time, ~1ms), so it
   should be revisited only after phase-level bottlenecks are confirmed.
5. The remaining gap is likely in batched matmul load/dequant efficiency.

## Decision

- Keep `AX_METAL_BATCH_Q4K_V2` default OFF.
- Keep `AX_METAL_BATCH_Q4K_V2` as explicit opt-in only (`=1`).
- Do not promote additional global default/profile defaults from these tests.

## Execution Plan (Next Iteration)

### Phase 1: Profiling-first root-cause

1. Run Metal GPU frame capture for `ax-bench` prefill (`prompt=256`, `decode=0`) on the three models.
2. Run the same shape against `llama-bench` for side-by-side timing.
3. Export per-phase timing for:
   - dequant reads/writes
   - f16 input path
   - threadgroup loading and memory stalls

### Phase 2: Scope-limited optimization

Only target candidates with measurable load-phase gains:

- `dequant_batch_q4_k` loading order/tiling
- memory access layout in existing TG=32/64 kernels
- shape- and model-conditioned profile heuristics

## Guardrails

- Preserve decode and prefill stability across existing tracked families.
- New candidate defaults require:
  - >= 5% gain on at least one tracked family
  - <= 1% regression on the same family comparisons

## Validation Checklist

- `cargo build --release --package ax-bench`
- Re-run the A/B command set after each kernel/profile change
- Compare against this PRD’s baseline table before promoting any default change

## Exit Criteria

Promote a new default only when two or more tracked families show sustained gains
that beat current shipped values without measurable regression in the remaining
families.
