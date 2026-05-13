# W4 Prefill Forward-Path Trigger

Date: 2026-05-13
Status: Triggered, diagnostic implementation slice selected

## Summary

The latest prefill breakdown evidence shows AX MLX short-prompt prefill can be
ahead of `mlx_lm` while still behind the shape-compatible `llama.cpp Metal`
external reference. The same rows attribute roughly all prefill wall time to the
model-forward path, not prefix-cache storage, generation-state setup, or generic
serving overhead.

This triggers W4 in `MLX-PREFILL-IMPROVEMENT-PRD.md`: collect and render
diagnostic forward-stage evidence before touching kernels or graph structure.

## Evidence

Source artifacts:

- `benchmarks/results/mlx-inference/2026-05-13-ttft-breakdown/`
- `benchmarks/results/llama-cpp-metal/2026-05-13-full-sweep/`

Key rows from the prefill breakdown renderer:

| Model | Prompt tok | AX/MLX | AX/llama.cpp | Prefill wall ms | Forward share |
|---|---:|---:|---:|---:|---:|
| `qwen3_6_35b_a3b_8bit` | 128 | 2.373x | 0.580x | 406.0 | 100.0% |
| `qwen3_coder_next_4bit` | 128 | 2.357x | 0.617x | 507.4 | 100.0% |
| `glm_4_7_flash_4bit` | 128 | 1.760x | 0.624x | 451.0 | 100.0% |

## Decision

Implement a diagnostic MLX forward-profile report before making runtime changes.
The report must:

- read `AX_MLX_LINEAR_ATTENTION_PROFILE=1` artifacts;
- stay out of normal README throughput reporting;
- reject known stale token counters such as `4294967295`;
- show stage timings and stage shares for projection, conv, qk-norm, recurrent,
  and output work;
- produce a clear next-action hint, but not a public performance claim.

## Non-Decisions

- No scheduler or continuous-batching change is justified by this evidence.
- No prefix-cache policy change is justified by this evidence.
- No README headline should be updated from timing-barrier profile artifacts.
- No GatedDelta, MLA, or sliding-window rewrite should start until the diagnostic
  report names a dominant stage for the concrete slow row.
