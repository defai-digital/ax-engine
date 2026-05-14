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

## 2026-05-13 Follow-Up: Qwen Projection Split

After adding projection substage counters and refreshing the Qwen profile
artifact, the concrete slow row now reads:

| Model | Prompt tok | Layout | Offline pack candidate | Projection ms | QKV ms | Z ms | A ms | B ms | QKV share | Split tail share |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen3_6_35b_a3b_8bit` | 128 | `split_qkv_z_a_b` | yes | 129.1 | 110.7 | 7.7 | 5.6 | 5.1 | 85.7% | 14.2% |

Interpretation:

- Projection remains the dominant profiled forward stage at 76.9% of diagnostic
  stage time.
- The QKV matmul is the dominant projection substage. The separate Z/A/B
  projections are measurable but are not the majority of projection time.
- Offline packing is still a valid experiment because the artifact marks the
  model as `offline_pack_candidate=true`, but the expected packed layout is not a
  simple row-block concat.

Decision:

- Do not implement ad hoc loader-time packing by simply concatenating
  `in_proj_qkv` with `in_proj_z`, or `in_proj_b` with `in_proj_a`.
- The packed path expects `qkvz` arranged per key head as `q,k,v,z`, and `ba`
  arranged per key head as `b,a`. A row-block concat can be shape-compatible
  while still semantically wrong.
- Any pack implementation must be a converter or loader helper with explicit
  layout tests for row order, quantization sidecar order, group size, bit width,
  and bias presence.
- The next safe runtime implementation target is therefore not the pack itself;
  it is a correctness-gated packer prototype or manifest-level converter path.

## Non-Decisions

- No scheduler or continuous-batching change is justified by this evidence.
- No prefix-cache policy change is justified by this evidence.
- No README headline should be updated from timing-barrier profile artifacts.
- No GatedDelta, MLA, or sliding-window rewrite should start until the diagnostic
  report names a dominant stage for the concrete slow row.
- No loader-time packed projection rewrite should land without a row-order
  equivalence gate against the split path.

## 2026-05-14 Follow-Up: Pack Comparison Reporting

The W4 projection-pack work now has the diagnostic plumbing needed to compare
split and packed AX rows without turning barrier-profile data into a public
throughput claim:

- the loader pack path remains opt-in behind
  `AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS=1`;
- `bench_mlx_inference_stack.py` can emit matched direct split and packed rows
  via `--ax-compare-linear-attention-projection-pack`;
- `render_mlx_forward_profile_report.py` now renders a `Pack Comparison` table
  with packed/split prefill throughput, projection wall-time ratio, projection
  substage ratio, and a diagnostic verdict.

Decision:

- Use the new `Pack Comparison` section as the next W4 artifact gate.
- Treat `candidate win` as a prompt to run a repeated, cooled artifact before
  promoting the path.
- Treat `neutral/noisy`, `candidate regression`, or missing substage data as a
  stop signal for public claims.
