# Qwen3.6 linear attention projection pack A/B

This report renders diagnostic `AX_MLX_LINEAR_ATTENTION_PROFILE=1` stage counters from inference-stack artifacts. Timing barriers perturb latency, so use this to choose the next kernel experiment, not as a headline throughput claim.

| Model | Engine | Prompt tok | AX prefill tok/s | AX/MLX | AX/SwiftLM | Prefill ms | Forward ms | LA layers | LA tokens | Projection ms | Conv ms | QK norm ms | Recurrent ms | Output ms | Dominant | Dominant % | Next hint | Artifact |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| .internal/models/Qwen3.6-35B-A3B-8bit | ax_engine_mlx_linear_pack | 512 | 1,865.8 | 1.437x | 0.786x | 1,379.8 | 1,379.7 | 150 | 76,800 | 994.5 | 86.5 | 35.7 | 118.5 | 92.0 | projection | 74.9% | compare packed vs split projection delta | `benchmarks/results/mlx-inference/2026-05-14-qwen36-linear-pack-ab/qwen3_6-35b-a3b-8bit-linear-pack-ab.json` |
| .internal/models/Qwen3.6-35B-A3B-8bit | ax_engine_mlx | 512 | 1,897.4 | 1.461x | 0.799x | 1,342.8 | 1,342.6 | 150 | 76,800 | 980.0 | 76.3 | 33.4 | 113.4 | 89.7 | projection | 75.8% | evaluate offline packed qkvz/ba projection | `benchmarks/results/mlx-inference/2026-05-14-qwen36-linear-pack-ab/qwen3_6-35b-a3b-8bit-linear-pack-ab.json` |
| .internal/models/Qwen3.6-35B-A3B-8bit | ax_engine_mlx | 128 | 722.1 | 1.777x | 1.081x | 888.3 | 888.1 | 150 | 19,200 | 661.6 | 46.6 | 30.1 | 50.7 | 64.3 | projection | 77.5% | evaluate offline packed qkvz/ba projection | `benchmarks/results/mlx-inference/2026-05-14-qwen36-linear-pack-ab/qwen3_6-35b-a3b-8bit-linear-pack-ab.json` |
| .internal/models/Qwen3.6-35B-A3B-8bit | ax_engine_mlx_linear_pack | 128 | 738.3 | 1.816x | 1.105x | 869.9 | 869.7 | 150 | 19,200 | 645.6 | 43.6 | 29.9 | 50.0 | 65.6 | projection | 77.4% | compare packed vs split projection delta | `benchmarks/results/mlx-inference/2026-05-14-qwen36-linear-pack-ab/qwen3_6-35b-a3b-8bit-linear-pack-ab.json` |

## Projection Breakdown

| Model | Engine | Prompt tok | Layout | Runtime pack | Offline pack candidate | Projection ms | QKVZ ms | BA ms | QKV ms | Z ms | A ms | B ms | QKV share | Split tail share |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| .internal/models/Qwen3.6-35B-A3B-8bit | ax_engine_mlx_linear_pack | 512 | split_qkv_z_a_b | yes | yes | 994.5 | 917.2 | 34.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% | 0.0% |
| .internal/models/Qwen3.6-35B-A3B-8bit | ax_engine_mlx | 512 | split_qkv_z_a_b | no | yes | 980.0 | 0.0 | 0.0 | 861.4 | 61.2 | 28.2 | 28.8 | 87.9% | 12.1% |
| .internal/models/Qwen3.6-35B-A3B-8bit | ax_engine_mlx | 128 | split_qkv_z_a_b | no | yes | 661.6 | 0.0 | 0.0 | 568.6 | 38.9 | 27.4 | 26.4 | 85.9% | 14.0% |
| .internal/models/Qwen3.6-35B-A3B-8bit | ax_engine_mlx_linear_pack | 128 | split_qkv_z_a_b | yes | yes | 645.6 | 584.9 | 28.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% | 0.0% |

## Pack Comparison

| Model | Prompt tok | Split tok/s | Packed tok/s | Packed/Split tok/s | Split projection ms | Packed projection ms | Projection ms ratio | Split substage ms | Packed substage ms | Substage ratio | Verdict | Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| .internal/models/Qwen3.6-35B-A3B-8bit | 128 | 722.1 | 738.3 | 1.023x | 661.6 | 645.6 | 0.976x | 661.3 | 613.8 | 0.928x | candidate win | `benchmarks/results/mlx-inference/2026-05-14-qwen36-linear-pack-ab/qwen3_6-35b-a3b-8bit-linear-pack-ab.json` |
| .internal/models/Qwen3.6-35B-A3B-8bit | 512 | 1,897.4 | 1,865.8 | 0.983x | 980.0 | 994.5 | 1.015x | 979.6 | 951.7 | 0.971x | neutral/noisy | `benchmarks/results/mlx-inference/2026-05-14-qwen36-linear-pack-ab/qwen3_6-35b-a3b-8bit-linear-pack-ab.json` |

## Reading Notes

- Lowest AX/MLX row in this diagnostic set: `.internal/models/Qwen3.6-35B-A3B-8bit` prompt=512, 1.437x.
- Strongest single-stage concentration: `.internal/models/Qwen3.6-35B-A3B-8bit` prompt=128, projection=77.5%.
- Compare this with the prefill breakdown report first: if forward is not dominant, do not use this report to justify kernel work.
- Projection substage cells are `n/a` for artifacts captured before the projection split counters existed.
- `Runtime pack` means the AX row used `AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS=1`; compare those rows against otherwise-identical split rows before making throughput claims.
- `split_qkv_z_a_b` pack candidates need row-order equivalence tests; simple row-block concatenation can be shape-compatible but semantically wrong.
- Reject stale artifacts with `ax_mlx_linear_attention_profile_tokens=4294967295`; that value came from an old signed/unsigned clamp bug.
- Keep barrier-profile artifacts out of README headline tables.

