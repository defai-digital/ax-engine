# 2026-05-06 MLX Community Checks

Host: Apple M5 Max, 128 GB, macOS 26.4.1.

These artifacts track requested community models by evidence level. GLM now has
the `mlx_lm.benchmark` primary reference, an `mlx_swift_lm` secondary reference,
and repo-owned AX Engine MLX direct and n-gram acceleration benchmarks. DeepSeek
V4 remains reference-only and fail-closed because the available local references
are incomplete for AX promotion.

The latest AX Engine rows below were refreshed after rebuilding the release
server with the GLM MoE dtype cast-back fix. The prior AX direct decode rows
(49.2 / 47.1 tok/s) were collected before `moe_experts_forward` cast the
weighted-sum output back to the residual hidden-state dtype.

Support-contract classification now comes from:

```bash
python3 scripts/probe_mlx_model_support.py --model-dir <model-dir>
```

The probe reads the model config, safetensors index, manifest readiness, and
local reference implementations. It currently classifies GLM as
`repo_owned_runtime_ready` and DeepSeek V4 as fail-closed partial-reference
only.

## Models

| Model | Hugging Face revision | Local size | Config model_type | AX model-manifest | Outcome |
|---|---|---:|---|---|---|
| `mlx-community/GLM-4.7-Flash-4bit` | `1454cffb1a21737e162f508e5bc70be9def89276` | 16 GB | `glm4_moe_lite` | runtime-ready manifest with MLA attention and router metadata | `mlx_lm.benchmark`, `mlx_swift_lm`, `ax_engine_mlx` direct, and `ax_engine_mlx_ngram_accel` benchmarks passed; probe says `repo_owned_runtime_ready` |
| `mlx-community/DeepSeek-V4-Flash-2bit-DQ` | `722bf559b7de93575b2320973cf2002e05bfe6c9` | 90 GB | `deepseek_v4` | absent | blocked by upstream `mlx_lm`; probe says fail-closed partial reference |

## GLM-4.7-Flash-4bit AX Result

Command:

```bash
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/GLM-4.7-Flash-4bit \
  --prompt-tokens 128,512 \
  --generation-tokens 128 \
  --repetitions 3 \
  --cooldown 3 \
  --prefill-step-size 2048 \
  --reuse-reference-results-from benchmarks/results/mlx-inference/2026-05-06/glm-4.7-flash-4bit-mlx-swift-lm.json \
  --output benchmarks/results/mlx-inference/2026-05-06/glm-4.7-flash-4bit-ax-engine-post-latest-4.json
```

| Prompt tok | Generation tok | mlx_lm prefill tok/s | mlx_lm decode tok/s | ax engine prefill tok/s | ax engine decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 502.9 | 93.0 | 841.8 | 94.8 |
| 512 | 128 | 1,584.7 | 90.4 | 2,299.9 | 94.1 |

Artifacts:

- `glm-4.7-flash-4bit-ax-engine-post-latest-4.json`
- `glm-4.7-flash-4bit-ax-engine-post-latest-4-prompts/`

## GLM-4.7-Flash-4bit AX N-Gram Result

Command:

```bash
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/GLM-4.7-Flash-4bit \
  --prompt-tokens 128,512 \
  --generation-tokens 128 \
  --repetitions 3 \
  --cooldown 3 \
  --prefill-step-size 2048 \
  --reuse-reference-results-from benchmarks/results/mlx-inference/2026-05-06/glm-4.7-flash-4bit-mlx-swift-lm.json \
  --ax-ngram-accel \
  --output benchmarks/results/mlx-inference/2026-05-06/glm-4.7-flash-4bit-ax-engine-ngram-post-latest-4.json
```

| Prompt tok | Generation tok | mlx_lm decode tok/s | ax engine direct decode tok/s | ax engine n-gram decode tok/s | N-gram accept rate |
|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 93.0 | 94.8 | 260.7 | 100.0% |
| 512 | 128 | 90.4 | 94.1 | 253.5 | 100.0% |

This confirms GLM 4.7 can run through AX Engine's n-gram acceleration policy.
In this latest refresh, both prompt shapes accepted all drafted tokens and
improved over direct AX decode: 94.8 -> 260.7 tok/s at 128 prompt tokens, and
94.1 -> 253.5 tok/s at 512 prompt tokens.

Artifacts:

- `glm-4.7-flash-4bit-ax-engine-ngram-post-latest-4.json`
- `glm-4.7-flash-4bit-ax-engine-ngram-post-latest-4-prompts/`

## GLM-4.7-Flash-4bit mlx-swift-lm Result

Command:

```bash
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/GLM-4.7-Flash-4bit \
  --prompt-tokens 128,512 \
  --generation-tokens 128 \
  --repetitions 3 \
  --cooldown 3 \
  --prefill-step-size 2048 \
  --skip-ax-engine \
  --mlx-swift-lm-command 'scripts/mlx-swift-bench/.build/release/mlx-swift-bench --model {model} --prompt-token-ids {prompt_token_ids_path} --generation-tokens {generation_tokens} --trials {trials} --delay {delay} --prefill-step-size {prefill_step_size}' \
  --output benchmarks/results/mlx-inference/2026-05-06/glm-4.7-flash-4bit-mlx-swift-lm.json
```

| Prompt tok | Generation tok | mlx_lm prefill tok/s | mlx_lm decode tok/s | mlx_swift_lm prefill tok/s | mlx_swift_lm decode tok/s | Peak memory |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 502.9 | 93.0 | 1,045.0 | 88.0 | 15.723 GB |
| 512 | 128 | 1,584.7 | 90.4 | 2,588.8 | 84.5 | 15.772 GB |

Artifacts:

- `glm-4.7-flash-4bit-mlx-swift-lm.json`
- `glm-4.7-flash-4bit-mlx-swift-lm-prompts/`

## GLM-4.7-Flash-4bit Reference Result

Command:

```bash
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/GLM-4.7-Flash-4bit \
  --prompt-tokens 128,512 \
  --generation-tokens 128 \
  --repetitions 3 \
  --cooldown 3 \
  --prefill-step-size 2048 \
  --skip-ax-engine \
  --output benchmarks/results/mlx-inference/2026-05-06/glm-4.7-flash-4bit-mlx-lm.json
```

| Prompt tok | Generation tok | mlx_lm prefill tok/s | mlx_lm decode tok/s | Peak memory |
|---:|---:|---:|---:|---:|
| 128 | 128 | 487.5 | 89.7 | 17.063 GB |
| 512 | 128 | 1,517.5 | 85.5 | 17.495 GB |

Artifacts:

- `glm-4.7-flash-4bit-mlx-lm.json`
- `glm-4.7-flash-4bit-mlx-lm.log`
- `glm-4.7-flash-4bit-mlx-lm-prompts/`

## DeepSeek-V4-Flash-2bit-DQ Result

Command:

```bash
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/DeepSeek-V4-Flash-2bit-DQ \
  --prompt-tokens 128,512 \
  --generation-tokens 128 \
  --repetitions 3 \
  --cooldown 3 \
  --prefill-step-size 2048 \
  --skip-ax-engine \
  --output benchmarks/results/mlx-inference/2026-05-06/deepseek-v4-flash-2bit-dq-mlx-lm.json
```

Outcome: no JSON result was emitted because the harness failed closed when
`mlx_lm.benchmark` returned:

```text
ValueError: Model type deepseek_v4 not supported.
```

Artifacts:

- `deepseek-v4-flash-2bit-dq-mlx-lm.log`
- `deepseek-v4-flash-2bit-dq-mlx-lm-prompts/`

Re-run this check when upstream `mlx-lm` adds `deepseek_v4` architecture
support, then decide separately whether AX should implement a repo-owned graph.
