# 2026-05-06 MLX Community Reference Checks

Host: Apple M5 Max, 128 GB, macOS 26.4.1.

These artifacts are reference-only `mlx_lm.benchmark` checks for requested
community models. They do not claim repo-owned AX Engine MLX support because
neither model directory contains an AX `model-manifest.json`, and no
hand-written AX model graph or server smoke exists for these architectures yet.

Support-contract classification now comes from:

```bash
python3 scripts/probe_mlx_model_support.py --model-dir <model-dir>
```

The probe reads the model config, safetensors index, draft manifest metadata,
and local reference implementations. It currently classifies GLM as an
implementation candidate with draft manifest mapping and DeepSeek V4 as
fail-closed partial-reference only.

## Models

| Model | Hugging Face revision | Local size | Config model_type | AX model-manifest | Outcome |
|---|---|---:|---|---|---|
| `mlx-community/GLM-4.7-Flash-4bit` | `1454cffb1a21737e162f508e5bc70be9def89276` | 16 GB | `glm4_moe_lite` | absent during benchmark; converter can now emit a draft manifest with MLA attention and router metadata | `mlx_lm.benchmark` passed; probe says implementation candidate |
| `mlx-community/DeepSeek-V4-Flash-2bit-DQ` | `722bf559b7de93575b2320973cf2002e05bfe6c9` | 90 GB | `deepseek_v4` | absent | blocked by upstream `mlx_lm`; probe says fail-closed partial reference |

## GLM-4.7-Flash-4bit Result

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
