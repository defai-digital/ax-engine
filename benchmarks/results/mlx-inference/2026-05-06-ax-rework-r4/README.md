# 2026-05-06 AX Rework R4 MLX Inference Refresh

This directory records the completed AX-only refresh requested for the
2026-05-06 engine rework. The harness reused the existing `mlx_lm` and
`mlx_swift_lm` reference rows from the matching checked-in artifacts, then
reran only the `ax_engine_mlx` and `ax_engine_mlx_ngram_accel` rows.

Host and workload contract:

- Apple M5 Max, 128 GB, macOS 26.4.1.
- `target/release/ax-engine-server`, release profile.
- Random-token prompts matching the `mlx_lm.benchmark` seed=0 contract.
- Prompt lengths: 128 and 512 tokens.
- Generation length: 128 tokens.
- Repetitions: 3 timed trials with 1 warmup.
- Cooldown: 3 seconds.
- Prefill step size: 2048.

## Command

Each model was run with the same shape and policy:

```bash
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir <local-mlx-model-dir> \
  --prompt-tokens 128,512 \
  --generation-tokens 128 \
  --repetitions 3 \
  --cooldown 3 \
  --prefill-step-size 2048 \
  --reuse-reference-results-from <previous-reference-artifact.json> \
  --ax-compare-policies \
  --output benchmarks/results/mlx-inference/2026-05-06-ax-rework-r4/<model>.json
```

The full command log is stored in `ax-rework-r4-refresh.log`.

## Result Summary

The rework is not an across-the-board direct-decode improvement. Direct AX
decode is positive only for Qwen 3.6 UD 4-bit and the 512-token Qwen 3.6 8-bit
shape. GLM 4.7, Qwen Coder Next, and Gemma remain below the matching `mlx_lm`
decode baseline in direct mode.

N-gram acceleration remains highly positive for Gemma, Qwen Coder Next, GLM
4.7, and several Qwen 3.6 shapes, but it is not universally effective. Qwen
3.5 is negative for both prompt shapes, Qwen 3.6 5-bit is negative for both
prompt shapes, and Qwen 3.6 8-bit is negative at the 128-token shape.

| Model | Artifact | Direct decode vs mlx_lm | N-gram effective decode vs mlx_lm | Direct prefill vs mlx_lm |
|---|---|---:|---:|---:|
| Gemma 4 E2B 4-bit | `gemma-4-e2b-it-4bit.json` | 128 -11.8%; 512 -13.9% | 128 +165.6%; 512 +175.5% | 128 +38.4%; 512 -2.7% |
| Gemma 4 E2B 5-bit | `gemma-4-e2b-it-5bit.json` | 128 -14.5%; 512 -15.8% | 128 +129.1%; 512 +132.5% | 128 +34.9%; 512 -15.2% |
| Gemma 4 E2B 6-bit | `gemma-4-e2b-it-6bit.json` | 128 -13.5%; 512 -12.7% | 128 +141.4%; 512 +150.1% | 128 +39.5%; 512 -3.4% |
| Gemma 4 E2B 8-bit | `gemma-4-e2b-it-8bit.json` | 128 -11.3%; 512 -10.7% | 128 +196.9%; 512 +207.1% | 128 +57.1%; 512 +7.6% |
| Gemma 4 26B A4B | `gemma-4-26b-a4b-it-4bit.json` | 128 -7.1%; 512 -5.4% | 128 +105.6%; 512 +75.2% | 128 +119.3%; 512 +66.3% |
| Gemma 4 31B | `gemma-4-31b-it-4bit.json` | 128 -10.6%; 512 -10.6% | 128 +119.9%; 512 +104.4% | 128 +49.8%; 512 +14.6% |
| Qwen 3.5 9B | `qwen3_5-9b-mlx-4bit.json` | 128 -8.4%; 512 -12.3% | 128 -14.1%; 512 -14.7% | 128 +62.8%; 512 +16.4% |
| Qwen 3.6 UD 4-bit | `qwen3_6-35b-a3b-ud-mlx-4bit.json` | 128 +0.4%; 512 +3.2% | 128 +26.2%; 512 +14.6% | 128 +71.0%; 512 +42.3% |
| Qwen 3.6 5-bit | `qwen3_6-35b-a3b-5bit.json` | 128 -1.7%; 512 -3.6% | 128 -29.7%; 512 -5.5% | 128 +81.0%; 512 +43.9% |
| Qwen 3.6 6-bit | `qwen3_6-35b-a3b-6bit.json` | 128 -1.1%; 512 -0.2% | 128 +26.2%; 512 +13.4% | 128 +95.1%; 512 +50.8% |
| Qwen 3.6 8-bit | `qwen3_6-35b-a3b-8bit.json` | 128 -1.5%; 512 +0.6% | 128 -2.9%; 512 +24.7% | 128 +99.6%; 512 +71.1% |
| Qwen Coder Next | `qwen3-coder-next-4bit.json` | 128 -1.5%; 512 -0.5% | 128 +160.6%; 512 +164.5% | 128 +203.9%; 512 +218.9% |
| GLM 4.7 Flash | `glm-4.7-flash-4bit.json` | 128 -1.9%; 512 -0.1% | 128 +169.4%; 512 +170.8% | 128 +63.6%; 512 +40.0% |

## Artifacts

| Model | Artifact |
|---|---|
| Gemma 4 E2B 4-bit | `gemma-4-e2b-it-4bit.json` |
| Gemma 4 E2B 5-bit | `gemma-4-e2b-it-5bit.json` |
| Gemma 4 E2B 6-bit | `gemma-4-e2b-it-6bit.json` |
| Gemma 4 E2B 8-bit | `gemma-4-e2b-it-8bit.json` |
| Gemma 4 26B A4B | `gemma-4-26b-a4b-it-4bit.json` |
| Gemma 4 31B | `gemma-4-31b-it-4bit.json` |
| Qwen 3.5 9B | `qwen3_5-9b-mlx-4bit.json` |
| Qwen 3.6 UD 4-bit | `qwen3_6-35b-a3b-ud-mlx-4bit.json` |
| Qwen 3.6 5-bit | `qwen3_6-35b-a3b-5bit.json` |
| Qwen 3.6 6-bit | `qwen3_6-35b-a3b-6bit.json` |
| Qwen 3.6 8-bit | `qwen3_6-35b-a3b-8bit.json` |
| Qwen Coder Next | `qwen3-coder-next-4bit.json` |
| GLM 4.7 Flash | `glm-4.7-flash-4bit.json` |
