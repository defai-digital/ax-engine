# MLX Inference Stack Results: 2026-05-04

Hardware: Apple M5 Max, 128 GB unified memory, macOS 26.4.1.

Contract: random-token prompts reproduced from `mlx_lm.benchmark`
(`mx.random.seed(0)`, `mx.random.randint(0, vocab_size, (1, prompt_tokens))`),
batch=1, `prefill_step_size=2048`, 128 generated tokens, 3 measured trials.
AX and the Swift adapter perform one untimed warmup.

`mlx_lm` is the required primary baseline. `mlx_swift_lm` is a secondary
BenchmarkHelpers/MLXLMCommon adapter reference. `ax_engine_mlx_greedy` is the
direct same-policy AX comparison. `ax_engine_mlx_speculative` is effective
throughput from AX's n-gram speculative policy and must not be read as raw model
decode speed.

## Artifacts

| Model | Result JSON | Prompt artifacts | Log |
|---|---|---|---|
| Gemma 4 E2B | `gemma-4-e2b-it-4bit.json` | `gemma-4-e2b-it-4bit-prompts/` | `gemma-4-e2b-it-4bit.log` |
| Qwen 3 4B | `qwen3-4b-4bit.json` | `qwen3-4b-4bit-prompts/` | `qwen3-4b-4bit.log` |
| Qwen 3.5 9B | `qwen3_5-9b-mlx-4bit.json` | `qwen3_5-9b-mlx-4bit-prompts/` | `qwen3_5-9b-mlx-4bit.log` |

Qwen 3 4B used the local Hugging Face snapshot
`mlx-community/Qwen3-4B-4bit@4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`.
Gemma 4 E2B and Qwen 3.5 9B used repo-local `.internal/models/...` artifact
directories; each JSON records the manifest-derived model config used for the
run.

## Summary

| Model | MLX quantization | Prompt tok | Engine | Prefill tok/s | Decode tok/s | vs mlx_lm |
|---|---|---:|---|---:|---:|---:|
| **Gemma 4 E2B** | 4-bit · group=64 · affine | 128 | mlx_lm | 2,265.8 | 197.5 | -- |
|  |  | 512 | mlx_lm | 7,634.1 | 191.9 | -- |
|  |  | 128 | mlx_swift_lm | 2,450.4 | 192.4 | 0.97x |
|  |  | 512 | mlx_swift_lm | 6,664.3 | 179.5 | 0.94x |
|  |  | 128 | ax greedy | 3,248.7 | 176.0 | 0.89x |
|  |  | 512 | ax greedy | 7,640.2 | 170.9 | 0.89x |
|  |  | 128 | ax speculative | 3,715.6 | 467.6 | 2.37x |
|  |  | 512 | ax speculative | 8,394.1 | 464.8 | 2.42x |
| **Qwen 3 4B** | 4-bit · group=64 | 128 | mlx_lm | 1,581.1 | 169.6 | -- |
|  |  | 512 | mlx_lm | 3,726.0 | 169.8 | -- |
|  |  | 128 | mlx_swift_lm | 3,627.8 | 168.7 | 0.99x |
|  |  | 512 | mlx_swift_lm | 6,173.7 | 161.0 | 0.95x |
|  |  | 128 | ax greedy | 3,077.7 | 167.7 | 0.99x |
|  |  | 512 | ax greedy | 5,428.9 | 158.9 | 0.94x |
|  |  | 128 | ax speculative | 3,401.5 | 311.5 | 1.84x |
|  |  | 512 | ax speculative | 5,628.1 | 289.5 | 1.70x |
| **Qwen 3.5 9B** | 4-bit · group=64 · affine | 128 | mlx_lm | 1,038.5 | 92.6 | -- |
|  |  | 512 | mlx_lm | 2,161.4 | 94.8 | -- |
|  |  | 128 | mlx_swift_lm | 2,101.1 | 93.7 | 1.01x |
|  |  | 512 | mlx_swift_lm | 3,165.8 | 91.4 | 0.96x |
|  |  | 128 | ax greedy | 1,912.0 | 95.2 | 1.03x |
|  |  | 512 | ax greedy | 2,735.7 | 94.5 | 1.00x |
|  |  | 128 | ax speculative | 2,181.2 | 168.7 | 1.82x |
|  |  | 512 | ax speculative | 2,847.3 | 87.5 | 0.92x |

## Commands

The same command shape was used for each model, changing only `--model-dir` and
`--output`:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir <model-dir> \
  --prompt-tokens 128,512 \
  --generation-tokens 128 \
  --repetitions 3 \
  --cooldown 3 \
  --prefill-step-size 2048 \
  --ax-both-modes \
  --mlx-swift-lm-command './scripts/mlx-swift-bench/.build/release/mlx-swift-bench \
    --model {model} --prompt-token-ids {prompt_token_ids_path} \
    --generation-tokens {generation_tokens} --trials {trials} \
    --delay {delay} --prefill-step-size {prefill_step_size}' \
  --output benchmarks/results/mlx-inference/2026-05-04/<model>.json
```
