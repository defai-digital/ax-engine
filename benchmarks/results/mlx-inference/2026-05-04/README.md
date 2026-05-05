# MLX Inference Stack Results: 2026-05-04

Hardware: Apple M5 Max, 128 GB unified memory, macOS 26.4.1.

Contract: random-token prompts reproduced from `mlx_lm.benchmark`
(`mx.random.seed(0)`, `mx.random.randint(0, vocab_size, (1, prompt_tokens))`),
batch=1, `prefill_step_size=2048`, 128 generated tokens, 3 measured trials.
AX and, when present, the Swift adapter perform one untimed warmup.

`mlx_lm` is the required primary baseline. `mlx_swift_lm` is a secondary
BenchmarkHelpers/MLXLMCommon adapter reference. `ax_engine_mlx_greedy` is the
direct same-policy AX comparison. `ax_engine_mlx_speculative` is effective
throughput from AX's n-gram speculative policy and must not be read as raw model
decode speed. Result JSON files include fixed-schema speculative telemetry with
zero-valued draft, accept/reject, complete-miss, no-draft, and cooldown counters.

## Artifacts

| Model | Result JSON | Prompt artifacts | Log |
|---|---|---|---|
| Gemma 4 E2B | `gemma-4-e2b-it-4bit.json` | `gemma-4-e2b-it-4bit-prompts/` | `gemma-4-e2b-it-4bit.log` |
| Gemma 4 E2B 5-bit | `gemma-4-e2b-it-5bit.json` | `gemma-4-e2b-it-5bit-prompts/` | `gemma-4-e2b-it-5bit.log` |
| Gemma 4 E2B 6-bit | `gemma-4-e2b-it-6bit.json` | `gemma-4-e2b-it-6bit-prompts/` | `gemma-4-e2b-it-6bit.log` |
| Gemma 4 E2B 8-bit | `gemma-4-e2b-it-8bit.json` | `gemma-4-e2b-it-8bit-prompts/` | `gemma-4-e2b-it-8bit.log` |
| Gemma 4 26B A4B | `gemma-4-26b-a4b-it-4bit.json` | `gemma-4-26b-a4b-it-4bit-prompts/` | `gemma-4-26b-a4b-it-4bit.log` |
| Gemma 4 31B | `gemma-4-31b-it-4bit.json` | `gemma-4-31b-it-4bit-prompts/` | `gemma-4-31b-it-4bit.log` |
| Qwen 3 4B | `qwen3-4b-4bit.json` | `qwen3-4b-4bit-prompts/` | `qwen3-4b-4bit.log` |
| Qwen 3.5 9B | `qwen3_5-9b-mlx-4bit.json` | `qwen3_5-9b-mlx-4bit-prompts/` | `qwen3_5-9b-mlx-4bit.log` |
| Qwen 3.6 35B A3B | `qwen3_6-35b-a3b-ud-mlx-4bit.json` | `qwen3_6-35b-a3b-ud-mlx-4bit-prompts/` | `qwen3_6-35b-a3b-ud-mlx-4bit.log` |
| Qwen 3.6 35B A3B 5-bit | `qwen3_6-35b-a3b-5bit.json` | `qwen3_6-35b-a3b-5bit-prompts/` | `qwen3_6-35b-a3b-5bit.log` |
| Qwen 3.6 35B A3B 6-bit | `qwen3_6-35b-a3b-6bit.json` | `qwen3_6-35b-a3b-6bit-prompts/` | `qwen3_6-35b-a3b-6bit.log` |
| Qwen 3.6 35B A3B 8-bit | `qwen3_6-35b-a3b-8bit.json` | `qwen3_6-35b-a3b-8bit-prompts/` | `qwen3_6-35b-a3b-8bit.log` |
| Qwen Coder Next | `qwen3-coder-next-4bit.json` | `qwen3-coder-next-4bit-prompts/` | `qwen3-coder-next-4bit.log` |

Qwen 3.6 35B A3B 5/6/8-bit AX server smoke evidence is stored in
`qwen3_6-35b-a3b-5_6_8bit-smoke.json`.

Qwen 3 4B used the local Hugging Face snapshot
`mlx-community/Qwen3-4B-4bit@4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`.
Gemma 4 E2B, Gemma 4 26B A4B, Gemma 4 31B, Qwen 3.5 9B,
Qwen 3.6 35B A3B, and Qwen Coder Next used repo-local `.internal/models/...`
artifact directories; each JSON records the manifest-derived model config used
for the run. Gemma 4 E2B 5/6/8-bit rows were downloaded from
`mlx-community/gemma-4-e2b-it-5bit@7e2d6526209badeacaf09510e86528a107369316`,
`mlx-community/gemma-4-e2b-it-6bit@6fe8c3cfab2910e5bc3439568f6f89413b4d1dca`,
and
`mlx-community/gemma-4-e2b-it-8bit@0cc7ae1721072b5bcd2716d161e4a3b5e786a11e`.
Gemma 4 31B was downloaded from
`mlx-community/gemma-4-31b-it-4bit@dcb78c3f5d6becacbfce71cd4851ad98c4f08a05`.
Gemma 4 26B A4B was downloaded from
`mlx-community/gemma-4-26b-a4b-it-4bit@695690b33533b1f8b0395c1d6b4f00dc411353ef`.
Qwen 3.6 35B A3B was downloaded from
`unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit@6700c3e5bdeb050a379c8d2a4133f43f3647f20f`.
The Qwen 3.6 35B A3B 5/6/8-bit rows were downloaded from
`mlx-community/Qwen3.6-35B-A3B-5bit@c90c1a8b7e7a6a9e39e549f0361a96eb05bbb499`,
`mlx-community/Qwen3.6-35B-A3B-6bit@cb7e092ef8efe540bc3672c8929c4adbe5f4f759`,
and
`mlx-community/Qwen3.6-35B-A3B-8bit@e06a74e6236a60c8367e1a3214e83d8b61b637b0`.

Qwen Coder Next is included as an AX MLX preview runtime row after validating
the Qwen3Next gated-delta linear attention, sparse top-k MoE, shared expert, and
quantized gather paths. Its MLX quantization is affine 4-bit globally, with
8-bit overrides for router and shared-expert gate tensors.

Gemma 4 26B A4B is the Gemma 4 MoE MLX checkpoint. Its `mlx_swift_lm` row is
intentionally absent: the local Swift reference has Gemma4 dense/text support
but no Gemma4 MoE Router/Experts implementation, so it is not an admitted
secondary reference for this model.

Gemma 4 E2B 5/6/8-bit rows are AX quantization-support checks. They use
`mlx_lm` as the primary reference plus `mlx_swift_lm` secondary reference and
AX greedy/speculative rows.

Qwen 3.6 35B A3B 5/6/8-bit rows are AX quantization-support checks against the
MLX-community `qwen3_5_moe` checkpoints. The 5/6-bit checkpoints are affine
5/6-bit globally with 8-bit router and shared-expert gate overrides; the 8-bit
checkpoint is affine 8-bit throughout. The exact AX `/v1/generate` smoke
request/response for all three is recorded in
`qwen3_6-35b-a3b-5_6_8bit-smoke.json`; the full benchmark JSON files then record
the reference-runtime and AX throughput rows.

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
| **Gemma 4 E2B** | 5-bit · group=64 · affine | 128 | mlx_lm | 2,267.5 | 182.9 | -- |
|  |  | 512 | mlx_lm | 8,405.7 | 178.1 | -- |
|  |  | 128 | mlx_swift_lm | 2,393.9 | 174.1 | 0.95x |
|  |  | 512 | mlx_swift_lm | 6,742.6 | 167.0 | 0.94x |
|  |  | 128 | ax greedy | 3,197.3 | 162.2 | 0.89x |
|  |  | 512 | ax greedy | 7,507.0 | 155.2 | 0.87x |
|  |  | 128 | ax speculative | 3,730.1 | 380.8 | 2.08x |
|  |  | 512 | ax speculative | 8,372.2 | 377.2 | 2.12x |
| **Gemma 4 E2B** | 6-bit · group=64 · affine | 128 | mlx_lm | 2,156.3 | 161.3 | -- |
|  |  | 512 | mlx_lm | 7,320.7 | 154.2 | -- |
|  |  | 128 | mlx_swift_lm | 3,436.8 | 153.0 | 0.95x |
|  |  | 512 | mlx_swift_lm | 7,962.3 | 147.1 | 0.95x |
|  |  | 128 | ax greedy | 3,122.3 | 145.4 | 0.90x |
|  |  | 512 | ax greedy | 7,315.7 | 139.6 | 0.91x |
|  |  | 128 | ax speculative | 3,645.2 | 344.0 | 2.13x |
|  |  | 512 | ax speculative | 7,937.5 | 346.0 | 2.24x |
| **Gemma 4 E2B** | 8-bit · group=64 · affine | 128 | mlx_lm | 1,911.7 | 139.4 | -- |
|  |  | 512 | mlx_lm | 6,582.8 | 134.5 | -- |
|  |  | 128 | mlx_swift_lm | 3,082.0 | 134.9 | 0.97x |
|  |  | 512 | mlx_swift_lm | 6,758.1 | 130.8 | 0.97x |
|  |  | 128 | ax greedy | 3,070.4 | 128.2 | 0.92x |
|  |  | 512 | ax greedy | 7,356.1 | 123.4 | 0.92x |
|  |  | 128 | ax speculative | 3,672.5 | 369.9 | 2.65x |
|  |  | 512 | ax speculative | 8,194.4 | 368.2 | 2.74x |
| **Gemma 4 26B A4B** | 4-bit · group=64 · affine, dense/router 8-bit overrides | 128 | mlx_lm | 545.3 | 118.3 | -- |
|  |  | 512 | mlx_lm | 1,620.7 | 113.1 | -- |
|  |  | 128 | ax greedy | 691.3 | 115.2 | 0.97x |
|  |  | 512 | ax greedy | 788.3 | 111.3 | 0.98x |
|  |  | 128 | ax speculative | 721.6 | 234.7 | 1.98x |
|  |  | 512 | ax speculative | 788.4 | 211.4 | 1.87x |
| **Gemma 4 31B** | 4-bit · group=64 · affine | 128 | mlx_lm | 336.5 | 26.2 | -- |
|  |  | 512 | mlx_lm | 563.5 | 24.9 | -- |
|  |  | 128 | mlx_swift_lm | 641.6 | 24.8 | 0.94x |
|  |  | 512 | mlx_swift_lm | 760.6 | 24.7 | 0.99x |
|  |  | 128 | ax greedy | 541.8 | 25.5 | 0.97x |
|  |  | 512 | ax greedy | 727.2 | 25.5 | 1.02x |
|  |  | 128 | ax speculative | 642.7 | 53.3 | 2.03x |
|  |  | 512 | ax speculative | 755.2 | 50.4 | 2.02x |
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
| **Qwen 3.6 35B A3B** | UD-MLX 4-bit · group=64 · affine, selected 8-bit overrides | 128 | mlx_lm | 531.7 | 107.6 | -- |
|  |  | 512 | mlx_lm | 1,594.2 | 103.3 | -- |
|  |  | 128 | mlx_swift_lm | 963.2 | 103.6 | 0.96x |
|  |  | 512 | mlx_swift_lm | 2,546.5 | 101.4 | 0.98x |
|  |  | 128 | ax greedy | 930.3 | 108.5 | 1.01x |
|  |  | 512 | ax greedy | 1,169.3 | 110.0 | 1.07x |
|  |  | 128 | ax speculative | 1,044.6 | 211.0 | 1.96x |
|  |  | 512 | ax speculative | 1,150.9 | 207.7 | 2.01x |
| **Qwen 3.6 35B A3B** | MLX 5-bit · group=64 · affine, gate 8-bit overrides | 128 | mlx_lm | 474.4 | 116.8 | -- |
|  |  | 512 | mlx_lm | 1,484.5 | 113.7 | -- |
|  |  | 128 | mlx_swift_lm | 861.8 | 110.2 | 0.94x |
|  |  | 512 | mlx_swift_lm | 2,416.7 | 108.7 | 0.96x |
|  |  | 128 | ax greedy | 810.2 | 119.8 | 1.03x |
|  |  | 512 | ax greedy | 939.6 | 120.2 | 1.06x |
|  |  | 128 | ax speculative | 859.3 | 215.9 | 1.85x |
|  |  | 512 | ax speculative | 956.1 | 212.5 | 1.87x |
| **Qwen 3.6 35B A3B** | MLX 6-bit · group=64 · affine, gate 8-bit overrides | 128 | mlx_lm | 420.0 | 102.9 | -- |
|  |  | 512 | mlx_lm | 1,377.9 | 101.1 | -- |
|  |  | 128 | mlx_swift_lm | 762.4 | 99.1 | 0.96x |
|  |  | 512 | mlx_swift_lm | 2,350.6 | 98.0 | 0.97x |
|  |  | 128 | ax greedy | 758.0 | 105.6 | 1.03x |
|  |  | 512 | ax greedy | 872.0 | 105.4 | 1.04x |
|  |  | 128 | ax speculative | 809.1 | 197.5 | 1.92x |
|  |  | 512 | ax speculative | 891.9 | 195.8 | 1.94x |
| **Qwen 3.6 35B A3B** | MLX 8-bit · group=64 · affine | 128 | mlx_lm | 393.1 | 93.6 | -- |
|  |  | 512 | mlx_lm | 1,202.2 | 91.4 | -- |
|  |  | 128 | mlx_swift_lm | 617.7 | 89.3 | 0.95x |
|  |  | 512 | mlx_swift_lm | 2,305.2 | 89.1 | 0.97x |
|  |  | 128 | ax greedy | 812.0 | 103.1 | 1.10x |
|  |  | 512 | ax greedy | 931.1 | 102.4 | 1.12x |
|  |  | 128 | ax speculative | 874.4 | 208.9 | 2.23x |
|  |  | 512 | ax speculative | 970.8 | 205.6 | 2.25x |
| **Qwen Coder Next** | 4-bit · group=64 · affine, router/shared-gate 8-bit | 128 | mlx_lm | 267.1 | 92.2 | -- |
|  |  | 512 | mlx_lm | 815.4 | 90.4 | -- |
|  |  | 128 | mlx_swift_lm | 384.9 | 89.4 | 0.97x |
|  |  | 512 | mlx_swift_lm | 1,417.0 | 89.2 | 0.99x |
|  |  | 128 | ax greedy | 698.7 | 95.1 | 1.03x |
|  |  | 512 | ax greedy | 801.2 | 95.4 | 1.06x |
|  |  | 128 | ax speculative | 748.1 | 204.7 | 2.22x |
|  |  | 512 | ax speculative | 826.9 | 200.7 | 2.22x |

## Commands

The same command shape was used for models with an admitted Swift adapter,
changing only `--model-dir` and `--output`.

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

Gemma 4 26B A4B used the same harness without `--mlx-swift-lm-command` because
the Swift reference does not implement the Gemma4 MoE path:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/gemma-4-26b-a4b-it-4bit \
  --prompt-tokens 128,512 \
  --generation-tokens 128 \
  --repetitions 3 \
  --cooldown 3 \
  --prefill-step-size 2048 \
  --ax-both-modes \
  --output benchmarks/results/mlx-inference/2026-05-04/gemma-4-26b-a4b-it-4bit.json
```

Gemma 4 E2B and Qwen 3.6 35B A3B 5/6/8-bit rows used the Swift-admitted
command shape above, changing only `--model-dir` and `--output`.
