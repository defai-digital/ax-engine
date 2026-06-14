# Qwen3-Coder-Next 4-bit — llama.cpp Metal reference (build b9620, flash-attn)

Shape-compatible external GGUF baseline for the README Qwen3-Coder-Next
direct-mode charts. This run replaces the earlier split llama.cpp rows
(`2026-05-13-full-sweep` build b9100 for p128/p512 and `2026-06-13-...-2048`
build b9590 for p2048) with **one consistent llama.cpp build across all three
prompt sizes**, so the chart can carry a single truthful version label.

## Setup

| Field | Value |
| --- | --- |
| Host | Apple M5 Max, 128 GB, macOS 26.5.1 |
| Engine | llama.cpp **b9620**, ggml **0.15.1** (Homebrew), Metal backend |
| Model | `Qwen_Qwen3-Coder-Next-Q4_K_M.gguf` (qwen3next 80B.A3B, Q4_K_M) |
| Shapes | prompt 128 / 512 / 2048, generation 128, batch 1 |
| llama-bench | `-ngl 99 -b 2048 -ub <prompt-sized> -fa 1 -r 5 --delay 15` |
| Flash attention | **on** (`-fa 1`) for all three sizes |
| Repetitions | 5 |

> `-ub` is sized to the prompt (512 for prompts ≤512, 2048 for the 2048 row) so
> each prefill is a single micro-batch pass — the standard llama-bench prefill
> contract. Build, flash-attn, and `-b` are identical across all three rows.

Reproduce:

```bash
python3 scripts/bench_mlx_inference_stack.py \
  --model-repo-id mlx-community/Qwen3-Coder-Next-4bit \
  --prompt-tokens 128,512,2048 --generation-tokens 128 \
  --repetitions 5 --cooldown 15 \
  --skip-ax-engine --skip-mlx-lm \
  --llama-cpp-bench "$(which llama-bench)" \
  --llama-cpp-gguf .internal/models/Qwen3-Coder-Next-Q4_K_M-GGUF/Qwen_Qwen3-Coder-Next-Q4_K_M.gguf \
  --llama-cpp-extra-args "-fa 1" \
  --output benchmarks/results/llama-cpp-metal/2026-06-13-qwen3-coder-next-9620-fa/qwen3-coder-next-4bit.json
```

## Results (median over 5 reps)

| prompt | prefill tok/s | decode tok/s | TTFT ms |
| ---: | ---: | ---: | ---: |
| 128 | 1253.9 | 86.2 | 102.1 |
| 512 | 2150.4 | 85.1 | 238.1 |
| 2048 | 2554.7 | 85.2 | 801.6 |

llama-bench consumes its own internal synthetic tokens, so these rows are a
shape-compatible reference only (no prompt-hash parity with the MLX/AX rows).
The MLX/AX rows for the same chart live in
`benchmarks/results/mlx-inference/2026-06-13-qwen3-coder-next-prefill-probe/`.
