# Qwen3.6 AX Direct-Only Benchmark

Date: 2026-06-26
Commit: `ab5a9040315bd5bcf91200c6b074549b9c4535d5`
Host: Apple M5 Max, 128 GB, macOS 26.5.1
Baseline: `benchmarks/results/mlx-inference/2026-06-22-ax-direct-readme-direct-only/`

## Method

Ran AX Engine only with direct decode:

```bash
cargo build -p ax-engine-server --release
python3 scripts/bench_mlx_inference_stack.py \
  --model <model> \
  --model-repo-id <repo> \
  --hf-cache-root /Volumes/Ext4T/models/hub \
  --prompt-tokens 128,512,2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 15 \
  --ax-direct \
  --skip-mlx-lm \
  --no-build-ax-engine \
  --output benchmarks/results/mlx-inference/2026-06-26-qwen36-ax-direct-only/<model>.json
```

This disables n-gram acceleration and prefix-cache reuse for cold direct-mode measurement. Because `--skip-mlx-lm` was used, these artifacts contain AX rows only and compare against the saved README baseline rather than an in-run `mlx_lm` row.

## Results

| Model | Prompt | Prefill tok/s | Decode tok/s | TTFT ms | Decode vs README | Prefill vs README | TTFT vs README |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3_6-27b-4bit` | 128 | 561.0 | 33.2 | 228 | -3.3% | -3.3% | +3.5% |
| `qwen3_6-27b-4bit` | 512 | 829.5 | 34.3 | 617 | -0.5% | -0.4% | +0.4% |
| `qwen3_6-27b-4bit` | 2048 | 923.9 | 33.8 | 2217 | -1.1% | -0.8% | +0.8% |
| `qwen3_6-27b-6bit` | 128 | 503.5 | 25.2 | 254 | -0.9% | -0.6% | +0.6% |
| `qwen3_6-27b-6bit` | 512 | 755.0 | 25.2 | 678 | -0.9% | -0.2% | +0.3% |
| `qwen3_6-27b-6bit` | 2048 | 833.6 | 23.6 | 2457 | -6.4% | -3.4% | +3.5% |
| `qwen3_6-35b-a3b-4bit` | 128 | 1103.4 | 148.8 | 116 | -2.9% | -1.7% | +1.8% |
| `qwen3_6-35b-a3b-4bit` | 512 | 2498.8 | 148.9 | 205 | -2.5% | -4.1% | +4.3% |
| `qwen3_6-35b-a3b-4bit` | 2048 | 3610.9 | 147.8 | 567 | -2.0% | -3.8% | +4.0% |

## Verdict

Against the current README AX direct baseline, this Qwen3.6-only rerun is slightly slower overall: median decode is -2.0%, median prefill is -1.7%, and median TTFT is +1.8%. The largest decode regression is `qwen3_6-27b-6bit` at prompt 2048 (-6.4%). The `qwen3_6-35b-a3b-4bit` prompt 128 run had two slow decode repetitions, so its median is usable but the row has noticeable variance.
