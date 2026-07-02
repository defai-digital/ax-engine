# Qwen3.6 MTP Benchmark Matrix Plan

Scope: Qwen3.6 27B and 35B-A3B, 4-bit and 6-bit, MTP-only.
Required metrics: decode tok/s, prefill tok/s, TTFT ms, and MTP accept rate.

| Target | Suite | Engine | Status | Metric contract | Reason / command |
|---|---|---|---|---|---|
| Qwen3.6 27B 4-bit | `flappy` | `ax_engine` | supported | decode_tok_s=measured, prefill_tok_s=measured, ttft_ms=measured, accept_rate=measured | `/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/akiralam/code/ax-engine/scripts/bench_mlx_inference_stack.py --model-dir /Users/akiralam/.cache/huggingface/hub/models--ax-local--Qwen3.6-27B-MTP/snapshots/v1 --prompt-source real --real-prompt-suite benchmarks/results/mtp-qwen36-matrix/2026-07-01-27b4-ax-pack-dense-ffn-signal-r1/prompt-subsets/flappy-first1.jsonl --generation-tokens 300 --repetitions 2 --warmup-repetitions 1 --cooldown 0.0 --inter-case-cooldown 0.0 --ax-sampling '{"temperature":0.6,"top_p":0.95,"top_k":20}' --skip-mlx-lm --no-thinking --capture-output-token-ids --ax-ngram-accel --ax-mtp-disable-ngram-stacking --ax-mtp-max-depth 3 --output benchmarks/results/mtp-qwen36-matrix/2026-07-01-27b4-ax-pack-dense-ffn-signal-r1/27b-4bit/flappy/ax_engine.json --no-build-ax-engine` |
