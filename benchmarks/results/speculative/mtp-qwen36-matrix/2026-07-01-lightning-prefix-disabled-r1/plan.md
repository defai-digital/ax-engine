# Qwen3.6 MTP Benchmark Matrix Plan

Scope: Qwen3.6 27B and 35B-A3B, 4-bit and 6-bit, MTP-only.
Required metrics: decode tok/s, prefill tok/s, TTFT ms, and MTP accept rate.

| Target | Suite | Engine | Status | Metric contract | Reason / command |
|---|---|---|---|---|---|
| Qwen3.6 27B 4-bit | `flappy` | `lightning_mlx` | supported | decode_tok_s=measured, prefill_tok_s=approx_prompt_tokens_over_client_ttft_s, ttft_ms=client_stream_ttft_s, accept_rate=measured_if_server_exposes_request_telemetry | `/opt/homebrew/var/mtplx/venv-1.0.4/bin/python /Users/akiralam/code/ax-engine/scripts/bench_rapid_mlx_prompt_suites.py --model /Users/akiralam/.cache/huggingface/hub/models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed/snapshots/be5190f2349594ec941753efc90a4ca5641af174 --suite flappy --prompts benchmarks/results/mtp-qwen36-matrix/2026-07-01-lightning-prefix-disabled-r1/prompt-subsets/flappy-first4.jsonl --output benchmarks/results/mtp-qwen36-matrix/2026-07-01-lightning-prefix-disabled-r1/27b-4bit/flappy/lightning_mlx.json --rapid-source /Users/akiralam/code/ax-engine/.internal/reference/lightning-mlx --lightning-source /Users/akiralam/code/ax-engine/.internal/reference/lightning-mlx --rapid-mtp-patch lightning --lightning-mode --depth 3 --temperature 0.6 --top-p 0.95 --top-k 20 --max-tokens 1000 --repetitions 5 --warmup-repetitions 2 --cooldown 15.0 --inter-case-cooldown 10.0 --ignore-eos --require-full-output-tokens --port 18765 --disable-thinking --mtp-draft-temperature 0.5 --disable-prefix-cache` |
