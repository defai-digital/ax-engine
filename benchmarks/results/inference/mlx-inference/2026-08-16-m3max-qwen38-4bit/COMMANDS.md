# qwen3.8-27b-axq-4bit M3 Max lever hunt — command log
date: 2026-08-17T00:18:04Z
host: Apple M3 Max, 128 GB, macOS 26.6.1
engine: dab32b41 Prepare v7.0.2 release + WIP (9 modified files, unrelated V4/expert-stream work)
model: /Volumes/huggingface/hub/models--AutomatosX--AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP/snapshots/1327acde70f0480cc10ab7dc8ffe043dce9b5de5
== baseline p2048 : env  ==
AX_PROMPT_LEN=2048 AX_PREFILL_QUANTUM=0 AX_WARMUPS=1 AX_REPETITIONS=5 AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0  /Users/akiralam/code/ax-engine/target/release/fair_prefill_bench_probe $MODEL > baseline-p2048.json
== bf16sdpa p2048 : env AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA=1 ==
AX_PROMPT_LEN=2048 AX_PREFILL_QUANTUM=0 AX_WARMUPS=1 AX_REPETITIONS=5 AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0 AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA=1 /Users/akiralam/code/ax-engine/target/release/fair_prefill_bench_probe $MODEL > bf16sdpa-p2048.json
== nativecausal p2048 : env AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL=1 ==
AX_PROMPT_LEN=2048 AX_PREFILL_QUANTUM=0 AX_WARMUPS=1 AX_REPETITIONS=5 AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0 AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL=1 /Users/akiralam/code/ax-engine/target/release/fair_prefill_bench_probe $MODEL > nativecausal-p2048.json
== stacked p2048 : env AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA=1 AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL=1 ==
AX_PROMPT_LEN=2048 AX_PREFILL_QUANTUM=0 AX_WARMUPS=1 AX_REPETITIONS=5 AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0 AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA=1 AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL=1 /Users/akiralam/code/ax-engine/target/release/fair_prefill_bench_probe $MODEL > stacked-p2048.json
== baseline p10240 : env  ==
AX_PROMPT_LEN=10240 AX_PREFILL_QUANTUM=0 AX_WARMUPS=1 AX_REPETITIONS=5 AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0  /Users/akiralam/code/ax-engine/target/release/fair_prefill_bench_probe $MODEL > baseline-p10240.json
== bf16sdpa p10240 : env AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA=1 ==
AX_PROMPT_LEN=10240 AX_PREFILL_QUANTUM=0 AX_WARMUPS=1 AX_REPETITIONS=5 AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0 AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA=1 /Users/akiralam/code/ax-engine/target/release/fair_prefill_bench_probe $MODEL > bf16sdpa-p10240.json
== nativecausal p10240 : env AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL=1 ==
AX_PROMPT_LEN=10240 AX_PREFILL_QUANTUM=0 AX_WARMUPS=1 AX_REPETITIONS=5 AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0 AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL=1 /Users/akiralam/code/ax-engine/target/release/fair_prefill_bench_probe $MODEL > nativecausal-p10240.json
== stacked p10240 : env AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA=1 AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL=1 ==
AX_PROMPT_LEN=10240 AX_PREFILL_QUANTUM=0 AX_WARMUPS=1 AX_REPETITIONS=5 AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0 AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA=1 AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL=1 /Users/akiralam/code/ax-engine/target/release/fair_prefill_bench_probe $MODEL > stacked-p10240.json
== single2048 p10240 : env AX_MLX_QWEN_PREFILL_SINGLE_2048=1 ==
AX_PROMPT_LEN=10240 AX_PREFILL_QUANTUM=0 AX_WARMUPS=1 AX_REPETITIONS=5 AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0 AX_MLX_QWEN_PREFILL_SINGLE_2048=1 /Users/akiralam/code/ax-engine/target/release/fair_prefill_bench_probe $MODEL > single2048-p10240.json
== chunk1536 p10240 : env AX_MLX_QWEN_PREFILL_CHUNK_1536=1 ==
AX_PROMPT_LEN=10240 AX_PREFILL_QUANTUM=0 AX_WARMUPS=1 AX_REPETITIONS=5 AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0 AX_MLX_QWEN_PREFILL_CHUNK_1536=1 /Users/akiralam/code/ax-engine/target/release/fair_prefill_bench_probe $MODEL > chunk1536-p10240.json
== decode decode-direct ==
python3 scripts/bench_mlx_inference_stack.py --ax-direct --output decode-direct.json
== decode decode-mtp ==
python3 scripts/bench_mlx_inference_stack.py --ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact --output decode-mtp.json
== decode decode-direct ==
/Users/akiralam/code/ax-engine/.venv/bin/python scripts/bench_mlx_inference_stack.py --ax-direct --output decode-direct.json
== decode decode-mtp ==
/Users/akiralam/code/ax-engine/.venv/bin/python scripts/bench_mlx_inference_stack.py --ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact --output decode-mtp.json
== decode decode-mtp ==
/Users/akiralam/code/ax-engine/.venv/bin/python scripts/bench_mlx_inference_stack.py --ax-ngram-accel --ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact --output decode-mtp.json
== decode decode-mtp ==
/Users/akiralam/code/ax-engine/.venv/bin/python scripts/bench_mlx_inference_stack.py --ax-ngram-accel --ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact --output decode-mtp.json
== AFTER: prefill probe default flags (bf16sdpa now default-on, exact scope narrowed) ==
== AFTER: pure-MTP decode row via CLI flag only (was contract-aborted before fix) ==
== AFTER re-run: p2048 202.1 (was thermal-noise 171.9) ==
