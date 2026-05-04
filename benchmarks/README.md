# AX Engine v4 Benchmarks

This directory contains the canonical workload manifests and benchmark outputs
for AX Engine v4.

Initial layout:

```text
benchmarks/
  manifests/
    matrix/
    scenario/
    replay/
  results/
```

The first planning goal is not exhaustive workload coverage.
It is to freeze a small, trustworthy set of canonical workloads that represent
the Phase 1 dense-path goals:

- short chat
- coding
- long context
- concurrent scheduling
- shared prefix
- replay and churn

The manifest schema direction is documented internally.

The checked-in MLX scenario manifests now include one dense Qwen target and
one dense Gemma target for bring-up route evidence:

- `benchmarks/manifests/scenario/chat_qwen_short.json`
- `benchmarks/manifests/scenario/chat_gemma_short.json`

You can validate both MLX dense families through the repo-owned smoke path:

```text
bash scripts/check-bench-mlx.sh
```

For real-model MLX-mode benchmarking, scenario manifests can carry `runtime.mlx_model_artifacts_dir`.
Manifest-relative values such as `../../../models/qwen-mlx` are supported.
If the fields are omitted, `ax-bench` still falls back to the SDK defaults,
including repo-owned Metal build detection and the
`AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR` environment variable.

The checked-in native replay set now covers:

- `benchmarks/manifests/replay/shared_prefix_long_churn.json`
- `benchmarks/manifests/replay/retained_prefix_after_cleanup.json`
- `benchmarks/manifests/replay/mixed_live_and_retained_prefix_paths.json`
- `benchmarks/manifests/replay/full_prefix_to_decode_branch.json`
- `benchmarks/manifests/replay/memory_blocked_prefix_recovery.json`

You can validate those replay workloads through the repo-owned smoke path:

```text
bash scripts/check-bench-replay.sh
```

For model-inference performance comparisons, use the MLX stack harness rather
than `ax-bench` scenario or replay manifests:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/Qwen3.5-9B-MLX-4bit \
  --prompt-tokens 512,2048 \
  --generation-tokens 128
```

That harness uses upstream `mlx_lm.benchmark` as the primary standard and can
optionally ingest a local `mlx-swift-lm` JSON adapter. The older SwiftLM app
server path is intentionally retired. Its JSON output labels the reference
runtime and AX Engine decode mode explicitly, so greedy AX MLX, speculative AX
MLX, `mlx_lm.benchmark`, and a Swift JSON adapter do not collapse into one
ambiguous throughput row.

llama.cpp examples also live alongside the MLX bring-up manifests:

- `benchmarks/manifests/scenario/llama_cpp_chat_qwen_short.json`
- `benchmarks/manifests/replay/llama_cpp_submit_cancel_dual.json`
- `benchmarks/manifests/replay/llama_cpp_prompt_cache_reuse_dual.json`

The `llama.cpp /completion` manifests still carry the broader delegated
stepwise replay and prompt-cache coverage. Treat those as non-MLX delegation
contract checks, not AX-owned MLX model-inference throughput baselines.
Scenario manifests are single-request blocking llama.cpp examples. Update their
`runtime.backend_adapter.server_url` before running them directly, or use the
repo-owned smoke script for the `llama.cpp`-backed path:

```text
bash scripts/check-bench-preview.sh
```

The checked-in frozen MLX Tier 2 scenario matrix lives at:

- `benchmarks/manifests/matrix/mlx_dense_phase7.json`

You can validate the full matrix roll-up through:

```text
bash scripts/check-bench-matrix.sh
```

You can validate matrix-to-matrix regression roll-up through:

```text
bash scripts/check-bench-matrix-compare.sh
```
