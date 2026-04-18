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

The checked-in native scenario manifests now include one dense Qwen target and
one dense Gemma target for bring-up route evidence:

- `benchmarks/manifests/scenario/chat_qwen_short.json`
- `benchmarks/manifests/scenario/chat_gemma_short.json`

You can validate both native dense families through the repo-owned smoke path:

```text
bash scripts/check-bench-native.sh
```

For real-model native benchmarking, scenario manifests can now carry:

- `runtime.native_runtime_artifacts_dir`
- `runtime.native_model_artifacts_dir`

Those paths support both manifest-relative values such as `../../../build/metal`
and environment-backed values such as `$AX_ENGINE_NATIVE_MODEL_DIR`.
If the fields are omitted, `ax-bench` still falls back to the SDK defaults,
including repo-owned Metal build detection and the `AX_ENGINE_NATIVE_MODEL_DIR`
environment variable.

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

Compatibility examples also live alongside the native bring-up manifests:

- `benchmarks/manifests/scenario/compatibility_chat_qwen_short.json`
- `benchmarks/manifests/scenario/compatibility_chat_qwen_short_vllm.json`
- `benchmarks/manifests/scenario/compatibility_chat_qwen_short_mistral_rs.json`
- `benchmarks/manifests/scenario/compatibility_chat_qwen_short_mlx.json`
- `benchmarks/manifests/replay/compatibility_submit_cancel_dual.json`
- `benchmarks/manifests/replay/compatibility_prompt_cache_reuse_dual.json`

The `llama.cpp /completion` manifests still carry the broader stepwise replay
and prompt-cache coverage. The checked-in `vLLM`, `mistral.rs`, and MLX
scenario manifests are single-request blocking compatibility examples. Update
their `runtime.backend_adapter.server_url` before running them directly, or use
the repo-owned smoke script for the `llama.cpp`-backed path:

```text
bash scripts/check-bench-preview.sh
```

The checked-in frozen native Tier 2 scenario matrix lives at:

- `benchmarks/manifests/matrix/frozen_native_dense_phase7.json`

You can validate the full matrix roll-up through:

```text
bash scripts/check-bench-matrix.sh
```

You can validate matrix-to-matrix regression roll-up through:

```text
bash scripts/check-bench-matrix-compare.sh
```
