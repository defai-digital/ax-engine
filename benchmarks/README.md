# AX Engine v4 Benchmarks

This directory contains checked-in workload manifests and generated benchmark
outputs for AX Engine v4.

```text
benchmarks/
  corpora/
  manifests/
    matrix/
    replay/
    scenario/
  results/
```

The manifests are workload contracts. They are not the MLX reference-inference
comparison harness. Use `docs/BENCHMARKS.md` for the full benchmarking model.

Prompt corpora under `benchmarks/corpora/` are inputs for serving-oriented
benchmarks. They are deliberately separate from scenario/replay manifests
because they model user request mixes rather than engine contract gates.

## Manifest Families

### MLX scenario manifests

The checked-in MLX scenarios cover the dense preview path:

- `benchmarks/manifests/scenario/chat_qwen_short.json`
- `benchmarks/manifests/scenario/chat_gemma_short.json`
- `benchmarks/manifests/scenario/coding_qwen_medium.json`
- `benchmarks/manifests/scenario/long_context_qwen_8k.json`
- `benchmarks/manifests/scenario/concurrent_qwen_dual.json`
- `benchmarks/manifests/scenario/shared_prefix_qwen_enterprise.json`

Validate the MLX dense smoke set with:

```text
bash scripts/check-bench-mlx.sh
```

Scenario manifests can carry `runtime.mlx_model_artifacts_dir` for real-model
MLX mode. Manifest-relative values are supported. If the field is omitted,
`ax-engine-bench` falls back to SDK defaults, including
`AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR`.

### MLX replay manifests

The checked-in replay set validates route, replay, churn, prefix, and memory
contract behavior:

- `benchmarks/manifests/replay/shared_prefix_long_churn.json`
- `benchmarks/manifests/replay/retained_prefix_after_cleanup.json`
- `benchmarks/manifests/replay/mixed_live_and_retained_prefix_paths.json`
- `benchmarks/manifests/replay/full_prefix_to_decode_branch.json`
- `benchmarks/manifests/replay/memory_blocked_prefix_recovery.json`

Validate them with:

```text
bash scripts/check-bench-replay.sh
```

### Frozen matrix

The frozen dense matrix lives at:

- `benchmarks/manifests/matrix/mlx_dense_phase7.json`

Run it with:

```text
bash scripts/check-bench-matrix.sh
```

Compare two successful matrix result directories with:

```text
bash scripts/check-bench-matrix-compare.sh
```

### Delegated llama.cpp manifests

The llama.cpp examples validate non-MLX delegation contracts only:

- `benchmarks/manifests/scenario/llama_cpp_chat_qwen_short.json`
- `benchmarks/manifests/scenario/llama_cpp_shared_prefix_qwen_short.json`
- `benchmarks/manifests/replay/llama_cpp_submit_cancel_dual.json`
- `benchmarks/manifests/replay/llama_cpp_prompt_cache_reuse_dual.json`

They cover the SDK-owned `llama.cpp /completion` route, submit/cancel behavior,
safe delegated preset metadata, and backend-reported prompt-cache evidence.
Artifacts preserve `runtime.llama_cpp_preset` plus delegated prompt/decode
throughput, KV usage when available, processing/deferred request events, and
cache reuse. They are not AX-owned MLX model-inference throughput baselines.

When running them directly, update `runtime.backend_adapter.server_url`. The
repo smoke path is:

```text
bash scripts/check-bench-preview.sh
```

### Delegated mlx-lm compatibility

`mlx_lm_delegated` is for explicit upstream `mlx-lm` text-model compatibility
through a user-provided `mlx_lm.server`. Those checks should be stored and
described as delegated route-contract evidence only. They are not AX-owned MLX
model-inference throughput baselines, and they should not be merged into
`ax_engine_mlx` performance tables.

## Model-Inference Comparisons

For AX Engine MLX mode versus upstream MLX-family references, use:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/Qwen3.5-9B-MLX-4bit \
  --prompt-tokens 512,2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 5
```

That harness requires `mlx_lm.benchmark` as the primary standard and fails
closed if the matching baseline cannot be produced. It mirrors the upstream
random-token prompt standard and writes prompt token JSON artifacts for AX
Engine and any secondary reference. It can optionally ingest an explicit
`mlx-swift-lm` `BenchmarkHelpers` / `MLXLMCommon` generation adapter. The older
SwiftLM application-server benchmark is retired for current AX Engine
decisions.

Use `--ax-compare-policies` when both direct AX MLX and n-gram acceleration
rows matter. Every AX or optional Swift row is compared against the matching
`mlx_lm.benchmark` random-token prompt/decode shape.

Checked-in MLX inference-stack result sets live under
`benchmarks/results/mlx-inference/<date>/`. Each set should include the JSON
result document, prompt-token JSON artifacts, and the command log used to
produce the run.

Validate the harness contract without loading a model:

```text
bash scripts/check-bench-inference-stack.sh
```

## Online Serving Benchmarks

For user-visible serving latency and request-mix evidence, use:

```text
python3 scripts/bench_ax_serving.py \
  --base-url http://127.0.0.1:8080 \
  --model-id qwen3_dense \
  --corpus benchmarks/corpora/serving/smoke.jsonl \
  --input-kind tokens \
  --requests 12 \
  --warmup-requests 2 \
  --concurrency 2 \
  --output benchmarks/results/serving/$(date +%F)-qwen3-dense-smoke.json
```

This writes `ax.serving_benchmark.v1` artifacts with TTFT, client TPOT,
streaming step intervals, E2E latency, request throughput, output-token
throughput, queue delay, category summaries, and SLO goodput. Serving artifacts
are not model-inference throughput baselines and should not be merged into the
MLX inference-stack table.

For disk-durable prefix-cache serving promotion, first generate a deterministic
shared-prefix token corpus. The preferred path is the soak runner:

```text
python3 scripts/run_disk_prefix_serving_soak.py \
  --base-url http://127.0.0.1:8080 \
  --model-id qwen3_dense \
  --run-id <run-id>
```

It assumes the AX server is already running with `AX_MLX_PREFIX_CACHE_DIR` set,
then writes `corpus.jsonl`, `artifact.json`, `report.md`, and `commands.json`
under `benchmarks/results/serving/<run-id>/`. Use a fresh single-component
`--run-id`; existing non-empty run directories fail closed to protect evidence
from accidental overwrite.

For manual runs, generate the corpus first:

```text
python3 scripts/build_serving_shared_prefix_corpus.py \
  --output benchmarks/results/serving/disk-prefix-cache-soak-corpus.jsonl \
  --prompts 8 \
  --prefix-tokens 8192 \
  --suffix-tokens 64
```

Then run `bench_ax_serving.py` with `--input-kind tokens`, at least one warmup
corpus pass, and validate the artifact with
`--require-route-decision-min ax_mlx_prefix_cache_disk_hits=1`. The route gate
is required because a long-prompt serving artifact alone does not prove the
disk-cache path was exercised.

## Generated Outputs

`ax-engine-bench` writes result directories under the chosen `--output-root`.
Successful scenario and replay runs emit:

- `manifest.json`
- `environment.json`
- `metrics.json`
- `routes.json`
- `trace.json`
- `summary.md`

Contract failures emit `contract_failure.json` plus `summary.md` instead of
synthetic metrics. Compare, matrix, matrix-compare, baseline, and autotune
commands emit command-specific structured artifacts.
