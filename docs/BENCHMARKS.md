# Benchmarks

AX Engine v4 treats benchmarks as a contract surface. A benchmark result is
useful only when the workload, runtime route, reference engine, host, and
artifact schema are explicit.

## Benchmark Model

There are three benchmark surfaces:

| Surface | Purpose | Tooling | Result status |
|---|---|---|---|
| AX workload contract | Validate checked-in scenario, replay, matrix, baseline, compare, and delegated-route artifacts | `ax-bench` plus `benchmarks/manifests/` | Repo-owned contract evidence |
| MLX inference stack | Compare AX Engine MLX mode against MLX-family reference runtimes | `scripts/bench_mlx_inference_stack.py` | Model-inference performance evidence |
| llama.cpp delegation | Validate non-MLX route behavior and backend-reported prompt-cache evidence | llama.cpp manifests through `ax-bench` | Delegation contract evidence only |

Do not mix these surfaces in one throughput table. AX-owned inference
performance claims come from the MLX inference stack. llama.cpp results explain
delegation behavior, not AX MLX throughput.

## MLX Inference Stack

Use the MLX stack harness for model-inference performance comparisons:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/Qwen3.5-9B-MLX-4bit \
  --prompt-tokens 512,2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 5
```

The standard reference order is:

1. `mlx_lm.benchmark` as the canonical Python MLX baseline.
2. `mlx-swift-lm` only through an explicit JSON-emitting adapter.
3. AX Engine MLX mode through `ax-engine-server` SSE `runner_time_us`.

The older SwiftLM application-server benchmark is retired. It can remain useful
as historical design input, but it is not a valid baseline for AX Engine MLX
inference because it measures server and application behavior in addition to the
MLX library runtime.

For AX Engine MLX mode, run both decode modes when the question involves
speculative speedups:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/Qwen3.5-9B-MLX-4bit \
  --prompt-tokens 256,512,2048 \
  --generation-tokens 128 \
  --ax-both-modes
```

The JSON output labels AX rows as `ax_engine_mlx_greedy` and
`ax_engine_mlx_speculative`, so greedy correctness/performance and speculative
speedups do not collapse into one ambiguous number.

## AX Workload Contract

Canonical workload manifests live under:

- `benchmarks/manifests/scenario/`
- `benchmarks/manifests/replay/`
- `benchmarks/manifests/matrix/`

Use `ax-bench` for workload artifacts:

```text
ax-bench scenario --manifest benchmarks/manifests/scenario/chat_qwen_short.json --output-root benchmarks/results
ax-bench replay --manifest benchmarks/manifests/replay/shared_prefix_long_churn.json --output-root benchmarks/results
ax-bench matrix --manifest benchmarks/manifests/matrix/mlx_dense_phase7.json --output-root benchmarks/results
ax-bench compare --baseline benchmarks/results/<baseline> --candidate benchmarks/results/<candidate> --output-root benchmarks/results
```

Successful execution emits structured artifacts such as `manifest.json`,
`environment.json`, `metrics.json`, `routes.json`, `trace.json`, and
`summary.md`. Compare and matrix commands emit their own regression and summary
artifacts.

`ax-bench` is also the place for:

- correctness and determinism gates
- route identity and support-tier reporting
- prefix-reuse and prompt-cache provenance
- trusted-baseline snapshots
- matrix roll-ups and matrix comparison
- local readiness diagnostics through `ax-bench doctor`

## Delegated llama.cpp Checks

The repo carries llama.cpp scenario and replay examples, including:

- `benchmarks/manifests/scenario/llama_cpp_chat_qwen_short.json`
- `benchmarks/manifests/scenario/llama_cpp_shared_prefix_qwen_short.json`
- `benchmarks/manifests/replay/llama_cpp_submit_cancel_dual.json`
- `benchmarks/manifests/replay/llama_cpp_prompt_cache_reuse_dual.json`

These are delegated non-MLX route checks. They validate the SDK-owned
llama.cpp adapter contract, submit/cancel behavior, and backend-reported
prompt-cache evidence. They must not be described as AX-owned model-inference
throughput benchmarks.

When running them directly, update `runtime.backend_adapter.server_url` for the
local llama.cpp server. The repo smoke path is:

```text
bash scripts/check-bench-preview.sh
```

## Readiness Rules

Decision-grade AX MLX benchmark claims require:

- Apple Silicon M4-or-newer macOS/aarch64 host.
- MLX runtime availability.
- stable model, tokenizer, prompt shape, and sampling parameters.
- explicit runtime identity: selected backend, support tier, resolution policy,
  and feature flags.
- enough repetitions and cooldown to avoid one-off thermal or startup noise.

`ax-bench doctor` reports whether the local host is ready, bring-up-only, or
not ready for AX-owned MLX benchmarking:

```text
ax-bench doctor --json
```

llama.cpp adapters do not widen AX-owned MLX host support. They are separate
delegated runtime checks.

## Required Interpretation

Use this interpretation table when reading benchmark output:

| Evidence | Can support | Cannot support |
|---|---|---|
| `bench_mlx_inference_stack.py` MLX rows | AX MLX model-inference performance claims | scheduler/replay correctness claims |
| `ax-bench scenario` / `replay` MLX artifacts | route, correctness, determinism, and workload-contract claims | direct comparison to upstream MLX unless the MLX stack harness was also run |
| llama.cpp delegated artifacts | non-MLX route-contract and backend prompt-cache claims | AX-owned MLX throughput claims |
| `ax-bench compare` / `matrix-compare` | regression evidence inside the same manifest/runtime family | cross-runtime ranking without matching reference contract |

If the reference identity or runtime route is unclear, treat the number as
exploratory only.
