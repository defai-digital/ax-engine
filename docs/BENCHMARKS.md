# Benchmarks

AX Engine v4 keeps benchmarking deliberately split by evidence type. A result is
useful only when the workload, runtime route, reference engine, host, model,
sampling policy, and artifact schema are explicit.

## Which Tool To Use

| Question | Use | Evidence produced |
|---|---|---|
| How fast is AX Engine MLX mode against upstream MLX? | `scripts/bench_mlx_inference_stack.py` | Required `mlx_lm.benchmark` primary baseline rows, AX Engine MLX greedy/speculative rows, optional `mlx-swift-lm` secondary baseline adapter rows, canonical prompt-token artifacts, and ratio-to-baseline fields |
| Did a checked-in workload still pass route, correctness, determinism, replay, or regression gates? | `ax-engine-bench` | Workload-contract artifacts under `benchmarks/results` |
| Is the local host ready for AX-owned MLX benchmarking? | `ax-engine-bench doctor` | Human or JSON readiness report |
| Did a bounded runtime knob improve a frozen workload? | `ax-engine-bench autotune` | Autotune trial artifacts and warm-start history |
| Does the non-MLX delegated route still behave correctly? | llama.cpp manifests through `ax-engine-bench` | Delegated route-contract evidence only |

Do not merge these rows into one unlabeled throughput table. AX-owned
model-inference claims come from the MLX inference stack, and every such claim
must include a matching `mlx_lm.benchmark` baseline for the same random-token
prompt/decode shape, with prompt-token provenance recorded in the artifact.
`ax-engine-bench` owns workload contracts. llama.cpp manifests validate
delegation behavior, not AX MLX runtime speed.

## MLX Model-Inference Comparison

Use the MLX inference-stack harness when the question is performance versus an
MLX-family reference:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/Qwen3.5-9B-MLX-4bit \
  --prompt-tokens 512,2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 5
```

The reference contract is:

1. `mlx_lm.benchmark` is mandatory and is the canonical upstream Python MLX
   baseline.
2. The harness mirrors the `mlx_lm.benchmark` prompt standard:
   `mx.random.seed(0)` followed by
   `mx.random.randint(0, vocab_size, (1, prompt_tokens))`.
3. The generated prompt token IDs are written as JSON artifacts and reused by
   AX Engine and any admitted `mlx-swift-lm` adapter.
4. AX Engine MLX mode is measured through `ax-engine-server` SSE
   `runner_time_us`.
5. `mlx-swift-lm` is optional and only admitted as a secondary reference
   through an explicit JSON-emitting adapter command built on the
   `mlx-swift-lm` `BenchmarkHelpers` / `MLXLMCommon` generation APIs.

The harness fails closed if `mlx_lm.benchmark` cannot run. It does not support
AX-only throughput tables. Every non-baseline row records the matching
`mlx_lm.benchmark` random-token prompt/decode shape plus prefill/decode ratios.
The direct AX comparison row is greedy by default. Speculative AX rows are
feature-speedup evidence and must not be treated as the same decode policy as
the primary MLX baseline.

Use `--ax-both-modes` when speculative decode is part of the question:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/Qwen3.5-9B-MLX-4bit \
  --prompt-tokens 256,512,2048 \
  --generation-tokens 128 \
  --ax-both-modes
```

The harness labels AX rows as `ax_engine_mlx_greedy` and
`ax_engine_mlx_speculative`. Greedy throughput and speculative speedups must not
collapse into one AX number.

The `mlx-swift-lm` reference checkout currently exposes `BenchmarkHelpers` for
loading, tokenization, decoding, download-cache timing, and shared integration
benchmark scaffolding, but not a repo-stable standalone LLM inference
benchmark CLI equivalent to `python3 -m mlx_lm.benchmark`. For AX Engine
inference comparisons, treat `mlx-swift-lm` as a secondary baseline only when
the adapter:

- is built against the reference `mlx-swift-lm` package, not a custom
  application server,
- uses `MLXLMCommon` generation APIs with `GenerateParameters(maxTokens: ...,
  temperature: 0, prefillStepSize: ...)`,
- reads the harness-emitted `{prompt_token_ids_path}`,
- reports prefill and decode throughput for the same random-token prompt/decode
  shape, and
- emits raw trial rows when possible so the harness can compute shared
  mean/median/min/max summaries.

The older SwiftLM application-server benchmark is retired for current AX Engine
decisions. It measures application/server behavior around MLX, not the
`mlx-swift-lm` library benchmark contract. Keep it as historical design input
only.

## AX Workload Contracts

`ax-engine-bench` is the repo-owned CLI for checked-in workload artifacts:

```text
ax-engine-bench scenario --manifest benchmarks/manifests/scenario/chat_qwen_short.json --output-root benchmarks/results
ax-engine-bench replay --manifest benchmarks/manifests/replay/shared_prefix_long_churn.json --output-root benchmarks/results
ax-engine-bench matrix --manifest benchmarks/manifests/matrix/mlx_dense_phase7.json --output-root benchmarks/results
ax-engine-bench compare --baseline benchmarks/results/<baseline> --candidate benchmarks/results/<candidate> --output-root benchmarks/results
ax-engine-bench matrix-compare --baseline benchmarks/results/<baseline-matrix> --candidate benchmarks/results/<candidate-matrix> --output-root benchmarks/results
ax-engine-bench baseline --source benchmarks/results/<run> --name "Dense Qwen Trusted" --output-root benchmarks/baselines
```

Canonical manifests live under:

- `benchmarks/manifests/scenario/`
- `benchmarks/manifests/replay/`
- `benchmarks/manifests/matrix/`

Successful execution emits structured artifacts such as:

- `manifest.json`
- `environment.json`
- `metrics.json`
- `routes.json`
- `trace.json`
- `summary.md`

Compare and matrix commands emit regression and roll-up artifacts. Baseline
commands copy a successful result into a named trusted baseline and fail closed
instead of overwriting an existing baseline.

Use this surface for:

- correctness and determinism gates
- route identity and support-tier reporting
- prefix-reuse and prompt-cache provenance
- replay and churn validation
- matrix roll-ups
- matrix-to-matrix regression review
- bounded autotune trials over frozen manifests

## Bounded Autotune

`ax-engine-bench autotune` is part of the workload-contract surface, not a broad
architecture search system:

```text
ax-engine-bench autotune \
  --manifest benchmarks/manifests/scenario/chat_qwen_short.json \
  --output-root benchmarks/results \
  --iterations 8 \
  --max-batch-token-options 2048,4096,8192 \
  --kv-total-block-options 256,512 \
  --prefix-cache-options true,false
```

Autotune may explore bounded knobs already represented in manifests. It must not
turn runtime-family selection, backend ownership, or scheduler architecture into
implicit search dimensions. Treat the output as candidate evidence that still
needs the normal scenario/replay/compare gates.

## Delegated llama.cpp Checks

The repo carries delegated non-MLX examples:

- `benchmarks/manifests/scenario/llama_cpp_chat_qwen_short.json`
- `benchmarks/manifests/scenario/llama_cpp_shared_prefix_qwen_short.json`
- `benchmarks/manifests/replay/llama_cpp_submit_cancel_dual.json`
- `benchmarks/manifests/replay/llama_cpp_prompt_cache_reuse_dual.json`

These validate the SDK-owned llama.cpp adapter contract, submit/cancel behavior,
and backend-reported prompt-cache evidence. They must not be described as
AX-owned model-inference throughput benchmarks.

When running them directly, update `runtime.backend_adapter.server_url` for the
local llama.cpp server. The repo smoke path is:

```text
bash scripts/check-bench-preview.sh
```

## Readiness

Decision-grade AX MLX benchmark claims require:

- Apple Silicon M4-or-newer macOS/aarch64 host.
- MLX runtime availability.
- Stable model, tokenizer, prompt shape, and sampling parameters.
- Prompt-token provenance matching the `mlx_lm.benchmark` random-token
  standard, including the saved prompt JSON path and SHA-256 hash.
- Explicit runtime identity: selected backend, support tier, resolution policy,
  feature flags, and AX decode mode.
- Enough repetitions and cooldown to avoid one-off thermal or startup noise.
- Raw JSON artifacts preserved next to any summarized table.

Inspect local readiness with:

```text
ax-engine-bench doctor --json
```

llama.cpp adapters do not widen AX-owned MLX host support. They are separate
delegated runtime checks.

## Smoke Checks

Use these repo-owned checks to validate the benchmark surfaces:

```text
bash scripts/check-bench-inference-stack.sh
bash scripts/check-bench-doctor.sh
bash scripts/check-bench-mlx.sh
bash scripts/check-bench-replay.sh
bash scripts/check-bench-preview.sh
bash scripts/check-bench-matrix.sh
bash scripts/check-bench-matrix-compare.sh
```

## Interpretation Rules

| Evidence | Can support | Cannot support |
|---|---|---|
| `bench_mlx_inference_stack.py` rows with matching `mlx_lm.benchmark` baseline | AX MLX model-inference performance claims against named MLX references | Scheduler/replay correctness claims |
| `ax-engine-bench scenario` / `replay` MLX artifacts | Workload-contract, route, correctness, determinism, replay, and regression claims | Direct upstream MLX comparison unless the MLX stack harness was also run |
| `ax-engine-bench autotune` artifacts | Candidate evidence for bounded manifest knobs | Architecture selection or cross-runtime ranking |
| llama.cpp delegated artifacts | Non-MLX route-contract and backend prompt-cache claims | AX-owned MLX throughput claims |
| `ax-engine-bench compare` / `matrix-compare` | Regression evidence inside the same manifest/runtime family | Cross-runtime ranking without matching reference contract |

If the reference identity or runtime route is unclear, treat the number as
exploratory only.
