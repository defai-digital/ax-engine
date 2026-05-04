# CLI

AX Engine v4 currently exposes:

- a benchmark-focused bring-up CLI through `ax-engine-bench`
- a thin direct inference CLI through `ax-engine-bench generate` / `ax-engine-bench stream`
- a preview local HTTP adapter through `ax-engine-server`

## Commands

The current command surface is:

- `ax-engine-bench scenario`
- `ax-engine-bench replay`
- `ax-engine-bench compare`
- `ax-engine-bench matrix-compare`
- `ax-engine-bench baseline`
- `ax-engine-bench matrix`
- `ax-engine-bench generate`
- `ax-engine-bench stream`
- `ax-engine-bench doctor`
- `ax-engine-bench metal-build`
- `ax-engine-server`
- `scripts/bench_mlx_inference_stack.py`

Example usage:

```text
ax-engine-bench scenario --manifest benchmarks/manifests/scenario/coding_qwen_medium.json --output-root benchmarks/results
ax-engine-bench replay --manifest benchmarks/manifests/replay/shared_prefix_long_churn.json --output-root benchmarks/results
ax-engine-bench compare --baseline benchmarks/results/<baseline> --candidate benchmarks/results/<candidate> --output-root benchmarks/results
ax-engine-bench matrix-compare --baseline benchmarks/results/<baseline-matrix> --candidate benchmarks/results/<candidate-matrix> --output-root benchmarks/results
ax-engine-bench baseline --source benchmarks/results/<run> --name "Dense Qwen Trusted" --output-root benchmarks/baselines
ax-engine-bench matrix --manifest benchmarks/manifests/matrix/mlx_dense_phase7.json --output-root benchmarks/results
ax-engine-bench generate --tokens 1,2,3 --max-output-tokens 4
ax-engine-bench generate --prompt "Hello from AX" --support-tier llama_cpp --llama-server-url http://127.0.0.1:8081
ax-engine-bench generate --tokens 1,2,3 --mlx --mlx-model-artifacts-dir /tmp/mlx-model-artifacts
ax-engine-bench stream --tokens 1,2,3 --support-tier llama_cpp --llama-server-url http://127.0.0.1:8081 --json
ax-engine-bench doctor --json
ax-engine-bench metal-build
cargo run -p ax-engine-server -- --model-id qwen3_dense --mlx --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts --port 8080
python3 scripts/bench_mlx_inference_stack.py --model-dir .internal/models/Qwen3.5-9B-MLX-4bit --prompt-tokens 512,2048 --generation-tokens 128
```

## Current State

The CLI currently validates:

- required subcommands
- required flags
- manifest path existence
- output-root shape checks
- manifest class and required shape
- local host and Metal-toolchain readiness reporting through `ax-engine-bench doctor`
- repo-owned Metal kernel artifact compilation through `ax-engine-bench metal-build`
- runtime correctness and determinism gates for the current bring-up path
- structured artifact emission for execution and compare flows, including
  prefix-reuse provenance, explicit prefix-cache evidence, delegated cached
  prompt-token counts, and compare-time runtime identity on delegated check
  paths
- preview server request translation through the SDK-backed generation and
  request lifecycle paths
- preview SSE transport through the SDK-backed lifecycle path
- direct blocking and streaming inference through the SDK-owned session
  contract, including llama.cpp backend resolution

`ax-engine-bench` is not the MLX-family reference-inference comparator; that role
belongs to `scripts/bench_mlx_inference_stack.py`. Production release gates are
still future work.
`ax-engine-bench` owns workload-contract artifacts rather than upstream model-runtime
comparison. The current implementation executes the deterministic SDK/MLX
bring-up path for checked-in workloads, so artifacts reflect request progress,
route metadata, runtime identity, and measured runner timing instead of
placeholder files.

Model-inference performance comparisons are intentionally outside the
`ax-engine-bench scenario` / `replay` surface. Use
`scripts/bench_mlx_inference_stack.py` to compare AX Engine MLX mode against
`mlx_lm.benchmark`, with optional `mlx-swift-lm` JSON adapter support. The
retired SwiftLM application-server benchmark should not be used as a current AX
MLX baseline.

`ax-engine-bench` also supports delegated llama.cpp contract checks through the
SDK-owned backend contract. The stepwise `llama.cpp /completion` adapter carries
the broader delegated coverage for multi-request scenario and replay manifests,
including submit / cancel shapes driven through one shared SDK session. Replay
and llama.cpp `shared_prefix` scenarios still surface backend-managed
prompt-cache reuse only through that stepwise delegated telemetry. Unsupported
delegated shapes still emit a `contract_failure.json` artifact plus
`summary.md` instead of synthetic execution metrics.
Successful compare runs now also carry the resolved compare runtime identity in
`regression.json` and `comparison.md`, so llama.cpp results do not read like
MLX compares; delegated prompt-cache runs also surface the
backend-reported cached-token count directly instead of hiding it only inside
route crossover metadata.
`ax-engine-bench baseline` now snapshots one successful benchmark result into a named
trusted baseline directory, copies the core execution artifacts, emits
`trusted_baseline.json` plus `trusted_baseline.md`, and fails closed instead of
overwriting an existing baseline name. Compare summaries also surface that
trusted baseline name when present, so later regression reviews can anchor back
to an intentionally frozen result rather than an arbitrary prior run folder.
`ax-engine-bench matrix` now executes the checked-in MLX scenario matrix as
one Tier 2 roll-up, writes per-member execution artifacts beneath one matrix
result directory, and emits `matrix.json` plus `summary.md` so dense-path
performance work can reference a canonical scenario set instead of ad-hoc
single-run folders.
`ax-engine-bench matrix-compare` now compares two successful matrix result directories
that come from the same frozen manifest fingerprint, reuses the existing
per-member compare contract for each scenario, and emits a matrix-level
`matrix_regression.json` plus `summary.md` so frozen-matrix drift can be
reviewed as one roll-up instead of manually opening each member folder.
The checked-in llama.cpp examples live at
`benchmarks/manifests/scenario/llama_cpp_chat_qwen_short.json` and
`benchmarks/manifests/scenario/llama_cpp_shared_prefix_qwen_short.json`,
`benchmarks/manifests/replay/llama_cpp_submit_cancel_dual.json`, plus the
delegated prompt-cache reuse replay example at
`benchmarks/manifests/replay/llama_cpp_prompt_cache_reuse_dual.json`; update their `server_url` before running them directly.
Those manifests validate delegated non-MLX route behavior only; do not compare
their throughput rows against AX Engine MLX rows as if both were AX-owned
inference runtimes.
`ax-engine-bench doctor` now turns the SDK-owned host and Metal-toolchain diagnostics
into one human-readable or JSON readiness report. It distinguishes fully
supported M4-or-newer MLX hosts from unsupported-host override bring-up, and it
keeps the rule explicit that llama.cpp adapters do not widen AX-owned MLX host
support.
`ax-engine-bench metal-build` now owns the checked-in Phase 1 Metal build contract in
Rust, writing `doctor.json`, `build_report.json`, and `summary.md` into the
target build directory while driving the explicit `xcrun metal -> metal-ar ->
metallib` pipeline when the local toolchain is actually available; when that
same output directory already contains validated compiled artifacts for the
same checked-in contract, the command now reuses them instead of recompiling
unnecessarily.
That `metal-ar` stage is an AX-owned artifact-contract choice for the checked-in
bring-up path, not a claim that every reviewed upstream Metal stack uses the
same intermediate.
The checked-in Phase 1 MLX Metal contract also keeps KV block-size support narrow:
Metal-backed MLX sessions currently validate only `block_size_tokens=16`
until more than one MLX Metal block-size shape is proven by real kernels.
The preview server is intentionally narrow and currently exposes token-based
bring-up endpoints rather than a broad llama.cpp API surface, including a
shared-session stepwise request lifecycle for local integration.

## Direct Inference

`ax-engine-bench generate` and `ax-engine-bench stream` are intentionally thin wrappers over
`ax-engine-sdk`.

They support:

- MLX preview requests with `--mlx --tokens`
- llama.cpp requests with `--prompt` or `--tokens`
- llama.cpp targets via `--llama-server-url` or local CLI fallback flags
- explicit MLX model artifact selection via `--mlx-model-artifacts-dir`

Examples:

```text
ax-engine-bench generate --tokens 1,2,3 --max-output-tokens 4
```

```text
ax-engine-bench generate \
  --tokens 1,2,3 \
  --mlx \
  --mlx-model-artifacts-dir /tmp/mlx-model-artifacts
```

```text
ax-engine-bench generate \
  --prompt "Hello from AX" \
  --support-tier llama_cpp \
  --llama-server-url http://127.0.0.1:8081
```

```text
ax-engine-bench stream \
  --tokens 1,2,3 \
  --support-tier llama_cpp \
  --llama-server-url http://127.0.0.1:8081 \
  --json
```

By default, `generate` prints the primary payload first and then a compact
metadata suffix that includes request id, status, finish reason,
execution-plan identity, and per-token logprob data when present.

`stream` now also includes route identity, finish reason, `delta_text`, and
`delta_token_logprobs` in its human-readable step output instead of hiding that
information behind JSON-only mode.

Use `--json` on either command for the full structured payload, including
`output_token_logprobs`, request-level `finish_reason` /
`terminal_stop_reason`, and streaming `delta_token_logprobs`.

## Design Intent

The CLI exists to support:

- reproducible benchmark execution
- replay workload validation
- regression comparison
- frozen matrix execution and roll-up reporting
- frozen matrix regression roll-up reporting
- local readiness diagnosis before benchmark or kernel bring-up work
- local transport-layer integration against the SDK contract

As the runtime matures, this CLI will become the main entry point for benchmark
and replay workflows.
