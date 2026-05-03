# CLI

AX Engine v4 currently exposes:

- a benchmark-focused bring-up CLI through `ax-bench`
- a thin direct inference CLI through `ax-bench generate` / `ax-bench stream`
- a preview local HTTP adapter through `ax-engine-server`

## Commands

The current command surface is:

- `ax-bench scenario`
- `ax-bench replay`
- `ax-bench compare`
- `ax-bench matrix-compare`
- `ax-bench baseline`
- `ax-bench matrix`
- `ax-bench generate`
- `ax-bench stream`
- `ax-bench doctor`
- `ax-bench metal-build`
- `ax-engine-server`

Example usage:

```text
ax-bench scenario --manifest benchmarks/manifests/scenario/coding_qwen_medium.json --output-root benchmarks/results
ax-bench replay --manifest benchmarks/manifests/replay/shared_prefix_long_churn.json --output-root benchmarks/results
ax-bench compare --baseline benchmarks/results/<baseline> --candidate benchmarks/results/<candidate> --output-root benchmarks/results
ax-bench matrix-compare --baseline benchmarks/results/<baseline-matrix> --candidate benchmarks/results/<candidate-matrix> --output-root benchmarks/results
ax-bench baseline --source benchmarks/results/<run> --name "Dense Qwen Trusted" --output-root benchmarks/baselines
ax-bench matrix --manifest benchmarks/manifests/matrix/frozen_native_dense_phase7.json --output-root benchmarks/results
ax-bench generate --tokens 1,2,3 --max-output-tokens 4
ax-bench generate --prompt "Hello from AX" --support-tier compatibility --compat-backend vllm --compat-server-url http://127.0.0.1:8000
ax-bench generate --tokens 1,2,3 --native-runtime-artifacts-dir build/metal --native-model-artifacts-dir /tmp/ax-model
ax-bench stream --tokens 1,2,3 --support-tier compatibility --compat-server-url http://127.0.0.1:8081 --json
ax-bench doctor --json
ax-bench metal-build
cargo run -p ax-engine-server -- --model-id qwen3_dense --compat-cli-path python3 --compat-model-path /absolute/path/to/mlx-model --port 8080
```

## Current State

The CLI currently validates:

- required subcommands
- required flags
- manifest path existence
- output-root shape checks
- manifest class and required shape
- local host and Metal-toolchain readiness reporting through `ax-bench doctor`
- repo-owned Metal kernel artifact compilation through `ax-bench metal-build`
- runtime correctness and determinism gates for the current bring-up path
- structured artifact emission for execution and compare flows, including
  prefix-reuse provenance, explicit prefix-cache evidence, delegated cached
  prompt-token counts, and compare-time runtime identity on delegated
  benchmark paths
- preview server request translation through the SDK-backed generation and
  request lifecycle paths
- preview SSE transport through the SDK-backed lifecycle path
- direct blocking and streaming inference through the SDK-owned session
  contract, including compatibility backend resolution

It does not yet perform full production benchmarking.
The current implementation executes the deterministic engine bring-up path, so
the artifacts reflect real request progress, route metadata, and synthetic
runtime observations rather than placeholder files.
`ax-bench` now also supports delegated compatibility benchmarking through the
SDK-owned backend contract. Server-backed `vLLM`, `mistral.rs`, and MLX routes
currently run scenario manifests through a blocking compatibility runtime, with
the explicit contract that those manifests stay single-request
(`shape.concurrency=1`) and do not require delegated prefix reuse. The
stepwise `llama.cpp /completion` adapter still carries the broader delegated
coverage for multi-request scenario and replay manifests, including submit /
cancel shapes driven through one shared SDK session. Replay and compatibility
`shared_prefix` scenarios still surface backend-managed prompt-cache reuse only
through that stepwise delegated telemetry. Unsupported delegated shapes still
emit a `contract_failure.json` artifact plus `summary.md` instead of synthetic
execution metrics.
Successful compare runs now also carry the resolved compare runtime identity in
`regression.json` and `comparison.md`, so compatibility results do not read like
native bring-up compares; delegated prompt-cache runs also surface the
backend-reported cached-token count directly instead of hiding it only inside
route crossover metadata.
`ax-bench baseline` now snapshots one successful benchmark result into a named
trusted baseline directory, copies the core execution artifacts, emits
`trusted_baseline.json` plus `trusted_baseline.md`, and fails closed instead of
overwriting an existing baseline name. Compare summaries also surface that
trusted baseline name when present, so later regression reviews can anchor back
to an intentionally frozen result rather than an arbitrary prior run folder.
`ax-bench matrix` now executes the checked-in frozen native scenario matrix as
one Tier 2 roll-up, writes per-member execution artifacts beneath one matrix
result directory, and emits `matrix.json` plus `summary.md` so dense-path
performance work can reference a canonical scenario set instead of ad-hoc
single-run folders.
`ax-bench matrix-compare` now compares two successful matrix result directories
that come from the same frozen manifest fingerprint, reuses the existing
per-member compare contract for each scenario, and emits a matrix-level
`matrix_regression.json` plus `summary.md` so frozen-matrix drift can be
reviewed as one roll-up instead of manually opening each member folder.
The checked-in compatibility examples live at
`benchmarks/manifests/scenario/compatibility_chat_qwen_short.json` and
`benchmarks/manifests/scenario/compatibility_chat_qwen_short_vllm.json`, plus
`benchmarks/manifests/scenario/compatibility_chat_qwen_short_mistral_rs.json`,
plus `benchmarks/manifests/scenario/compatibility_chat_qwen_short_mlx.json`,
plus
`benchmarks/manifests/scenario/compatibility_shared_prefix_qwen_short.json`,
plus
`benchmarks/manifests/replay/compatibility_submit_cancel_dual.json`, plus the
delegated prompt-cache reuse replay example at
`benchmarks/manifests/replay/compatibility_prompt_cache_reuse_dual.json`;
update their `server_url` before running them directly.
`ax-bench doctor` now turns the SDK-owned host and Metal-toolchain diagnostics
into one human-readable or JSON readiness report. It distinguishes fully
supported M4-or-newer native hosts from unsupported-host override bring-up, and
it keeps the rule explicit that compatibility adapters do not widen AX native
host support.
`ax-bench metal-build` now owns the checked-in Phase 1 Metal build contract in
Rust, writing `doctor.json`, `build_report.json`, and `summary.md` into the
target build directory while driving the explicit `xcrun metal -> metal-ar ->
metallib` pipeline when the local toolchain is actually available; when that
same output directory already contains validated compiled artifacts for the
same checked-in contract, the command now reuses them instead of recompiling
unnecessarily.
That `metal-ar` stage is an AX-owned artifact-contract choice for the checked-in
bring-up path, not a claim that every reviewed upstream Metal stack uses the
same intermediate.
The checked-in Phase 1 native contract also keeps KV block-size support narrow:
Metal-backed native sessions currently validate only `block_size_tokens=16`
until more than one native block-size shape is proven by real kernels.
The preview server is intentionally narrow and currently exposes token-based
bring-up endpoints rather than a broad compatibility API surface, including a
shared-session stepwise request lifecycle for local integration.

## Direct Inference

`ax-bench generate` and `ax-bench stream` are intentionally thin wrappers over
`ax-engine-sdk`.

They support:

- native preview requests with `--tokens`
- compatibility requests with `--prompt` or `--tokens`
- delegated backend selection via `--compat-backend`
- delegated targets via `--compat-server-url` or local CLI fallback flags
- explicit native runtime/model artifact selection via
  `--native-runtime-artifacts-dir` and `--native-model-artifacts-dir`

Examples:

```text
ax-bench generate --tokens 1,2,3 --max-output-tokens 4
```

```text
ax-bench generate \
  --tokens 1,2,3 \
  --native-runtime-artifacts-dir build/metal \
  --native-model-artifacts-dir /tmp/ax-model
```

```text
ax-bench generate \
  --prompt "Hello from AX" \
  --support-tier compatibility \
  --compat-backend mistral-rs \
  --compat-server-url http://127.0.0.1:8000
```

```text
ax-bench stream \
  --tokens 1,2,3 \
  --support-tier compatibility \
  --compat-server-url http://127.0.0.1:8081 \
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
