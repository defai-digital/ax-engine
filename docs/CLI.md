# CLI

AX Engine currently exposes three command surfaces:

- `ax-engine-bench` for workload contracts, readiness, bounded autotune, Metal
  build checks, and thin direct SDK inference helpers.
- `scripts/bench_mlx_inference_stack.py` for repo-owned MLX runtime
  model-inference comparison against MLX-family references.
- `ax-engine-server` for the preview local HTTP adapter.
- `ax-engine-manager` for the read-only Ratatui cockpit over those contracts.

For a step-by-step manager workflow, including server metadata, benchmark
artifacts, support bundles, and release checks, see [`MANAGER.md`](MANAGER.md).

## `ax-engine-bench`

`ax-engine-bench` is the repo-owned workload-contract CLI. It is not the
upstream MLX reference comparator.

Current command surface:

- `ax-engine-bench scenario`
- `ax-engine-bench replay`
- `ax-engine-bench autotune`
- `ax-engine-bench compare`
- `ax-engine-bench matrix-compare`
- `ax-engine-bench baseline`
- `ax-engine-bench matrix`
- `ax-engine-bench doctor`
- `ax-engine-bench metal-build`
- `ax-engine-bench generate`
- `ax-engine-bench stream`

Examples:

```text
ax-engine-bench scenario --manifest benchmarks/manifests/scenario/coding_qwen_medium.json --output-root benchmarks/results
ax-engine-bench replay --manifest benchmarks/manifests/replay/shared_prefix_long_churn.json --output-root benchmarks/results
ax-engine-bench matrix --manifest benchmarks/manifests/matrix/mlx_dense_phase7.json --output-root benchmarks/results
ax-engine-bench compare --baseline benchmarks/results/<baseline> --candidate benchmarks/results/<candidate> --output-root benchmarks/results
ax-engine-bench matrix-compare --baseline benchmarks/results/<baseline-matrix> --candidate benchmarks/results/<candidate-matrix> --output-root benchmarks/results
ax-engine-bench baseline --source benchmarks/results/<run> --name "Dense Qwen Trusted" --output-root benchmarks/baselines
ax-engine-bench autotune --manifest benchmarks/manifests/scenario/chat_qwen_short.json --output-root benchmarks/results --iterations 8
ax-engine-bench doctor --json
ax-engine-bench metal-build
```

Add `--json` to `scenario`, `replay`, `matrix`, `compare`,
`matrix-compare`, or `baseline` to emit an `ax.benchmark_artifact.v1` summary
with the written `result_dir`. This is the stable automation path for CI and
TUI callers; the legacy text output remains available for shell use.

`ax-engine-bench doctor --json` also reports workflow discovery. When run from a
source checkout it points callers at `cargo run ...`, `scripts/download_model.py`,
and the checkout root; outside a checkout it points callers at installed
`ax-engine-bench` and `ax-engine-server` binaries. TUI code should use this
contract instead of guessing Homebrew versus source mode from paths.

When `--mlx-model-artifacts-dir <path>` is provided, the doctor JSON includes a
structured `model_artifacts` report for `config.json`, `model-manifest.json`,
safetensors presence, `model_type`, quantization metadata, and readiness
blockers. TUI code should display those fields directly instead of parsing
performance-advice text.

Run `bash scripts/check-cli-tui-phase0.sh` to verify the full Phase 0 contract
set before adding Ratatui code. The check covers download JSON,
manifest-generation JSON, doctor workflow discovery, model-artifact readiness,
benchmark artifact summaries, and server `/health`, `/v1/runtime`, and
`/v1/models` metadata.

Successful scenario and replay runs emit `manifest.json`, `environment.json`,
`metrics.json`, `routes.json`, `trace.json`, and `summary.md`. Contract failures
emit `contract_failure.json` plus `summary.md` instead of synthetic metrics.

`baseline`, `compare`, `matrix`, `matrix-compare`, and `autotune` build on
those artifacts. They should only compare or tune results inside the same
manifest/runtime family.

## `ax-engine-manager`

`ax-engine-manager` is the Phase 1 read-only TUI shell. It does not start
downloads, benchmarks, or servers. It may run doctor, read existing JSON
contracts, poll server metadata, and browse local artifact directories.

```text
ax-engine-manager --check
ax-engine-manager --model-dir /path/to/model
ax-engine-manager --server-url http://127.0.0.1:8080
ax-engine-manager --doctor-json /path/to/doctor.json --artifact-root benchmarks/results
ax-engine-manager --doctor-json /path/to/doctor.json --support-bundle /tmp/ax-support
```

`--check` prints a non-interactive summary for doctor, server, benchmark, and
artifact readiness without entering terminal raw mode.

The `Jobs` tab is still non-mutating in Phase 2. It projects the doctor workflow
into guarded job plans with explicit job kind, evidence class, process ownership,
command, and benchmark artifact requirements.

`--phase2-check` verifies the local job-runner foundation without launching real
downloads, benchmarks, or servers. It builds the job plan from doctor workflow
JSON, writes and reads a manager profile under `--profile-dir`, runs a fake
short-lived job, cancels a fake owned server process, and checks that benchmark
results cannot be presented without an artifact path.

Run `bash scripts/check-cli-tui-phase1.sh` before extending the TUI. The check
covers contract parsers, tab rendering snapshots, server polling against a local
test HTTP server, non-interactive `--check`, and unsupported/missing-state UI.
Run `bash scripts/check-cli-tui-phase2.sh` before adding interactive job
controls.
Run `bash scripts/check-cli-tui-phase3.sh` before publishing Homebrew artifacts.

See [`MANAGER.md`](MANAGER.md) for user-facing usage examples and troubleshooting.

## MLX Inference Stack

Use `scripts/bench_mlx_inference_stack.py` when the question is repo-owned MLX
runtime throughput versus upstream MLX-family runtimes:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 512,2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 5
```

`mlx_lm.benchmark` is mandatory. The script fails closed if the matching
baseline cannot be produced, and every AX or `mlx-swift-lm` row carries
ratio-to-`mlx_lm.benchmark` fields for the same random-token prompt/decode
shape. The harness mirrors the upstream prompt standard (`mx.random.seed(0)`
plus random token IDs from the model vocabulary) and writes the prompt token
JSON path and hash into the artifact. AX direct decode is the default direct
comparison; use `--ax-compare-policies` to also emit `ax_engine_mlx_ngram_accel`
feature-speedup rows. `mlx-swift-lm` is accepted only as a secondary baseline
through an explicit `BenchmarkHelpers` / `MLXLMCommon` generation adapter that
reads the emitted prompt token JSON. The older SwiftLM application-server
benchmark is retired.

For TurboQuant evidence collection, the MLX inference-stack harness can pass
through the server's experimental shadow policy:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 8192 \
  --generation-tokens 256 \
  --experimental-mlx-kv-compression turboquant-shadow
```

This mode is optional, disabled by default, and records TurboQuant route
metadata only when the runtime emits it. The current implementation is the
full-precision shadow path: generation still uses the existing MLX KV decode
path, while the optional side path records eligibility, estimated saved KiB,
runtime shadow-storage writes, shadow-storage sync calls and wall time, current
compression decode path, and fused decode candidate/attempt/success/fallback
counters. It is not a production support claim and should be paired with
`scripts/check_turboquant_quality_artifact.py` before being used as promotion
evidence.

For runner-route experiments, `--experimental-mlx-kv-compression
turboquant-fused-experimental` requests compressed decode selection and may
report `fused_compressed_decode` when the eligible K8/V4 single-token path uses
the two-stage Metal cold decode plus full-precision hot-tail merge. If Metal is
unavailable but the reference fallback works, it reports
`cpu_oracle_compressed_decode`. A fallback reason label of
`runner_not_integrated` means no runtime decode attempt was observed yet;
`cpu_oracle_unavailable` means both compressed-decode attempts fell back to the
full-precision MLX KV path. Only `fused_compressed_decode` route evidence with
successful attempts and zero fallbacks can feed the internal quality artifact
gate; shadow and CPU oracle rows are diagnostic only.

## Delegated llama.cpp Checks

`ax-engine-bench` can run delegated llama.cpp scenario and replay manifests
through the SDK-owned backend contract. Those runs validate non-MLX route
behavior, submit/cancel behavior, safe delegated preset metadata, and backend
prompt-cache evidence. The manifests expose parallel slots, continuous
batching, logical/physical batch sizing, cache-prompt intent, slot
save/restore path state, speculative decode mode, and metrics endpoint capture.
Artifacts keep the preset in `runtime.llama_cpp_preset` and record delegated
prompt/decode throughput, KV usage when available, processing/deferred request
events, and cache reuse under `metrics.delegated_llama_cpp`. They are not
repo-owned MLX model-inference benchmarks.

Checked-in examples:

- `benchmarks/manifests/scenario/llama_cpp_chat_qwen_short.json`
- `benchmarks/manifests/scenario/llama_cpp_shared_prefix_qwen_short.json`
- `benchmarks/manifests/replay/llama_cpp_submit_cancel_dual.json`
- `benchmarks/manifests/replay/llama_cpp_prompt_cache_reuse_dual.json`

Update `runtime.backend_adapter.server_url` before running them directly, or use:

```text
bash scripts/check-bench-preview.sh
```

## Delegated mlx-lm Compatibility

For an MLX text model that is supported by upstream `mlx-lm` but not yet by
the repo-owned MLX runtime, run `mlx_lm.server` yourself and select the explicit
delegated route:

```text
mlx_lm.server --model /absolute/path/to/mlx-model --host 127.0.0.1 --port 8090

ax-engine-bench generate \
  --prompt "Hello from mlx-lm" \
  --support-tier mlx_lm_delegated \
  --mlx-lm-server-url http://127.0.0.1:8090
```

This path is text-only delegated compatibility through `mlx_lm.server`.
Streaming surfaces forward delegated text deltas through AX envelopes, but they
do not provide AX-owned token IDs, KV state, or MLX kernel throughput. CLI
benchmark rows remain delegated route-contract evidence, not repo-owned MLX
performance claims, and should not be mixed into `ax_engine_mlx` throughput
tables.

## Direct Inference Helpers

`ax-engine-bench generate` and `ax-engine-bench stream` are thin wrappers over
`ax-engine-sdk`.

Examples:

```text
ax-engine-bench generate --tokens 1,2,3 --max-output-tokens 4
ax-engine-bench generate --tokens 1,2,3 --mlx --mlx-model-artifacts-dir /tmp/mlx-model-artifacts
ax-engine-bench generate --prompt "Hello from AX" --support-tier llama_cpp --llama-server-url http://127.0.0.1:8081
ax-engine-bench generate --prompt "Hello from mlx-lm" --support-tier mlx_lm_delegated --mlx-lm-server-url http://127.0.0.1:8090
ax-engine-bench stream --tokens 1,2,3 --support-tier llama_cpp --llama-server-url http://127.0.0.1:8081 --json
```

By default, `generate` prints the primary payload first and then compact
metadata with request id, status, finish reason, execution plan, and token
logprob data when present. Use `--json` for the full structured payload.

## Server

Start the preview server with the repo-owned MLX runtime:

```text
cargo run -p ax-engine-server -- --model-id qwen3_dense --mlx --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts --port 8080
```

For comparable repo-owned MLX inference numbers, use
`scripts/bench_mlx_inference_stack.py` over manual server timing. The script
starts the server, captures AX SSE `runner_time_us`, runs the required
`mlx_lm.benchmark` baseline, and records the reference runtime identity.

## Design Intent

The CLI exists to support:

- reproducible workload-contract execution
- replay workload validation
- regression comparison
- frozen matrix execution and roll-up reporting
- bounded autotune over explicit manifest knobs
- local readiness diagnosis before benchmark or kernel bring-up work
- local transport-layer integration against the SDK contract

The CLI should fail closed when provenance is incomplete. It should never turn a
delegated route check into an unlabeled repo-owned MLX throughput number.
