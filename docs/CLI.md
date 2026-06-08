# CLI

AX Engine currently exposes four command surfaces:

- `ax-engine` for common local serving workflows. It is the recommended entrypoint
  for starting a local server and preparing Qwen3.6 MTP sidecars.
- `ax-engine-server` for the backward-compatible low-level HTTP/gRPC server
  process when callers need explicit runtime flags.
- `ax-engine-bench` for workload contracts, readiness, bounded autotune, Metal
  build checks, and thin direct SDK inference helpers.
- `scripts/bench_mlx_inference_stack.py` for repo-owned MLX runtime
  model-inference comparison against MLX-family references.

## `ax-engine`

`ax-engine` is the product-level orchestration CLI. It wraps existing AX Engine
components but does not replace them: `serve` launches `ax-engine-server`, and
`convert-mtplx` wraps the sidecar packaging/provenance tools.

Use `serve` as the normal local-server entrypoint:

```text
ax-engine serve /path/to/mlx-model --port 8080
ax-engine serve qwen36-35b --download --port 8080
ax-engine serve qwen36-35b --dry-run --json
ax-engine serve qwen36-35b -- --max-batch-tokens 1024
```

`serve --dry-run --json` emits an `ax.local_serve_plan.v1` document with the
resolved model/preset and exact `ax-engine-server` argv. A local filesystem path
wins over alias lookup. Extra flags after `--` are passed through to
`ax-engine-server`. `serve --download` is explicit: it downloads supported aliases
or raw Hugging Face repo ids before launch, and fails closed if the downloader
does not return ready AX artifacts.

Use `download` when you want model acquisition as a separate step:

```text
ax-engine download --list
ax-engine download qwen36-35b
ax-engine download qwen36-27b
ax-engine download qwen36-27b-5bit
ax-engine download qwen36-27b-6bit
ax-engine download qwen36-27b-8bit
ax-engine download gemma4-e2b
ax-engine download gemma4-e2b-5bit
ax-engine download gemma4-e2b-6bit
ax-engine download gemma4-e2b-8bit
ax-engine download gemma4-12b
ax-engine download gemma4-12b-6bit
ax-engine download gemma4-31b
ax-engine download mlx-community/Qwen3.6-35B-A3B-4bit --json
ax-engine download qwen36-35b --dest /path/to/explicit-copy
```

`download` wraps the same workflow as `scripts/download_model.py`: download
through `mlx-lm`, validate `config.json` and safetensors, and run
`ax-engine-bench generate-manifest` when available. The JSON output is the
`ax.download_model.v1` summary. The built-in download aliases target Qwen3.6 and
Gemma 4 MLX models, including Qwen3.6 27B, Gemma 4 E2B 5/6/8-bit, and Gemma 4
12B 4/6-bit variants where repo support is already tracked; other models should
use an explicit repo id or local path. If the model argument is missing or an
alias is unknown, `download` prints the same target list. The JSON form of
`download --list` emits an `ax.download_options.v1` document for automation.

Best practice is to keep the default Hugging Face Hub cache destination. That
cache is shared with `mlx-lm` and `huggingface_hub`, and its location is
controlled by `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME`. Use `--dest` only
when you need an explicit copied model directory outside the shared cache.

Use `convert-mtplx` to package standard Qwen3.6 MTP source shards with a
quantized MLX serving base:

```text
ax-engine convert-mtplx mlx-community/Qwen3.6-27B-4bit \
  --mtp-source Qwen/Qwen3.6-27B \
  --fair-base-only \
  --json
```

`convert-mtplx` writes `mtp.safetensors`, `mtplx_runtime.json`, patched
`config.json`, and `ax_mtp_sidecar_manifest.json`, then runs the provenance
checker before reporting success. Optional knobs use model-specific defaults
when omitted: Qwen3.6 27B uses MTP depth 3, and Qwen3.6 35B-A3B uses depth 1.
The current `--mtp-source` contract is a Hugging Face repo id that ships `mtp.*`
tensors; local MTP source directories must fail closed unless local shard
discovery is implemented.

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
- `ax-engine-bench generate-manifest`
- `ax-engine-bench serving-stress`
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
other command-line callers; the legacy text output remains available for shell use.

`ax-engine-bench doctor --json` also reports workflow discovery. When run from a
source checkout it points callers at `cargo run ...`, `scripts/download_model.py`,
and the checkout root; outside a checkout it points callers at installed
`ax-engine-bench` and `ax-engine-server` binaries. Automation should use this
contract instead of guessing Homebrew versus source mode from paths.

When `--mlx-model-artifacts-dir <path>` is provided, the doctor JSON includes a
structured `model_artifacts` report for `config.json`, `model-manifest.json`,
safetensors presence, `model_type`, quantization metadata, and readiness
blockers. Callers should use those fields directly instead of parsing
performance-advice text.

Use `bash scripts/check-bench-doctor.sh`, `bash scripts/check-server-preview.sh`,
and the relevant `check-bench-*.sh` gate before changing CLI workflow contracts.
Those checks cover doctor readiness, workflow discovery, benchmark artifacts,
and server `/health`, `/v1/runtime`, and `/v1/models` metadata.

Successful scenario and replay runs emit `manifest.json`, `environment.json`,
`metrics.json`, `routes.json`, `trace.json`, and `summary.md`. Contract failures
emit `contract_failure.json` plus `summary.md` instead of synthetic metrics.

`baseline`, `compare`, `matrix`, `matrix-compare`, and `autotune` build on
those artifacts. They should only compare or tune results inside the same
manifest/runtime family.

## MLX Inference Stack

Use `scripts/bench_mlx_inference_stack.py` when the question is repo-owned MLX
runtime throughput versus upstream MLX-family runtimes:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /path/to/local/mlx-model \
  --prompt-tokens 512,2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 15
```

`mlx_lm.benchmark` is mandatory. The script fails closed if the matching
baseline cannot be produced, and every AX or `mlx-swift-lm` row carries
ratio-to-`mlx_lm.benchmark` fields for the same random-token prompt/decode
shape. The harness mirrors the upstream prompt standard (`mx.random.seed(0)`
plus random token IDs from the model vocabulary) and writes the prompt token
JSON path and hash into the artifact. AX n-gram decode acceleration is the
server/user default; benchmark artifacts keep the direct AX row as the
same-policy baseline and use `--ax-compare-policies` to also emit the
`ax_engine_mlx_ngram_accel` default-policy row. `mlx-swift-lm` is accepted only
as a secondary baseline through an explicit `BenchmarkHelpers` / `MLXLMCommon`
generation adapter that reads the emitted prompt token JSON. The older SwiftLM
application-server benchmark is retired.

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
unavailable, the runtime falls back to the full-precision MLX KV path instead
of using the CPU oracle. A fallback reason label of `runner_not_integrated`
means no runtime decode attempt was observed yet; `cpu_oracle_unavailable`
identifies legacy/debug artifacts where the compressed-decode oracle path was
not available. Only `fused_compressed_decode` route evidence with successful
attempts, Metal fused decode successes, and zero fallbacks can feed the internal
quality artifact gate; shadow and legacy CPU oracle rows are diagnostic only.

For offline TurboQuant policy search, use the diagnostic grid harness:

```text
python3 scripts/search_turboquant_kv_policy.py \
  --metadata /path/to/metadata.json \
  --baseline /path/to/baseline-shape.json \
  --kv-presets disabled,TurboQuantK8V4 \
  --hot-window-tokens 256,512 \
  --fallback-policies fail_closed
```

This writes an `ax.offline_policy_search.v1` artifact to
`benchmarks/results/offline-policy-search/<date>/` when `--output` is omitted.
It enumerates candidate policies only; it does not run inference, benchmarks,
or promote TurboQuant support. Validate checked-in artifacts with
`bash scripts/check-offline-policy-search-artifacts.sh`.

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
