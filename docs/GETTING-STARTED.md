# Getting Started

AX Engine v4 is currently in active development.

The repository provides:

- a working inference engine core (request lifecycle, scheduler, KV cache, runner integration)
- benchmark tooling split between `ax-engine-bench` workload contracts and the
  MLX inference-stack comparison harness
- a preview SDK, local HTTP server, Python bindings, and a JavaScript preview client
- repo-owned MLX inference plus `llama.cpp` bypass support for non-MLX inference

## Current Scope

The near-term target is:

- Mac-first runtime for Apple M4-or-newer Macs
- single-machine execution
- dense Qwen and Gemma families first
- benchmark and replay discipline before broad optimization or autotune

The repository is not yet a polished end-user product.

## If You Used Earlier AX Engine

This v4 repository does not yet provide feature parity with the earlier
`ax-engine` workspace.

What exists here today:

- engine-core request lifecycle, scheduler, KV, and deterministic bring-up loop
- checked-in Metal kernel manifest/build artifacts plus a core-owned Metal
  asset loader and validation-only asset boundary
- benchmark manifests
- `ax-engine-bench` scenario / replay / matrix / compare / baseline / autotune
  workload-contract runtime plus thin direct inference commands
- preview `ax-engine-sdk` backend-resolution and session contract surface
- preview `ax-engine-server` local HTTP adapter over the SDK
- preview `ax-engine-py` / `python/ax_engine` Python access layer for
  token-based generation, request lifecycle control, and in-process streaming
- repo-local `javascript/ax-engine` preview client over the checked-in HTTP
  and OpenAI-compatible server endpoints

What is not yet present in this repository:

- broad production or orchestrated HTTP server surface
- fully migrated Rust SDK facade
- broad JavaScript bindings beyond the thin preview HTTP client
- broad transport-level Python ergonomics beyond the current local preview layer
- broad model coverage beyond the current MLX-first path and delegated
  llama.cpp route

Do not assume that user-facing surfaces from the earlier AX repo have already
migrated into v4.

These client-facing surfaces are still expected to return, but as thin layers
above the engine core rather than as the primary architecture driver.

Model support is also expected to be reported through support tiers and backend
selection, rather than only through a flat yes-or-no model list.

## Repository Areas

- `crates/ax-engine-core`: core runtime contracts and bring-up execution loop
- `crates/ax-engine-bench`: workload-contract CLI and bring-up runtime harness
- `crates/ax-engine-sdk`: SDK facade with backend resolution and session management
- `crates/ax-engine-server`: local HTTP server adapter over the SDK
- `javascript/`: repo-local JavaScript preview client package
- `crates/ax-engine-py`: Python extension crate (PyO3)
- `benchmarks/`: canonical benchmark manifests
- `python/`: Python package wrapper, type stubs, tests, examples
- `scripts/`: E2E smoke check scripts
- `docs/`: public-facing documentation

For the current crate layering and dependency-boundary guidance, see
`docs/ARCHITECTURE.md`.

## Build Prerequisites

Current development assumes:

- Rust toolchain
- an Apple Silicon M4-or-newer target environment for the eventual runtime

The benchmark CLI and core workspace compile on a normal Rust setup, but
decision-grade AX-owned MLX inference claims require the supported Apple
Silicon MLX runtime environment.

AX Engine v4 MLX mode depends on the available Apple Silicon MLX runtime;
non-MLX inference uses delegated llama.cpp routes.
Runtime surfaces fail closed on pre-M4 hosts instead of pretending degraded
support exists.

## First Commands

To inspect the workload-contract CLI:

```text
cargo run -p ax-engine-bench -- help
```

To inspect whether the local machine is inside the supported M4-or-newer native
target contract, or only allowed through an internal bring-up override:

```text
cargo run -p ax-engine-bench -- doctor
```

To run one thin direct inference request through the SDK-owned session surface:

```text
cargo run -p ax-engine-bench -- generate --tokens 1,2,3 --max-output-tokens 4
```

To run a llama.cpp-backed text request through a delegated server:

```text
cargo run -p ax-engine-bench -- generate \
  --prompt "Hello from AX" \
  --support-tier llama_cpp \
  --llama-server-url http://127.0.0.1:8081
```

To run a checked-in scenario manifest through the current workload-contract
path:

```text
cargo run -p ax-engine-bench -- scenario --manifest benchmarks/manifests/scenario/chat_qwen_short.json --output-root benchmarks/results
```

The checked-in delegated llama.cpp manifests are route-contract examples, not
AX-owned model-inference benchmarks. They validate the stepwise
`llama.cpp /completion` delegation path and backend-reported prompt-cache
evidence.

To compare AX Engine MLX mode against the upstream MLX-family inference
standard:

```text
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/Qwen3.5-9B-MLX-4bit \
  --prompt-tokens 512,2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 5
```

That harness requires `mlx_lm.benchmark` as the primary reference and fails
closed if the matching baseline cannot be produced. Add `--ax-both-modes` when
you need both greedy and speculative AX MLX rows. Each AX or optional
`mlx-swift-lm` row is compared against the matching `mlx_lm.benchmark`
prompt/decode shape. Use `--mlx-swift-lm-command` only for an explicit
JSON-emitting `mlx-swift-lm` harness. Do not use the retired SwiftLM
application-server benchmark as a current AX Engine baseline.

To run a bounded autotune pass over explicit manifest knobs:

```text
cargo run -p ax-engine-bench -- autotune \
  --manifest benchmarks/manifests/scenario/chat_qwen_short.json \
  --output-root benchmarks/results \
  --iterations 8
```

Autotune output is candidate evidence. It still needs the normal
scenario/replay/compare gates before it influences architecture or release
decisions.

To validate checked-in MLX dense Qwen and Gemma scenario manifests
through one repo-owned smoke command:

```text
bash scripts/check-bench-mlx.sh
```

That smoke path now emits the repo-owned Metal build report into the default
`build/metal` directory, or the explicit `AX_ENGINE_METAL_BUILD_DIR` override
if set, and when that report reaches `status=compiled` it also requires
benchmark-visible `metal_dispatch_completed` evidence instead of silently
accepting a CPU-only fallback.

To validate the checked-in readiness-report contract itself:

```text
bash scripts/check-bench-doctor.sh
```

To emit the checked-in Phase 1 Metal kernel build report and compile the
compiled Metal preview artifacts (`.air`, `.metalar`, and `.metallib`) when the
local toolchain is actually ready:

```text
cargo run -p ax-engine-bench -- metal-build
bash scripts/build-metal-kernels.sh
```

The repo-owned `ax-engine-bench metal-build` subcommand is now the canonical build
entrypoint. `scripts/build-metal-kernels.sh` remains as a thin wrapper over
that Rust-owned path for smoke checks and automation.
When the same output directory already holds validated compiled assets for the
current checked-in contract, that build command now reuses them instead of
rerunning the full toolchain pipeline.
That checked-in build graph keeps the explicit `metal-ar` archive stage as part
of AX's own artifact contract, and the current MLX Metal bring-up contract
stays intentionally narrow by validating only `block_size_tokens=16`.

To validate the checked-in Metal kernel inventory, manifest, and gated build
contract in one repo-owned smoke check:

```text
bash scripts/check-metal-kernel-contract.sh
```

When a compiled Phase 1 `metallib` is later loaded through the core-owned
macOS bring-up path, AX also treats a sibling
`ax_phase1_dense_path.binary_archive.metallib` as a best-effort pipeline cache:
valid archives are reused, stale ones are recreated, and required compute
pipeline descriptors are serialized back out without turning cache misses into
hard runtime failures.

That bring-up path also now keeps one process-local Metal dispatch arena for
KV cache buffers, so repeated dispatches can reuse previously materialized
slot-backed cache storage while refreshing only the per-step metadata/input
buffers that describe the current workload.

To validate the checked-in MLX replay manifests for live-share, retained
reuse, mixed-path, full-prefix decode, and memory-blocked recovery behavior:

```text
bash scripts/check-bench-replay.sh
```

To run the repo-owned llama.cpp delegated-contract smoke path for the
checked-in scenario and replay example manifests:

```text
bash scripts/check-bench-preview.sh
```

To start the preview local server:

```text
cargo run -p ax-engine-server -- --model-id qwen3_dense --mlx --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts --port 8080
```

To install the checked-in JavaScript preview client from this repository:

```text
npm install ./javascript/ax-engine
```

That package is intentionally thin: it targets the preview server's
`/v1/runtime`, `/v1/generate`, `/v1/generate/stream`, `/v1/completions`, and
`/v1/chat/completions` endpoints rather than bypassing the SDK/server contract.

To run a repo-owned end-to-end server smoke check instead of driving that path
manually:

```text
bash scripts/check-server-preview.sh
```

To query runtime metadata from that server:

```text
curl http://127.0.0.1:8080/v1/runtime
```

That runtime payload now includes backend-resolution metadata plus host and
Metal-toolchain diagnostics.

To submit and inspect a request through the shared preview server session:

```text
curl http://127.0.0.1:8080/v1/requests -H 'content-type: application/json' -d '{"model":"qwen3_dense","input_tokens":[1,2,3],"max_output_tokens":2}'
curl -X POST http://127.0.0.1:8080/v1/step
curl http://127.0.0.1:8080/v1/requests/1
```

To stream preview lifecycle events from the local server:

```text
curl -N http://127.0.0.1:8080/v1/generate/stream -H 'content-type: application/json' -d '{"model":"qwen3_dense","input_tokens":[1,2,3],"max_output_tokens":2}'
```

To compile the current workspace:

```text
cargo check
```

To build and install the preview Python package into the active environment:

```text
maturin develop
```

If you want one repo-owned command that bootstraps a temporary virtualenv,
installs `maturin`, builds the extension, runs the checked-in Python examples,
and then runs both the installed-package preview tests and the wrapper tests,
use:

```text
bash scripts/check-python-preview.sh
```

To run the checked-in Python preview examples after installation:

```text
python examples/python/basic.py
python examples/python/stepwise.py
python examples/python/streaming.py
```

See `docs/PYTHON.md` for the current Python preview scope.

## Stability Note

Public command surfaces and runtime behavior are still evolving.
Expect interface changes while the v4 engine loop, KV manager, sampler
boundary, and benchmark system continue to mature.
