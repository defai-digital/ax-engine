# Architecture

AX Engine is organized as a small set of crates with intentionally different
dependency boundaries.

The project goal is not to make every crate depend on the same general-purpose
Rust stack. The goal is to keep the execution core lean, keep transport and
serialization concerns at the edges, and make observability and error handling
consistent across the workspace.

**Related:** [Scheduler](SCHEDULER.md) · [KV Cache](KV-CACHE.md) ·
[MLX Backend](MLX-BACKEND.md) · [CUDA Backends](CUDA-BACKENDS.md) ·
[Server](SERVER.md) · [Roadmap](ROADMAP.md)

## Current Crate Layers

- `ax-engine-core`: request lifecycle, scheduler, KV cache, runner integration,
  and deterministic bring-up loop
- `ax-engine-mlx`: MLX model graphs, KV cache, n-gram acceleration, MTP,
  and runner dispatch (only crate where `unsafe` is permitted, via
  `mlx-sys`)
- `mlx-sys`: bindgen FFI over `ax_shim.h` to MLX C++; safe `MlxArray` RAII
  wrappers and type-tagged handle system
- `ax-engine-sdk`: backend resolution, session management, request lifecycle
  contract, and delegated backend bridges for `mlx_lm.server`, llama.cpp,
  vLLM, TensorRT-LLM, and TensorRT Edge-LLM
- `ax-engine-server`: HTTP/SSE adapter over the SDK; default Mac builds include
  native MLX, while the Linux `delegated-server` feature omits MLX linkage
- `ax-engine-py`: Python binding surface over the SDK contract
- `ax-engine-bench`: workload-contract CLI, replay harness, reporting,
  bounded autotune, readiness, and bring-up checks
- `ax-engine-microbench`: isolated microbenchmarks and kernel dispatch probes
  (RMSNorm, MoE, diffusion, MLA, disk-prefix-cache); depends on
  `ax-engine-core`, `ax-engine-mlx`, and `mlx-sys` only

This means AX Engine already has a practical split between:

- execution core
- runtime/session contract
- transport adapters
- tooling and benchmark surfaces

Benchmarking is intentionally split at the project boundary:
`ax-engine-bench` records workload-contract evidence, while
`scripts/bench_mlx_inference_stack.py` records repo-owned MLX runtime
model-inference comparison against the required `mlx_lm.benchmark` primary baseline and
optional `mlx-swift-lm` secondary baseline adapter rows.
Delegated `mlx_lm_delegated` and llama.cpp checks stay outside repo-owned MLX
throughput claims.

## Dependency Boundaries

### `ax-engine-core`

The core should stay focused on engine behavior and state transitions.

Good fit:

- `tracing` for structured instrumentation
- `thiserror` for typed domain errors
- small deterministic utility crates that support execution behavior directly

Avoid by default:

- web frameworks
- async runtimes as a design center
- generic middleware abstractions
- JSON-specific transport concerns

Serialization should only enter the core when a specific type truly needs to
cross a crate or process boundary. Core internals should not derive
serialization traits just for convenience.
AX Engine currently uses a small amount of core-level serialization for
public Metal manifest and build-report contracts that are shared across
workspace surfaces.

### `ax-engine-sdk`

The SDK is the runtime-facing contract layer. It is a good place for:

- `serde` and `serde_json`
- typed error boundaries
- backend metadata plus delegated `mlx_lm.server`, llama.cpp, vLLM, and
  TensorRT payload translation
- session-level request and response types

If future work introduces a more explicit "runtime" naming convention, the
first question should be whether the current SDK responsibilities need to be
renamed or split. In the current repository, the SDK already plays the
runtime/session-contract role.

### `ax-engine-server`

The server owns HTTP, SSE, request parsing, and async orchestration glue.

Good fit:

- `tokio`
- `axum`
- `tower` in transport or test-only contexts
- serialization and route-local response models

These dependencies should stay in the server shell instead of flowing inward
into the execution core.

## Platform And Provider Ownership

AX Engine is Mac-first, not Mac-only. Platform support is split by ownership:

- macOS 26+ Apple Silicon uses the repo-owned MLX/Metal execution path;
- AX OCR treats Mac and NVIDIA Thor as co-primary deployment targets, with
  certified Linux x86_64 CUDA PCs as the secondary support platform; target
  priority does not bypass profile-specific release gates;
- Linux CUDA uses a portable AX Engine control plane and an external GPU
  worker;
- vLLM is one logical provider shared by x86_64 and Thor through distinct,
  fail-closed runtime profiles;
- TensorRT-LLM and TensorRT Edge-LLM are independent optimized providers, not
  aliases or automatic fallbacks for vLLM.

The Rust process owns backend selection, wire contracts, security,
observability, and public identity. The worker owns model loading, GPU
scheduling, KV/cache implementation, kernels, and engine-specific tuning.
Python/PyTorch/vLLM are packaged independently and never become transitive Mac
dependencies or in-process server imports.

Delegated transport also stays provider-neutral. HTTP agents are cached by
policy and `Accept` contract so JSON responses and SSE streams cannot evict
each other's keep-alive connection. Accepted AX HTTP sockets use
`TCP_NODELAY`; this avoids delayed delivery of small first-stream events while
leaving scheduling and model execution entirely with the selected worker.

See [CUDA Backends](CUDA-BACKENDS.md) for the deployment topology, runtime
profiles, product ownership boundary, and hardware release gates.

## Runtime Ownership And Capability Gates

MLX stream registration is thread-local. The native server therefore passes an
`EngineSessionConfig` to its generation service and constructs, executes, and
drops the resulting `EngineSession` on one dedicated worker thread. Model
replacement follows the same rule: native caches are cleared and the
replacement session is loaded on its eventual owner thread before the service
becomes live. Transport and stateless request context may be prepared outside
that worker, but native model and stream ownership may not cross it.

Optimized execution routes must be enabled from typed capabilities derived
from the loaded model structure plus an explicit certification, not from a
model-family name alone. Unsupported or incomplete graphs fail closed to the
existing sequential execution path. Continuous dense decode is currently
certified only for dense, full-attention Qwen3; additional model families need
equivalence, KV/preemption, and server-path performance evidence before their
certification is promoted.

## Architecture Composition And Generation Strategies (ADR-038)

Portable structural views live in `ax-engine-core` and are derived from
`NativeModelManifest` without a second on-disk schema:

- `ArchitectureSpec` / `StructuralCapabilities` — layer attention/FFN/cache kinds
- `GenerationKind` — autoregressive, block diffusion, or encoder embed
- `WorkUnitKind` on each `ExecutionItem.planned_work_unit` — prefill chunk,
  token decode, denoise step, block commit, or embed forward
- `ARCHITECTURE_REGISTRY` / `LayerForwardRoute` — static convert/default route
  and layer-forward dispatch (prefer over open family-string allowlists)
- `MultimodalPrefillAdapter` — vision/audio feed the same generation strategy

Native sessions bind `EngineCore::set_generation_kind` from the manifest so the
scheduler plans `DenoiseStep` for DiffusionGemma after prefill. Runners emit
`ax_mlx_generation_kind`, `ax_mlx_generation_work_unit`, and
`ax_mlx_layer_forward_route` on step telemetry. Diffusion monoblock generation
still runs denoise+commit inside one MLX block; schedule progress fields and
`DiffusionScheduleUpdate` exist so multi-step denoise/commit planning can
extend without another schema break.

### `ax-engine-bench` and `ax-engine-py`

Tooling and binding crates can use convenience dependencies when they help with
reporting, transport, or packaging, as long as those choices do not redefine
the core API surface.

## Error Model

AX Engine should prefer typed domain errors for core and SDK surfaces.

That keeps it possible to distinguish:

- request validation failures
- state transition violations
- unsupported host or backend conditions
- delegated backend failures
- transport-level failures

`anyhow` can still be useful in one-off tooling or local utilities, but it
should not replace public error enums in `ax-engine-core` or `ax-engine-sdk`.

## Observability

`tracing` is the workspace-standard instrumentation layer.

Use it for:

- scheduler and step lifecycle spans
- KV allocation and prefix reuse decisions
- runner dispatch timing
- backend routing and fallback paths
- benchmark execution diagnostics

For performance-sensitive runs, tracing should stay opt-in and narrowly scoped.
The benchmark CLI only enables tracing when `AX_BENCH_LOG` or `RUST_LOG` is
set. The preview server follows the same rule with `AX_ENGINE_SERVER_LOG`
first and then `RUST_LOG`.

For throughput and latency measurements, prefer leaving tracing disabled, or
using narrow `info` and `warn` filters instead of `debug` or `trace`.

## Guidance For New Dependencies

When adding a crate, ask:

1. Does this dependency belong in the execution core, or only at a boundary?
2. Is this solving a real runtime need, or only making one outer surface more convenient?
3. Can the dependency stay in `ax-engine-sdk`, `ax-engine-server`, `ax-engine-py`, or `ax-engine-bench` instead of entering `ax-engine-core`?
4. Will this make error handling and observability clearer, or blur crate responsibilities?

For AX Engine, a smaller and clearer core is usually the better default.

## First-Party Product Clients

AX Engine is consumed in two supported ways:

1. **In-process** via `ax-engine-sdk` (and language bindings that embed the session API)
2. **Sidecar HTTP** via `ax-engine-server` / `ax-engine serve` exposing OpenAI-compatible `/v1/*`

Product defaults, lifecycle phases, and non-goals are specified in
[LOCAL-ENGINE-CLIENTS.md](./LOCAL-ENGINE-CLIENTS.md). AX Studio and AX Code
default to managed sidecar HTTP and can explicitly attach to an existing local
server without taking over its lifecycle. In-process SDK embedding remains
available to hosts that deliberately own a native `EngineSession`.

AX OCR consumes the sidecar HTTP surface when using CUDA. Its document
workflow, accuracy corpus, model artifact policy, and release acceptance stay
in AX OCR; generic provider transport and worker launch/runtime profiles stay
in AX Engine.
