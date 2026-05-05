# Architecture

AX Engine is organized as a small set of crates with intentionally different
dependency boundaries.

The project goal is not to make every crate depend on the same general-purpose
Rust stack. The goal is to keep the execution core lean, keep transport and
serialization concerns at the edges, and make observability and error handling
consistent across the workspace.

## Current Crate Layers

- `ax-engine-core`: request lifecycle, scheduler, KV cache, runner integration,
  and deterministic bring-up loop
- `ax-engine-sdk`: backend resolution, session management, request lifecycle
  contract, and delegated backend bridges for `mlx_lm.server` and llama.cpp
- `ax-engine-server`: local HTTP and SSE adapter over the SDK
- `ax-engine-py`: Python binding surface over the SDK contract
- `ax-engine-bench`: workload-contract CLI, replay harness, reporting,
  bounded autotune, readiness, and bring-up checks

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
- backend metadata plus delegated `mlx_lm.server` and llama.cpp payload
  translation
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
