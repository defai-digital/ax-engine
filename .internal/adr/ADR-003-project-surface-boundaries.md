# ADR-003: Project Surface Boundaries

**Status**: Accepted
**Date**: 2026-05-25

---

## Context

AX Engine's active workspace is CLI/server/runtime first. The old
server-vs-manager ADR is no longer valid because the manager/TUI surface has
been removed from the active workspace.

The current project surfaces are:

- `ax-engine-server` for HTTP/gRPC serving
- `ax-engine-sdk` for Rust session and backend lifecycle APIs
- `ax-engine-py` and language SDKs as adapters over stable runtime contracts
- `ax-engine-bench` for stable benchmark workflows
- `ax-engine-microbench` for low-level runtime probes
- scripts and docs as orchestration and evidence surfaces

Earlier ADRs separately described benchmark, SDK/Python, and server/manager
boundaries. Their reusable content is the ownership model, not the obsolete
phase status or removed manager path.

## Decision

Keep AX Engine centered on a small serving/runtime core with clear adapters.

### Server

`ax-engine-server` is the data-plane serving binary. It owns:

- HTTP and gRPC routes
- OpenAI-compatible request/response/SSE translation
- serving runtime state
- generation, chat, embeddings, metadata, health, and lifecycle endpoints

It must not become a local model manager, installer, browser UI, or process
supervisor.

### SDK and Python

`ax-engine-sdk` is the Rust facade for session configuration, backend lifecycle,
streaming, route metadata, and artifact reporting. Backend-specific behavior
belongs in focused modules behind the facade.

`ax-engine-py` is a thin Python adapter over SDK behavior. It may own Python
exception classes, request parsing, stream iterator ownership, and Python object
conversion, but it must not fork runtime/session behavior that belongs in the
SDK or runtime crates.

### Benchmarking

`ax-engine-bench` owns repeatable, user-facing benchmark workflows, including
scenario runs, compare modes, doctor checks, manifest validation, environment
capture, and artifact writing.

`ax-engine-microbench` owns source-level probes and low-level measurements. It
may depend on `ax-engine-mlx`; production runtime crates must not depend on
microbenchmark crates.

Benchmark evidence belongs under `benchmarks/` or `.internal/` according to
publication status. Machine-specific logs, local model artifacts, and generated
build outputs must not be committed.

## Design Rules

- Server code owns transport behavior. SDK/runtime crates own execution and
  session behavior. Python and language SDKs adapt these contracts instead of
  duplicating them.
- Do not add a control-plane UI, model installer, or process supervisor to
  `ax-engine-server`. A future control-plane product surface needs a separate
  ADR and crate ownership decision.
- Shared code should start where the real dependency already exists. Do not add
  a new crate until at least two active surfaces need the same contract and a
  module-level boundary is no longer enough.
- Benchmark claims must name their route: repo-owned MLX runtime,
  `mlx_lm_delegated`, `llama_cpp`, or external reference. Do not blend delegated
  compatibility evidence with repo-owned runtime throughput.
- Public API shape changes require endpoint, SDK, or language-binding tests
  before the ADR can be considered implemented.

## Validation

Use the checks for the surface being changed:

- server routes and OpenAI/gRPC behavior: `cargo test -p ax-engine-server`
- SDK lifecycle and stream behavior: `cargo test -p ax-engine-sdk`
- Python binding shape: `bash scripts/check-python-preview.sh`
- stable benchmark workflows: `cargo test -p ax-engine-bench` and
  `bash scripts/check-bench-doctor.sh`
- low-level probes: `cargo test -p ax-engine-microbench`
- script hygiene after benchmark or packaging changes:
  `bash scripts/check-scripts.sh`

## Consequences

- Server work should preserve deployability without local UI or download
  orchestration.
- SDK/Python changes should preserve public behavior while moving internals into
  narrower modules.
- Benchmark and microbenchmark changes should not expand the production runtime
  surface unless a runtime contract requires it.
- Any future control-plane UX must be proposed as a separate product surface and
  must not be smuggled back into `ax-engine-server`.

## Rejected Alternatives

### Restore a manager/TUI boundary ADR

Rejected. There is no active manager/TUI crate in the workspace, so that ADR no
longer fits the project.

### Merge benchmark probes back into runtime crates

Rejected. Probe lifecycle follows validation and promotion workflows, not the
production runtime lifecycle.

### Let Python bindings become a second SDK implementation

Rejected. Python should adapt the SDK, not reimplement runtime behavior.
