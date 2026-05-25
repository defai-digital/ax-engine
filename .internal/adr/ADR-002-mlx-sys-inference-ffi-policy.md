# ADR-002: mlx-sys Inference FFI Policy

**Status**: Accepted
**Date**: 2026-05-25

---

## Context

`mlx-sys` is not a complete Rust wrapper for MLX. It exists to expose the MLX C
surface that AX Engine needs for inference, validation, benchmark probes, and
runtime diagnostics.

An older ADR listed missing top-5-family operations in phased batches. The
specific blocker list is now stale: the active binding surface already includes
inference operators such as `topk`, `topk_axis`, `flatten`, `repeat`,
`stop_gradient`, `pad`, and `unflatten`.

The useful long-term decision is not the old batch list. The useful decision is
the policy for when `mlx-sys` grows.

## Decision

`mlx-sys` remains a narrow inference-first binding crate.

Add MLX bindings only when there is a concrete AX Engine consumer in one of
these categories:

- native model-family forward pass
- weight loading or tensor conversion
- KV-cache/runtime correctness
- benchmark or microbenchmark probe that validates runtime behavior
- diagnostic tooling needed to explain a production runtime decision

Every new binding should include:

- a safe Rust wrapper in the appropriate `mlx-sys` module
- explicit shape, dtype, stream, and ownership expectations in tests or caller
  validation
- a nearby consumer, not a speculative wrapper with no project use

## Design Rules

- Unsafe MLX calls stay contained in `mlx-sys`. The workspace forbids unsafe
  code by default; `mlx-sys` is the explicit FFI exception.
- Wrappers should expose Rust types and ownership behavior that make the common
  caller path safe. Any caller-side precondition that cannot be encoded in the
  type must be validated near the boundary.
- Shape and dtype assumptions belong in tests or explicit runtime validation,
  not in comments alone.
- New operations should be named after the MLX operation unless the wrapper
  intentionally narrows semantics for AX Engine.
- A binding added for a microbenchmark must still identify the runtime behavior
  the probe is validating.

## Validation

For FFI changes, use checks that exercise both the binding and at least one
consumer:

- `cargo test -p mlx-sys`
- the relevant consumer crate test, usually `cargo test -p ax-engine-mlx` or
  `cargo test -p ax-engine-microbench`
- `cargo clippy --all-targets --all-features -- -D warnings` before handoff

If a binding needs real MLX device execution, document the local artifact/device
requirement in the PRD, script, or benchmark evidence that consumes it.

## Non-Goals

- Do not expose the full MLX training, linalg, FFT, random, or optimizer surface
  unless AX Engine adds a concrete inference/runtime consumer.
- Do not add bindings only because a reference project imports the same MLX
  Python symbol.
- Do not hide expensive two-pass substitutes in hot paths when MLX provides a
  direct operation that the model architecture requires.

## Consequences

- `mlx-sys` stays small enough to audit.
- Missing MLX APIs are evaluated through model/runtime evidence instead of a
  generic parity checklist.
- Benchmark probes can justify low-level bindings, but they must still explain
  the runtime behavior being measured.

## Rejected Alternatives

### Keep phased missing-op ADRs as active architecture

Rejected. Phase-specific gap lists expire quickly and are already stale in this
workspace.

### Generate or expose all MLX C APIs

Rejected. A broad wrapper increases audit and maintenance cost without matching
AX Engine's current inference-only product shape.
