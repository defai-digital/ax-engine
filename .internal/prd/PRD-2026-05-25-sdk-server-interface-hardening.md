# PRD: SDK and Server Interface Hardening

**Status**: Active
**Date**: 2026-05-25
**Current ADR**: `.internal/adr/ADR-003-project-surface-boundaries.md`
**Scope**: `crates/ax-engine-server`, `crates/ax-engine-sdk`,
`crates/ax-engine-py`, `python/`, external SDK contract tests

---

## Problem

The old SDK, Python, and server PRDs described large monolithic files that have
since been split. The remaining project need is interface hardening: keep server
transport behavior, SDK session behavior, Python bindings, and chat/OpenAI
compatibility aligned as backend routes evolve.

## Goals

- Keep `ax-engine-server` a deployable data-plane server, not a local manager or
  model installer.
- Keep `ax-engine-sdk` as the Rust session facade and backend lifecycle owner.
- Keep `ax-engine-py` as a thin adapter over SDK behavior.
- Keep OpenAI-compatible chat/completions/embeddings behavior testable by module.
- Prefer backend-native chat where delegated backends support it.
- Keep native MLX honest about token-native versus text/chat ownership.

## Non-Goals

- No control-plane UI or process supervisor in `ax-engine-server`.
- No public endpoint rename without a separate API compatibility plan.
- No Python API redesign in this hardening PRD.
- No new shared crate until at least two active surfaces require the same
  contract and a module boundary is insufficient.

## Current Evidence

- `crates/ax-engine-server/src/main.rs` is now startup-sized.
- Server modules exist for routes, app state, OpenAI, generation, embeddings,
  gRPC, backend adapters, errors, and tests.
- `crates/ax-engine-sdk/src/session/` exists with focused modules for config,
  routes, stream, native, delegated, llama lifecycle, artifacts, and errors.
- `crates/ax-engine-py/src/lib.rs` is module registration glue, with Python
  behavior split into focused files.
- Chat policy exists under server code, with explicit unsupported fallback
  behavior for families that should not silently use a plain prompt.

## Plan

### Phase 1: Contract Tests

Keep or add tests that prove current public behavior:

- OpenAI request validation and error shape
- OpenAI streaming chunks and finish reasons
- delegated llama.cpp and mlx-lm chat behavior
- Python dict shape and stream iterator behavior
- SDK route retention, cancellation, and stream progression

### Phase 2: Chat Correctness

Maintain a clear backend mode for chat requests:

- backend-native chat when available
- AX-rendered fallback only when there is a tested template
- explicit unsupported error when AX cannot honestly render the chat contract

User stop sequences must merge with, not replace, model-native chat terminators.

### Phase 3: Shared Contract Extraction

Extract server protocol, chat policy, model catalog, or typed client modules only
when a second active surface needs them. Start inside the existing owning crate
when that keeps dependency direction clear.

### Phase 4: Python/SDK Drift Reduction

Reduce manual Python serialization only behind dict-shape tests that cover
omitted optionals, defaults, enum labels, integer widths, nested reports, and
stream event objects.

## Acceptance Criteria

- Server route/API changes have module-level tests that cover request and
  response shape.
- SDK lifecycle changes pass default and `mlx-native` feature tests.
- Python binding changes pass preview packaging checks and keep stub/runtime
  signatures aligned.
- Chat behavior never silently falls back to a generic prompt for a known
  unsupported chat family.
- Any newly shared module has a named owner and at least two real consumers.

## Validation

```bash
cargo test -p ax-engine-server
cargo test -p ax-engine-sdk
cargo test -p ax-engine-sdk --features mlx-native
cargo test -p ax-engine-py
bash scripts/check-python-preview.sh
```

Before public API cleanup, run:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --quiet --no-fail-fast
```
