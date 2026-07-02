# ADR-001: Clippy Restriction Lints Strategy

**Status:** Accepted
**Date:** 2026-07-02
**Deciders:** Engineering
**Superseded by:** —

---

## Context

ax-engine is a production-grade LLM inference runtime with a Rust workspace. The codebase already enforces `cargo clippy --all-targets --all-features -- -D warnings` in CI, which catches most code-quality issues. However, this does not surface **high-risk panic patterns** like `unwrap()`, `expect()`, and `panic!`, which are in clippy's `restriction` group and are not enabled by default.

The release profile uses `panic = "abort"` (Cargo.toml L46), which means any panic aborts the entire server process. This raises the stakes on panic removal: a single panic in the server boundary (e.g., `multimodal.rs`, `chat.rs`) can kill all in-flight requests.

A review of the codebase revealed ~150+ instances of `unwrap()`/`expect()`/`panic!` in production code (excluding test modules), with hotspots in:

- `crates/ax-engine-core/src/kv.rs` (158 instances)
- `crates/ax-engine-core/src/engine.rs` (157 instances)
- `crates/ax-engine-core/src/request_manager.rs` (103 instances)
- `crates/ax-engine-server/src/multimodal.rs` (65 instances)

Many of these are `.expect()` with justification (e.g., `.expect("invariant: block must exist after allocation")`), but there is no automated enforcement to distinguish justified uses from accidental ones.

---

## Decision

We will introduce **clippy restriction lints** at the workspace level, with a **warn-first escalation strategy**:

1. **Workspace-wide `warn`** — Add `[workspace.lints.clippy]` to `Cargo.toml` with `unwrap_used`, `expect_used`, `panic`, `dbg_macro`, and `large_enum_variant` at `warn` level. This surfaces all uses without breaking the build.

2. **Test module `allow`** — Test modules use `#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]` to suppress warnings in test setup code, where `unwrap()` is idiomatic.

3. **Core hot path `deny`** — After an initial audit, escalate to `deny` in core hot paths (`kv.rs`, `engine.rs`, `request_manager.rs`) where panics are most dangerous. This is done via module-level `#![deny(clippy::unwrap_used, clippy::expect_used)]` in those files.

4. **Server boundary `deny` (future)** — Phase 4 will audit and escalate server-boundary code (`multimodal.rs`, `chat.rs`) to `deny`, replacing panics with `Result` + graceful error responses.

---

## Consequences

### Positive

- **Automated visibility** — Every `unwrap()`/`expect()`/`panic!` is now surfaced as a warning, making it visible in CI logs and IDE diagnostics.
- **Gradual adoption** — The `warn` level allows the team to audit and justify uses incrementally, without breaking the build.
- **Hot path safety** — Core hot paths (`kv.rs`, `engine.rs`, `request_manager.rs`) are escalated to `deny`, preventing new unjustified panics.
- **Aligns with `panic = "abort"`** — Reduces the risk of server-wide aborts from panics in critical paths.

### Negative

- **Warning noise** — The initial `cargo clippy` run will produce many warnings. This is expected and will be reduced as uses are audited and justified.
- **Test module boilerplate** — Test modules require `#![cfg_attr(test, allow(...))]`, which is minor boilerplate.

### Neutral

- **No immediate build breakage** — The `warn` level does not break the build. Only the `deny` escalation in hot paths will block new unjustified panics.

---

## Implementation

See [tech-spec.md](tech-spec.md) Phase 1 for implementation details.

### Configuration

```toml
# Cargo.toml
[workspace.lints.clippy]
unwrap_used = "warn"
expect_used = "warn"
panic = "warn"
dbg_macro = "warn"
large_enum_variant = "warn"
```

### Test Module Allow

```rust
// In test modules (e.g., crates/ax-engine-core/src/kv/tests.rs)
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
```

### Hot Path Deny

```rust
// In core hot paths (e.g., crates/ax-engine-core/src/kv.rs)
#![deny(clippy::unwrap_used, clippy::expect_used)]
```

---

## Alternatives Considered

### Alternative 1: Immediate `deny` workspace-wide

**Rejected.** This would break the build immediately due to the ~150+ existing uses. The `warn`-first strategy allows gradual adoption.

### Alternative 2: Use `clippy::pedantic` group

**Rejected.** The `pedantic` group is too noisy for production code (e.g., `missing_errors_doc`, `must_use_candidate`). The `restriction` group is more targeted to the specific risk (panics).

### Alternative 3: Custom clippy plugin

**Rejected.** Over-engineering. The built-in `restriction` lints are sufficient.

---

## References

- [Clippy lint groups](https://rust-lang.github.io/rust-clippy/stable/index.html)
- [Cargo.toml L46: `panic = "abort"`](../Cargo.toml)
- [PRD: Rust Code Quality Improvement Initiative](PRD.md)
