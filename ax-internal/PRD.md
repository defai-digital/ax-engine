# PRD: Rust Code Quality Improvement Initiative

**Status:** Approved
**Author:** Engineering
**Date:** 2026-07-02
**Last Updated:** 2026-07-02

---

## Problem Statement

ax-engine is a production-grade LLM inference runtime with a Rust workspace spanning 8 crates. The codebase already enforces strict quality standards:

- `unsafe_code = "forbid"` workspace-wide (with a single FFI carve-out in `ax-engine-mlx` via `mlx-sys`)
- `thiserror` for error types in core/sdk, `anyhow` correctly absent from library crates
- CI gates for `cargo fmt`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test`
- Clear module layering: `core` has no web/async/serde dependencies

However, a review of a generic "Rust code quality improvement" guide revealed two **actionable gaps**:

1. **No clippy restriction lints** — The workspace lacks a `[workspace.lints.clippy]` config to surface high-risk patterns like `unwrap()`, `expect()`, and `panic!`. While the codebase uses `.expect()` with justification in many places, there is no automated enforcement to distinguish justified uses from accidental ones.

2. **No supply-chain security** — The project publishes to PyPI via OIDC trusted publishing (`.github/workflows/pypi.yml`), but has no `cargo-audit` (RUSTSEC advisories) or `cargo-deny` (license/ban policy) in CI. This is a P0 gap for any project with a public release surface.

---

## Goals

### Primary Goals

1. **Surface high-risk panic patterns** — Introduce clippy restriction lints (`unwrap_used`, `expect_used`, `panic`) at `warn` level workspace-wide, with `allow` in test modules and escalation to `deny` in core hot paths (`kv.rs`, `engine.rs`, `request_manager.rs`).

2. **Enforce supply-chain security** — Add `cargo-audit` and `cargo-deny` to CI, with a `deny.toml` policy file that blocks known-vulnerable dependencies and enforces license compliance.

3. **Document future work** — Define Phase 3 (fuzzing for parsers) and Phase 4 (server boundary panic audit) as future initiatives, with clear success criteria.

### Non-Goals

- **Do not add `criterion` benchmarks** — ax-engine already uses `ax-engine-microbench` probe binaries + Python `bench_*.py` scripts with "2 warmup + 5 measure, report median," which is the correct strategy for an LLM inference engine where GPU dispatch dominates. Criterion microbenchmarks of Rust-side logic are not meaningful for this workload.

- **Do not refactor module boundaries** — The existing layering (core → mlx-sys → ax-engine-mlx → sdk → server) is already strict and well-documented in `AGENTS.md`.

- **Do not change the `unsafe_code = "forbid"` policy** — This is already stricter than the generic guide's recommendations and is the correct stance for an FFI-heavy inference engine.

---

## Success Criteria

### Phase 1 (Clippy Lints)

- [x] `[workspace.lints.clippy]` added to `Cargo.toml` with `unwrap_used`, `expect_used`, `panic` at `warn` level
- [x] Test modules use `#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]`
- [x] `cargo clippy --all-targets --all-features -- -D warnings` passes (warnings are allowed, not denied)
- [x] Core hot paths (`kv.rs`, `engine.rs`, `request_manager.rs`) audited and escalated to `deny` where justified

### Phase 2 (Supply Chain)

- [x] `.github/workflows/audit.yml` added, runs `cargo audit` and `cargo deny check` on push/PR
- [x] `deny.toml` added with policy: block `GPL-3.0`, allow `Apache-2.0`/`MIT`/`BSD-*`, ban known-bad crates
- [x] CI passes with no RUSTSEC advisories
- [x] PyPI release workflow (`pypi.yml`) unchanged (OIDC trusted publishing remains the canonical path)

### Phase 3 (Fuzzing, Future)

- [ ] `cargo-fuzz` targets added for: tokenizer, model config parsing, weight format conversion
- [ ] Fuzzing runs for 1 hour per target with no crashes
- [ ] Fuzzing integrated into CI (nightly run)

### Phase 4 (Server Panic Audit, Future)

- [ ] Manual audit of `unwrap()`/`expect()`/`panic!` in `ax-engine-server` (especially `multimodal.rs`, `chat.rs`)
- [ ] All server-boundary panics replaced with `Result` + graceful error responses
- [ ] Regression test: malicious request payloads do not trigger server-wide panic (critical given `panic = "abort"` in release profile)

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Clippy restriction lints produce excessive warnings in test code | Use `#![cfg_attr(test, allow(...))]` in test modules |
| `cargo-audit` blocks on false-positive RUSTSEC advisories | Use `audit.toml` to ignore specific advisories with justification |
| `cargo-deny` license policy is too restrictive | Start with permissive policy (allow Apache-2.0/MIT/BSD-*), tighten incrementally |
| Phase 3 fuzzing requires significant setup effort | Document as future work, prioritize Phase 1–2 (P0) |

---

## Dependencies

- **Tools:** `cargo-audit`, `cargo-deny` (installed in CI via `cargo install`)
- **CI:** GitHub Actions (`.github/workflows/audit.yml`)
- **No new Rust dependencies** — This initiative adds CI config and lint config, not new crates

---

## Timeline

- **Phase 1 (Clippy lints):** Implemented 2026-07-02
- **Phase 2 (Supply chain):** Implemented 2026-07-02
- **Phase 3 (Fuzzing):** TBD (requires cargo-fuzz infrastructure)
- **Phase 4 (Server panic audit):** TBD (requires manual review)

---

## References

- [ADR-001: Clippy Restriction Lints Strategy](ADR-001-clippy-restriction-lints.md)
- [ADR-002: Supply Chain Security](ADR-002-supply-chain-security.md)
- [Tech Spec: Phased Implementation Plan](tech-spec.md)
- [AGENTS.md](../AGENTS.md) — Existing code style and architecture guidelines
