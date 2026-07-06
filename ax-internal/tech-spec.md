# Tech Spec: Phased Implementation Plan

**Status:** In Progress
**Author:** Engineering
**Date:** 2026-07-02
**Last Updated:** 2026-07-02

---

## Overview

This tech spec defines the phased implementation plan for the Rust Code Quality Improvement Initiative. The initiative addresses two P0 gaps identified in the [PRD](PRD.md):

1. **Phase 1:** Add clippy restriction lints to surface `unwrap()`/`expect()`/`panic!` usage
2. **Phase 2:** Add supply-chain security via `cargo-audit` + `cargo-deny`

Phase 3 (fuzzing) and Phase 4 (server panic audit) are documented as future work.

---

## Phase 1: Clippy Restriction Lints (P0)

**Status:** ✅ Implemented
**Date:** 2026-07-02

### Objective

Introduce clippy restriction lints at the workspace level to surface high-risk panic patterns, with a warn-first escalation strategy.

### Changes

#### 1.1 Add `[workspace.lints.clippy]` to `Cargo.toml`

```toml
# Cargo.toml (after [workspace.lints.rust])
[workspace.lints.clippy]
unwrap_used = "warn"
expect_used = "warn"
panic = "warn"
dbg_macro = "warn"
large_enum_variant = "warn"
```

**Rationale:** These lints are in clippy's `restriction` group and are not enabled by default. Setting them to `warn` surfaces all uses without breaking the build.

#### 1.2 Add `#![cfg_attr(test, allow(...))]` to test modules

For each test module (e.g., `crates/ax-engine-core/src/kv/tests.rs`), add:

```rust
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
```

**Rationale:** Test setup code uses `unwrap()` idiomatically. Suppressing warnings in test modules avoids noise.

#### 1.3 Escalate core hot paths to `deny`

For core hot paths (`kv.rs`, `engine.rs`, `request_manager.rs`), add module-level `deny`:

```rust
// At the top of crates/ax-engine-core/src/kv.rs
#![deny(clippy::unwrap_used, clippy::expect_used)]
```

**Rationale:** These are the most critical paths where panics are most dangerous (especially given `panic = "abort"` in the release profile). Escalating to `deny` prevents new unjustified panics.

**Note:** This may require auditing existing uses and replacing unjustified ones with `Result` + `?` propagation. For this phase, we will add the `deny` directive and fix any immediate breakage, but a full audit is deferred to Phase 4.

### Validation

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo test --quiet --no-fail-fast
```

**Expected:** `cargo clippy` passes (warnings are allowed, not denied). Core hot paths may produce errors if unjustified panics exist; these will be fixed incrementally.

---

## Phase 2: Supply Chain Security (P0)

**Status:** ✅ Implemented
**Date:** 2026-07-02

### Objective

Add `cargo-audit` and `cargo-deny` to CI to enforce supply-chain security for dependencies.

### Changes

#### 2.1 Add `.github/workflows/audit.yml`

```yaml
name: Supply Chain Audit

on:
  push:
  pull_request:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: dtolnay/rust-toolchain@stable
      - name: Install cargo-audit
        run: cargo install cargo-audit --locked
      - name: Run cargo-audit
        run: cargo audit
      - name: Install cargo-deny
        run: cargo install cargo-deny --locked
      - name: Run cargo-deny
        run: cargo deny check
```

**Rationale:** This runs on every push/PR to catch new vulnerabilities, and weekly to catch newly disclosed vulnerabilities in existing dependencies.

#### 2.2 Add `deny.toml` policy file

```toml
# deny.toml
[advisories]
vulnerability = "deny"
unmaintained = "warn"
yanked = "warn"
notice = "warn"

[licenses]
unlicensed = "deny"
allow = ["Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib", "Unicode-DFS-2016"]
copyleft = "deny"

[bans]
multiple-versions = "warn"
wildcards = "deny"

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
```

**Rationale:** This enforces:
- **Advisories:** Block dependencies with known RUSTSEC vulnerabilities.
- **Licenses:** Allow permissive licenses compatible with Apache-2.0; block copyleft (GPL-3.0, AGPL-3.0).
- **Bans:** Warn on multiple versions of the same crate (dependency bloat); deny wildcard version specs.
- **Sources:** Only allow dependencies from crates.io (not arbitrary git repos).

#### 2.3 Add `.cargo/audit.toml` ignore list

```toml
# .cargo/audit.toml
[advisories]
ignore = [
  # Add RUSTSEC IDs here with justification, e.g.:
  # "RUSTSEC-2023-0001",  # False positive for our use case
]
```

**Rationale:** Some RUSTSEC advisories may not apply to our use case. This file allows ignoring them with justification.

### Validation

```bash
cargo audit
cargo deny check
```

**Expected:** Both commands pass with no vulnerabilities or license violations. If advisories are surfaced, they will be addressed or ignored (with justification in `audit.toml`).

---

## Phase 3: Fuzzing for Parsers (P1, Future)

**Status:** 📋 Documented, not started
**Priority:** P1

### Objective

Add `cargo-fuzz` targets for parsers that handle untrusted input (tokenizer, model config, weight format conversion).

### Proposed Changes

1. **Install `cargo-fuzz`** — Add to CI toolchain.
2. **Define fuzz targets** — Create `fuzz/fuzz_targets/` with targets for:
   - Tokenizer (input: arbitrary byte sequences)
   - Model config parsing (input: arbitrary JSON/TOML)
   - Weight format conversion (input: arbitrary weight blobs)
3. **Run fuzzing** — Execute each target for 1 hour with `cargo fuzz run <target> -- -max_total_time=3600`.
4. **Integrate into CI** — Add a nightly CI job that runs fuzzing for 1 hour per target.

### Success Criteria

- [ ] No crashes after 1 hour of fuzzing per target
- [ ] Fuzzing integrated into nightly CI
- [ ] Fuzz corpus committed to repo (or stored in CI artifacts)

### Risks

- **Setup effort** — `cargo-fuzz` requires nightly Rust and libFuzzer. This is non-trivial to set up in CI.
- **False positives** — Fuzzing may surface issues that are not bugs (e.g., expected error paths). These need manual triage.

---

## Phase 4: Server Boundary Panic Audit (P1, Future)

**Status:** 📋 Documented, not started
**Priority:** P1

### Objective

Manually audit `unwrap()`/`expect()`/`panic!` in `ax-engine-server` (especially `multimodal.rs`, `chat.rs`) and replace with `Result` + graceful error responses.

### Proposed Changes

1. **Audit server code** — Review all `unwrap()`/`expect()`/`panic!` in `crates/ax-engine-server/src/`.
2. **Replace with `Result`** — For each unjustified panic, replace with `Result<T, ServerError>` and return a graceful HTTP error response (e.g., 400 Bad Request, 500 Internal Server Error).
3. **Add regression tests** — For each replaced panic, add a test that sends a malicious request payload and verifies the server does not panic.

### Success Criteria

- [ ] All server-boundary panics replaced with `Result` + graceful error responses
- [ ] Regression tests for malicious request payloads
- [ ] `#![deny(clippy::unwrap_used, clippy::expect_used)]` added to server modules

### Risks

- **Manual effort** — This requires manual review of ~65 instances in `multimodal.rs` and ~18 in `chat.rs`.
- **Behavior changes** — Replacing panics with error responses may change the server's behavior in edge cases. This needs careful testing.

### Critical Context

The release profile uses `panic = "abort"` (Cargo.toml L46), which means any panic aborts the entire server process. This makes server-boundary panics a **DoS vector**: a single malicious request that triggers a panic can kill all in-flight requests. This phase is critical for production reliability.

---

## Rollback Plan

If Phase 1 or Phase 2 causes issues, revert the changes:

- **Phase 1 rollback:** Remove `[workspace.lints.clippy]` from `Cargo.toml` and the `#![cfg_attr(test, allow(...))]` directives from test modules.
- **Phase 2 rollback:** Remove `.github/workflows/audit.yml`, `deny.toml`, and `.cargo/audit.toml`.

Both phases are additive (they add CI checks and lint config, not new runtime behavior), so rollback is safe and has no impact on production.

---

## References

- [PRD: Rust Code Quality Improvement Initiative](PRD.md)
- [ADR-001: Clippy Restriction Lints Strategy](ADR-001-clippy-restriction-lints.md)
- [ADR-002: Supply Chain Security](ADR-002-supply-chain-security.md)
- [AGENTS.md](../AGENTS.md) — Existing code style and architecture guidelines
