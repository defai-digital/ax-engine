# ax-internal

Internal engineering documentation for ax-engine. This folder contains product requirements, architecture decision records, and technical specifications for cross-cutting initiatives.

## Contents

- **[PRD.md](PRD.md)** — Product Requirements Document: Rust Code Quality Improvement Initiative
- **[ADR-001-clippy-restriction-lints.md](ADR-001-clippy-restriction-lints.md)** — Architecture Decision: Clippy Restriction Lints Strategy
- **[ADR-002-supply-chain-security.md](ADR-002-supply-chain-security.md)** — Architecture Decision: Supply Chain Security (cargo-audit + cargo-deny)
- **[tech-spec.md](tech-spec.md)** — Technical Specification: Phased Implementation Plan

## Scope

This initiative addresses gaps identified during a review of a generic "Rust code quality improvement" guide. The review revealed that ax-engine already implements most best practices (strict `unsafe_code = "forbid"`, `thiserror` error types, module layering, CI gates), but has two actionable gaps:

1. **Missing clippy restriction lints** — no `[workspace.lints.clippy]` config to surface `unwrap()`/`expect()`/`panic!` usage
2. **Missing supply-chain security** — no `cargo-audit` or `cargo-deny` in CI, no `deny.toml` policy file

The tech spec defines a phased rollout, with Phase 1–2 (P0) implemented immediately and Phase 3–4 (P1) documented for future work.

## Status

- **Phase 1** (Clippy lints): ✅ Implemented
- **Phase 2** (Supply chain): ✅ Implemented
- **Phase 3** (Fuzzing): 📋 Documented, not started
- **Phase 4** (Server panic audit): 📋 Documented, not started
