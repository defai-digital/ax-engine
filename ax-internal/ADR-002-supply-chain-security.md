# ADR-002: Supply Chain Security (cargo-audit + cargo-deny)

**Status:** Accepted
**Date:** 2026-07-02
**Deciders:** Engineering
**Superseded by:** —

---

## Context

ax-engine publishes to PyPI via OIDC trusted publishing (`.github/workflows/pypi.yml`). This means the project has a **public release surface** — any vulnerability in a transitive dependency can affect downstream users. However, the project currently has:

- **No `cargo-audit`** — RUSTSEC advisories (known vulnerabilities in Rust crates) are not checked in CI.
- **No `cargo-deny`** — There is no policy enforcement for licenses, banned crates, or dependency sources.
- **No `deny.toml`** — No declarative policy file for supply-chain rules.

This is a P0 gap for any project with a public release surface, especially one that publishes to PyPI (where supply-chain attacks are increasingly common).

---

## Decision

We will introduce **supply-chain security** via `cargo-audit` and `cargo-deny`, with a declarative `deny.toml` policy file:

1. **`cargo-audit` in CI** — Add a GitHub Actions workflow (`.github/workflows/audit.yml`) that runs `cargo audit` on every push and PR. This checks for known vulnerabilities in dependencies (RUSTSEC advisories).

2. **`cargo-deny` in CI** — The same workflow runs `cargo deny check advisories bans licenses sources`, which enforces:
   - **Advisories:** Block dependencies with known RUSTSEC advisories (unless explicitly ignored with justification).
   - **Bans:** Block known-bad crates (e.g., `openssl` in favor of `rustls`).
   - **Licenses:** Enforce license compliance (allow Apache-2.0, MIT, BSD-*, ISC; block GPL-3.0).
   - **Sources:** Enforce that dependencies come from crates.io (not arbitrary git repos).

3. **`deny.toml` policy file** — A declarative policy file at the repo root that defines the rules for `cargo-deny`.

4. **`audit.toml` ignore list** — A config file (`.cargo/audit.toml`) that allows ignoring specific RUSTSEC advisories with justification (e.g., false positives, not applicable to our use case).

---

## Consequences

### Positive

- **Automated vulnerability detection** — Known vulnerabilities in dependencies are caught in CI before merge.
- **License compliance** — Ensures all dependencies use permissive licenses compatible with Apache-2.0 (the project's license).
- **Dependency source control** — Prevents accidental inclusion of dependencies from untrusted git repos.
- **Aligns with PyPI OIDC publishing** — The project already uses OIDC trusted publishing (no API tokens), which is a supply-chain best practice. Adding `cargo-audit`/`cargo-deny` completes the supply-chain security story.

### Negative

- **CI runtime increase** — `cargo-audit` and `cargo-deny` add ~1–2 minutes to CI. This is acceptable for a P0 security control.
- **False positives** — Some RUSTSEC advisories may not apply to our use case (e.g., vulnerabilities in optional features we don't use). These can be ignored via `audit.toml` with justification.

### Neutral

- **No immediate build breakage** — The initial run may surface advisories, but these can be addressed incrementally (or ignored with justification).

---

## Implementation

See [tech-spec.md](tech-spec.md) Phase 2 for implementation details.

### CI Workflow

```yaml
# .github/workflows/audit.yml
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
        run: cargo install cargo-audit
      - name: Run cargo-audit
        run: cargo audit
      - name: Install cargo-deny
        run: cargo install cargo-deny
      - name: Run cargo-deny
        run: cargo deny check
```

### `deny.toml` Policy

```toml
# deny.toml
[advisories]
vulnerability = "deny"
unmaintained = "warn"
yanked = "warn"
notice = "warn"

[licenses]
unlicensed = "deny"
allow = ["Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib"]
copyleft = "deny"  # Block GPL-3.0, AGPL-3.0

[bans]
multiple-versions = "warn"
wildcards = "deny"

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
```

### `audit.toml` Ignore List

```toml
# .cargo/audit.toml
[advisories]
ignore = [
  # Example: RUSTSEC-2023-0001 is a false positive for our use case
  # "RUSTSEC-2023-0001",
]
```

---

## Alternatives Considered

### Alternative 1: Use only `cargo-audit`

**Rejected.** `cargo-audit` only checks for known vulnerabilities. `cargo-deny` adds license compliance, ban lists, and source control, which are also important for supply-chain security.

### Alternative 2: Use `cargo-supply-chain`

**Rejected.** `cargo-supply-chain` is useful for auditing crate authors, but it is not a CI gate (it is a manual audit tool). `cargo-audit` + `cargo-deny` are the standard CI tools.

### Alternative 3: Run audits only on schedule (not on push/PR)

**Rejected.** Vulnerabilities can be introduced by new dependencies. Running on push/PR ensures every change is audited. The weekly schedule is a backup to catch newly disclosed vulnerabilities in existing dependencies.

---

## References

- [cargo-audit](https://github.com/rustsec/rustsec/tree/main/cargo-audit)
- [cargo-deny](https://embarkstudios.github.io/cargo-deny/)
- [RUSTSEC advisories](https://rustsec.org/)
- [PyPI OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/)
- [PRD: Rust Code Quality Improvement Initiative](PRD.md)
