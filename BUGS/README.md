# BUGS

Bug reports and findings for the ax-engine-v4 project.

## 2026-04-20: No TUI or SolidJS code found

**Scope:** TUI (terminal UI) and SolidJS-related code.

**Result:** After exhaustive search of the entire codebase, no TUI or SolidJS code exists.

### What was checked

- **TUI libraries:** All 6 `Cargo.toml` files and `Cargo.lock` searched for `ratatui`, `crossterm`, `termion`, `dialoguer`, `indicatif`, `cursive`, `ansi_term`, `termcolor`, `colored` -- zero matches.
- **SolidJS / web frameworks:** No `.jsx`, `.tsx`, `.vue`, `.svelte`, `.html`, `.css` files. No `solid`, `react`, `vue`, `svelte` references in source code. No framework dependencies in `package.json`.
- **Terminal rendering:** No ANSI escape codes, no progress bars, no interactive prompts. Only standard `tracing` logging and `clap` argument parsing.
- **All directories scanned:** `crates/`, `javascript/`, `python/`, `examples/`, `build/`, `metal/`, `scripts/`, `.internal/`, `docs/`, `.github/`.

### Project composition

| Component | Description |
|-----------|-------------|
| `ax-engine-core` | Rust inference engine with Metal GPU compute |
| `ax-engine-sdk` | Backend/session API layer |
| `ax-engine-server` | HTTP adapter (axum) |
| `ax-engine-py` | Python bindings (PyO3) |
| `ax-bench` | Benchmark CLI (clap) |
| `javascript/ax-engine` | Vanilla JS HTTP client (no framework) |

**Conclusion:** There is no TUI or SolidJS code to report bugs on. If TUI or SolidJS components are planned, they have not yet been added to this repository.
