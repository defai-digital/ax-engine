# Benchmark and Microbenchmark Reorganization Implementation Plan

**Status**: In progress
**Date**: 2026-05-16
**Companions**:
- `.internal/prd/PRD-2026-05-25-benchmark-evidence-tooling.md`
- `.internal/adr/ADR-003-project-surface-boundaries.md`

---

## Implemented in First Slice

- Added workspace crate `crates/ax-engine-microbench`.
- Moved source-level MLX probe binaries out of `crates/ax-engine-mlx/src/bin`.
- Preserved binary names through explicit `[[bin]]` entries:
  - `turboquant-microbench`
  - `dequant-dtype-probe`
  - `kernel-chain-batching-probe`
  - `mla-warm-extend-drift-probe`
  - `rmsnorm-fused-probe`
  - `residual-rmsnorm-fused-probe`
  - `disk-prefix-cache-stress`
- Updated docs and script README command examples to use `-p ax-engine-microbench`.
- Added `crates/ax-engine-microbench/README.md`.

---

## Validation

Passing:

```bash
cargo fmt -p ax-engine-microbench
cargo check -p ax-engine-microbench
cargo test -p ax-engine-microbench
```

Known blocker outside this slice:

```bash
cargo check -p ax-engine-mlx --all-targets
```

This currently fails in the pre-existing `crates/ax-engine-mlx/src/model/`
split because test modules in `model/mod.rs` are missing imports. The
microbench move does not introduce those errors.

---

## Next Slice

1. Add shared microbench helpers in `crates/ax-engine-microbench/src/`:
   - `artifact.rs`
   - `host.rs`
   - `stats.rs`
   - `timing.rs`
2. Define and validate `ax.microbench.v1`.
3. Convert one binary first, preferably `dequant-dtype-probe`, because it is
   small and has a simple correctness/timing shape.
4. Convert `turboquant-microbench` last because it already has its own mature
   schema and external validator.
5. Begin splitting `crates/ax-engine-bench/src/main.rs` only after the model
   split worktree compiles cleanly.
