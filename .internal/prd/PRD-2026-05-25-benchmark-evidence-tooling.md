# PRD: Benchmark Evidence and Tooling

**Status**: Active
**Date**: 2026-05-25
**Current ADR**: `.internal/adr/ADR-003-project-surface-boundaries.md`
**Scope**: `crates/ax-engine-bench`, `crates/ax-engine-microbench`,
`benchmarks/`, benchmark scripts, performance documentation

---

## Problem

Benchmark tooling is no longer in the state described by the old PRD. The
microbenchmark crate exists and source-level probes have moved out of
`ax-engine-mlx/src/bin`, but `ax-engine-bench/src/main.rs` is still large and
new benchmark-result layouts are inconsistent.

The project needs benchmark evidence that is easy to audit, repeat, and connect
to a specific performance claim.

## Goals

- Keep stable user-facing benchmark workflows in `ax-engine-bench`.
- Keep low-level probes in `ax-engine-microbench`.
- Make performance claims route-explicit: repo-owned MLX runtime,
  `mlx_lm_delegated`, `llama_cpp`, or external reference.
- Standardize new benchmark artifacts without rewriting historical results.
- Keep benchmark publication honest in dirty worktrees and local-model setups.

## Non-Goals

- No mass migration of historical benchmark directories.
- No benchmark result deletion as part of this PRD.
- No tuning-only changes without before/after evidence.
- No broad script folder reshuffle unless wrappers and CI consumers are known.

## Current Evidence

- `crates/ax-engine-microbench` exists in the workspace.
- Microbench binaries now live under `crates/ax-engine-microbench/src/bin`.
- `crates/ax-engine-bench/src/main.rs` is still large and remains the main
  stable benchmark CLI entrypoint.
- Existing benchmark results under `benchmarks/results/` use several layouts.
- Recent MTP compare artifacts exist under `benchmarks/results/`, but they
  should not automatically become published performance claims.

## Plan

### Phase 1: Result Index Policy

Add a lightweight index for new benchmark result directories. Each indexed run
should record:

- claim being evaluated
- route/surface
- model source and quantization
- host
- command
- git commit and dirty state
- artifact path
- validation status

### Phase 2: `ax-engine-bench` Review Slices

Reduce benchmark harness complexity by moving one concern at a time out of
`main.rs` only when it improves testability or ownership:

- artifact writing and summaries
- compare JSON/schema logic
- route readiness and metadata
- workload definitions
- doctor/environment probing

Do not chase a line-count target without a review or testability win.

### Phase 3: Microbench Artifact Contract

For new or touched probes, converge on an `ax.microbench.v1`-style artifact with
command, git, host, config, correctness, and measurements. Existing mature probe
schemas may keep compatibility wrappers until a reader exists.

### Phase 4: Publication Gate

Before updating public performance docs, require:

- clean or explicitly documented dirty worktree state
- artifact path
- benchmark command
- route/surface label
- host/model metadata
- summary of regressions or unsupported comparisons

## Acceptance Criteria

- New benchmark artifacts have an index entry or an explicit reason they are
  internal-only scratch output.
- Public docs distinguish repo-owned runtime performance from delegated
  compatibility evidence.
- Microbenchmark outputs used for claims include correctness and host metadata.
- `ax-engine-bench` refactors preserve CLI behavior and artifact schemas unless
  a PRD explicitly changes them.

## Validation

```bash
cargo test -p ax-engine-bench
cargo test -p ax-engine-microbench
bash scripts/check-bench-doctor.sh
bash scripts/check-scripts.sh
```

Long-running model benchmarks should record completed rows only. Pending or
aborted rows must stay explicit.
