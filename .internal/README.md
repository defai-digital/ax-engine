# AX Engine v4 Internal Docs

This directory contains internal engineering documents that should not be mixed
into the public documentation surface.

## Structure

- `product/` for product-level PRDs and product-shaping documents.
- `adr/` for accepted and superseded architecture decision records.
- `architecture/` for system-shape and interface documents.
- `benchmark/` for benchmark contracts, schemas, and benchmark planning.
- `planning/` for active implementation plans and engineering best practices.
- `archive/` for completed or superseded internal planning documents.
- `reference/` for archived upstream codebases and research snapshots.
- `models/` for local model assets used during development.

## Current Reading Path

Start here for current strategy:

1. `product/PRD.md`
2. `adr/README.md`
3. `planning/README.md`
4. `planning/MLX-RUNTIME-PERFORMANCE-PRD.md`
5. `planning/TURBOQUANT-PROMOTION-PRD.md`
6. `planning/CLI-TUI-PRD.md`
7. `benchmark/BENCHMARK-LAB-PRD.md`

## Active Planning Entry Points

- `planning/MLX-RUNTIME-PERFORMANCE-PRD.md`
- `planning/TURBOQUANT-PROMOTION-PRD.md`
- `planning/IMPLEMENTATION-PLAN.md`
- `planning/CLIENT-SURFACES-PLAN.md`
- `planning/MODEL-SUPPORT-TIERS.md`
- `planning/NEXT-STEP-OPTIONS.md`
- `planning/MIGRATION-STATUS.md`
- `planning/CLI-TUI-PRD.md`
- `adr/0012-retire-ax-native-and-route-mlx-or-llama.md`
- `adr/0013-mlx-kv-cache-improvement-strategy.md`
- `adr/0017-mlx-runtime-optimization-governance.md`
- `adr/0019-cli-tui-workflow-cockpit.md`

## Housekeeping Rule

Do not delete internal planning history unless explicitly requested. Move
superseded files to `archive/superseded/<date>-<reason>/` and completed files to
`archive/completed/<date>-<reason>/`, then update the active index files.

These files are for contributors and maintainers. They are not meant to be the
primary reading path for repository users.
