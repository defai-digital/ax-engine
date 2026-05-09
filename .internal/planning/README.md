# AX Engine v4 Planning Index

Status: Active
Date: 2026-05-09
Owner: AX Engine

This directory is the active internal planning surface. Detailed historical notes
and superseded PRDs are preserved under `../archive/` and should not be used as
the first reading path.

## Canonical PRDs

- `../product/PRD.md` — product-level PRD and product boundary.
- `MLX-RUNTIME-PERFORMANCE-PRD.md` — repo-owned MLX runtime performance plan.
- `KV-SCHEDULER-REVAMP-PRD.md` — bounded KV cache and scheduler revamp plan.
- `TURBOQUANT-PROMOTION-PRD.md` — TurboQuant experiment-to-promotion plan.
- `CLI-TUI-PRD.md` — local CLI TUI workflow cockpit plan.
- `../benchmark/BENCHMARK-LAB-PRD.md` — benchmark product/system plan.

## Active Supporting Plans

- `API-INTEGRATION-ROADMAP.md`
- `CLIENT-SURFACES-PLAN.md`
- `IMPLEMENTATION-PLAN.md`
- `MODEL-SUPPORT-TIERS.md`
- `NEXT-STEP-OPTIONS.md`
- `MIGRATION-STATUS.md`
- `TRAINING-AND-QUANTIZATION.md`
- `METAL-KERNEL-REVIEW.md`
- `BEST-PRACTICES.md`

## Housekeeping Rule

When a new PRD supersedes several planning files, move the old files to
`../archive/superseded/<date>-<reason>/` instead of deleting them. When a plan is
complete and no longer an active steering surface, move it to
`../archive/completed/<date>-<reason>/`.
