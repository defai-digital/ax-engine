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
- `GEMMA-QWEN-MLX-PERFORMANCE-PRD.md` — Gemma/Qwen MLX direct-decode, n-gram fallback, and memory-attribution execution plan (under ADR 0024).
- `KV-SCHEDULER-REVAMP-PRD.md` — bounded KV cache and scheduler revamp plan.
- `TURBOQUANT-PROMOTION-PRD.md` — TurboQuant experiment-to-promotion plan.
- `WEIGHT-QUANT-AND-SPECULATION-PRD.md` — weight-side quant, MTP, FlashDecoding boundary (under ADR 0022).
- `DS4-REFERENCE-LEARNINGS-PRD.md` — ds4 reference-only learnings (under ADR 0023); REQ-6 ngram observability shipped 2026-05-11.
- `INSTRUMENTS-PROFILING-RUNBOOK.md` — Phase 2 dispatch profiling runbook under DS4-LEARNINGS PRD.
- `WEIGHT-ROTATION-IMPLEMENTATION-PLAN.md` — ADR 0022 D2 implementation plan, **Closed 2026-05-11** (see retrospective).
- `WEIGHT-ROTATION-RETROSPECTIVE.md` — closes WEIGHT-QUANT-AND-SPECULATION-PRD §W2; five negative-result findings + re-open conditions.
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
