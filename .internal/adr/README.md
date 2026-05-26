# AX Engine ADRs

This directory contains active architecture decisions for the current
`ax-engine_v5` workspace.

## Active Decisions

- `ADR-001-runtime-model-boundaries.md` - runtime, manifest, model-family, and
  MLX model-code ownership.
- `ADR-002-mlx-sys-inference-ffi-policy.md` - inference-first MLX binding policy.
- `ADR-003-project-surface-boundaries.md` - server, SDK, Python, benchmark, and
  microbenchmark ownership.
- `ADR-004-cache-local-speculative-serving.md` - cache-local serving,
  speculative decoding, and offline policy-search boundaries.

## Policy

- ADRs describe current project architecture, not phase checklists.
- PRDs and planning notes own delivery phases, task lists, and temporary status.
- Expired ADRs are removed during cleanup when they no longer fit the active
  project shape.
- Do not create an ADR archive directory for this repo cleanup policy.

## When To Add Or Replace An ADR

Create or replace an ADR when a change affects one of these project contracts:

- crate ownership or dependency direction
- model-family, manifest, tensor-role, or runtime execution boundaries
- public server, SDK, Python, or benchmark behavior
- MLX FFI surface growth or unsafe-code containment
- benchmark evidence policy for published performance claims

Do not create an ADR for a phase checklist, a one-off implementation note, a
temporary bugfix, or a benchmark run log. Put those in PRDs, planning notes, or
benchmark artifacts instead.

## Required ADR Shape

Each ADR should answer these questions directly:

- **Context**: what project state or pressure forced the decision?
- **Decision**: what boundary or rule is now authoritative?
- **Design Rules**: what must future changes preserve?
- **Validation**: what local evidence proves the boundary still holds?
- **Consequences**: what trade-offs are accepted?
- **Rejected Alternatives**: what plausible options should not be reopened
  without new evidence?

## Lifecycle

- Active ADRs stay in this directory.
- Expired or incorrect ADRs are removed, not archived.
- If a new ADR replaces an old one, update direct references in `.internal/prd`
  and `.internal/planning` during the same cleanup.
- If a historical PRD still references a removed project surface, mark that PRD
  as historical or refresh its references before using it for implementation.
