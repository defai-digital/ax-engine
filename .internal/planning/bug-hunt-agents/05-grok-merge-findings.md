# Grok F2 — merge dual-agent findings

You are the Grok orchestrator merging Codex and ax-code inspect outputs for one
family into a single prioritized work queue.

## Assignment

```text
FAMILY_ID={{FAMILY_ID}}
CODEX_REPORT={{OUT_DIR}}/codex-inspect.md
AXCODE_REPORT={{OUT_DIR}}/axcode-scan.md
OUTPUT={{OUT_DIR}}/findings.md
```

## Mission

1. Read both reports fully.
2. De-duplicate overlapping findings; keep the stronger evidence.
3. Drop low-confidence items that cannot be symbol-anchored (or park as `needs-proof`).
4. Assign stable IDs: `DI-{{FAMILY_ID}}-001`, `002`, …
5. Order queue: P0 → P1 → P2 → P3; within severity prefer BUG/MTP/IMPL before DEAD/DOC.
6. Resolve Codex vs ax-code disagreements with explicit reasoning (fail-closed bias).
7. Write `findings.md` and update the ledger finding log.

## Output structure for findings.md

```markdown
# Findings — {{FAMILY_ID}}

## Merge notes
- Codex findings: N
- ax-code findings: M
- Unique after merge: K
- Dropped / needs-proof: …

## Work queue
| ID | Class | Sev | Title | Source | Status |
| --- | --- | --- | --- | --- | --- |
| DI-…-001 | MTP | P0 | … | both | open |

## Finding details
### DI-…-001
…
```

## Rules

- Do not start fixes in this phase.
- Do not open another family.
- If zero P0/P1 remain after merge, proceed to exit-gate prep (`06-family-exit-gate.md`).
