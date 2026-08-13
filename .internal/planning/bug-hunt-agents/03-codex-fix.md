# Codex F4 — implement one finding

You are Codex with **maximum reasoning effort** implementing a **single**
accepted finding for the active direct-inference family.

## Assignment

```text
REPO={{REPO}}
FAMILY_ID={{FAMILY_ID}}
MANIFEST_FAMILY={{MANIFEST_FAMILY}}
FINDING_ID={{FINDING_ID}}
FINDING_TITLE={{FINDING_TITLE}}
```

## Inputs you must read

1. Merged findings file for this family (`findings.md` in OUT_DIR).
2. The specific finding section for `FINDING_ID`.
3. Surrounding code at the listed symbols.
4. Existing tests near the hot path.
5. PRD constraints: no exactness gate weakening; surgical diffs; tests required for P0/P1.

## Mission

1. Reproduce or statically prove the defect.
2. Implement the **minimal** correct fix.
3. Add/adjust regression tests.
4. Run targeted tests for touched crates.
5. Summarize residual risk and any cross-family impact.

## Implementation rules

- Match project style (`rustfmt`, no new unsafe, avoid unwrap/expect/panic in production paths).
- Prefer capability/manifest-driven logic over new string allowlists.
- MTP: keep fail-closed defaults; exact arithmetic independent of request flags where required.
- Do not edit unrelated families “while here.”
- Do not rewrite large files unless the finding proves no smaller safe fix.

## Output format

```markdown
# Fix report — {{FINDING_ID}}

## Root cause
## Change summary
## Files touched
## Tests added/updated
## Commands run and results
## Cross-family impact
## Residual risk
## Suggested ledger disposition
fixed | needs-follow-up | blocked
```

## Stop conditions

- If fix requires a design decision (OD-*), stop and write the decision needed.
- If fix would lower product gates, stop and refuse.
- If finding is not reproducible and not statically proven, reclassify and stop.
