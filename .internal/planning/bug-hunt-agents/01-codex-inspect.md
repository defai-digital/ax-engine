# Codex F1 — deep inspect (read-only)

You are Codex with **maximum reasoning effort** auditing one AX Engine
**direct-inference** model family. Work in **read-only** mode first: do not
edit files unless the operator upgrades the sandbox after this report.

## Assignment

```text
REPO={{REPO}}
FAMILY_ID={{FAMILY_ID}}
MANIFEST_FAMILY={{MANIFEST_FAMILY}}
MODEL_ARTIFACTS_DIR={{MODEL_ARTIFACTS_DIR}}
```

## Context to load

- `.internal/prd/PRD-DIRECT-INFERENCE-MODEL-BUG-HUNT.md` §5 taxonomy, §7 checklist
- `docs/SUPPORTED-MODELS.md` for this family
- `crates/ax-engine-core/src/architecture_registry.rs` entry for `MANIFEST_FAMILY`
- Convert mapping: `crates/ax-engine-core/src/convert/model_family.rs`
- Graph: `crates/ax-engine-mlx/src/model/families/*` relevant modules
- Runner / MTP: `runner/mtp_*.rs`, `mtp.rs`, `mtp_adaptive_gate.rs`, family-specific MTP
- Server presets/aliases only as they affect this family

## Mission

Find **real** defects and design errors. Prefer depth and proof over volume.

Search specifically for:

1. **BUG** — wrong tokens risk, crash, KV corruption, silent weight drops
2. **IMPL** — graph disagrees with manifest/config/paper contract
3. **MTP** — wrong mode design, exactness coupled to flags, bad defaults, short-budget policy
4. **PERF** — only if you can point to a concrete hot path anti-pattern
5. **DEAD** — unreachable routes / obsolete knobs on this family's path
6. **LIMIT** — intentional fail-closed edges (not bugs)

## Method

1. Map the end-to-end path: convert → manifest → load → prefill → decode → sample → (MTP) → server.
2. Trace family dispatch from registry through `model` forward to runner.
3. For MTP-capable families, audit eligibility, exact arithmetic scope, adaptive gate, and product default.
4. Compare checklist §7 item-by-item; mark pass/fail/n/a with evidence.
5. Reject speculative findings without symbol anchors.

## Output format (markdown)

```markdown
# Codex inspect — {{FAMILY_ID}}

## Executive summary
(3–8 sentences)

## Path map
(bullet flow with key symbols)

## Checklist
| Item | Result | Evidence |
| --- | --- | --- |
| ... | pass/fail/n/a | symbol / note |

## Findings
### DI-{{FAMILY_ID}}-001
- Class: BUG|IMPL|MTP|PERF|DEAD|DOC|LIMIT
- Severity: P0|P1|P2|P3
- Title:
- Symbols:
- Why it is real:
- Reproduction sketch:
- Suggested fix direction:
- Risk if unfixed:

(repeat)

## Non-findings / residual risk
## Recommended fix order
## Tests that should exist but do not
```

## Constraints

- Do not propose lowering exactness or formal MTP gates.
- Do not expand into unrelated families unless a shared root cause is proven; then label impact list.
- If `MODEL_ARTIFACTS_DIR=UNAVAILABLE`, still do static audit; mark smoke-dependent items.
