# Findings — W0-shared

## Merge notes

- Codex inspect: `.internal/reports/bug-hunt/20260811/W0-shared/codex-inspect.md` (gpt-5.6-sol, reasoning=max)
- ax-code scan: `.internal/reports/bug-hunt/20260811/W0-shared/axcode-scan.md` (zai-coding-plan/glm-5.2[1m])
- Agents: Codex high-confidence P0/P1 tooling + registry; ax-code reported clean MTP defaults with LOW guard gaps
- Unique after merge: 7 findings; P0/P1 accepted fixed in this program; LOW/medium hardening closed where cheap

## Work queue

| ID | Class | Sev | Title | Source | Status |
| --- | --- | --- | --- | --- | --- |
| DI-W0-001 | IMPL | P1 | Whisper convert family orphaned from ARCHITECTURE_REGISTRY | Codex | fixed |
| DI-W0-002 | IMPL | P1 | gemma4_unified converted as Certified `gemma4` | Codex | fixed |
| DI-W0-003 | DEAD/DOC | P2 | convert/registry parity test list is handwritten | both | fixed (list+uniqueness test; convert tests pin family) |
| DI-W0-004 | BUG | P1 | smoke `ensure_binary` reuses stale binaries | Codex | fixed |
| DI-W0-005 | BUG | P1 | `--models-dir` falls through to ambient HF cache | Codex | fixed |
| DI-W0-006 | DOC | P3 | tier test Compatible wildcard incomplete | Codex | parked (low; Compatible list still explicit) |
| DI-W0-007 | IMPL | P2 | registry duplicate-label ambiguity | Codex | fixed (uniqueness unit test) |

## Finding details

### DI-W0-001 — fixed

Whisper was convert-supported (`family_name=whisper`) but missing from
`ARCHITECTURE_REGISTRY`, so `lookup_architecture` returned `None` and tier
fell through Compatible silently.

**Fix:** registry row + Compatible tier assertion + convert-family parity list includes whisper.

### DI-W0-002 — fixed

`gemma4_unified` / `gemma4_unified_text` model types emitted `family_name=gemma4`
(Certified), overriding the Compatible `gemma4_unified` registry row.

**Fix:** convert emits `gemma4_unified`; Gemma split-prefill optim accepts both labels; convert test expects `gemma4_unified`.

### DI-W0-004 — fixed

`ensure_binary` returned any existing binary without rebuild.

**Fix:** always `cargo build` unless `--no-build`.

### DI-W0-005 — fixed

`resolve_snapshot` searched HF cache after a miss under `--models-dir`.

**Fix:** when `models_dir` is set, `allow_hf_cache_fallback=False`; unit test added.

### DI-W0-006 — parked

Tier completeness hardening deferred; not a runtime defect.

## Exit

Wave 0 closed with code fixes + dual-agent artifacts. Residual: DI-W0-006 parked.
