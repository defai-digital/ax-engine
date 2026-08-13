# Codex inspect — qwen36-27b

**Agent:** Codex CLI · model `gpt-5.6-sol` · `model_reasoning_effort=max`  
**Mode:** static / code-only (weights UNAVAILABLE)  
**Batch report:** `../wave1-4-codex-batch.md` (program dual-agent batch)  
**Shared substrate:** `../W0-shared/codex-inspect.md`

## Executive summary

Static direct-inference audit for `qwen36-27b` (`qwen3_5`). Graph anchors and MTP policy
reviewed against Wave 0 shared substrate findings (already fixed: whisper registry,
gemma4_unified tier honesty, smoke harness DI-W0-004/005). No new family-local P0
defects identified that block `closed-code-only` without weights.

## Path map

- Manifest family: `qwen3_5`
- Anchors: families/qwen3_linear.rs + standard hybrid; runner mtp_model_policy
- MTP: Qwen linear MTP fail-closed default (ADR-020); cert candidate env opt-in

## Findings

_none open P0/P1 from static family audit after Wave 0 fixes_

## Residual LIMIT

Weights smoke deferred; MTP formal gates remain ADR-020

## Disposition

`closed-code-only` — dual-agent program artifacts present; real-weight smoke deferred.
