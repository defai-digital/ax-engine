# Codex inspect — gemma4-e-series-26-31

**Agent:** Codex CLI · model `gpt-5.6-sol` · `model_reasoning_effort=max`  
**Mode:** static / code-only (weights UNAVAILABLE)  
**Batch report:** `../wave1-4-codex-batch.md` (program dual-agent batch)  
**Shared substrate:** `../W0-shared/codex-inspect.md`

## Executive summary

Static direct-inference audit for `gemma4-e-series-26-31` (`gemma4/gemma4_vl`). Graph anchors and MTP policy
reviewed against Wave 0 shared substrate findings (already fixed: whisper registry,
gemma4_unified tier honesty, smoke harness DI-W0-004/005). No new family-local P0
defects identified that block `closed-code-only` without weights.

## Path map

- Manifest family: `gemma4/gemma4_vl`
- Anchors: standard + gemma4_vl.rs towers
- MTP: Assistant-MTP where packaged

## Findings

_none open P0/P1 from static family audit after Wave 0 fixes_

## Residual LIMIT

Media capability fail-closed without tower tensors

## Disposition

`closed-code-only` — dual-agent program artifacts present; real-weight smoke deferred.
