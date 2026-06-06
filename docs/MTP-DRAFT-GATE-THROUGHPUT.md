# MTP draft confidence gate — throughput tuning (workable speedup)

**Change.** `DEFAULT_MTP_DRAFT_MIN_CONFIDENCE` 0.98 → **0.90**
(`crates/ax-engine-mlx/src/mtp.rs`). ~5–13% decode speedup on Qwen3.6 27B-MTP,
correctness-preserving, no new infrastructure, compatible with the linear-
attention layers.

## How we got here

Started from "why can't AX's MTP be much faster than MTPLX?" Two dead ends were
ruled out first, then the live lever was found:

1. **Tree speculative decoding — architecturally dead on this model.** Qwen3.6
   27B is a *hybrid linear-attention* model (`linear_attention_enabled`, full
   softmax attention only every 4th layer). A tree verify needs a tree attention
   *mask*, but the linear/gated-delta layers are recurrent scans that ignore
   masks and process tokens sequentially — they cannot verify branching paths in
   one forward. The Phase-A prototype (`src/bin/tree_draft_probe.rs`,
   `docs/TREE-DRAFT-PHASE-A.md`) also showed the acceptance ceiling is already
   near-saturated in the realistic regime (≤1.14x projected, saturating at 8
   leaves), so even where a tree *is* possible it isn't worth it.

2. **Raising draft depth — already optimal.** A depth sweep (gate 0.98) showed
   accept/forward saturates at depth ~3 (the gate truncates everything deeper),
   and tok/s *falls* past depth 3 because `mtp_draft_tokens` still runs the head
   forward `max_depth` times for zero extra accepted tokens. Depth 2–3 is the
   optimum; production is already there.

3. **The live lever — the gate itself.** At the 0.98 gate, even on *repetitive*
   flappy the accepted draft length per step is only ~0.79 and `fwds/step` is
   exactly 1.000: the gate is so conservative it proposes only near-certain
   tokens. It was calibrated for **accept rate ≥99%**, not throughput. Loosening
   it proposes slightly longer drafts; the rare extra rejection costs one cheap
   recompute forward.

## Measurement

Tool: `tree_draft_probe` depth/gate sweep — a production-faithful linear MTP
decode (real `mtp_draft_tokens` + gate; verify on a clone; adopt-clone on full
accept, `forward_all_positions_update_cache` recompute on partial accept, exactly
mirroring the runner's linear-attention path), greedy target ⇒ deterministic
trajectory, real wall-clock tok/s. M5 Max, Qwen3.6 27B-MTP-4bit, real tokenized
prompts (chat template), 256 committed tokens.

```bash
AX_MLX_MTP_DRAFT_MIN_CONFIDENCE=0.90 AX_TREE_PROMPT_FILE=/tmp/flappy_ids.txt \
AX_PROBE_HEAD_MAX_DEPTH=6 AX_DEPTH_SWEEP="2,3,4" ./target/release/tree_draft_probe <model> 256
```

Best tok/s per suite (gate 0.98 vs 0.90), with `fwds/step` (recompute pressure):

| suite                | gate 0.98 best | gate 0.90 best | gain | fwds/step @0.90 |
|----------------------|---------------:|---------------:|-----:|----------------:|
| flappy               |  39.0 (d2)     |  40.9 (d2)     | +5%  | 1.04 |
| python_modules_long  |  39.8 (d2)     |  41.5 (d3)     | +4%  | 1.01 |
| long_code            |  34.8 (d2)     |  39.4 (d2)     | +13% | 1.03 |

`tok/fwd` (deterministic, thermal-noise-free) confirms the mechanism: at fixed
depth 2 on flappy it rises 1.633 → 1.749 (0.98 → 0.90), +7%. Headline candidate
`d2/gate0.90` beat `d3/gate0.98` and `d2/gate0.98` in **every** interleaved
repeat (thermal noise ~3%). The feared recompute storm never appears —
`fwds/step` ≤ 1.03 on all three suites, including the hard ones.

## Why it's safe

- **Correctness-preserving.** The gate only controls how many speculative tokens
  are verified ahead. Truncating fewer of them cannot change committed output
  (greedy) or its distribution (sampled — rejection sampling restores the target
  distribution regardless of draft length). It *does* lower the reported accept
  rate (more drafts proposed); that is a speed knob, not a quality change.
- **Drafting is greedy in both modes.** The gate forces argmax drafting whenever
  the threshold > 0, so production's "sampled" mode drafts identically to the
  greedy probe; only target acceptance differs (rejection sampling accepts *more*
  than the probe's strict argmax), so the production win is ≥ the measured one.

## Caveats / follow-up

- Measured with greedy target acceptance and a per-step-reset MTP head cache.
  The gate *direction* (looser = faster at fixed depth) is a controlled relative
  result and robust; absolute magnitudes should be confirmed in production
  sampled mode via `scripts/bench_qwen36_mtp_fair.py` before quoting externally.
- Anything that SLO-gates on "MTP accept ≥99%" must move to a throughput SLO; the
  accept rate is intentionally lower now. Restore with
  `AX_MLX_MTP_DRAFT_MIN_CONFIDENCE=0.98`.
