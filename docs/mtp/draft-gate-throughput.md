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
   `docs/mtp/tree-draft-phase-a.md`) also showed the acceptance ceiling is already
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
| -------------------- | -------------- | -------------- | ---- | --------------- |
| flappy               | 39.0 (d2)      | 40.9 (d2)      | +5%  | 1.04            |
| python_modules_long  | 39.8 (d2)      | 41.5 (d3)      | +4%  | 1.01            |
| long_code            | 34.8 (d2)      | 39.4 (d2)      | +13% | 1.03            |

`tok/fwd` (deterministic, thermal-noise-free) confirms the mechanism: at fixed
depth 2 on flappy it rises 1.633 → 1.749 (0.98 → 0.90), +7%. Headline candidate
`d2/gate0.90` beat `d3/gate0.98` and `d2/gate0.98` in **every** interleaved
repeat (thermal noise ~3%). The feared recompute storm never appears —
`fwds/step` ≤ 1.03 on all three suites, including the hard ones.

## Per-workload gate sweep

Gate × depth sweep on all three fair-MTP suites (256 committed tokens). Because
the run is fully deterministic (greedy, fixed seed/prompt), `tok/fwd` and
`accept/st` are *exactly* reproducible — so the cross-gate variation is a real
step-boundary effect, not noise. `tok/fwd` counts ALL target forwards (verify +
recompute), so at a fixed depth it is the clean, thermal-free efficiency metric
for the gate (it omits only head-compute, which is what makes depth 2 win on
wall-clock — see below). Depth 2 is the throughput optimum on every suite.

Best `tok/fwd` at depth 2, gate 0.98 vs each suite's optimal gate:

| suite                | gate 0.98 | optimal gate | tok/fwd @opt | Δ tok/fwd |
| -------------------- | --------: | :----------: | -----------: | --------: |
| flappy               |     1.647 |   **0.90**   |        1.779 |     +8.0% |
| python_modules_long  |     1.571 |   **0.85**   |        1.785 |    +13.6% |
| long_code            |     1.369 |   **0.80**   |        1.610 |    +17.6% |

**Trend: the harder / more diverse the workload, the looser the optimal gate.**
On repetitive content (flappy) the head is confident, so even a tightish gate
keeps long drafts and 0.90 is best; on diverse code (long_code) a tight gate
truncates nearly everything, so a looser 0.80 recovers far more speculation. The
old 0.98 is worst everywhere, and worst by the largest margin on the hardest
suite (where speculation matters most). Looser still (<0.80) was not better —
extra rejections start costing recompute forwards faster than they add accepted
tokens.

Depth: deeper drafts raise `tok/fwd` slightly but cost one extra head forward per
step; past the gate's effective truncation depth (~2-3) that head-compute exceeds
the marginal acceptance, so wall-clock tok/s peaks at depth 2 in this probe.
Production's adaptive-depth controller already picks depth per request, so the
shippable knob is the gate, not a fixed depth.

> NOTE: raw tok/s from a *sequential* sweep (one config after another) is
> thermally confounded on a fanless/heat-soaked box — later runs throttle. Use
> `tok/fwd` (deterministic) for cross-config ranking, or interleave configs
> round-robin when a wall-clock number is needed (as in the headline table above).

## Best practices

- **Default (shipped): `AX_MLX_MTP_DRAFT_MIN_CONFIDENCE = 0.90`.** Best on
  repetitive workloads, strong everywhere, cleanly validated (+~8% vs 0.98 on
  flappy, interleaved). Use this when the workload mix is unknown.
- **Per-workload override** (`AX_MLX_MTP_DRAFT_MIN_CONFIDENCE`), looser for harder
  content:
  - Repetitive / templated generation (boilerplate, structured code): **0.90**.
  - General code / mixed prose+code: **0.85**.
  - Long, diverse code generation (the hardest, lowest-base-acceptance case):
    **0.80** — this is where the gate change pays the most (+17% tok/fwd).
- **Do not exceed ~0.95** for throughput — that regime is the accept-rate-tuned
  setting and leaves the most speed on the table. **Restore 0.98** only if a
  consumer SLO-gates on "MTP accept ≥99%" (the accept rate is intentionally lower
  now; output is unchanged).
- **Do not chase a fixed deep draft depth** to go faster — depth is already
  adaptive in the runner, and past depth ~3 extra head forwards cost more than
  they return. The gate, not the depth, is the throughput lever.
- **Linear-attention models only get the gate lever**, not tree speculation — see
  `docs/mtp/tree-draft-phase-a.md`
- **Per-request tuning** is available in code via `mtp::mtp_draft_tokens_gated`
  (an explicit-gate variant of `mtp_draft_tokens`); the runner still defaults to
  the env value. Wire it through the request/session API if a single deployment
  must serve mixed workloads at different gates.

## Adaptive gate (evaluated, NOT adopted)

An auto-tuning controller was prototyped and validated before deciding the gate
must be set, not learned online. The controller hill-climbs the gate per request
on a deterministic throughput proxy `committed / (target_forwards + steps*depth*r)`
(r≈0.10), bounded [0.80, 0.95] (`tree_draft_probe` `AX_ADAPTIVE_GATE=1`).

Result (proxy; deterministic, so thermal-noise-free):

| suite                 | best fixed   | ADAPTIVE   | verdict         |
| --------------------- | ------------ | ---------- | --------------- |
| flappy                | 1.453 (0.90) | 1.453      | matched         |
| python_modules_long   | 1.383 (0.80) | 1.328      | −4%, mis-tuned  |
| long_code             | 1.337 (0.85) | 1.282      | −4%             |

It matched the optimum on easy/repetitive content but **mis-tuned on the harder
suites** — short hill-climb windows have enough per-region variance that the
search wanders to the tight end exactly where loose is optimal. A fixed gate (or
per-deployment override) beats the controller where it matters most, so the
adaptive path is not worth the hot-path complexity. Revisit only with a
much-longer-window or feed-forward (head-confidence) signal, and only after
sampled-mode validation.

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
