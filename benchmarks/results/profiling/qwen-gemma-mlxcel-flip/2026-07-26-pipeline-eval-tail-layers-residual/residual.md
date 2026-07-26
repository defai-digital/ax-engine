# Path B residual: `AX_MLX_PIPELINE_EVAL_TAIL_LAYERS` (mbp-m5, 2026-07-26)

**Decision: `reject_tail_overlay` / not_yet.** Gates unchanged. No S0–S3.

## Residual

Force per-layer blocking eval on the last `N` multi-token layers (final layer
still exempt) while the thr-oriented base stack keeps `block:8` on early
layers. Intent: monopolize early prefill for thr, then yield the tail to a
sibling decode process for stream-gap fairness.

Env: `AX_MLX_PIPELINE_EVAL_TAIL_LAYERS=N` (default **0** / off).

Pure helpers unit-tested:

- `parse_pipeline_eval_tail_layers` — fail-closed to 0
- `pipeline_eval_layer_in_tail` — last N multi-token layers before final
- `pipeline_eval_should_fire` — tail overlay OR base granularity

## Concurrent dual-target S1 smokes (1-rep, thr-b8 + Qwen util)

Base target: `ax-qwen-gemma-m5max-multiprocess-cache-eval-thr-b8-qwen-util`.
Peer: `mlxcel-v0.4.2-qwen-gemma-m5max`. Scenario S1 prefill isolation.
Build/log: `run_tail_eval.log` on mbp-m5 (`AKMBPM5MAX`).

| config | thr | gap | ttft | ax thr | ax gap |
|--------|----:|----:|-----:|-------:|-------:|
| **tail0 (baseline)** | **1.146** | 1.172 | **0.869** | **20.96** | 42.2 |
| tail4 | 1.064 | 1.175 | 0.938 | 19.41 | 40.7 |
| tail8 | 1.080 | **1.016** | 0.923 | 20.17 | 39.0 |
| tail12 | 1.105 | 1.044 | 0.902 | 20.66 | 38.6 |
| tail16 | 1.119 | 1.112 | 0.890 | 20.58 | 38.2 |
| tail24 | 1.077 | 1.131 | 0.926 | 19.86 | 39.5 |
| tail12-only (granularity off + tail12) | 1.077 | 1.167 | 0.926 | 19.71 | 40.2 |

Need: thr ≥ **1.15**, gap ≤ **0.90**, TTFT ≤ **0.90**.

## Physics

- Every `N>0` tail overlay **regresses thr** vs baseline (1.146 → 1.06–1.12).
- Best gap among overlays is tail8 at **1.016** — still **≫ 0.90**, and thr
  only **1.080** (far from gate).
- Dual score cannot clear thr+gap together; cool formal correctly **skipped**
  (`need thr>=1.14 and gap<=1.12` not met by best dual-score pick).
- Absolute stream-gap stays ~38–42 ms vs mlxcel ~34–38 ms; relative ratio never
  reaches the 0.90 gate.

## Conclusion

Keep default **tail overlay OFF**. Ship residual as fail-closed opt-in for
future probes only — not product-on. No cool formal, no S0–S3. Flip remains
**not_yet**. Gates file unchanged.

## Artifacts

- Ladder log: `run_tail_eval.log`
- Smoke dirs: `2026-07-26-s1-smoke-thr-b8-util-tail0`,
  `2026-07-26-s1-smoke-thr-b8-tail{4,8,12,16,24}-util`,
  `2026-07-26-s1-smoke-thr-b8-tail12-only-util`
- Targets: `ax-qwen-gemma-m5max-multiprocess-cache-eval-thr-b8-tail*-qwen-util.json`
