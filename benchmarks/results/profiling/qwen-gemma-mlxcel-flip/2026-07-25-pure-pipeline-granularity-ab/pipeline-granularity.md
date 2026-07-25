# Lever: layer-boundary pipeline granularity (mlxcel M5 residual)

## Residual

Best formal S1 (multi-process AX + Gemma `AX_MLX_CACHE_ONLY_CHUNK_EVAL=1`):

| metric | AX | mlxcel | ratio | gate |
|--------|---:|-------:|------:|------|
| thr tok/s | 19.92 | 17.97 | **1.109** | need ≥1.15 |
| gap p95 ms | 39.4 | 35.4 | **1.113** | need ≤0.90 |
| TTFT p95 | — | — | **0.899** | PASS |
| abs gap | 39.4 | — | — | PASS ≤50 |

Need ~**3.7%** thr (scenario wall ≲9.08s from ~9.43s) and gap ratio fix.
Pure compose under cache_eval keep_base: norot/qmmrms reject.

Pure Gemma cache_eval cold ~**8.22s**; multi-process Gemma e2e ~**9.3–9.5s**
(~14% concurrent Metal tax). Cutting pure ~4% **or** concurrent tax is
required for thr≥1.15.

## mlxcel source

- `mlxcel_core::utils::pipeline_hint` (`.internal/reference/mlxcel/.../utils.rs`)
- Env `MLXCEL_PIPELINE_GRANULARITY`: `off` (default) | `layer` | `block:N`
- Called from Gemma4 / Gemma / Qwen3 / Llama3 layer loops after each non-final
  layer: `async_eval(hidden)` so layer N can run while host builds N+1 /
  weights prefetch.
- Comment: **“On M5 (Neural Accelerator + GPU shader cores), this can improve
  throughput by overlapping NA compute for layer N with weight loads for layer
  N+1.”**

AX had no equivalent. Dual Metal / dual_gate_up compile / #705 shaped /
host fuses already rejected for gate_up residual.

## AX change

1. `AX_MLX_PIPELINE_GRANULARITY` (default off) + `pipeline_hint_should_fire`.
2. After each non-final layer in standard multi-layer prefill
   (`forward_and_logits_mode` + media-range path): `async_eval(&hidden)`.
3. Pure 13.8k cold 3-rep under **cache_eval ON** baseline (matches multi-process
   Gemma): variants `block:4`, `block:2`, `layer`.

Keep if median pure ratio ≤ **0.96** vs base (physics for thr ~1.15 under
multi-process). Else reject; default stays off.

## Result (mbp-m5, 2026-07-25, 3-rep interleaved)

Under cache_eval keep_base pure Gemma 13.8k cold:

| variant | median cold ms | ratio vs base |
|---------|---------------:|--------------:|
| base (off) | 10340 | 1.000 |
| block:4 | 11068 | **1.070** |
| block:2 | 10752 | **1.040** |
| layer | 10847 | **1.049** |

Decision: **reject_keep_off** (`keep_base`). No pure cut; multi-process S1
not re-run for this lever. Thermal noise high (base r1 8.4s → r3 11.1s) but
every candidate median ≥ base. Gates unchanged; flip remains **not_yet**.

## Success

Cool multi-process S1 thr ≥1.15 **and** gap ratio ≤0.90 **and** TTFT ≤0.90
→ full S0–S3 flip. Else not_yet; gates unchanged.
