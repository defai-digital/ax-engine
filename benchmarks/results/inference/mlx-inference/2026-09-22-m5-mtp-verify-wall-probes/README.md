# Qwen 3.8 27B MTP verify: bandwidth-wall probes (M5 Max, 2026-09-22)

Follow-up to `../2026-09-22-m5-invariant-qmv-bf16-q4/` (bf16/4-bit invariant
kernel, opt-in depth-4 throughput profile, width-aware depth controllers).
This directory records the probes that did **not** move the default MTP
decode cycle, so the next session does not repeat them, plus the one that
did (the 3-bit draft lm_head, last section). No route, policy, or
certification claim changes; nothing here is a README performance claim.

Host: MacBook Pro M5 Max 128 GB, macOS 27, MLX 0.32.2, server binary built
from `a8adcaac`. Lane: `scripts/bench_mlx_inference_stack.py` on the flappy
real-prompt suite, 256 generated tokens, greedy, MTP depth 3 (default
throughput profile), n-gram stacking off, `--ax-qwen-linear-mtp-exact` as in
the earlier campaigns. Lane scripts are in `lanes/`.

## Pack composition (corrects an assumption in the earlier README)

`AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` is a mixed-precision recipe whose global
setting is affine **4-bit, group 32**: every trunk projection (gate/up/down,
in_proj_qkv/z, out_proj, q/k/v/o) is 4-bit/g32 except a handful of
4-bit/g64 FFN layers, `embed_tokens` at 8-bit, and 9 of the 48 gated-delta
layers whose `in_proj_a`/`in_proj_b` are 6-bit/g64 (plus one layer with a
6-bit/g32 `in_proj_b`). Those 9 layers are the ones the direct C++
linear-attention inputs route refuses at `seq == 1` (mixed-bit packing is
decode-excluded by design), which is what the earlier
`ax_mlx_direct_cpp_linear_attention_inputs_fallbacks` counter was showing.

## Cycle anatomy (head lane, `lanes/flappy_head*.json`)

| lane | decode tok/s (median of 3) | verify eval / cycle | draft / cycle | rollback / cycle |
| --- | ---: | ---: | ---: | ---: |
| head | 83.67 / 83.71 / 83.69 (three runs) | 43.05 ms | 0.055 ms | 0.58 ms |

3.5 tokens are accepted per cycle on average (acceptance ~100% on flappy).
The trunk streams ~16.5 GB of weights per verify pass (4-bit weights, bf16
scale/bias sidecars, 8-bit lm_head), so the S=4 verify runs at ~384 GB/s
effective against a ~530 GB/s sustained ceiling, and ~10 ms above a singleton
step (~33 ms, ~500 GB/s). MLX `quantized_matmul` on the verify shapes runs
at 440-520 GB/s at M=4 (`../2026-09-22-m5-invariant-qmv-bf16-q4/`), so the
projection efficiency loss explains 3-4 ms; the remaining 6-7 ms is spread
across the S-scaling kernels (gated-delta scan, conv, norms, lm_head rows,
argmax) and dispatch.

## Probes that are neutral (all within 0.4% of head, `lanes/`)

| probe | decode tok/s | verify / cycle | what it tests |
| --- | ---: | ---: | --- |
| `AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS=0` | 83.97 | 42.84 ms | packed QKVZ/BA staging via MLX ops instead of the direct C++ route |
| both direct linear-attention routes off | 83.63 | 43.02 ms | plus the post-input (conv/norm/gate) direct route |
| `--ax-pack-dense-ffn-gate-up` | 83.65 | 43.07 ms | route stays split (Qwen keeps split gate/up by design; the flag does not engage) |
| `AX_MLX_LINEAR_ATTENTION_WHOLE_LAYER_METAL=1` | 83.58 | 43.09 ms | compositional whole-layer gated-delta path |
| `AX_MLX_MTP_GDN_TGY=8` / `16` / `32` | 83.59 / 83.71 / 83.65 | 43.04-43.08 ms | verify gated-delta threadgroup width (fewer redundant key-head preps per value row) |

Conclusion: at S=4 the cycle is bound by weight streaming; per-op dispatch
savings, the direct FFI routes, and the gated-delta launch geometry do not
register. The 8/46 per-step direct-route fallbacks are therefore not a lever
on this workload.

## 6-bit microbatch kernel probe (`microbench/`)

Measured before the pack composition above was checked; kept because it
bounds what a repo-owned 6-bit microbatch kernel can do on this host.

`q6_baseline.txt`: MLX `quantized_matmul` at 6-bit/g64 on the verify shapes
is 460-540 GB/s at M=1..3, 400-500 at M=4, 360-470 at M=5 and collapses to
240-320 GB/s at M=6 (the depth-5 profile pays this cliff). The generic
invariant `qmv_fast` kernel at 6-bit matches MLX at M=1..2 and falls to
300-340 GB/s at M=4 (register cliff from the f32 x copies).

`q6_proto_run1.txt`, `q6_proto3.py`: a bf16-bit-pair / unscaled-6-bit
microbatch kernel (16 values per thread, 4 rows per simdgroup) reaches parity
with MLX at M=1..3 and loses at M=4 (0.75-0.83x) and M=5 (0.32-0.54x): the
16-entry unpacked weight array plus the x bit pairs spill. An 8-value variant
(`E8`, three ushort loads per lane) is at parity at M=4 (1.00-1.03x) and
0.92x at M=5; 8 rows per simdgroup is worse (0.61-0.98x); inline unpacking
without the array (`A16`) is 0.70-0.83x at M>=4. No variant beats MLX by more
than 3% at any M, so a 6-bit microbatch kernel is not a lever either.

## Blocking stage profiles (`lanes/flappy_decode_profile.json`, `lanes/flappy_la_profile.json`)

Recorded for completeness. `AX_MLX_DECODE_PROFILE` and
`AX_MLX_LINEAR_ATTENTION_PROFILE` force a blocking eval per stage, which
drains everything queued before the stage and mis-attributes time (the
gate/up stage absorbs 58% of the profiled wall). They are not usable for the
~6-7 ms attribution question above; a per-kernel GPU timeline is.

## Joint review: split-K verify QMM below 16384 outputs (`split_k_min_n/`)

The reviewers' first proposal was to admit the 5120-6144-10240-wide
projections to the split-K verify QMM (`AX_MLX_MTP_VERIFY_QMM_MIN_N`,
default 16384). End to end, same binary:

| suite | head | `MIN_N=4096` | `MIN_N=2048` |
| --- | ---: | ---: | ---: |
| flappy | 83.69 | 84.51 (+1.0%, verify 43.05 -> 42.45 ms) | 84.01 (+0.4%) |
| long_code | 77.78 | 78.50 (+0.9%, verify 44.21 -> 43.62 ms) | - |
| python_modules_long | 75.33 | **71.34 (-5.3%)**, tokens/cycle 3.99 -> 3.93 | - |

Greedy identity (`identity_probe_minn.py`, 8 flappy + long_code prompts,
256 tokens, two repeats per arm): 7/8 identical; `long_code_api_audit`
diverges at token 221, the same near-tie prompt that flipped under the
depth-4 probe. The split-K reduction order changes the relaxed verify's
logits at the ULP level, and on python_modules_long the changed stream
loses more acceptance than the 0.6 ms/cycle projection saving is worth.
Not a default candidate; the env override stays for per-host experiments.

The opposite proposal, routing the >= 16384-wide projections back to stock
MLX `quantized_matmul` (`MIN_N=200000`: gate/up to stock, lm_head keeps the
multi-simdgroup kernel; `MIN_N=1000000000`: everything stock), loses 1.4%
(82.53 / 82.59 vs 83.69, verify 43.06 -> 43.75 ms). The current default
threshold is the measured optimum on this host in both directions.

## Joint review: 3-bit draft lm_head for dense targets (`draft_lm_head/`)

The kimi reviewer's lever. The pack's `lm_head` is dense bf16 (2.5 GB); for
dense targets the loader already requantizes a draft-only copy at 4-bit
group 64 (`AX_MLX_MTP_DENSE_HEAD_DRAFT_Q4`), and the draft step re-reads it
once per depth. Its bytes hide inside the verify eval (the async draft is
materialized by the next verify), which is why the `draft` counter shows
0.05 ms while the verify eval carries the cost. Same binary, env override
`AX_MLX_MTP_DRAFT_LM_HEAD_BITS=3 AX_MLX_MTP_DRAFT_LM_HEAD_GROUP_SIZE=64`:

| suite | head (4-bit g64 draft head) | 3-bit g64 draft head | tokens / cycle |
| --- | ---: | ---: | --- |
| flappy | 83.69 | **85.10 (+1.7%)**, verify 43.06 -> 42.53 ms | 4.02 -> 4.02 |
| long_code | 77.78 | **79.06 (+1.6%)**, verify 44.21 -> 43.70 ms | 4.00 -> 4.00 |
| python_modules_long | 75.33 | **76.68 (+1.8%)**, verify 43.12 -> 42.60 ms | 3.99 -> 3.97 |

4-bit group 128 is +0.2% (`flappy_draft_head_q4g128.json`). Greedy identity
(`identity_probe_q3g64.py`, 8 prompts, two repeats per arm): **8/8
identical**, as expected for a draft-side change (the target head decides
acceptance). Acceptance is unchanged on all three suites, so the 2-bit
acceptance collapse recorded for the decode overlay does not appear at
3 bits. Landed as the new dense-target default
(`AX_MLX_MTP_DENSE_HEAD_DRAFT_BITS`, default 3, `4` restores the previous
width; `AX_MLX_MTP_DENSE_HEAD_DRAFT_Q4=0` still disables the requantized
draft head entirely).

## What is left

- The opt-in depth-4 profile (`AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP_DEPTH=4`,
  +9.5% flappy) remains the only structural lever measured on this pack;
  promoting it to the default needs the ADR-033 MTP-D evidence set, not a
  session A/B.
- Attributing the remaining ~6-7 ms of S-scaling cost needs a GPU timeline
  (Metal System Trace or per-kernel timestamps), not the blocking profiles.
