# Qwen 3.8 27B MXFP4 on pre-M3 Apple GPUs: activation dtype and dense-GEMM prefill — 2026-10-10

Why the primary pack prefilled at the MLX kernel floor on a Mac Studio M2
Ultra, what the host lacks, which two runtime policies now adapt to it, and
what was measured on the M2 Ultra and on the M5 Max bench host. Single-host
engineering evidence plus a bench-host regression check; **not a SKU
qualification** (the 27B SKU is the Mac mini M4 Pro 64 GB).

Pack: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP` @
`b2c5354f779e430d0c1733143db848a72b71c16e`. MLX 0.32.3 (pip wheel).
Harness: `scripts/bench_mlx_inference_stack.py` (greedy, random-token
prompts for direct rows; the `flappy` real-prompt suite for the MTP row).

## Finding

On the M2 Ultra (MLX architecture `applegpu_g14d`) direct decode and prefill
were both already at the stock-MLX kernel floor, and ahead of `mlx_lm`
0.31.3 on the same pack:

| row (M2 Ultra 192 GB) | AX prefill tok/s | AX decode tok/s | mlx_lm prefill | mlx_lm decode |
| --- | ---: | ---: | ---: | ---: |
| p128 gen128 | 185 | 33.3 | 136 | 31.4 |
| p512 gen128 | 210 | 32.7 | 192 | 31.1 |
| p2048 gen128 | 215 | 32.3 | 210 | 30.7 |

`decode-trace` put 1.15 ms of a 30.6 ms step on the host; the rest is GPU.
A per-op probe of every projection at M=1024 (`quantized_matmul`, SDPA,
norms, conv, gated delta) summed to a 208 tok/s floor in bf16 — the runtime
has no overhead left to remove on this host. What the host lacks is native
bfloat16 arithmetic: GPU generations before 15 (M3) convert bf16 per op, and
the same probe with fp16 activations summed to 279 tok/s (+34%). Stock
`mlx_lm` with the model cast to fp16 confirmed it end to end: 206-213 →
274-285 prompt tok/s on a 1024-token prompt, with identical greedy tokens on
three real prompts.

The second host-specific result: dequantizing MXFP4 weights to the activation
dtype and running the steel dense GEMM (`AX_MLX_QWEN_PREFILL_DEQUANT_DENSE`)
is a documented 3% regression on the M5 Max, but on the M2 Ultra it measured
p2048 prefill 255 vs 215 tok/s (+19%) in bf16.

## What changed

| Policy | Env | Default | Where |
| --- | --- | --- | --- |
| Activation dtype | `AX_MLX_ACTIVATION_DTYPE` = `auto` / `bfloat16` / `float16` | `auto`: fp16 on GPU generations below 15 (M1/M2), bf16 elsewhere | `fastpath::activation_dtype`, cast pass in `weights.rs` (`qwen3_5` family only) |
| Dense-GEMM prefill | `AX_MLX_QWEN_PREFILL_DEQUANT_DENSE` = `auto` / `0` / `1` (resident copy) / `transient` (per-call); `AX_MLX_QWEN_PREFILL_DEQUANT_DENSE_MIN_SEQ` | `auto`: transient on generations below 15 with a 512-token floor, off elsewhere (1024 floor when forced on) | `fastpath::qwen_prefill_dequant_dense_mode`, `model/shared/utils.rs` |
| Exact-MTP lm_head window | `AX_MLX_EXACT_LM_HEAD_WINDOW_INVARIANT` | on (`=0` restores the row loop) | `model/mod.rs::lm_head_verify_window_projection` |

The GPU generation comes from MLX's device architecture string
(`applegpu_g14d` on the M2 Ultra, `applegpu_g17s` on the M5 Max); an
unknown architecture keeps bf16 and the qmm.

The lm_head change is host-independent: under the exact Qwen linear MTP
profile a 2..=4-row verify window was projected through the quantized head
one row at a time (commit 5a19dd7a, because MLX's `qmv_wide` at M>=2 is not
row-invariant). The invariant affine qmv_fast kernel already keeps every row
on the singleton arithmetic whatever `Leading` is, so one launch over the
window reads the 1.3 GB head once per cycle instead of once per row. A unit
test (`invariant_projection_window_matches_row_by_row_bit_exact`) pins the
window output bit-for-bit to the row loop for 4- and 8-bit affine heads in
bf16 and fp16.

## Measured (A/B, same binary, env flags only)

### MacBook Pro M5 Max 128 GB (bench host, signal only; Homebrew cargo 1.98.0, MLX 0.32.3, commit e8c60bb1, 2 reps + 1 warmup unless noted)

| arm | p128 prefill / decode | p512 prefill / decode | p2048 prefill / decode | output tokens vs baseline |
| --- | ---: | ---: | ---: | --- |
| baseline (bf16, qmm; 3 reps + 2 warmups) | 626 / 35.6 | 911 / 35.5 | 931 / 35.0 | — |
| `AX_MLX_ACTIVATION_DTYPE=float16` | 620 / 35.7 | 911 / 35.6 | 978 / 35.2 | diverge at token 3-4 (random-token prompts) |
| `AX_MLX_QWEN_PREFILL_DEQUANT_DENSE=1` (resident) | 623 / 35.7 | 911 / 35.5 | 1004 / 35.1 | identical |
| `AX_MLX_QWEN_PREFILL_DEQUANT_DENSE=transient` | 630 / 35.6 | 912 / 35.5 | 905 / 35.1 | identical |

| MTP arm (flappy, depth 3, exact verifier, gen 256) | decode tok/s per prompt (264 / 273 / 374 / 432 tokens) | output tokens vs baseline |
| --- | --- | --- |
| baseline (3 reps + 2 warmups) | 77.0 / 76.6 / 75.3 / 77.0 | — |
| `AX_MLX_EXACT_LM_HEAD_WINDOW_INVARIANT=1` | 77.0 / 76.6 / 75.3 / 77.0 | identical (all 4 x 256) |
| `AX_MLX_ACTIVATION_DTYPE=float16` | 81.6 / 81.5 / 81.1 / 79.6 | identical (all 4 x 256) |

Reading: on the M5 Max the dense GEMM wins only with the resident copy (+8%
at p2048, ~54 GB extra memory), so `auto` keeps it off there; fp16 is +5%
prefill and +6-8% MTP decode with identical real-prompt tokens, which is
recorded as an opt-in candidate for a certification pass rather than a
default change on native-bf16 hosts.

### Mac Studio M2 Ultra 192 GB (campaign host; rustup 1.97.1, MLX 0.32.3, 2 reps + 1 warmup, every arm sets its flags explicitly)

| arm | p128 prefill / decode | p512 prefill / decode | p2048 prefill / decode | output tokens vs bf16 |
| --- | ---: | ---: | ---: | --- |
| bf16, qmm (control) | 186.5 / 33.2 | 210.6 / 32.7 | 215.4 / 32.2 | — |
| fp16 | 246.9 / 35.5 | 283.9 / 35.1 | 290.0 / 35.0 | p128 identical; p512/p2048 diverge at token 3-4 (random-token prompts) |
| bf16 + dequant dense (resident) | 185.9 / 33.1 | 210.7 / 32.6 | 255.6 / 32.4 | identical |
| fp16 + dequant dense (resident) | 244.0 / 35.6 | 284.1 / 35.1 | 342.9 / 35.0 | identical to fp16 |
| fp16 + dequant dense (transient) | 246.8 / 35.6 | 282.7 / 35.0 | 335.6 / 35.1 | identical to fp16 |
| fp16 + transient, floor 512 | 246.7 / 35.5 | 308.1 / 35.1 | 336.1 / 35.0 | identical to fp16 |
| `auto` (the new default on this host: fp16 + transient) | 247.6 / 35.6 | 284.1 / 35.0 | 335.7 / 34.7 | identical to fp16 |

| MTP arm (flappy, depth 3, exact verifier, gen 256) | decode tok/s per prompt (273 / 374 / 432 / 264 tokens) | output tokens vs control |
| --- | --- | --- |
| bf16 control (window off) | 52.2 / 52.1 / 51.0 / 52.3 | — |
| window on | 52.4 / 52.0 / 50.9 / 52.3 | identical (all 4 x 256) |
| fp16 | 55.7 / 55.7 / 54.8 / 56.0 | identical (all 4 x 256) |
| fp16 + window | 55.7 / 55.7 / 54.5 / 55.9 | identical |
| `auto` | 56.0 / 55.6 / 54.5 / 55.7 | identical |

Net on the M2 Ultra under the new defaults (the floor-512 default was set
after this run): prefill +33% at p128, +46% at p512, +56% at p2048; direct
decode +7-9%; MTP decode +7%. The decode gain is the kernel, not bytes: an
idle-GPU probe of a 64-layer M=1 MXFP4 FFN chain streams 505 GB/s in bf16
and 544 GB/s in fp16 on this host, against 664 GB/s for a plain 8 GiB
reduction. The earlier "61% of the published 800 GB/s" was therefore the
stock MXFP4 matvec on this two-die part (76-82% of what a streaming read
achieves here), not runtime overhead, and nothing in AX's graph could have
recovered it.

Peer context on the same host and pack: `mlx_lm` 0.31.3 decodes 30.7-31.4
tok/s; MTPLX 2.12.2 (sustained, depth 3) 37-41 tok/s on the flappy suite;
OMLX 0.7.1.dev1 on an imported snapshot 12-23 tok/s (unstable). None of
these numbers qualify anything; they locate AX relative to its peers here.

## Final commit (`3c4065b0`) under its defaults, 3 reps + 2 warmups

| host | p128 prefill / decode | p512 | p2048 | flappy MTP decode (273 / 374 / 432 / 264) | tokens vs the pre-change arm |
| --- | ---: | ---: | ---: | --- | --- |
| M2 Ultra 192 GB, before (fe13a04b) | 186 / 33.3 | 210 / 32.7 | 216 / 32.3 | 52.5 / 52.1 / 51.2 / 52.4 | — |
| M2 Ultra 192 GB, after | 242 / 35.5 | 307 / 35.1 | 335 / 35.1 | 55.7 / 55.3 / 54.6 / 55.8 | MTP identical; direct identical to the fp16 A/B arm |
| M5 Max 128 GB, after (defaults unchanged there) | 624 / 35.6 | 910 / 35.5 | 971 / 35.1 | 77.0 / 76.6 / 75.3 / 77.0 | identical to the M5 baseline (direct and MTP) |
| M5 Max 128 GB, `mlx_lm` 0.31.3 reference | 437 / 34.6 | 777 / 34.5 | 920 / 34.0 | — | — |

M2 Ultra net: prefill +31% (p128), +46% (p512), +56% (p2048); direct
decode +7-9%; MTP decode +6-7%. The M5 Max keeps its path and its numbers
(the p2048 prefill spread, 921-972 across runs, is within this host's
run-to-run variation).

## The same policies on Qwen 3.5 9B (community affine 4-bit, group 64)

`mlx-community/Qwen3.5-9B-MLX-4bit` @ `938d8919` (`qwen3_5`, 32 layers,
affine 4-bit g64, no MTP sidecar; `ax-engine-bench generate-manifest`
added the manifest), Mac Studio M2 Ultra 192 GB, final commit, 3 reps +
2 warmups, direct rows:

| arm | p128 prefill / decode | p512 | p2048 | tokens vs before |
| --- | ---: | ---: | ---: | --- |
| before (bf16, qmm, row loop forced) | 639 / 96.4 | 762 / 94.8 | 792 / 93.0 | — |
| `mlx_lm` 0.31.3 reference | 460 / 97.9 | 685 / 96.6 | 776 / 95.8 | — |
| fp16 only | 847 / 105.8 | 1039 / 104.1 | 1093 / 102.3 | |
| defaults (`auto`: fp16 + transient dense GEMM from 512) | 847 / 105.6 | 1036 / 104.0 | 1164 / 104.1 | p128 and p2048 identical; p512 diverges at token 75 (random-token prompt) |

Net on the M2 Ultra for the 9B: prefill +33% / +36% / +47%, decode
+10-12%. The affine pack takes the same two policies (the dense GEMM also
beats the affine qmm at M>=512 on this host); the lm_head window does not
apply because the pack has no MTP head.

## Numerics and scope

- fp16 has a narrower range than bf16 (max 65504). The RMS-normalized
  residual stream of this model stayed finite on every measured prompt and
  greedy tokens matched bf16 on the probed prompts, but fp16 is not promised
  to be token-identical to bf16 in general; the policy only engages on GPU
  generations that have no native bf16, where bf16 itself is emulated.
- Gated-delta recurrent state, decay and `A_log` / `dt_bias` stay f32.
- Quantized payloads and MXFP e8m0 scales are untouched by the cast pass;
  affine scales/biases (the 8-bit lm_head) follow the activation dtype.
- `transient` dequant holds no resident dense copy; `1` keeps ~2 bytes per
  parameter resident (~54 GB on the 27B) and is only for hosts with the
  memory to spare.
- The M4 Pro SKU (generation 16) and the M5 Max bench host keep bf16 and
  the steel qmm under `auto`; the policies are host adaptations, not a
  change to the qualified path.
