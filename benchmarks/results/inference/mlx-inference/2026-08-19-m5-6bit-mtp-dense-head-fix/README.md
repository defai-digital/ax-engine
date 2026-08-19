# 2026-08-19 M5: 6bit-MTP dense-head fix (0.80× → 1.15×)

Host `df-macbookpro-m5`, harness `bench_mlx_inference_stack.py`
(`--ax-ngram-accel --ax-qwen-linear-mtp-exact`, p128/gen256, 5 reps),
pack `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` (dense bf16 lm_head,
2.54 GB). Baselines from the 2026-08-18 campaign: direct 33.49 tok/s,
MTP-exact 26.82 tok/s (0.80×), MXFP4-MTP sibling 53.75 (its lm_head is
affine 8-bit).

## Root cause

Two stacked costs, both on the dense lm_head:

1. **Per-call re-materialization (bug).** The invariant projection's
   dense arm fed `qw.weight` — a lazy transpose view after
   `prepare_contiguous_decode_weight_t` — to its Metal kernel, whose
   `ensure_row_contiguous` copied the full 2.54 GB head on **every**
   draft and verify call. Walls: draft 18.6 ms/step, verify-eval
   22.9 ms/step (vs 4.1/6.4 on the quantized-head MXFP4 sibling).
2. **Dense draft logits (physics).** Even without the copy, every draft
   step reads the whole 2.54 GB head for logits that only propose.

## Fixes and measured steps

| run | change | decode tok/s | draft wall/step |
| --- | --- | --- | --- |
| R6 (08-18) | broken baseline | 26.82 | 18.6 ms |
| R10 | weight_t GEMV routing (S=1..8, one arithmetic) | 33.14 | 11.5 ms |
| R12 | + draft from 2-bit decode overlay — **rejected**: acceptance collapsed (3 accepted / 8 steps), MTP bypass gate tripped | 32.70 (MTP effectively off) | — |
| R14 | + derived **4-bit gs64 draft lm_head** (existing `build_draft_lm_head` machinery, ~320 MB) | **38.38** | **3.0 ms** |

Net: **26.82 → 38.38 (+43%)**; vs direct 33.49 = **1.146× MTP speedup**
(was 0.80×). Acceptance intact (117/123 full-accept, misses 27 → 6).
R15: MXFP4-MTP unchanged (53.75 → 53.76) — quantized heads take neither
new path.

## Parity

4-way greedy probe (256 tokens): MTP-off and MTP-off+gate-metal are
**bit-identical across the pre-fix and fixed binaries** (`fa559ab8…`) —
the weight_t GEMV reproduces the old kernel's stream. MTP-on diverges
from MTP-off on this pack **on both binaries** (pre-fix `15e3a10f`,
fixed `f546a26c`; gate-metal a third stream on both) — a pre-existing
exact-profile drift on the 6bit pack, unlike MXFP4-MTP whose probe is
4-way clean. Filed for the pending v7 re-cert pass; not introduced by
this fix.
