# Flash Next prefill and decode against the pinned reference — 2026-10-05

This note records why Qwen 3.8 Flash Next (`qwen4_exp`, MXFP4 MTP pack
`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`) prefilled and decoded slower than an
independent runtime on the same bytes, what changed, and what was measured. It is
a single-host engineering record, **not a speedup claim**: the reference is a
labelled, non-primary runtime and is timed in process while AX is timed through
an HTTP stream.

## Finding

The reviewed upstream `mlx_lm` cannot load `qwen4_exp`, so the comparison runtime is
the pinned, independently implemented MLX-VLM 0.7.0rc0 language graph (commit
`6102cb4a`, MLX 0.32.2, Transformers 5.18.0.dev0), run on the same pack bytes on
the Mac Studio M2 Ultra, 192 GiB, internal SSD. On the same fixed workload (uniform
random token IDs, 128 greedy output tokens, two warmups and three measured runs
per cell) the server built from `7efa5516` was far behind the reference: prefill
was 2.8-3x slower (8,192 tokens: 69 s against 25 s) and decode 0.78-0.84x.
Disabling expert streaming changed nothing, so paging was not the cause.

## What changed

Stage attribution (an ignored real-pack test with a barrier after each stage) of a
2,048-token chunk put 48% of the time in the sequential GDN recurrence and 21% in
an unsorted expert gather, and sampling the generation thread during decode put 77%
of its time inside the QSA indexer's per-layer host read-back, which also stops
the CPU from encoding the next layers while the GPU runs. Six changes follow; five
have a kill switch that restores the previous behaviour.

| Change | Where | Kill switch |
| --- | --- | --- |
| Gather routed experts in expert-id order for prefill chunks of 64 or more tokens (the SwitchGLU layout mlx-lm uses) and restore token order before the weighted sum | `qwen4_exp_moe.rs` | `AX_MLX_FLASH_NEXT_PREFILL_EXPERT_SORT=0` |
| Use MLX `gated_delta_update` for prefill chunks of 128-2,048 tokens (the adapter already existed, default off) | `flash_next_gdn.rs` | `AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX=0` |
| Choose QSA blocks on the device for single-query steps. The kept-token count depends only on position, so the index tensor has an exact host-known length and no read-back is needed. A stable sort of the negated scores reproduces the host order (score descending, lower block first on ties) | `qwen4_exp_qsa.rs` | `AX_MLX_FLASH_NEXT_QSA_DEVICE_SELECT=0` |
| Attend prefill chunks of 16 or more queries in padded, masked sub-batches (about 1 GiB of gathered K/V each) instead of one gather and attention call per query | `qwen4_exp_attention.rs` | `AX_MLX_FLASH_NEXT_QSA_BATCHED_ATTENTION=0` |
| Attend densely and causally while the whole context fits the QSA budget (2,051 tokens for this geometry): every query keeps every visible token, so selection is the identity | `qwen4_exp_attention.rs` | `AX_MLX_FLASH_NEXT_QSA_DENSE_SHORT_CONTEXT=0` |
| Skip the LM head for non-final prefill chunks and project only the last row of the final one (about 2 GiB and 0.26 s per 2,048-token chunk) | `qwen4_exp.rs`, `model/mod.rs` | none (verification and MTP keep all rows) |

## Numerics

Every change reorders floating-point work; none is an approximation in exact
arithmetic. Prefill arithmetic for long prompts therefore differs from the previous
build, single-row logits use the decode-shaped projection, and greedy token
identity with the previous arithmetic is not promised: eleven of sixteen natural
128-token trajectories diverge from the selected-attention path, as do most
random-token runs, at the ordinary bf16 level. The evidence that this is not a
quality regression is separate and bounded:

- unit tests compare device and host QSA selection over tie-heavy rows, batches and
  a pinned Transformers fixture, and single-query attention is bitwise identical
  between the two selections;
- the batched prefill attention matches the per-query path to 2e-5 (float32) and
  4e-3 (bfloat16) on ragged selections, and the dense path advances exactly the
  same K/V and index-key caches;
- the real-pack last-row logits differ from the full projection by 0.031 at most with
  the same greedy token;
- the frozen 105-item cohort passes 101/105 in explicit-direct, product-default and
  required-MTP modes with identical text across modes, and every failure it
  retains is a failure the reference runtime also has;
- the 29,774-token lookup still answers `1734` in every mode.

This is not a teacher-forced logit comparison against the official graphs on this
pack revision, which remains open.

## Results

Fixed workload, product-default arm and explicit-direct arm, each the lower-latency
of two independent server processes (see below). Cells are TTFT in seconds / decode in
tokens per second; ratios are AX over the reference.

| Input tokens | Reference | AX before (`7efa5516`) | AX now, default | AX now, explicit direct | Ratio, default (TTFT / decode) | Ratio, explicit direct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1.13 / 22.9 | 4.27 / 18.0 | 1.22 / 23.8 | 1.22 / 23.7 | 1.08 / 1.04 | 1.08 / 1.03 |
| 2,048 | 6.53 / 17.4 | 16.38 / 17.5 | 4.31 / 22.5 | 4.32 / 22.4 | 0.66 / 1.30 | 0.66 / 1.29 |
| 8,192 | 24.55 / 20.4 | 69.67 / 17.0 | 23.94 / 21.9 | 24.07 / 21.7 | 0.97 / 1.07 | 0.98 / 1.06 |

The frozen release bounds are AX decode at least 0.90x and TTFT at most 1.15x of
the reference in every cell. The 29,774-token lookup in the frozen cohort now takes
about 104 s (it took 268-272 s before). Required MTP still decodes at
0.93, 0.92 and 0.91 of direct speed at 512, 2,048 and 8,192
input tokens with identical tokens, so it remains an explicit opt-in and no MTP
speedup is claimed.

## Measurement notes

- The first server process after the host's file-cache or kernel-cache state changes
  serves its first requests 30-45% slower and settles over a few runs
  (2.09 s, 7.05 s and 30.54 s at 512, 2,048 and 8,192
  tokens). The harness runs a discarded pass over the identical workload first and keeps
  it as the cold-regime observation.
- The host also showed time-varying background interference: the same binary gave
  2,048-token TTFT of 5.8 s and 4.3 s within one hour, while three consecutive clean
  processes agreed to better than 1%. Each arm therefore runs in two interleaved
  server processes and keeps the lower-latency one; the repeats must emit identical
  tokens, and all of them are in the raw record. The frozen thresholds record this
  amendment. Timing runs also start only at 90% or more system CPU idle.
- AX is timed on the client through the native `/v1/generate/stream` event stream with
  `ignore_eos`; the reference is timed in process, so the comparison disadvantages AX by
  the transport. Neither number is a throughput benchmark of either project.

## Evidence

[Release-gate evidence](../../benchmarks/results/qualification/2026-10-05-flash-next-mxfp4-release-gates/README.md) holds the raw results, the pre-change
measurement, the reference run and the evaluator's verdict.
