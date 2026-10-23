# Flash Next n-gram reads and fused decode stretches — 2026-10-22

This note records why Qwen 3.8 Flash Next (`qwen4_exp`, MXFP4 MTP pack
`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`) prefilled novel prompts at 300-450 tok/s
and decoded at 31-34 tok/s on an Apple M5 Max 128 GB, what changed, and what was
measured. It is a single-host engineering record, **not a speedup claim and not a
certification run**: no comparison runtime is involved, and the host is a campaign
host, not the Ultra-class target SKU.

## Finding

The pack's 53.6 GiB PLE n-gram table is never loaded; each token reads 16 of its rows
(2 orders x 8 heads) with positional reads, and each row is three reads: packed
weight (160 B), scales (10 B) and biases (10 B), which sit in three different file
regions. The table is far larger than free memory next to the ~60 GiB of resident
weights, so a token the page cache has not seen costs 48 cold random SSD reads. They
ran one after another, followed by one dequantize and one blocking device round trip
per shard. Measured on the M5 Max:

| 48 cold positional reads (one decode token) | median |
| --- | ---: |
| sequential (before) | 3.30 ms |
| 16 concurrent readers | 1.10 ms |

| 32,768 cold rows = 98,304 reads (one 2,048-token prefill chunk) | time |
| --- | ---: |
| sequential | 6.7 s |
| 4 / 8 / 16 / 32 / 64 concurrent readers | 2.10 / 1.33 / 1.16 / 1.33 / 1.43 s |

The benchmark suites repeat their prompts, so their n-gram rows sit in the page cache;
that is why earlier prefill rates (640-775 tok/s) were higher than what a first-seen
prompt got. A stage profile of a 2,048-token chunk shows the same gap: the
synchronized stage total is 1.9 s while the unsynchronized control takes 6.7-8.8 s.

A second finding came from counting MLX ops per stage (new `ops` field in the
synchronized operator profile). A singleton decode step is 9,923 op wrappers: gated
residual read and write 3,840 (39%), gated delta network 3,168 (32%), MoE 1,776
(18%), QSA 1,032 (10%). About a third of the generation thread's time is the wait for
the device to finish the last command buffers, and MLX's command-buffer knobs
(`MLX_MAX_OPS_PER_BUFFER` 25-500, `MLX_METAL_FAST_SYNCH`, `MLX_MAX_MB_PER_BUFFER`)
moved nothing beyond +-1%.

## What changed

| Change | Where | Kill switch |
| --- | --- | --- |
| Read every row of a gather concurrently (16 scoped threads) into caller-ordered planes and dequantize them once, lazily, instead of per shard with a device round trip each. Equal to the per-shard path element for element; that path stays as the fallback for tables whose shards differ | `ngram_table.rs`, `mlx-sys` `io_rows.rs` | none (non-uniform tables use the old path) |
| Start the next prefill chunk's n-gram reads on a background thread while this chunk computes (row IDs depend only on token IDs); a wrong guess is discarded and read synchronously | `ngram_table.rs`, `model/qwen4_exp.rs`, `runner/mod.rs` | `AX_FLASH_NEXT_NGRAM_PREFETCH=0` |
| Read with 32 threads (omlx measured 98.8k random-read IOPS at 32 threads against 9k at queue depth 1) and, after the weights have loaded, pre-read the 6.4 GB of quantization scales and biases planes into the page cache on a background thread so a first-seen row needs about one cold read instead of three (the MTPLX approach; done before the weight load it is evicted and gains nothing) | `ngram_table.rs`, `mlx-sys` `io_rows.rs`, `runner/mod.rs` | `AX_FLASH_NEXT_NGRAM_WARM_AUX=0`; `AX_FLASH_NEXT_NGRAM_READ_THREADS=<n>` |
| Compile 13 pure elementwise stretches of the decode step (`hc_norm`, `hc_low`, `hc_mix`, `hc_gate`, `hc_write`, `gdn_beta`, `gdn_post`, `gdn_conv_act`, `gdn_decay_pre`, `gdn_decay_post`, `gdn_l2_sum`, `gdn_q_scale`, `gdn_k_scale`) into one fused call each. Projections, the recurrence, and the two ops MLX cannot reproduce inside a fused kernel (`log1p`, `power`) stay outside; only 1-row inputs fuse | `model/shared/qwen4_exp_fused.rs`, `qwen4_exp_residual.rs`, `qwen4_exp_gdn.rs` | `AX_FLASH_NEXT_FUSE=0` (a comma list selects stretches; `AX_FLASH_NEXT_FUSE_ROWS` raises the row limit to at most 8) |

The registry behind the third change tombstones a stretch on any failed or panicking
build, failed apply or failed first evaluation and runs the eager ops from then on,
rebuilds a closure after `AX_MLX_COMPILE_CACHE_REFRESH_THRESHOLD` applies, never holds
a borrow across an MLX call, and treats `MLX_DISABLE_COMPILE` as compilation being
unavailable. Compilation happens on the generation thread during the load-time
production warm-up, so the first request pays nothing extra.

## Numerics

Fused and eager outputs are compared **bit for bit**, not to a tolerance, because the
exact MTP profile and the verifier windows compare decode rows with singleton rows:

- on the real MXFP4 pack, logits and serialized state are identical for verifier
  windows of 1, 2, 3, 4 and 8 rows under both `Shared` and `RowExact`, and across a
  16-step singleton chain, on an Apple M5 Max 128 GB and an Apple M2 Ultra 192 GB;
  per-stretch counters prove every one of the 13 stretches ran (over the chain 1,552
  norm/low/mix, 1,536 gate/write, 576 for each GDN stretch except 1,152 for the sum of
  squares, which q and k both use);
- unit tests at synthetic and real geometry (f32, bf16, f16; 1 to 8 rows) compare raw
  bits, walk dtypes without clearing the registry, inject a failure into each stretch,
  check that 9- and 16-row inputs never fuse, and check closures are released at
  thread exit;
- server text is identical with and without fusion on eight prompts (one of 8,976
  tokens), under canonical MTP (equal to direct decode) and the opt-in batched MTP
  verifier, after an image prefill, with streamed experts, after a request cancelled
  mid-prefill, and at a refresh threshold of 200; 30 back-to-back 256-token requests
  stay flat in latency and resident memory.

The decay chain and the q/k L2 normalizations could not be compiled whole: a bisect on
the real pack showed the recurrent-state bits change. A probe over the chain prefixes
(40 random cases, M2 Ultra and M5 Max) put the first mismatch at `log1p` after any
preceding op and at `power(x, -0.5)`; the surrounding add, minimum, exp, multiply and
sum agree. The cause is in how MLX builds its fused kernels: Metal's `mathMode` and
`mathFloatingPointFunctions` are separate compile options, MLX sets only the math mode
to safe for JIT libraries, so FP32 math functions written without `precise::` (`pow`,
and the `log` inside MLX's `log1p`) stay on the fast variants, while the precompiled
kernels are built with `-fno-fast-math`. Ops spelled `precise::` (exp, log, rsqrt,
sigmoid) match. MLX documents compiled results only as equal "up to numerical
precision", so bit-exactness is our own gate, not an MLX guarantee. The fix that kept
the gate is to split each chain at those two ops, leave only them eager, and compile
the segments between them: every segment is bit-identical in isolation (probed in
bf16, f32 and f16), the three-call decay equals the original 14-op chain, and the
stage tests repeat the comparison on adversarial inputs (NaN, inf, subnormals, exp
thresholds, cancelling projection and bias, all-zero, tiny and square-overflowing rows)
and on the real non-contiguous q/k slices. Other engines either keep these chains eager
or compute them inside one hand-written kernel with explicit `precise::` calls and a
bitwise self-check at load (MTPLX, omlx, TensorFold); that single-dispatch GDN step is
the remaining larger candidate and needs its own record.

## Results (Apple M5 Max 128 GB)

Pack on the internal SSD, one request at a time, MTP off, `--prefill-chunk 2048`,
page cache purged before each arm, first-seen code prompts, greedy 128 tokens. Generated
text SHA-256 is identical in all three arms for every prompt.

| Prompt tokens | TTFT before | TTFT n-gram reads + read-ahead | TTFT + fused | Decode before | Decode n-gram reads | Decode + fused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,773 | 3.79 s | 1.57 s | 1.57 s | 33.6 | 39.3 | 42.4 |
| 1,905 | 4.39 s | 1.62 s | 1.61 s | 33.1 | 39.3 | 42.4 |
| 2,207 | 5.45 s | 2.49 s | 2.49 s | 31.2 | 37.1 | 40.1 |
| 2,560 | 7.29 s | 2.87 s | 2.87 s | 31.6 | 36.9 | 40.0 |
| 8,976 | 28.90 s | 13.95 s | 13.96 s | 30.4 | 35.3 | 37.9 |

A 39,225-token first-seen prompt (48 generated tokens, `--total-blocks 4096`): TTFT
123.0 s before and 76.4 s after both changes (prefill 319 to 513 tok/s), decode 26.9
to 31.6 tok/s, with identical generated text (SHA-256 of the 48 tokens).

The "fused" columns were measured with the first seven stretches; the six GDN segment
stretches added afterwards are +0.8% decode on top (same text), and the reader-thread and
aux-plane changes are in the Limits section.

Decode is tok/s after the first token. The read-ahead alone is 8-12% of TTFT on
multi-chunk prompts; the rest of the TTFT gain is the concurrent reads. The fused
stretches add 8-9% to decode (37.5-40.3 to 40.9-44.2 tok/s on eight prompts with
identical text; the last six stretches are 0.8% of that). A 16-step singleton chain with
logits evaluated at every step takes 403 ms eager and 363 ms fused on the M5 Max, 630 ms
and 546 ms on the M2 Ultra, under both `Shared` and `RowExact`.

## Limits

- One host per number; the M2 Ultra run is a bit-exactness check and a chain timing, not
  a TTFT or server record. The Ultra-class target's qualification evidence is unchanged.
- The first prefill chunk's rows cannot be read ahead (about 0.4-1.1 s at 2,048 tokens).
  Interleaving the three planes of each row in the pack would cut the reads per row
  from three to one; it needs a converter change and a re-converted pack, and the
  remaining n-gram cost is under 3% of decode, so it is not done.
- Prefill chunks, the opt-in batched MTP verifier's windows (unless
  `AX_FLASH_NEXT_FUSE_ROWS` is raised) and every model family other than `qwen4_exp`
  are unchanged.
- MLX 0.32 multiplies float32 on Neural Accelerator GPUs (M5 and later) in TF32, about
  1e-5 relative error (MLX issues #3534 and #3235 describe the same behaviour), and
  selects a different gated-delta kernel there. That, not the code in this note, made 18
  `ax-engine-mlx` unit tests fail on the M5 Max while they pass on the M2 Ultra. The cargo
  test environment now sets `MLX_ENABLE_TF32=0` and `GATED_DELTA_CHUNK=8` (the
  architecture-independent route), the pins MLX's own CI, mlx-vlm and omlx use;
  the lib suite is 1803 passed, 0 failed on both hosts. A test binary run directly
  rather than through cargo needs the same two variables.
- The reader-thread and aux-plane changes above were measured after the table: with
  the prefix cache off, first-seen prompts of 1,773-2,560 tokens got 2-6% lower TTFT,
  decode unchanged, text identical.
