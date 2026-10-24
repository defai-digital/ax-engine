# Reference-review wave, M5 Max 128 GB — 2026-10-24

Final build of the 2026-10-24 wave (memory-priced placement, Flash Next MTP cost
parking, qmv precision match, keep-warm tick, prefix restore by value and semantic
cuts, adaptive prefill / pressure requeue / cancel keep-prefix, norm contract, tiled
QSA scores, DFlash2 tree verify, batched state materialize, API polish) compared
with the pre-wave baseline (main `318d315d` plus the Rust 1.99.1 repin). Same host,
same packs, MLX 0.32.4. Every new behaviour except priced placement, MTP cost
parking, batched state materialize, norm contract and API polish is default off,
so the default-path rows measure regressions, not opt-in gains.

Harness: `scripts/bench_mlx_inference_stack.py --prompt-source real` (flappy and
long_code MTP suites, 256 generated tokens, 3 repetitions after 1 warm-up,
prefill step 2048, `--no-thinking`); medians. Long-context rows: one
streaming request per prompt after a page-cache purge, `--total-blocks 4096`,
MTP and n-gram acceleration off, 128 generated tokens.

## Decode / prefill, tok/s (median)

| Arm | Baseline decode | Final decode | Baseline prefill | Final prefill |
|---|---:|---:|---:|---:|
| 27B MXFP4 direct, flappy | 35.3 | 35.3 | 705.8 | 704.0 |
| 27B MXFP4 MTP, flappy | 96.9 | 96.8 | 738.4 | 741.5 |
| 27B MXFP4 DFlash2, flappy | 127.2 | 127.3 | 740.2 | 741.7 |
| 27B MXFP4 direct, long_code | 35.2 | 35.1 | 878.2 | 880.4 |
| 27B MXFP4 MTP, long_code | 89.4 | 89.5 | 888.5 | 887.3 |
| 27B MXFP4 DFlash2, long_code | 124.6 | 124.6 | 888.3 | 889.1 |
| Flash Next MXFP4 direct, flappy | 51.6 | 51.7 | 762.2 | 760.1 |
| Flash Next MXFP4 direct, long_code | 51.2 | 51.2 | 1180.5 | 1199.1 |
| Flash Next MXFP4 MTP (required), flappy | 47.4 | 47.5 | 720.5 | 723.5 |
| Flash Next 6-bit direct, flappy | not runnable (1) | 45.6 | — | 676.3 |
| Flash Next 6-bit direct, long_code | not runnable (1) | 45.2 | — | 1034.3 |

(1) Under the baseline Auto placement (fixed 48 GiB headroom) the 6-bit pack
streams experts on a 128 GB host (about 0.17 tok/s); the priced plan keeps it
resident.

## Long context, Flash Next

| Pack, prompt tokens | Baseline TTFT | Final TTFT | Final prefill | Final decode |
|---|---:|---:|---:|---:|
| MXFP4, 2,207 | 2.38 s | 2.38 s (2) | 926 | 47.2 |
| MXFP4, 8,976 | 13.67 s | 13.69 s | 656 | 45.8 |
| MXFP4, 39,225 | 76.16 s | 75.96 s | 516 | 44.7 |
| 6-bit, 2,207 | — | 2.97 s | 742 | 42.3 |
| 6-bit, 8,976 | — | 28.85 s | 311 | 41.1 |
| 6-bit, 39,225 | — | 163.3 s | 240 | 26.3 |

(2) The single final-run sample was 2.89 s; three alternating baseline/final
re-runs gave 2.366-2.387 s vs 2.376-2.389 s. Output text hashes match the
baseline for all three MXFP4 prompts.

6-bit long context: the priced plan admits a 114.5 GiB working set (ceiling
115.2 GiB, above the Metal recommendation of 107.5 GiB). A rerun of the 39k
prompt showed no new swap-outs, so the slowdown is not swapping; the most
likely cause is the smaller page cache left for n-gram row reads. Not yet
proven; MXFP4 stays the long-context recommendation.

## Opt-in features measured on this host

- Tiled QSA scores (`AX_MLX_FLASH_NEXT_QSA_TILED_SCORES=1`, MXFP4, one run per
  setting): at the default chunk TTFT is unchanged; with outer chunk 2048/4096/8192
  TTFT is 13.57 -> 13.16-13.21 s at 8,976 tokens and 75.3 -> 73.3 s at 39,225
  tokens (about 2.6%). Text identical in all settings. The 8,192-token real-pack
  test matches QSA indices and final-logit bits with tiling off.
- DFlash2 tree verify (`AX_MLX_TREE_VERIFY=1`, 27B): output identical to linear
  DFlash2, 2-6% slower on four prompts. Stays off.
- Keep-warm tick (`AX_MLX_IDLE_TICK_MS=1000`, 27B, cached 2.2k prompt): TTFT after
  10 s / 60 s idle 0.547 / 0.550 s off vs 0.436 / 0.432 s on (hot 0.413 s).
- Flash Next MTP: `required` and `auto` emit the same tokens as direct on four
  prompts; `required` stays about 8% slower than direct, `auto` matches direct.
