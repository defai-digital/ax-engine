# Qwen MXFP4 verifier and GDN replay controls on MLX

One ad hoc-signed `release-server` build ran on MLX 0.32.4. These are short,
fixed-output native HTTP serving controls. Each process warms two requests
before measuring. Greedy requests use the same token IDs in each paired
arm, with prefix reuse disabled. The direct and Flash Next arms suppress
independent n-gram speculation. The 27B DFlash2 arms keep the engine's
normal speculation admission and require packed DFlash2 route evidence. Host
CPU idle is sampled at entry and exit. The 90% endpoints mark timing
eligibility. Pressure samples are host-wide. Throughput includes prefill
and streaming. The benchmark corpus is small, so tail percentiles are
diagnostic.

## Single-request direct controls

These M5 Max 128 GB runs use the same prompt IDs and output budget as
the respective speculative arms. The 27B corpus has seven 96-token
requests and one 32-token request. Flash Next has three 64-token
requests. One request is active at a time. Each model uses its own
workload, so the elapsed-time measurements are model-specific.

| Model | Output tok/s | First token p95 | Stream gap p95 | CPU idle before / after |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.8 27B MXFP4 | 32.64 | 314.3 ms | 29.9 ms | 97.79% / 99.10% |
| Flash Next MXFP4 | 36.24 | 712.5 ms | 20.3 ms | 98.65% / 98.54% |

## Flash Next experimental required MTP on M5 Max 128 GB

The pinned resident MXFP4 pack was on the internal SSD. Each arm ran one
active request. The fixed corpus contains three code, math and chat
prompts with 64 output tokens each. The block arms used the legacy block
schedule. `AX_FLASH_NEXT_MTP_BLOCK_DRAFTS=2` caps that schedule at two
drafts. The canonical arm uses its ordinary one-draft verifier.

| Order | Arm | Output tok/s | First token p95 | Stream gap p95 | CPU idle before / after |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Three drafts | 28.64 | 725.9 ms | 121.7 ms | 81.15% / 89.27% |
| 2 | Two drafts | 29.43 | 724.0 ms | 92.1 ms | 90.25% / 90.17% |
| 3 | Two drafts | 29.67 | 719.4 ms | 90.6 ms | 89.80% / 92.69% |
| 4 | Three drafts | 29.23 | 719.4 ms | 119.8 ms | 93.25% / 93.12% |
| 5 | Canonical | 33.35 | 717.8 ms | 44.5 ms | 94.51% / 94.71% |

Both cap orders remove all third-draft proposals. The block arms emit
identical tokens to each other for every prompt, and each matches two of
three canonical outputs. The two-draft arm records smaller stream gaps
and slightly higher throughput in both orders. Each pair has a CPU
endpoint under the timing floor. Canonical has the highest measured
throughput and the smallest gaps on these prompts. Pressure stays normal
throughout. The product default remains resident direct decode with one
active request.

## Qwen 3.8 27B DFlash2

The M5 Max 128 GB ran the pinned target and admitted DFlash2 drafter
from network-mounted storage. The serving durations start after model
load and two warmup requests. Eight distinct prompts were offered with a
four-request concurrency cap. Each treatment starts a fresh server.
The incumbent uses its existing cohort policy. The cap sets
`AX_MLX_DFLASH2_COHORT_MAX_ROWS=4` and counts the anchor in those rows.

| Order | Arm | Output tok/s | First token p95 | Stream gap p95 | Packed rows / lanes | CPU idle before / after |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Incumbent | 85.07 | 941.1 ms | 186.3 ms | 117 / 26 | 99.26% / 97.48% |
| 2 | Four-row cap | 85.00 | 941.6 ms | 159.0 ms | 96 / 24 | 99.11% / 94.29% |
| 3 | Four-row cap | 82.15 | 942.6 ms | 140.4 ms | 96 / 24 | 97.71% / 97.61% |
| 4 | Incumbent | 88.05 | 944.9 ms | 145.7 ms | 117 / 26 | 98.69% / 98.18% |

The cap lowers p95 stream gaps in both orders. Aggregate throughput falls
by 0.07% and 6.70% respectively. Five and four of eight outputs agree
with the paired incumbent, and the two incumbent processes agree on
five of eight. These observations include route-dependent scheduling
and numerics. The evidence supports retaining the existing default.

On the M4 Pro 64 GB supported-tier check, the same eight-request
completed workload measured 28.58 tok/s and 392.3 ms p95 stream gap
with the incumbent. Four rows measured 26.43 tok/s and 417.3 ms, with
four outputs matching. Both runs satisfy the CPU endpoints and report
normal pressure. This is a supported-tier comparison.

### GDN partial replay

A separate trace pins four rows and uses projected replay on M5. Of
338 traced rounds, 111 materialized the replayed convolution and
recurrent state. Median graph construction took 0.313 ms. Median
evaluation took 1.730 ms, with a 9.968 ms maximum and 1.470 ms
amortized over all traced rounds. Evaluation can wait for dependent GPU
work. The trace retains all rounds emitted by the server. Five of eight
outputs match the separately launched trace control with materialization
disabled. Both processes use the same four-row pin and projected-replay
settings. These observations bound the recorded replay cost. The runtime
retains the existing pre-draft path.

## Reproduction

[Build receipt](build.json) identifies the exact candidate source,
binary and MLX library. [Capture driver](capture.py) uses
`scripts/bench_ax_serving.py` from the recorded build. Place both
scripts together, then supply the pinned local pack, benchmark corpus
and output directory. Use the checked-in
[27B corpus](../../corpora/serving/qwen38_dflash2_paired.jsonl) with the
admitted drafter, or the [Flash Next corpus](flash-prompts.jsonl).
The driver checks that its DFlash2 arms exercised packed verification.
Flash Next arms use a fixed 2,048-block pool.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
