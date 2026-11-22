# Flash Next selective prefill on M5 Max

MacBook Pro M5 Max, 128 GB unified memory, internal SSD, macOS 27.2.1.
The Rust 1.99.1 release-server experiment at
`70b6bb6da1efbf906a97fb58a4c2d0f070e2fa05` ran MLX 0.32.4 and the
pinned MXFP4 checkpoint `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
All arms used resident experts, two slots, MTP disabled, direct
coalescing disabled, and prefix reuse disabled.

Each of the two launch orders offered the same 220 requests at a mean
0.25 requests per second per arm. The corpus contained 132 prompts of
512 tokens, 66 of 2,048 and 22 of 8,192. Every request emitted 96 tokens
and matched the full solo output array. All 1,320 arrays passed in total.
Every arm drained its serving counters. All 5,947 memory samples showed
normal pressure. The largest sampled physical footprint was 75.96 GiB.

The split arms keep the same logical item and physical forward cuts.
The selective arm starts split items only for prompts of at least
8,192 tokens. Its raw steps confirm actual shared splits at that length
and no split below it in either order.

Short-stream p95 is the p95 across the 132 short requests' individual
p95 gaps. Arrival-to-first-token time includes waiting for a client lane.

| Arm | Shared split requests | Gaps ≥100 ms | Worst gap | Short-stream p95 | Arrival-to-first-token p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off, first | 0 | 200 | 3.604 s | 52.83 ms | 37.356 s |
| All, first | 57 | 411 | 0.936 s | 63.78 ms | 44.152 s |
| 8K only, first | 8 | 247 | 2.602 s | 404.20 ms | 42.708 s |
| 8K only, reverse | 1 | 123 | 2.607 s | 52.01 ms | 90.137 s |
| All, reverse | 51 | 363 | 0.908 s | 68.09 ms | 103.700 s |
| Off, reverse | 0 | 279 | 3.650 s | 55.30 ms | 55.303 s |

All-split reduced the worst gap and increased both the number of
100 ms gaps and the short-stream p95 in both orders. Selective
splitting also reduced the worst gap and increased arrival-to-first-token
p95 relative to off in both orders. Its short-stream p95 regressed in the
first order. The reverse arm saw one shared 8K split where the first saw
eight. Both launch orders show a latency tradeoff, so split policy
promotion stays closed. The experiment records two-slot functional and
performance evidence within this workload. The product profile is
single-stream resident with MTP off.

In the first selective arm, eight short streams had per-stream p95
gaps of at least 100 ms. All 117 long intervals in those streams ended
on a two-request step that advanced one 8K prefill slice and emitted one
short-stream token. The reverse selective arm saw 14 such intervals in
one stream. This associates the tail with actual shared work. Native
step timing is needed to measure its execution cost.

## Replay

The [archive](evidence.tar.xz) retains the six original terminal controls,
token requests, per-step traces, memory samples and both same-binary
physical holdouts. The solo and measured client files omit their loopback
URLs and absolute corpus paths. [Source hashes](sources.json) retain the
original client digests and every archived file digest. Offline replay
checks activation and exact outputs before regenerating
[the terminal comparison](comparison.json) byte for byte.

```sh
python3 benchmarks/results/flash-next-mxfp4-selective-prefill-m5-20261121/replay.py
```
