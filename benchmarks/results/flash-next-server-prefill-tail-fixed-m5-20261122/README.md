# Corrected Flash Next server-side tail deferral on M5 Max

The four resident two-slot runs used MacBook Pro M5 Max 128 GB, internal
NVMe, macOS 27.2.1, Rust 1.99.1 and MLX/MLX Metal 0.32.4. All runs used
the same codesign-verified `14235e84b` binary, including the deadline
recheck before KV resolution and the compact stepwise hold marker.
The pinned MXFP4 pack, launch options and measured request functions
match the [frozen M5 diagnostic](../flash-next-server-prefill-tail-m5-20261122/README.md).

Both modes submit a long 8,192-token input immediately after the short
512-token input emits token 84 of 96. The control mixes physical
prefill with short decode. The treatment holds the long input at the
start and subsequently runs its four ordinary 2,048-token prefills
after the short request completes. TTFT for the long request starts
at the trigger, including all time spent waiting on the server.

| Order | Mode | Short tail maximum gap | Long TTFT from trigger | Burst output tokens/s | CPU idle before / after |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Physical split | 785.13 ms | 11.045 s | 12.23 | 92.25% / 92.47% |
| 2 | Tail hold | 22.38 ms | 9.278 s | 14.08 | 92.36% / 92.48% |
| 3 | Tail hold | 20.66 ms | 9.179 s | 14.21 | 92.75% / 98.84% |
| 4 | Physical split | 767.11 ms | 10.635 s | 12.80 | 94.62% / 93.99% |

Both orders pass the 90% CPU idle criterion and all 584 pressure
samples report normal pressure. All 768 generated IDs match the
frozen solo outputs. All five work counters drain after each run.
The measured process footprint peaks at 75.74 GiB.

This result validates one late-arrival workload on the corrected
experimental two-slot scheduler. Promotion requires primary-tier
coverage for earlier arrivals, cancellation and sustained mixed
traffic. The product qualification uses the one-slot resident profile.

The raw archive binds the measured binary, launch inputs and all four
stream traces. The replay recomputes the reported timings, checks each
frozen output ID and evaluates the pair-level timing conditions.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
