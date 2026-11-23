# Flash Next guarded two-slot natural arrivals on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, and the signed
runtime `8cb06414c` from the
[paired 8K admission control](../flash-next-native-mixed-admission-m5-20261122/README.md).
The run uses the pinned resident MXFP4 pack with MTP off. Each arm
offers the same 220 arrivals at 0.25 requests/s, with 132 prompts at
512 tokens, 66 at 2,048, and 22 at 8,192. The client has two dispatch
lanes and each request generates 96 greedy tokens.

The physical and held arms use the same guarded two-slot admission
and physical prefill boundaries. The held arms may delay the first
long prefill while a 512-token peer emits its last 12 output tokens.
Replay binds the original request, arrival, and solo-output oracles.
Actual hold markers are checked against the scheduled short peer and
the long request's first prefill step.

| Run order | Arm | CPU idle before / after | Timing eligible |
| ---: | --- | ---: | --- |
| 1 | physical-1 | 96.80% / 84.31% | No |
| 2 | held-1 | 96.11% / 85.10% | No |
| 3 | held-2 | 93.19% / 88.44% | No |
| 4 | physical-2 | 91.73% / 92.66% | Yes |

All 84,480 greedy output token IDs match
the one-slot oracle and all five work counters return to zero in
every arm. All 4,181 sampled memory-pressure
observations are normal.

| Pair | Observed late long arrivals in control | Actual held long arrivals | Timing eligible |
| --- | ---: | ---: | --- |
| 1 | 2 | 2 | No |
| 2 | 1 | 1 | No |

Both pairs fall below the 90% CPU idle requirement in at least one arm.
They supply functional and exposure evidence.

This experiment uses a guarded two-slot profile. The standard Flash
Next product profile remains single-request resident.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
