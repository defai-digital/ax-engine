# Flash Next shared-step timing on M5 Max

The [sealed natural-arrival crossover](../flash-next-natural-short-drain-m5-20261123/README.md)
supplies four 8,192-token and 512-token prompt pairs and their exact
target steps. The replayer regenerates the fixtures from that archive.
The signed executable is the same as the crossover and the
[M2 stepwise control](../flash-next-shared-step-clock-m2-20261123/README.md).
MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, MLX 0.32.4.
The diagnostic uses resident experts, two slots, and MTP disabled.

All four target steps reproduce the archived long-prefill history,
physical width, logical boundary and short output depth. The 41
short output token IDs match both the archived SSE trace and the
independent same-host solo requests. Both requests drain after
cancellation. All 140 sampled memory pressure observations are
normal. CPU idle is 95.90% before and 94.28% after.

| Long-prefill width | Runner time | Long prefill forward | Engine time outside runner |
| ---: | ---: | ---: | ---: |
| 511 | 720.83 ms | 693.69 ms | 34 µs |
| 511 | 706.86 ms | 681.64 ms | 35 µs |
| 512 | 713.36 ms | 684.44 ms | 36 µs |
| 512 | 703.93 ms | 683.29 ms | 40 µs |

The same-step clock and retained prefill counter deltas place a
median 96.3% of runner time inside the long prefill forward.
The first three targets compare adjacent steps with identical
participants. The fourth uses the sum of the long bootstrap and
short-only prefill counters as the predecessor baseline. This
accounts for the long request's prefill when it enters the
combined counter total.

These are explicit stepwise API measurements with controlled
admission. The original asynchronous SSE gaps keep their own
clock and workload scope. The single-slot resident profile
remains the supported standard.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
