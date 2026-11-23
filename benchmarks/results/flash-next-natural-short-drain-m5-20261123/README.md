# Flash Next natural-arrival priority short decode drain on M5 Max

MacBook Pro M5 Max 128 GiB, internal SSD, macOS 27.2.1, MLX and
MLX Metal 0.32.4, and Rust and Cargo 1.99.1. The four processes use
the same signed experimental server `71f891b25` and MXFP4 pack
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. Both configurations use
two resident slots, mapped weights, short-first prefill, and MTP off.
The decode-drain switch is the sole configuration change.

The process order is mixed, drain, drain, mixed. Each process has
two warmups and serves the same 220 scheduled arrivals at 0.25
requests per second, with 96 greedy output tokens per request.
Offered latency begins at the scheduled arrival. Dispatch latency
begins when a client lane starts the request.

| Order | Policy | Endpoint CPU idle before / after | Client-window CPU idle | Output tok/s | 512 offered TTFT p95 | 2,048 dispatch TTFT p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| First | Mixed | 97.15% / 96.59% | 88.08% | 22.84 | 45.58 s | 3.03 s |
| First | Drain | 93.34% / 93.87% | 91.81% | 23.79 | 12.53 s | 13.03 s |
| Reverse | Mixed | 94.20% / 92.18% | 90.80% | 23.07 | 33.53 s | 3.05 s |
| Reverse | Drain | 97.70% / 91.39% | 79.63% | 21.68 | 96.42 s | 14.26 s |

All four arms meet the predeclared endpoint CPU criterion of at
least 90% at each check. Passive Darwin ticks cover each complete
client run and show unequal CPU conditions during service. Relative
throughput and offered TTFT change direction between orders. The
run supplies no repeatable whole-workload speedup.

Matching on short prompt, long peer and long cursor at the priority
turn gives 14 first-order and 15 reverse-order pairs. Every matched
short request has a smaller maximum output gap with drain. The median
of those maxima moves from 915.93 to 23.81 ms in the first order and
from 824.68 to 25.07 ms in the reverse order. All 33 observed drain
holds prevent long prefill during the short decode.

The same 29 matched cases also have a 2,048-token successor. In
both policies and orders, it starts prefill on the next engine step
after the original 8,192-token peer's final decode. The [M2 waiting
state control](../flash-next-medium-handoff-m2-20261123/README.md)
directly verifies zero prefill during that wait. Offered and dispatch
TTFT capture different portions of this queue. The matched median
offered TTFT is 34.60 to 14.99 s in the first order and 26.76 to
91.87 s in the reverse order.

All 84,480 output IDs match the frozen solo oracle. All five
native counters drain in all arms. All 4,107 memory pressure samples
are normal and peak process footprint is 75.85 GiB. The two-slot
policy remains experimental. A matched one-slot comparison and
comparable client-window conditions remain necessary to evaluate
a supported concurrency change.

The replay binds the earlier [targeted crossover](../flash-next-priority-short-drain-m5-20261123/README.md),
pack and binary receipts, per-request IDs and engine traces, and
the original CPU samples. The archive enumerates the two redacted
client metadata fields and retains the original capture hashes.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
