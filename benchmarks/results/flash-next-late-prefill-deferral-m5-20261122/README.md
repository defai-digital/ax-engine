# Flash Next late-prefill dispatch on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, MLX and MLX Metal 0.32.4,
Rust 1.99.1. The same experimental release-server binary runs all four
arms with two resident request slots. Drafting, generic n-gram
acceleration, direct coalescing, and layer checkpoints are disabled.
The fixed 512-token and 8,192-token prompts each request exactly 96
greedy output tokens.

The short stream's 84th output token sets the offered arrival time of the
long request in every arm. The control dispatches immediately. The
treatment waits for the short stream to close, then dispatches. The
long-request TTFT includes this entire client wait. The timed burst
starts when the short request is sent and ends when both requests finish.

TensorFold's Flash Next scheduler keeps decode and long-prompt fill
separate, and bounds both the fill work beside a decode round and its
own wait. That policy motivated measuring the cost of overlap during
the last few decode tokens. This diagnostic changes only client
dispatch and retains the engine's existing forward widths. The
[earlier overlap probe](../flash-next-prefill-late-arrival-m5-20261122/README.md)
binds the same model, prompt pair and 84th-token trigger.

| Process order | Largest gap in last 12 short tokens | Client wait before long dispatch | 8K first token from trigger | Burst output rate |
| --- | ---: | ---: | ---: | ---: |
| Immediate 1 | 748.62 ms | 0.04 ms | 10.482 s | 12.883 tok/s |
| Deferred 1 | 21.07 ms | 229.98 ms | 9.182 s | 14.149 tok/s |
| Deferred 2 | 22.77 ms | 231.98 ms | 9.344 s | 13.976 tok/s |
| Immediate 2 | 761.54 ms | 0.04 ms | 10.606 s | 12.825 tok/s |

Every arm matches all 192 generated IDs against the frozen solo oracle,
finishes healthy, and drains all five work counters. The immediate
arms overlap the short stream and use the original mixed-prefill
boundaries. The deferred arms dispatch after the short stream finishes
and use four complete 2,048-token prefill items. All four arms meet the 90% CPU-idle floor before and
after timing. All 528 pressure samples are normal. Peak server footprint
is 75.74 GiB.

The short-tail maximum falls 97.2% and 97.0% in the two orders.
Long-request first-token latency falls 12.4% and 11.9% with
the 230–232 ms client wait included. The two-request burst output
rate rises 9.8% and 9.0%. The result supports bounded scheduler
follow-up for late arrivals. It covers one prompt pair and trigger,
and leaves sustained serving, early arrivals and defaults outside
the measured scope.

The [archive](evidence.tar.xz) binds the receipt, the driver and
the original request traces. The replay recomputes the
[comparison](comparison.json), including trigger-relative latency,
prompt cuts, token identity and two-request burst rate.

```sh
python3 -B benchmarks/results/flash-next-late-prefill-deferral-m5-20261122/replay.py
```
