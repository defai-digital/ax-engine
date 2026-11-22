# Flash Next layered prefill over native HTTP on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, MLX 0.32.4 and Rust
1.99.1. The experiment built the isolated native runner at
`d4e69e64f` and used one release-server binary across the arms.
It verified all 49 publisher members and the delivered MXFP4 manifest
before and after execution. The server used two slots, resident
experts, greedy direct decoding, no prefix reuse and the existing
physical prefill split. The candidate additionally set
`AX_MLX_FLASH_NEXT_PREFILL_LAYER_YIELD=1` for groups of eight layers.

The inputs are the two frozen 512/8,192-token
[HTTP bursts](../flash-next-mxfp4-prefill-step-cost-m5-20261121/README.md).
A completed arm emits 192 output IDs per arrival case. Each is checked
against the [independent single-slot outputs](../flash-next-mxfp4-mixed-serving-m5-20261121/README.md).
The early arrival overlaps prefill and decode. Across both candidate
arms the long stream reports 77 suspended steps per early burst with
zero prompt-cursor advance. Each reported step publishes a peer decode
token. Both modes retain the same 20 early-burst physical widths.
The late request arrives after the short stream has finished in both
modes of the first pair.

The first pair meets the protocol's 90% idle requirement before and
after each arm. Results describe the early-overlap burst.

| Mode | Short p95 token gap | Long arrival to first token | Total throughput |
| --- | ---: | ---: | ---: |
| Physical forwards | 748.40 ms | 11.24 s | 12.46 tok/s |
| Eight-layer groups | 277.15 ms | 12.83 s | 12.41 tok/s |

In the first pair the short-stream tail gap decreases by 63.0%.
Long first-token latency increases by 14.2%. The long stream
first emits before short completion in the control and after completion
in the candidate. The reverse candidate ends at 79.90% CPU idle and
its elapsed times are diagnostic. The four arms match 1,536 output
IDs across the frozen bursts. The pair exercises a scheduling
tradeoff on this device. Product latency, throughput and two-slot
qualification remain open.

Both candidate disconnect controls close a socket after observing an
actual layer suspension. Closing either the long or short socket lets
the remaining stream finish with all 96 expected IDs. All tracked
service counters drain and sampled pressure stays normal.

The first executable matched the candidate's output. The SDK omitted
the request-local layer cursor from its progress-only frames. The
rebuilt SDK exposes progress on the owning request's
current SSE step and excludes transient layer keys from the stored
request route. The benchmark uses this rebuilt executable.

The [portable raw evidence](evidence.json.gz) retains stream records,
layer steps, pressure samples and the terminal receipt. The client
endpoint and local corpus path are omitted. The [driver sources](driver.tar.xz)
and [input receipt](inputs.sha256) bind the live run. The
[replay](replay.py) validates the preserved widths, token IDs, step
ownership and cleanup and recomputes the client timing metrics.

```bash
python3 benchmarks/results/flash-next-prefill-layer-http-m5-20261122/replay.py
```
