# Flash Next resident two-request coalescing comparison on M5 Max

MacBook Pro M5 Max 128 GB, macOS 27.2.1, internal SSD, MLX 0.32.4 and
Rust 1.99.1. The `b36b7b287` release-server binary and the pinned MXFP4
pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` were verified before
and after the run. The two archived 2K prompts used greedy direct
generation, resident experts, MTP and n-gram acceleration disabled,
and the [2,049-token reservation schedule](../flash-next-mxfp4-prefill-partition-m5-20261121/README.md#one-decode-token-and-one-2048-token-prefill).

After a one-slot baseline, two-slot processes ran with coalescing off,
on, on, and off. Each process served both prompts twice at each of
three arrivals: serial, simultaneous and after the first stream began
decoding. Every two-slot reply matched all 96 solo token IDs. The
enabled arms recorded coalesced decode barriers in all eight overlapping
trials. The disabled arms and the serial trials recorded none.

Each two-slot process also completed two disconnect and reuse trials. All eight
trials observed a shared step before disconnect. The survivor and the
replacement matched solo tokens, the disconnected prefix matched, and all
five lifecycle gauges drained after disconnect and after replacement.
Peak sampled process footprint across the arms was 72.293 GiB, with
normal memory pressure in every sample.

Each cell below is the median of two trials in a process. Aggregate
rate divides both completed outputs by the elapsed time for the pair.
Maximum gap is the largest interval between generated-token events
in either client for each trial.

| Process pair | Arrival | Off tok/s | On tok/s | Off max gap | On max gap |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Simultaneous | 26.41 | 24.89 | 1.634 s | 1.889 s |
| 1 | During decode | 26.57 | 24.90 | 1.614 s | 1.882 s |
| 2 | Simultaneous | 25.88 | 25.80 | 1.764 s | 1.785 s |
| 2 | During decode | 25.90 | 25.81 | 1.758 s | 1.773 s |

The two enabled processes measured lower aggregate rates and similar or
larger largest gaps. The first enabled process also ran its serial
trials more slowly with coalescing inactive, so the between-process
timing difference has more than one possible cause. The result supports
the existing opt-in route's correctness on these inputs. A performance
promotion remains open.

The [raw receipt](coalescing-result.json.gz) retains all token arrays,
arrival order, engine events, coalesced route readings, lifecycle
counters and memory samples.
