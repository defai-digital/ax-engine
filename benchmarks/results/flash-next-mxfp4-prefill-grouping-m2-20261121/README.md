# Flash Next grouped prefill on M2 Ultra

A Mac Studio with M2 Ultra, 192 GB of unified memory, an external SSD,
macOS 27.2.1, MLX 0.32.4, and Rust 1.99.1 ran the pinned resident MXFP4
pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. The isolated
`release-server` executable came from `2cfc3eccb` on top of `fe96b4803`.
The run verified the pack before and after all three arms. All 821
memory-pressure samples were normal, with unchanged swap usage across
the captured samples.

The replay used two 8,192-token prompts and swapped which request began
decoding before the other arrived. Two serial requests set the baseline.
All 12 paired replies reproduced their baseline sequences of 96 tokens.
Both split arms verified the cursor and the request-specific route for
every piece of all four logical 2,048-token items. The grouped arm also
canceled an active prefill at cursor 1,024 during a shared engine step.
The survivor and the replacement request reproduced their complete
baseline sequences and all runner drain counters returned to zero.

| Scheduler turns for an overlapping 2,048-token item | Median p95 gap for the first decoder | Largest first-decoder gap | Gaps of at least 100 ms per trial |
| --- | ---: | ---: | ---: |
| 2,048 | 73 ms | 4.996 s | 4 |
| 512 / 512 / 512 / 511 / 1 | 1.331 s | 1.382 s | 17 |
| 1,024 / 1,023 / 1 | 2.617 s | 2.686 s | 9 |

The grouped turns retained the five physical prompt forwards tested by
the isolated F32 and BF16 fixtures. Grouping reduced the number of long
gaps relative to the smaller split and increased their duration. Every
gap of at least 100 ms occurred next to peer prefill. This run provides
functional evidence on the best-effort M2 tier. The existing M5
single-request default remains the supported profile. An M5 decision
on overlapping prefill requires measured per-user p95 inter-token gaps
and first-token latency under mixed load.

Run `python3 benchmarks/results/flash-next-mxfp4-prefill-grouping-m2-20261121/replay.py`
from the repository root to reproduce [comparison.json](comparison.json)
from the [raw streams](result.json.gz). The replay checks the
[build identity](build.json), [source hashes](live-inputs.sha256),
[isolated patch](grouped-prefill.patch.gz), and [exact driver sources](driver-sources.tar.gz).
It uses the [same prompts](../flash-next-mxfp4-prefill-partition-m5-20261121/holdout-8192-prompts.jsonl.gz)
as the [M5 two-user replay](../flash-next-mxfp4-prefill-partition-m5-20261121/README.md).
