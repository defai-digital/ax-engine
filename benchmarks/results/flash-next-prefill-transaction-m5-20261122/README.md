# Flash Next suspended prefill forward on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, MLX 0.32.4 and Rust
1.99.1. The release-server test repeats the [prior fixed-width M5 probe](../flash-next-prefill-layer-peer-m5-20261122/README.md)
with the same delivered MXFP4 manifest, 49 verified publisher members,
8,192 long prompt tokens and 69 peer prompt tokens.

The model returns an evaluated, request-owned forward to its caller
after each group of eight layers. The caller executes one singleton
decode for the peer before resuming. The long request keeps all twenty
physical prefill widths. The test exercises ten suspensions across the
512-token cold and 511-token deep forwards. Both requests' logits and
serialized states match their solo controls bit for bit. The two state
hashes and all eleven peer outputs also match the previous executable.

An additional cold control suspends after eight layers, attempts an
advance with the wrong owner and discards the staged graph. The
original cache is byte-identical and the same physical forward retries
successfully. The transaction consumes the staged graph on error and
publishes the committed cache only after the final advance succeeds.

| Physical forward | Range of measured eight-layer advances |
| --- | ---: |
| Offset 0, width 512 | 60.21–64.65 ms |
| Offset 7,680, width 511 | 97.81–104.89 ms |

Each advance includes layer graph construction and its checkpoint
evaluation. Host idle reads 93.28% before and 98.16% after, with normal
sampled pressure throughout. These are model timings. The benchmark does not measure
HTTP token gaps, throughput or request cancellation in the scheduler.
Ordinary forwards use an unbounded layer budget. The server still
needs an explicit progress result and bounded KV reservation before
it can publish peer tokens between advances.

Replay the [result](result.json) against the [execution receipt](receipt.json)
and the earlier fixed-width control.

```bash
python3 benchmarks/results/flash-next-prefill-transaction-m5-20261122/replay.py
```
