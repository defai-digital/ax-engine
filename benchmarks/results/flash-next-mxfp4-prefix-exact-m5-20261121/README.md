# Flash Next exact prompt reuse on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, MLX and MLX
Metal 0.32.4. Both processes used the same Rust 1.99.1 release-server
executable from `cdc7ce2e9` and the resident MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. MTP, n-gram
acceleration, disk caching, direct coalescing and experimental
physical prefill splitting were disabled. All 49 published pack
members were checked before and after both processes.

The cold process allowed one request and a 2,048-token step budget.
The L1 process allowed two requests, a 2,049-token step budget and an
eight-entry, 2 GiB cache. It seeded two disjoint 2,048-token prompts.
The exact replay was the only active request when it ran.

| Control | Observed prefix route | Output identity versus cold |
| --- | --- | ---: |
| Exact 2,048-token replay | In-memory L1, 2,048 tokens restored | 96/96 |
| 2,112-token extension beside a decoding peer | Full prefill | 96/96 |
| 2,112-token extension that disconnects after output | Full prefill | 1/1 emitted |
| 2,112-token replacement with a distinct suffix | Full prefill | 96/96 |
| Two independent 512-token peers | No prefix reuse | 384/384 |

The exact replay reached its first token in 0.037 seconds with L1
and 3.858 seconds in the cold control. Time to the last response was
2.119 and 5.978 seconds, respectively. These are single observations
across two process configurations. The two overlapping extensions
each shared two prefill steps with their peer's decode. The observed
overlap used cold prefill for the extension. All seven drain checks
reached zero and health passed after replacement. Pressure was normal
on all 146 samples. Sampled footprint peaked at 72.01 GiB cold and
72.91 GiB with L1.

The exact replay's cache event carries its request ID and was retained
on its later decode report and final response. The overlap checks use
step IDs from both requests. The scheduler scalars on the terminal
route carry values from its last step, so they do not count earlier
mixed steps. The measured benefit applies to repeated complete
prompts with L1 enabled. The one-request, 32K, MTP-off product
envelope is the basis for release qualification.

## Evidence and replay

The [request receipt](receipt.json.gz) retains the original prompt and
output IDs, per-request step records and cache events. The [input
checksums](source.json) bind the receipt, memory samples, driver and
frozen inputs. Offline replay independently recomputes output identity,
cache boundaries, peer overlap, drain checks and memory peaks.

```sh
python3 benchmarks/results/flash-next-mxfp4-prefix-exact-m5-20261121/replay.py
```

For a device rerun, extract the named frozen inputs from the [mixed
serving archive](../flash-next-mxfp4-mixed-serving-m5-20261121/evidence.tar.xz)
into an isolated directory and copy in [the exact driver](run_warm_prefix.py).
Run the driver with the pinned pack, executable, build receipt,
publisher inventory, corpus and its SHA-256, and a fresh output
directory. Provide the shared-library search path for MLX 0.32.4.
