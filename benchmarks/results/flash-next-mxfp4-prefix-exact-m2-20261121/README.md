# Flash Next exact prompt cache and continuation control on M2 Ultra

The pinned Flash Next MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` ran on a Mac Studio M2
Ultra 192 GB with an external SSD, macOS 27.2.1 and MLX 0.32.4.
The `0f597f13c` release-server build used Rust 1.99.1.
The two server processes compared one cold request slot with two
request slots and a bounded 2 GiB L1 cache. Both kept experts resident. Both used direct greedy
text generation with MTP, generic n-gram acceleration, experimental
physical prefill splitting, and disk prefix caching disabled.
All 49 published pack members were verified before and after the run.

| Request | Observed cache route | Greedy output comparison |
| --- | --- | --- |
| Exact 2,048-token replay | L1 restored 2,048 tokens | 96/96 IDs matched cold |
| 2,112-token extension | Cold prefill | 96/96 IDs matched cold |
| 2,112-token extension disconnected after output | Cold prefill | First emitted ID matched cold |
| 2,112-token replacement | Cold prefill | 96/96 IDs matched cold |

Each extension tested beside an independent client shared two native
prefill steps with that client already decoding. Both independent
clients matched all 192 of their cold output IDs. Seven snapshots of
the lifecycle gauges drained to zero. Health passed after replacement.
Every memory-pressure sample was normal, and sampled physical
footprint peaked at 86.03 GiB in the cold process and 87.54 GiB in
the L1 process.

The single exact replay reached its first token in 0.046 seconds
after L1 restore. Its cold control took 4.526 seconds. These
are two observations on the M2 test device. Ordinary direct Flash
Next permits full-prompt reuse here. The observed extensions kept
the cold path under the existing prefix policy.

## Evidence and replay

[Source and checksums](source.json) bind the server, model metadata,
corpus, drivers and raw result. The
[request receipt](result.json.gz) retains native step IDs and request
route decisions. The two sample logs retain memory and pressure
observations. The [compact corpus](corpus.jsonl) selects the original
two seed inputs and five short inputs and produces the identical
control workload.

Extract [the runners](runners.tar.gz) into an isolated directory.
Build package `ax-engine-server` at the recorded commit with the
`release-server` profile and the pinned toolchain. Provide the
matching [publisher inventory](../flash-next-native-descriptor-reader-m2-20261121/publisher-inventory.json)
and the admitted MXFP4 model manifest. Run `run_warm_prefix.py`
with the staged pack, executable, build receipt, corpus and its
SHA-256, and a fresh output directory. The source receipt records
the exact inputs for this run.
