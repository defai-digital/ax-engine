# Flash Next 2,048-token native split audit on M5 Max

The [corrected SSE server](build.json) ran the pinned Flash Next MXFP4
revision `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` on a MacBook Pro
M5 Max with 128 GB unified memory, internal SSD, macOS 27.2.1 and
MLX 0.32.4. The diagnostic ran resident with MTP and n-gram
acceleration disabled. Its cold controls disabled prefix reuse.
All 49 publisher files passed validation before and after the run.

The [strict audit](audit.json) matched 28 arrays of 96 output IDs.
Four requests split as 512, 512, 512, 511 and 1 prompt tokens while
a peer was already decoding. Each physical step carried its own
request's exclusive logical boundary. All six disconnect and reuse
controls completed, including two disconnected during prefill.
The audit verified the route marker at disconnect, surviving and
replacement sequences, zero serving gauges and normal measured pressure.

| Nonserial observations | Split disabled | Split enabled |
| --- | ---: | ---: |
| Largest observed inter-token gap | 1.763 s | 0.515 s |
| Median combined output throughput | 26.37 tokens/s | 24.77 tokens/s |
| Median per-trial first-token latency | 2.018 s | 2.294 s |
| Peak sampled process footprint | 72.183 GiB | 72.018 GiB |

The timing comparison covers four nonserial trials per policy.
The smaller maximum gap and lower aggregate throughput support
continuing the controlled evaluation. The 2,049-, 2,050- and
8,192-token holdouts and sustained mixed-serving run are separate.

Decompress the [raw receipt](result.json.gz) and use
`verify_flash_next_prefill_split.py` from the
[replay archive](../../flash-next-mxfp4-sse-cuts-m2-20261121/replay-tools.tar.gz).
Pass the decompressed file and the `--server-sha256` value in
`build.json`. The verifier reproduces `audit.json`.
