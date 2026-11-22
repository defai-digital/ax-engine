# Flash Next persistent PLE reader on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1, Rust
1.99.1, MLX and MLX Metal 0.32.4. The [same binary and workload as
M2](../flash-next-ple-reader-pool-m2-20261122/README.md) measures
32-thread ordinary/pooled/pooled/ordinary arms and a final serial
control. All arms match the M2 payload digests and read volumes for
the 128 PLE table shards. All pressure samples are normal. Host CPU
idle before and after is 97.56% and 97.35%.

| Reader | Repeated decode p50 | Repeated decode p95 | Repeated prefill p50 |
| --- | ---: | ---: | ---: |
| 32 threads, ordinary | 0.1600 ms | 0.1958 ms | 25.155 ms |
| 32 threads, pooled | 0.3065 ms | 0.5425 ms | 26.338 ms |
| 32 threads, pooled | 0.1788 ms | 0.5025 ms | 25.793 ms |
| 32 threads, ordinary | 0.1620 ms | 0.1978 ms | 25.003 ms |
| 1 thread | 0.0208 ms | 0.0221 ms | 14.247 ms |

The repeated-row probe favors serial reads. A persistent pool does
not reduce repeated decode p95 or prefill p50 against the ordinary
32-thread reader in either order. The first-process initial prefill
is more expensive on the shared view. Cache state across processes
was uncontrolled. Keep the current default. The measurements cover
table staging, with separate HTTP and first-token evidence needed
for any serving decision.

The [comparison](comparison.json) replays the
[raw result](observed-m5.json.gz) using the shared M5 reader verifier.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
