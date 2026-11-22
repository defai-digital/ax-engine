# Flash Next aligned resident reader in default serving on M5 Max

MacBook Pro M5 Max 128 GB with internal NVMe storage, macOS 27.2.1,
and MLX and MLX Metal 0.32.4. The four processes use the same
Rust 1.99.1 `release-server` binary at `31e2b6e1f`, the frozen
MXFP4 publisher inventory, and a fresh APFS clone per arm. The
reader derived from TensorFold's aligned Darwin I/O changes only
the 15 resident trunk shards. All arms keep the mapped MTP
sidecar and one active resident request with 32,768 context tokens.
MTP is off throughout this comparison.

| First prompt | Mmap readiness | Aligned readiness | Mmap resident load | Aligned resident load |
| --- | ---: | ---: | ---: | ---: |
| 512 tokens | 192.400 s | 26.105 s | 170.242 s | 11.364 s |
| 8,192 tokens | 104.369 s | 26.079 s | 82.559 s | 11.357 s |

Each process matches all 448 greedy output IDs, all stop and route
fields, the 32K limit, and both health checks. All sampled pressure
levels remain normal and verifier steps stay at zero.

The short-first pair meets the 90% CPU-idle requirement at every
sampled boundary. Its warmed TTFT at 2,048 and 8,192 tokens rises by
14.39% and 8.96%, over the 3% guard. All short-first decode rates
meet the guard. The reverse-order aligned arm ends at 81.40% CPU
idle, so the reverse timing cannot resolve the regression. Keep
mmap as the ordinary default. This APFS-clone crossover evaluates
a runtime candidate. The delivered-copy board and Required policy
have separate evidence.

The archive retains all four process receipts, reader traces, resource
samples and checked measurement inputs. Paths and local listener
addresses are redacted in captured runtime logs. Replay validates the
hashes and recomputes the comparison from the recorded workload.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
