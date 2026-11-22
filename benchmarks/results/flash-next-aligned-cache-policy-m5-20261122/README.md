# Flash Next aligned reader cache-policy crossover on M5 Max

MacBook Pro M5 Max 128 GB with internal NVMe storage, macOS 27.2.1,
Rust 1.99.1 and MLX and MLX Metal 0.32.4. All processes use the same
`23c5c8cb8` release-server image and frozen publisher inventory.
Independent fresh APFS clones hold the 15 resident trunk shards.
Every arm uses the same aligned stage geometry and MLX allocation
path. Only the resident file descriptors' cache policy changes.
The MTP sidecar stays mapped, MTP stays off, and all arms use the
resident one-request 32K profile.

| Process order | File caching | HTTP ready | Resident load | CPU idle before / after |
| --- | --- | ---: | ---: | ---: |
| 1 | Off | 36.110 s | 11.514 s | 91.83% / 96.49% |
| 2 | On | 28.069 s | 7.809 s | 98.59% / 95.75% |
| 3 | On | 30.138 s | 7.613 s | 97.30% / 88.47% |
| 4 | Off | 36.140 s | 13.769 s | 96.13% / 91.91% |

Both buffered arms fill all 294,943 selected pages across three
sampled files at readiness. Both uncached arms leave their own
prelaunch selected-page counts unchanged. Each arm matches all 448
output IDs, stop reasons, and route fields against the earlier
[mapped control](../flash-next-aligned-native-default-m5-20261122/README.md).
Health, the 32K boundary, overflow rejection and normal sampled pressure
hold in all four processes.

In the timing-eligible first pair, buffering raises warmed 8K TTFT by
5.01% from 8.846 to 9.289 seconds. The 512 and 2K TTFT changes and the
three decode rates pass the 3% guard. In reverse order, the buffered
process falls below the host CPU-idle floor and the numerical 8K TTFT
difference reverses. The two uncached runs alone vary by 6.37% at 8K.
These observations do not isolate file caching as the cause of the
TTFT difference. Keep mmap as the ordinary reader. This diagnostic
uses APFS-cloned inputs and does not update the delivered-copy
readiness or independent comparison gate.

The archive contains all four arms, resource samples, reader traces
and checked experiment inputs. Local listener addresses and file paths
are redacted. Replay checks the archived serving inputs and the
earlier mapped trajectory and recomputes the comparison from raw data.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
