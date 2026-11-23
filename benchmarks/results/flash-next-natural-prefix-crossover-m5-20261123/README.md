# Flash Next natural prefix-learning crossover on M5 Max

A MacBook Pro M5 Max 128 GB served the pinned Flash Next MXFP4 pack on
macOS 27.2.1 with its internal SSD, Rust 1.99.1, and MLX and MLX Metal
0.32.4. Four fresh processes ran off, on, on, off with the same signed
server, resident experts, MTP off, two request slots, and no explicit
seed. The publisher inventory verified all 49 members before and after
the campaign. The [M2 eviction run](../flash-next-natural-prefix-eviction-m2-20261123/README.md)
binds the executable to the same pack.

All four arms produced the same 1,921 token IDs as the
[earlier disabled M5 arm](../flash-next-variable-prefix-suffix-http-m5-20261123/README.md).
In both enabled arms the third question reuses 2,048 tokens. Two later
related requests also hit, including the disconnect case. The first two
questions, the unrelated prompt, and the 2,049-token suffix retain the
cold path. The independent peer progresses on the same scheduler items,
and all 758 host-pressure samples are normal.

The first enabled arm ended at 87.40% CPU idle against the 90% floor.
This leaves the full four-arm timing comparison ineligible. The second
pair passes the timing guard for its own observations.

| Second pair, target first-token latency | Off | On |
| --- | ---: | ---: |
| Second question, cold prefill and optional snapshot write | 2.076 s | 2.112 s |
| Third question, learned-prefix restore | 3.123 s | 1.481 s |
| Aligned 2,048-token suffix boundary | 4.443 s | 2.623 s |
| Second and third questions, sum | 5.198 s | 3.593 s |

The replay retains request memory snapshots and per-arm process peaks.
These measurements cover all server allocations. The product release
board requires the current independent MLX-VLM reference.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
