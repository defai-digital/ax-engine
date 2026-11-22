# Flash Next MXFP4 reader startup and warmed requests on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, Rust 1.99.1,
and MLX 0.32.4. The [previous native loader study](../flash-next-native-loader-m5-20261121/README.md)
supplied the same diagnostic executable, pinned MXFP4 pack and loader
settings. Four fresh APFS clones ran in `name-short`, `lazy-short`,
`lazy-long`, and `name-long` order. Three selected ranges per clone
each began with one sampled page resident. The range samples give
partial coverage of host cache state.

| First prompt | Reader | HTTP ready | First TTFT | Launch to first token | CPU idle after run |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | Guarded name-order mmap | 70.194 s | 1.952 s | 72.146 s | 94.47% |
| 512 | MLX lazy CPU | 22.098 s | 0.969 s | 23.067 s | 91.55% |
| 8,192 | MLX lazy CPU | 20.058 s | 10.315 s | 30.373 s | 81.40% |
| 8,192 | Guarded name-order mmap | 48.104 s | 9.613 s | 57.717 s | 92.41% |

| First prompt | Warm prompt | Lazy CPU change in median TTFT | Lazy CPU change in median decode rate |
| ---: | ---: | ---: | ---: |
| 512 | 512 | +18.19% | −8.21% |
| 512 | 2,048 | +38.25% | −13.81% |
| 512 | 8,192 | +42.37% | −8.83% |
| 8,192 | 512 | −4.01% | +2.39% |
| 8,192 | 2,048 | +13.46% | +2.69% |
| 8,192 | 8,192 | +7.10% | +0.98% |

Each warmed median uses three matched prompts per arm. The run matched
all 1,792 output IDs to the same-prompt controls across the four arms.
It reproduced the ordinary resident single-request profile, 32,768-token
boundary, 32,769-token rejection, and post-run health. All 127 pressure
samples were normal. All requests reported no verifier steps and no
route errors.

The terminal comparator failed. The lazy long-first arm measured
81.40% post-run CPU idle against the 90% floor. The eligible short-first
pair crossed both warmed timing bands, +3% TTFT and −3% decode. Keep
guarded mmap as the current default. This comparison supplies no new
measurement under the frozen release delivery procedure.

The [sealed evidence](evidence.tar.xz) preserves the original terminal
receipt, memory samples, build and workload identities, and the four
server logs with local listen addresses redacted. Replay and compare
against [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-native-loader-first-request-m5-20261122/replay.py
```
