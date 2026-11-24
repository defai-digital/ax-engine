# Flash Next grouped QSA on the native HTTP path

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1,
Rust 1.99.1 and published MLX 0.32.4. Six fresh processes use the same
signed native server and the verified 49-member pinned MXFP4 pack.
One slot, resident experts, MTP off, and prefix reuse off. The wide
pairs set both the CLI chunk and long-prompt clamp to 2,048, changing
only the grouped evaluation switch within each pair. The 512-row
sparse control keeps the ordinary long-prompt clamp.

| Pair | Peak MLX, stock | Peak MLX, grouped | Peak physical, stock | Peak physical, grouped |
| --- | ---: | ---: | ---: | ---: |
| Wide first | 83.18 GiB | 75.33 GiB | 85.82 GiB | 77.33 GiB |
| Wide reverse | 83.18 GiB | 75.33 GiB | 86.02 GiB | 77.34 GiB |
| 512-row control | 73.44 GiB | 73.44 GiB | 75.64 GiB | 75.52 GiB |

All 448 paired completion IDs match, excluding replacement requests.
The widest request uses 32,704 input and 64 output tokens and observes
144 grouped layer forwards in each treatment arm. The 512-token
prompt and the 512-row sparse control use the existing evaluation.
Each wide server sees a native SSE token before disconnect. Every
replacement matches the original 8K completion, and lifecycle
gauges drain after each request. All 661 sampled pressure
levels remain normal and within the fixed 96 GiB footprint guard.
The 8K output IDs differ across the two physical chunk geometries.
Each crossover pair uses one fixed geometry.

At the maximum context, client TTFT is
43.97 to 41.30 seconds
in the first pair and
44.17 to 40.91 seconds
in reverse. Request-window host CPU idle is
80.67% / 91.90% and
83.56% / 91.91%.
The CPU conditions differ between arms. The experimental switch defaults to
off and the estimated working set retains the full scratch reserve.

Mac mini M4 Pro 64 GB, macOS 27.0,
replays the synthetic gathered-attention fixture. All 48 outputs
match within geometry. The 2,048-row incremental peak falls from
11.21 GiB to 2.81 GiB with two-tile groups at 8K and 32K history.
The component control has no model weights.

The archive binds source, build, exact requests, route markers and
observed memory. It includes no qualification comparison against the
independent ordinary reference.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
