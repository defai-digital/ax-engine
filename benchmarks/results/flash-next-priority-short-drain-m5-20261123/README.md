# Flash Next priority short decode drain on M5 Max

MacBook Pro M5 Max 128 GiB, internal NVMe SSD, macOS 27.2.1,
MLX and MLX Metal 0.32.4, and Rust and Cargo 1.99.1. All four arms
use the same signed release server from experimental commit
`71f891b25` and the pinned MXFP4 pack
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. Both configurations
use two mapped resident slots, MTP off, and short-first enabled.
Only the decode-drain switch changes.

Each process replays the same four 8,192/512-token pairs. The short
request arrives 500 ms after the long one. Both arms first run the
short prompt whole after the long reaches 2,048. During short
decode, the mixed arm physically splits the remaining long prefill.
The drain arm holds the original long cursor until the short
finishes, then resumes the next whole 2,048-token cut.

| Order | Arm | CPU idle before / after | Timing eligible | Holds | Median short maximum gap | Median burst output |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1 | Mixed | 93.93% / 91.82% | Yes | 0 | 731.81 ms | 12.15 tok/s |
| 2 | Drain | 97.41% / 99.42% | Yes | 4 | 22.65 ms | 13.66 tok/s |
| 3 | Drain | 99.20% / 99.38% | Yes | 4 | 22.55 ms | 13.66 tok/s |
| 4 | Mixed | 99.27% / 97.33% | Yes | 0 | 702.55 ms | 12.56 tok/s |

All 3,072 output IDs match the frozen solo oracle. Each held pair
has zero concurrent long-prefill and short-decode steps. Each mixed
pair has 15. All five counters drain in each arm. All 885
pressure samples are normal and peak footprint is 75.75 GiB.

Paired differences use the median of the four matched observations
within each timing-eligible process order.

| Order | Short offered TTFT | Short maximum gap | Long offered TTFT | Long offered completion | Burst output rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| First | -30.5 ms | -709.2 ms | +193.6 ms | -1579.2 ms | +11.09% |
| Second | -15.4 ms | -679.7 ms | +512.9 ms | -1095.7 ms | +7.72% |

The treatment remains experimental. The fixed pairs expose the hold
in each treatment burst. Natural-arrival performance and the standard
single-stream release each require their own evidence.

The portable replay checks exact output, prefill geometry, route
markers, full runtime flags, pack binding, pressure and all zero
counters. It retains original stream hashes and documents the two
client metadata redactions. The prior
[M2 controls](../flash-next-priority-short-drain-m2-20261123/README.md)
cover completion and cancellation with this executable.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
