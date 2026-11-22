# Flash Next sidecar-only positional reads on M5 Max

The MacBook Pro M5 Max 128 GB experiment used the pinned MXFP4 pack,
internal NVMe SSD, macOS 27.2.1 and MLX 0.32.4. Each arm used a fresh
APFS clone and the same instrumented `8204ff924` release-server binary.
The trunk used the guarded mmap reader throughout. Only the MTP
sidecar reader changed. The runner verified all 49 publisher files
before and after measurement and reversed reader order for the arm
starting with the 8,192-token prompt.

| First prompt | Mapped readiness | Positional readiness | Mapped sidecar | Positional sidecar |
| --- | ---: | ---: | ---: | ---: |
| 512 tokens | 164.453 s | 92.460 s | 12.731 s | 0.535 s |
| 8,192 tokens | 104.517 s | 94.470 s | 12.191 s | 0.532 s |

The 512-token pair also saw a 59.266-second trunk-load difference with
the same trunk reader. The 8,192-token pair saw a 0.093-second trunk
difference. The isolated sidecar timings provide the direct comparison.
The first 8,192-token request reached its first token 1.009 seconds
later from dispatch in the positional arm. Readiness plus this first
request remained 9.038 seconds shorter.

| First prompt | Warm 512 TTFT change | Warm 2,048 TTFT change | Warm 8,192 TTFT change |
| --- | ---: | ---: | ---: |
| 512 tokens | +0.62% | −1.68% | −0.32% |
| 8,192 tokens | −2.21% | −2.42% | −5.13% |

All six warm TTFT and decode medians meet the original 3% bands.
Decode changes range from −1.89% to +1.85%. Each arm keeps the ordinary
resident, one-request 32K profile and matches all 448 greedy output
IDs, including the 32K boundary. It records zero MTP verifier steps
and route errors. The health and overflow controls pass, pressure
remains normal and pre- and post-run CPU idle exceed 90% in every arm.

This bench result supports a candidate limited to the sidecar reader.
The [required-MTP functional comparison](../flash-next-owned-positional-native-sidecar-m5-20261122/README.md)
supplies separate output and verifier controls for the same experimental
route. Verify the isolated runtime change in a production build, on
the supported M4 tier, and against the frozen delivery readiness harness.

The [raw receipt](receipt.json.gz) retains the warm, first-request and
readiness measurements. The [measured runner](run_sidecar_warm.py)
validates the resident shard inventory and requires the matching sidecar
startup trace without a positional trunk trace. The offline replay
binds the runner and reproduces [the comparison](comparison.json).

```bash
python3 benchmarks/results/flash-next-sidecar-reader-warm-m5-20261122/replay.py
```
