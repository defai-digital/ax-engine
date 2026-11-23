# Flash Next guarded-reader native matrix on M5 Max

The signed `d2220c26f` server from the [fresh delivery](../flash-next-guarded-reader-fresh-m5-20261123/README.md)
ran two separate processes per direct mode and a warm-up process on MacBook
Pro M5 Max 128 GB with an internal SSD, macOS 27.2.1 and MLX 0.32.4. All
five launches use the ordinary resident 32K context with one active
request. Both repeats in both modes match the [prior signed native matrix](../flash-next-probe-followup-matrix-repeat-m5-20261123/README.md)
on all 124 output arrays, including fixed-length and trajectory prompts.
MTP verifies zero steps throughout the four measured processes.

| Input tokens | Default median TTFT | Default median decode |
| ---: | ---: | ---: |
| 512 | 0.527 s | 48.64 tokens/s |
| 2,048 | 1.837 s | 45.79 tokens/s |
| 8,192 | 10.510 s | 44.19 tokens/s |

CPU idle was 97.40% before and 95.94% after. The full native matrix
retains all repeats and the frozen fastest-repeat selection. The selected
processes reach HTTP within 30.276 seconds. The table reports native
measurements. Release quality and relative timing still require the
same-OS independent reference.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
