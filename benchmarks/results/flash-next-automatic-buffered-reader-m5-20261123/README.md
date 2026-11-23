# Flash Next automatic buffered resident reader on M5 Max

The signed candidate `d2220c26f` serves the exact pinned MXFP4 pack
on a MacBook Pro M5 Max 128 GB with macOS 27.2.1 and the internal SSD.
Every process uses the ordinary 32K single-request profile with MTP off.
The mapped arms explicitly set `AX_MLX_FLASH_NEXT_ALIGNED_BUFFERED=0`.
The buffered arms leave the reader overrides absent and verify that the
loader selected `AlignedBuffered`. All four arms recheck the 49 publisher
files before and after serving.

| Arm | HTTP ready | Resident load | CPU idle before / after |
| --- | ---: | ---: | ---: |
| Mapped first | 162.431 s | 140.932 s | 97.72% / 96.20% |
| Buffered first | 28.089 s | 8.124 s | 97.42% / 93.81% |
| Buffered second | 30.103 s | 7.777 s | 95.74% / 97.10% |
| Mapped second | 168.460 s | 146.734 s | 97.60% / 90.66% |

Both automatic starts meet the 120-second bound (28.089
and 30.103 seconds). Each arm matches 448 token IDs
against the signed M5 oracle, preserves the 32K limit and records normal
memory pressure throughout. Timing-eligible orders are first, second.
The fixed 512-, 2,048- and 8,192-token comparison passes the warmed timing
band. The raw receipts retain each metric and the two orders.

This measures the guarded 32K default on the stated source. Stock Required,
full QA, lifecycle, fixed timing matrix and fresh delivery retain separate
release evidence. The preceding independent heartbeat
[stopped under critical pressure](../flash-next-reference-heartbeat-pressure-m5-20261123/README.md).

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
