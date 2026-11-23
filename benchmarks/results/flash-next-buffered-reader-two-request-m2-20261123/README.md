# Flash Next buffered reader two-request controls on M2 Ultra

The signed reader candidate `d2220c26f` serves the pinned MXFP4 pack from
the internal SSD of a Mac Studio M2 Ultra with 192 GB on macOS 27.2.1.
The native executable uses Rust 1.99.1 and MLX and MLX Metal 0.32.4. Both
arms recheck the 49 publisher files before and after serving.

The two-request profile with reader overrides absent selects `Mapped`.
Setting `AX_MLX_FLASH_NEXT_ALIGNED_BUFFERED=1` selects `AlignedBuffered`.
Both fresh processes opt into learned document prefixes with MTP disabled.
Each arm matches all 1,921 output token IDs against the
[sealed cold HTTP oracle](../flash-next-natural-document-prefix-http-m2-20261123/README.md).
Both restore exactly 2,048 tokens in three cases, preserve independent
peer progress during overlapping prefill, and drain all five counters
after each case, including cancellation and replacement. All 245 host
pressure samples are normal.

This is functional evidence for the tested two-request workload on M2 Ultra.
The release decision requires its own M5 Max evidence.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
