# Flash Next two-slot baseline stopped on M5 memory pressure

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1. The
[same corrected binary and pinned MXFP4 pack](../flash-next-server-prefill-tail-m2-20261122/README.md)
begin the resident two-slot HTTP comparison. The first arm enables
physical prefill splitting and leaves tail deferral disabled.

The 8,192-token prompt overlaps the 512-token request's decoder
following the token-84 trigger. The observed bursts match the 192
pinned M5 solo token IDs, and all five generation counters drain.
The process footprint reaches 75.64 GiB. The monitor observes
165 normal and 142 warning pressure samples. The after-arm guard
rejects the baseline and the process exits before starting the
deferral treatment.

The [comparison](comparison.json) verifies the [failed arm
receipt](receipt.json), raw output/cursor trace and complete pressure
series. The retained arm supplies functional overlap evidence.
Performance evaluation requires a primary-tier comparison with
eligible host conditions.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
