# Flash Next sidecar pre-read after independent empty-cache deliveries

MacBook Pro M5 Max 128 GB, internal NVMe SSD, and the signed
`eb57f768a` native downloader and release-server.
Both arms fetch and validate the same pinned MXFP4 pack with 49
publisher members and 132,261,877,478 bytes. Each starts from a
distinct empty application cache. The mapped arm runs first.

| Reader | Native download | Disabled HTTP ready | Required HTTP ready | Default HTTP ready | Arm timing eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| Mapped | 987.007 s | 156.160 s | 122.857 s | 121.892 s | Yes |
| Prefetch | 974.658 s | 94.876 s | 114.087 s | 113.122 s | Yes |

All 42 native functional controls match the frozen HTTP choices,
stable usage and verifier counters. Each Required run verifies 107
steps. Disabled and default record zero, and each Required
kill-switch check fails before the sidecar payload route.

Both arms meet the host timing conditions. The mapped arm misses
the frozen delivery readiness gate. The pre-read arm passes it.

The portable archive verifies the independent transfer progress,
pinned runtime, native controls, and sidecar read markers. It retains
original digests and removes local downloader destinations.
The full release board also requires the pinned MLX-VLM reference.

Replay with macOS and Python 3.12 or newer.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
