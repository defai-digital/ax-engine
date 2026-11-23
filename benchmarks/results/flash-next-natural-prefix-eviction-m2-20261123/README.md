# Flash Next learned prefix eviction over HTTP on M2 Ultra

A Mac Studio M2 Ultra 192 GB served the pinned Flash Next MXFP4 pack
from ordinary external SSD files on macOS 27.2.1, using Rust 1.99.1 and
MLX and MLX Metal 0.32.4. The two fresh processes use the same signed
release-server executable. The harness verifies all 49 publisher
members before and after the runs.

The cases send three different questions with the same 2,048-token
prefix and no explicit seed. Each question overlaps an independent
peer. With two L1 entries, the third question restores exactly 2,048
tokens after the peer's intervening eviction. With one entry, the peer
displaces the checkpoint and the third question takes the cold route.
Both arms match 864 generated token IDs against the original
[cold HTTP oracle](../flash-next-natural-document-prefix-http-m2-20261123/README.md).
All monitored pressure samples are normal.

The HTTP route checks physical restore and request ownership. The
[M4 Pro runner controls](../flash-next-natural-prefix-controls-m4pro-20261123/README.md)
verify the exact keys of the two evicted full prompts, competing
publications, and cleanup on cancellation. These measurements cover
functional behavior under the stated capacities.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
