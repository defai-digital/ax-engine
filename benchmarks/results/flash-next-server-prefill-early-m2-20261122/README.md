# Flash Next early arrivals with tail deferral enabled on M2 Ultra

The Mac Studio M2 Ultra 192 GB used an external APFS storage, macOS 27.2.1,
Rust 1.99.1 and MLX and MLX Metal 0.32.4. The native release-server
executable, MXFP4 publisher inventory and single-slot token oracle
match the [M2 late-arrival campaign](../flash-next-server-prefill-tail-m2-20261122/README.md).
The resident server has two request slots, MTP disabled, physical
prefill splitting enabled and the experimental tail-deferral switch
enabled. Each burst sends the 8,192-token prompt on receipt of the
specified generated token from the 512-token prompt.

TensorFold at `f0cc0a134` gives ongoing long prefill bounded progress
between decode rounds and checks cancellation and admission between
prompt calls. That design motivated checking early arrivals that had
already entered prefill by the time the short request reached its tail.

Each burst matches all 192 generated IDs from the same-binary
single-slot oracle and drains all five generation counters. Each long
prompt starts prefill before short token 84 and completes four logical
items. Each 2,048-token item preserves the physical widths 512, 512,
512, 511 and 1. The token-48 case completes prefill before the short
tail. The token-65 case continues the fourth item across the start of
the short tail. Neither case records a scheduler hold or a partial
layer suspension.

The [comparison](comparison.json) replays the [raw HTTP traces,
receipt and pressure samples](evidence.tar.xz) against both pinned
prompts and every solo output ID. This evidence covers M2 functional
behavior under two controlled arrivals. Primary-tier timing,
sustained concurrency and the one-slot product readiness gate retain
their separate campaigns.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
