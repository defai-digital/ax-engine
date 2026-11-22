# Flash Next HTTP disconnect during mixed prefill on M2 Ultra

Mac Studio M2 Ultra 192 GB, external APFS storage, macOS 27.2.1.
The [same release-server binary, pack and solo oracle](../flash-next-server-prefill-early-m2-20261122/README.md)
run in resident two-slot mode with MTP disabled and experimental tail
deferral and physical splitting enabled.

TensorFold checks cancellation and admission between prompt calls.
For this native AX HTTP control, the long prompt arrives after short
output token 48. The client closes the long stream when the server
reports its first 512-token physical prefill cut. The cut shares a
scheduler step with short decode and the long request publishes no
generated tokens. The short request continues for 46 more generated
steps after the shared step and produces its exact 96-token oracle
output. The server drains five generation counters.

A fresh request with the same 8,192-token prompt starts its cursor
at zero, completes four 2,048-token prefill items, and matches all
96 solo output IDs. All five counters drain again and health remains
OK. All 57 pressure samples record normal pressure. This test covers
one mid-prefill disconnect and recovery sequence on M2.

The [comparison](comparison.json) replays the
[observations](evidence.tar.xz) with the sealed executable provenance,
request identities, token oracle, cursor and drain checks.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
