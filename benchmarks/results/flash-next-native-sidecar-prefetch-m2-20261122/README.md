# Flash Next mapped sidecar pre-read native controls on M2 Ultra

One codesign-verified release-server binary ran on Mac Studio M2 Ultra
192 GB with external APFS SSD, Rust 1.99.1, MLX 0.32.4, and MLX Metal
0.32.4. The build receipt binds the source base and exact experimental
patch. Each process uses the same 49-file pinned MXFP4 pack and
resident trunk. Publisher files are verified before and after.

The read-ahead uses one admitted sidecar descriptor and a 4 MiB
buffer. The final materializer stays on the guarded mapped copy.
Each enabled arm reads 5,214,301,696 bytes before that copy.
The [component-level control](../flash-next-mapped-sidecar-prefetch-m2-20261122/README.md)
isolates the same sidecar weights and preserves the allocation order.

| Order | Policy | Sidecar pre-read | Sidecar stage | Resident trunk | HTTP ready | Verifier steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Default, mapped | 0 s | 1.427 s | 9.961 s | 28.324 s | 0 |
| 2 | Default, pre-read | 0.332 s | 1.144 s | 240.848 s | 255.842 s | 0 |
| 3 | Required, pre-read | 0.331 s | 1.136 s | 252.588 s | 268.125 s | 107 |
| 4 | Required, mapped | 0 s | 18.869 s | 248.627 s | 282.979 s | 107 |

All 28 functional controls pass. Each mapped/pre-read pair matches
all seven request bodies, response choices, stable usage fields, and
verifier counters. Pressure is normal at the start and end of each
arm. Explicitly skipping the sidecar under Required MTP exits with
the activation error in 0.417 seconds and emits no pre-read marker.

The reused external-SSD runs have large resident trunk variation.
This campaign establishes native output and route parity. Delivered
readiness and warmed-serving timing require the primary-tier
comparisons on M5 Max.

The archive contains the exact source patch, build receipt, raw
native responses, and de-identified phase and activation traces.
Replay binds inputs and recomputes both policy comparisons and the
explicit skip control.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
