# Flash Next stock Required on the guarded reader candidate

The signed `d2220c26f` server runs the pinned MXFP4 pack on MacBook Pro
M5 Max 128 GB with macOS 27.2.1 and an internal SSD. Both launches
use the stock native-support commands on the retained publisher
files, and no read-policy overrides are supplied.

| Mode | Advertised context | HTTP ready | Functional cases | Verified steps |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 32,768 | 29.129 s | 7 | 0 |
| Required | 16,384 | 113.450 s | 7 | 107 |

CPU idle is 91.86% before and 90.91% after. Required meets the
120-second retained target; its stock 16K geometry remains outside
the automatic buffered rule. All 14 responses match the earlier signed
[delivery corpus](../flash-next-probe-followup-fresh-m5-20261123/README.md)
in stable choices, usage and verifier counters. The incompatible
Required override still fails the activation check.

This replay covers retained-file behavior on the same executable as
the [automatic-reader crossover](../flash-next-automatic-buffered-reader-m5-20261123/README.md).
Fresh delivery and release qualification have separate results.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
