# Flash Next aligned buffered reader crossover on M5 Max

A signed release-server binary on source `bde3b0319` compares two
resident readers on a MacBook Pro M5 Max 128 GB with an internal SSD.
The launch uses the pinned MXFP4 pack, ordinary one-request 32K
serving, and MTP off. The experiment explicitly selects the reader.

| Arm | HTTP ready | Resident tensor load | CPU idle before / after |
| --- | ---: | ---: | ---: |
| Mapped first | 172.461 s | 149.243 s | 97.31% / 96.32% |
| Buffered first | 30.091 s | 8.222 s | 96.80% / 95.94% |
| Buffered repeat | 30.089 s | 8.034 s | 94.64% / 96.86% |
| Mapped second | 170.465 s | 148.227 s | 97.55% / 90.84% |

Both orders use the same 512-token first prompt and fixed 512-,
2,048-, and 8,192-token warmed cases. Buffered readiness improves
by 82% in both orders. The largest warmed TTFT increase is 1.05%,
and the largest decode-rate reduction is 2.33%. Each arm matches
448 signed-oracle output IDs and preserves the 32K boundary. All
164 sampled memory-pressure readings are normal.

The first three arms are the
[previously sealed evidence](../flash-next-current-buffered-reader-first-m5-20261123/README.md).
The independent reference took the device before the final arm and then
[stopped under critical pressure](../flash-next-reference-heartbeat-pressure-m5-20261123/README.md).
After memory pressure returned to normal, the supervisor verified the
nine prior raw files and ran only the missing mapped arm. The original
and resumed runs remain separately identifiable.

This measures explicit startup and warmed behavior on the stated build.
The guarded automatic-selection build has a separate crossover.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
