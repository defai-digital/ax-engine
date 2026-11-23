# Flash Next guarded-reader native QA on M5 Max

The signed `d2220c26f` server from the [fresh delivery](../flash-next-guarded-reader-fresh-m5-20261123/README.md)
ran the frozen 105-item cohort in each direct mode on MacBook Pro M5 Max
128 GB with an internal SSD, macOS 27.2.1 and MLX 0.32.4. Both modes use
the ordinary 32,768-token profile. All 210 responses match the
[prior signed cohort](../flash-next-probe-followup-qa-m5-20261122/README.md)
in text, usage and stop reason. Both modes pass the same 101 answer
checks, with the same four recorded failures. Every request stops
normally. The native route records zero MTP verification steps and
zero route errors.

| Direct mode | HTTP readiness | Peak server RSS | 29,774-token lookup |
| --- | ---: | ---: | ---: |
| Disabled | 29.145 s | 69.106 GiB | 42.532 s |
| Default | 31.147 s | 70.603 GiB | 43.137 s |

The observed CPU idle is 97.28% before and 91.85% after. Readiness,
peak RSS and the lookup pass the frozen gates for this QA receipt.
Replay regrades all raw responses and checks the archive against the
sealed delivery, build, grader inputs and prior cohort. The separate
independent reference comparison governs release quality.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
