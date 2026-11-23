# Flash Next MTP counters across cancellation

Run the pinned MXFP4 pack `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`
on Mac Studio M2 Ultra 192 GB, internal SSD, macOS 27.2.1, and MLX
and MLX Metal 0.32.4. The two arms share the same signed release
server. Both use one resident slot, mapped weights, a 512-token
prompt, and a 64-token greedy output budget. The launcher binds all
49 publisher files, the derived manifest, and the loaded MLX library.
Prefix reuse and generic n-gram acceleration are disabled. Required
MTP runs with its experimental cost controller disabled.

The SDK's sparse internal decode report previously retained no
Flash Next MTP counters after the first full report. The earlier signed
server exposed a six-token prefix before an SSE disconnect. Its visible
decode frame widths were 1, 1, 1, and 2, while its post-disconnect
metrics recorded one verification and zero acceptances. The exact
same-prompt recovery still succeeded.

The fixed SDK forwards the request-set identity and cumulative Flash
Next MTP counters on sparse engine steps. A stored singleton request
keeps those counters when explicit cancellation returns its terminal
snapshot. The full route is materialized at its existing report
boundaries.

| Required control | Verifications / accepts at the interruption | Verifications / accepts after drain |
| --- | ---: | ---: |
| Native SSE disconnect | 4 / 1 | 6 / 2 |
| Stepwise explicit cancel | 4 / 1 | 4 / 1 |

Both Required interruption prefixes contain six tokens. The stepwise cancellation
returns the same prefix and verified and accepted counts in its terminal
snapshot. Each arm completes the baseline and two same-prompt recoveries
with identical 64-token outputs. Required completes 41 verifier steps and
accepts 22 drafts in each completed request. Disabled reports zero
verification steps. All five admission and stream counters drain after
each action. MTP error counters stay at zero. All 76 pressure
samples are normal and peak process footprint is 80.20 GiB.

This is a request-lifecycle and counter-publication control on M2 Ultra.
Private verifier rollback, M5 qualification and throughput remain
separate evidence.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
