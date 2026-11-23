# Flash Next guarded-reader native lifecycle on M5 Max

The signed `d2220c26f` server from the [fresh delivery](../flash-next-guarded-reader-fresh-m5-20261123/README.md)
ran the seven frozen lifecycle actions in each direct mode on MacBook
Pro M5 Max 128 GB. The ordinary resident launch advertises a 32,768-token
context and one active request. Both modes match all 14 stable HTTP actions
from the [prior signed run](../flash-next-probe-followup-lifecycle-m5-20261122/README.md),
drain all five generation counters after each action, verify zero MTP steps,
and exit cleanly.

Disabled reaches HTTP in 29.150 seconds and Default in 29.140 seconds.
CPU idle before and after is 90.70% and 96.29%. This receipt passes the
frozen lifecycle and readiness gates. Generation-matrix and independent
reference comparisons complete the remaining release evidence.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
