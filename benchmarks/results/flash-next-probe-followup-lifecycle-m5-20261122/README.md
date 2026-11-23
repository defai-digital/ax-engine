# Flash Next storage-probe follow-up lifecycle on M5 Max

The signed `26c0ccefe` release-server binary used the same pinned
resident MXFP4 pack and internal SSD as the
[native HTTP](../flash-next-probe-followup-http-m5-20261122/README.md)
and [QA](../flash-next-probe-followup-qa-m5-20261122/README.md) runs.
The MacBook Pro used M5 Max, 128 GB unified memory, macOS 27.2.1,
and MLX 0.32.4.

Both modes reproduce all seven stable HTTP actions from the published
baseline. SSE, two token budgets, stop, unfinished stream disconnect, and
subsequent recovery pass. All five generation counters drain after
every action, and both server processes exit cleanly.

| MTP mode | HTTP ready | Actions matched |
| --- | ---: | ---: |
| Disabled | 94.915 s | 7 |
| Default | 111.074 s | 7 |

Both starts meet the 120-second staged readiness bound. CPU idle was
98.90% before and 94.26% after the run. The captured startup logs retain
the memory plan, completed sidecar pre-read, runner admission, and HTTP
listening markers. This receipt covers lifecycle on the retained pack.
The independent empty-cache delivery and ordinary MLX-VLM comparison
remain separate.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
