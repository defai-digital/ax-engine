# Flash Next first text by prompt length on M5 Max

**Date:** 2026-11-25 · **Host:** MacBook Pro M5 Max 128 GB, macOS 27.2,
pinned MLX 0.32.4 · **Pack:** Qwen 3.8 Flash Next AXQ MXFP4 at revision
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` ·
**Tool:** [`scripts/check_first_text_by_length.py`](../../../scripts/check_first_text_by_length.py)

Cold first-text latency of a direct greedy request with one output token,
by prompt length. Every prompt starts with a marker unique to the run, the
length and the repeat, so nothing is reused between requests. Three warmups,
then three prompts per length; a row's `median_seconds` is their median and
`input_tokens` is the server's own count.

| File | AX Engine source | Use |
| --- | --- | --- |
| [`m5-max-ax-c3ce06f75.json`](m5-max-ax-c3ce06f75.json) | `c3ce06f75` | Baseline |
| [`m5-max-ax-c3ce06f75-repeat.json`](m5-max-ax-c3ce06f75-repeat.json) | `c3ce06f75`, server restarted | Run-to-run spread |
| [`m5-max-ax-b04a5f2ef.json`](m5-max-ax-b04a5f2ef.json) | `b04a5f2ef`, the commit before | A build the gate must reject |

| Input tokens | `c3ce06f75` (s) | Repeat (s) | `b04a5f2ef` (s) |
| ---: | ---: | ---: | ---: |
| 59 | 0.213 | 0.211 | 0.305 |
| 150 | 0.228 | 0.226 | 0.320 |
| 255 | 0.262 | 0.261 | 0.394 |
| 437 | 0.345 | 0.344 | 0.433 |
| 827 | 0.514 | 0.517 | 0.634 |
| 1,517 | 0.829 | 0.834 | 0.963 |
| 2,258 | 1.375 | 1.384 | 1.756 |
| 2,986 | 2.015 | 2.045 | 2.570 |
| 4,468 | 3.545 | 3.583 | 4.340 |

The repeat stays within 1.6% of the baseline at every length. The earlier
commit is 16 to 51% slower and fails the gate at all nine lengths with the
default margins (10% and 20 ms, both exceeded):

```text
$ python scripts/check_first_text_by_length.py compare \
    --baseline m5-max-ax-c3ce06f75.json --candidate m5-max-ax-b04a5f2ef.json
FAIL: 9 of 9 lengths regressed or are missing
```

What made `c3ce06f75` faster is described in
[Flash Next prefill schedule](../../../docs/performance/flash-next-prefill-schedule-2026-11-25.md).
Token counts here are 23 higher than in that note's table because the
default marker is longer.

## Replay

Start one server on the pack and nothing else on the host:

```bash
ax-engine-server --host 127.0.0.1 --port 8080 --mlx --model-id flash-next \
  --mlx-model-artifacts-dir /path/to/flash-next-pack \
  --max-concurrent-requests 1 --mlx-mtp-policy disabled \
  --disable-ngram-acceleration --stream-experts off
python scripts/check_first_text_by_length.py measure \
  --base-url http://127.0.0.1:8080 --model flash-next \
  --sentences 1,8,16,30,60,113,170,226,340 --output candidate.json
python scripts/check_first_text_by_length.py compare \
  --baseline benchmarks/results/flash-next-first-text-by-length-m5-20261125/m5-max-ax-c3ce06f75.json \
  --candidate candidate.json
```

Measure a server on its own. Started right after another engine had served
the same pack, the same binary measured 5 to 9% slower from 2,961 tokens up
than it did alone.
