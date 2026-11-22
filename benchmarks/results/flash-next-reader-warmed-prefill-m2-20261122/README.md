# Flash Next warmed reader prefill on M2 Ultra

The staged experiment at `3130d9d12` runs the native greedy prefill
entry point with fresh per-request cache state. It verifies the same
49-member resident MXFP4 pack and uses one process per reader with
Rust 1.99.1 and MLX 0.32.4. Both processes use the same 512-token
warmup. They each measure three 2K/8K pairs before and after two
additional unmeasured 8K prefills. A final traced replay captures the
actual physical widths and stage times.

| Period | Prompt | Guarded mmap | Positional |
| --- | ---: | ---: | ---: |
| Before added warmup | 2,048 | 4.144 s | 4.138 s |
| Before added warmup | 8,192 | 22.413 s | 22.466 s |
| After added warmup | 2,048 | 4.133 s | 4.145 s |
| After added warmup | 8,192 | 22.357 s | 22.644 s |

The first tokens, serialized states and full forward geometry match
between readers for both lengths. The 2K prompt uses 512, 512, 512,
511 and one token. The 8K prompt uses fifteen 512-token forwards, a
511-token forward and a final singleton. Each isolated traced pass
matches its untraced control. The largest median reader difference
is 1.29% on the reused external-SSD snapshot.

This is a model-only check on M2. The two warmup periods occur in
sequence, so time drift can contribute to their differences. The
existing M5 serving regression still governs the reader decision. The
M5 follow-up runs the same executable in both reader orders.

Replay the [comparison](comparison.json) from the [mapped](mmap.json)
and [positional](pread.json) observations.

```bash
python3 benchmarks/results/flash-next-reader-warmed-prefill-m2-20261122/replay.py
```
