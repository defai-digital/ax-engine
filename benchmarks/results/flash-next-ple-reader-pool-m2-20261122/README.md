# Flash Next persistent PLE read workers

An isolated experimental branch reuses worker threads for Flash Next
synchronous PLE row reads. Its caller processes the first chunk, and
a bounded process-wide `rayon-core` pool processes the remaining
disjoint chunks. Speculative prefetch still uses its original scoped
workers. Each demand gather waits for all its chunks and keeps the
existing error and cancellation precedence. The pool is enabled
explicitly with `AX_FLASH_NEXT_NGRAM_READ_POOL=1`.

TensorFold reuses parked host workers for cold PLE gathers. The pinned
MLX positional loader also reuses workers and waits for disjoint
destination ranges. Those implementations informed the experiment.
The [existing reader baseline](../flash-next-ple-reader-dispatch-m2-20261122/README.md)
showed that per-gather spawns contribute to repeated-row staging time.

The five real-pack arms run on a Mac Studio M2 Ultra with 192 GB,
external SSD, macOS 27.2.1, Rust 1.99.1 and MLX 0.32.4. They exercise
the pinned MXFP4 checkpoint and verify all 49 publisher files. The
arm order is scoped, pooled, pooled, scoped at 32 configured threads,
then serial. Every arm matches the raw payload and read volume for
288 singleton decode stages and four 512-token prefill stages,
including a second pass of identical rows. The result confirms actual
pool activation on both candidate arms and normal pressure throughout.

| Repeated staging, p50 | Scoped 32 | Pooled 32 | Serial |
| --- | ---: | ---: | ---: |
| One decode token | 0.367–0.377 ms | 0.095–0.096 ms | 0.035 ms |
| 512-token prefill | 29.903–30.087 ms | 29.966–30.119 ms | 21.543 ms |

The time covers native PLE staging. The data labels distinguish the
initial pass after publisher verification and repeated rows. Physical
cache residency and end-to-end serving need separate measurements.
The M5 comparison is queued behind the original reader baseline.
Commit `566cac1d6` passed 4,786 Rust tests, 345 Python tests,
strict Clippy, the script suite and both qualification dry runs.

The M4 Pro 64 GB support-tier software control passed 20 native tests
with each backend using the same executable and wheel. Both paths also
passed the workstation controls, including progress during a blocked
speculative prefetch. The [comparison](comparison.json) includes the
measured stages, and the [probe patch](probe.patch.gz) contains the exact
experimental source and test instrumentation.

From this directory, run `shasum -a 256 -c SHA256SUMS` and
`python3 -B replay.py` to verify the committed result.
