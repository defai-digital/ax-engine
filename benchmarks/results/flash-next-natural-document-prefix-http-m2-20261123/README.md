# Flash Next natural document-prefix reuse on M2 Ultra

Two signed experimental servers served the same pinned resident Flash Next
MXFP4 pack from an external SSD on Mac Studio M2 Ultra 192 GB. Both
verified all 49 publisher members before and after execution. The
fixtures contain separate questions that share a 2,048-token prefix.
Each question overlaps the decode of an independent peer. Neither
server receives an explicit seed request.

The control records zero physical reuse throughout. The learning arm
keeps the first two questions cold. The third reuses exactly 2,048 tokens,
as do the later cancellation and suffix-boundary cases. The unrelated
document and the 2,049-token suffix stay cold. The replay requires
request-owned restore telemetry for completed targets and verifies the
physical counters, peer progress and drained gauges. All 1,921 generated
token IDs agree across the two processes. All 273 host-pressure samples
are normal.

The source uses the observe-then-retain behavior exercised by TensorFold's
prompt-cache fork test as design input. This is functional evidence for
the original isolated patch. A later revision addresses its failed Rust
fastpath switch classification check. Primary-tier timing, reuse
after full-prompt eviction, competing learners and qualification have
separate checks.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
