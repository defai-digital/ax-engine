# Flash Next server-side decode-tail hold over native HTTP

The experimental release-server build from `2b905c012` runs on Mac
Studio M2 Ultra 192 GB, macOS 27.2.1, external APFS storage and MLX
0.32.4. It uses the pinned resident MXFP4 checkpoint and verifies all
49 publisher members before and after the run. A one-slot process first
records exact greedy output for the 512-token and 8,192-token prompts.
Both prompts request 96 generated tokens. The M2 outputs differ from
the earlier archived M5 arrays at both lengths, so the functional
comparison binds the same binary and same-host single-slot oracle.

Both measured two-slot processes receive the long HTTP request as
soon as the short stream emits its 84th token. The physical control
exercises mixed-prefill cuts alongside the short decoder. The enabled
process records one singleton decode step with the scheduler hold
marker, then completes the long prefill as four 2,048-token items
after the short stream ends. Each arm matches all 192 solo output IDs
and drains all five work counters. All 545 pressure samples across the
three processes are normal.

This verifies the server-side mechanism and per-host output identity
in one late-arrival case. The reverse-order M5 timings and mixed-load
holdouts remain separate. The [earlier M5 comparison](../flash-next-late-prefill-deferral-m5-20261122/README.md)
used client-side deferral and supplies the frozen prompts and helper
sources.

The initial exact-binary functional run stopped during resident
weight loading when the external pack volume detached. This receipt
comes from a fresh run after the volume remounted. The runtime commit,
driver, publisher and request hashes are checked by the
[replay](replay.py). A replay with one changed output ID and its
recomputed measurement checksum fails the same-host equality check.

```sh
python3 -B benchmarks/results/flash-next-server-prefill-tail-m2-20261122/replay.py
```
