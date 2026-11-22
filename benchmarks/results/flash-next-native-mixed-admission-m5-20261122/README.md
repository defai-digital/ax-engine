# Flash Next native mixed admission on M5 Max

A MacBook Pro M5 Max 128 GB with internal NVMe SSD and macOS 27.2.1
ran the codesign-verified `8cb06414c` release-server build with the
pinned resident MXFP4 pack, Rust 1.99.1, and MLX and MLX Metal
0.32.4. The input and publisher file digests match the frozen pack
and workload controls.

The first build, `c7f82e300`, advanced a second 8,192-token prompt to
`Runnable` with a zero cursor while the first was still in prefill.
The resident run configured two sequences and the memory plan had no
explicit sequence cap. The correction applies the measured physical
two-slot envelope even under that memory plan.

The corrected run keeps the second prompt waiting at zero progress
while the first reaches 2,048 and 4,096 tokens. After cancellation
of the first request, the next scheduler step advances the waiting
prompt to 2,048. The promoted request completes with all 96 expected
output IDs. The existing 512/8,192-token short/long control also
completes with all 192 expected IDs and one held decode step. The
five work counters drain after both controls.

All 145 memory-pressure samples report normal pressure. CPU idle
was 91.46% before and 86.10% after. This run supplies functional
evidence for the experimental resident two-slot profile. The mixed
load timing and the single-slot product gates retain their original
measurement requirements.

The archive includes the initial failure, corrected execution,
signed build receipt, and frozen input digests. Replay checks the
waiting cursors, release, route, output IDs, and resource drainage.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
