# Flash Next independent reference materialization batches on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD and macOS 27.2.1. Both
diagnostics used the same pinned ordinary MLX-VLM source and packages
as the [previous materialization attempt](../flash-next-ordinary-reference-m5-20261121/README.md#materialization-only-diagnostic).
The instrumentation matched the shapes and dtypes of the real `mx.eval`
arguments against the [pinned lazy graph](../flash-next-reference-parameter-graph-m2-20261121/README.md).
Both began at normal system pressure and above 90% CPU idle.

The first trial completed batches 0–14, including the entire PLE table,
in 4.946 seconds. All 29 pressure samples were normal. Observed peak
MLX allocation was 56.95 GiB and peak process physical footprint was
57.43 GiB. The harness stopped at the selected PLE boundary.

The second trial completed 46 of 73 batches. The last started batch
was 46, with zero-based indexing. Its arrays contain no PLE parameters.
Pressure went from level 1 to level 2. The monitor stopped its owned
process 0.126 seconds after the last recorded batch-start event. At the
warning sample the process had a 90.45 GiB physical footprint and the
host reported a 70.94 GiB compressor. The two counters describe
different scopes. The recorded batch identifies the timing interval
of the sampled warning, and cannot establish the allocation that caused it.

Both trials ended before prefill or QA. The short trial demonstrates
that the entire PLE portion can materialize under normal sampled
pressure. The full attempt shows that later batches can still hit the
host pressure guard. The ordinary reference release gates remain open.

The [sealed evidence](evidence.tar.xz) retains both raw phase streams,
pressure samples, partial and terminal receipts, and hash manifests.
Replay and compare with [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-reference-materialization-batches-m5-20261122/replay.py
```
