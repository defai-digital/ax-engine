# Flash Next independent MLX-VLM reference on M5 Max

The ordinary MLX-VLM runner at `6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb`
used MLX and MLX Metal 0.32.2 to load the pinned Flash Next MXFP4
pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. It ran on a
MacBook Pro M5 Max with 128 GB unified memory, internal SSD and
macOS 27.2.1. The launcher validated its bound inputs and code
before startup. [Source hashes](source.json) identify the frozen
runner, suite, matrix input and monitor.

The [partial result](partial-result.json) contains 3 of 105 planned
QA rows and no timing matrix. The monitor stopped the owned process
after 1,203.5 seconds without a new result. [Terminal status](status.json)
records an exit by SIGTERM and an incomplete qualification.
The [147 resource samples](samples.jsonl.gz) include 42 normal
pressure and 105 pressure-warning observations. The peak system
compressor size was 78,706,638,848 bytes. These measurements record
host conditions during the run and do not establish the cause of the
stall.

This run supplies no fresh integrity, quality or reference-relative
performance result for the native release board.

## Four-case recovery diagnostic

The subsequent [heartbeat attempt](critical-pressure/status.json)
used the same pinned ordinary MLX-VLM runtime without a wired-residency
override. It began after sampled normal pressure and 92.93% CPU idle.
The [partial receipt](critical-pressure/partial-result.json) records
materialization of 2,326 arrays in 22.383 seconds. The first QA case
started and emitted zero tokens before the monitor observed critical
system pressure and stopped the process. The full ordinary repeat
therefore did not launch.

The [resource samples](critical-pressure/samples.jsonl) record normal
pressure, a 26.75 GB compressor and 55.43 GB process RSS before the last
sample. The last sample recorded pressure level 4, a 106.18 GB
compressor and 21.09 GB swap in use. The host-wide counters describe
conditions around the transition. Per-allocation attribution would
require a separate measurement.

## Materialization-only diagnostic

The [guarded loader diagnostic](materialization-only/status.json)
reused the frozen independent source and MLX 0.32.2 profile. It stopped
on the first sampled system pressure warning. Before its stop, the
runner began explicit weight evaluation but did not reach the end of
materialization or the first-prefill boundary. Its [phase trace](materialization-only/phase-events.jsonl)
records 89 ms for lazy model construction. The last sample came 14.09 s
after explicit materialization began, with system pressure level 2,
79.20 GB occupied by the compressor and 22.65 GB reference process RSS.
The run started with normal pressure and 90.20% CPU idle.

This observation places the warning before first forward in this
attempt. The host-wide compressor count cannot attribute individual
allocations. The diagnostic stops before the first prompt by design
and has no qualification results.
