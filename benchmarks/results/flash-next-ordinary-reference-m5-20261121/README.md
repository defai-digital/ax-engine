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
