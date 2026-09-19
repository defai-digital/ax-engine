# Tiel MTP prefill timing

`AX_MLX_PREFILL_TIME_DEBUG=1` adds phase timing to the native MTP history
prefill path used by the two Tiel MXFP4 MTP packs. Set it before starting
the process; its value is cached. It is disabled by default.

This diagnostic adds clock readings and stderr output, but no evaluation
barriers, tensor operations, cache-policy changes, or sampling changes. When
disabled, the new phase marks do not read the clock. The existing flag also
enables other verbose prefill/decode diagnostics, so **enabled runs are not
throughput acceptance measurements**.

Each `AX_PREFILL_TIME_DEBUG mtp_chunk` line identifies `prompt_len`,
`chunk_len`, the prompt-relative `chunk_offset`, and one of `cache_only`,
`cache_mid`, `retained_mid`, or `retained_final`. Durations are integer
microseconds:

| Field | Observed work |
| --- | --- |
| `graph_build_us` | Forward graph construction and cache position advance |
| `history_prep_us` | Retained-history slicing and collecting chunk references |
| `eval_wait_us` | Existing intermediate evaluation or asynchronous submission call |
| `sample_us` | Final chunk's existing evaluation and token sampling/readback |
| `retained_mat_us` | History concatenation, token-vector construction, and existing materialization |
| `clear_cache_us` | Existing allocator cache cleanup call |
| `chunk_total_us` | Elapsed chunk wall time up to logging |

`eval_wait_us` can measure asynchronous submission; it is not necessarily GPU
completion time. `sample_us` includes work built lazily in earlier phases;
it is not just the sampling kernel. Unused fields are zero. Phase sums need
not equal the total because of bookkeeping and timer overhead.

The `mtp_final` row records total elapsed time, chunk size/count, retained
chunk count, history cap/start, and cache-only prefix/end. It describes this
prefill function, not complete request TTFT or MTP warmup/decode. Logs contain
geometry and timing rather than prompt text or token IDs.

## Measured scope

The [M5 Max evidence](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19-prefill/README.md)
records phase attribution, default-off regression controls, matched MTPLX
comparisons, and rejected performance experiments. None of that diagnostic
campaign's runtime experiments met its 10% TTFT target; they were reverted.
The diagnostic does not change MTP defaults, certify performance, or establish
a hardware/driver root cause.

The observed latency depends strongly on inter-request idle time. Keep idle
time, input token arrays, fixed output count, warmups, cache policy, model
revision, and timing boundaries identical when comparing engines. Report
native-call-to-last-token completion, TTFT, and decode excluding the first
emitted batch separately. A shorter idle interval is a workload change, not
an engine optimization.

## Residency policy for the audited M5 Max exports

The subsequent [residency campaign](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19-wired/README.md)
measures a separate runtime change: releasing wired residency after model load.
It applies automatically only when the audited Tiel/Cyber export metadata
fingerprints match, the CPU is Apple M5 Max with at least 128 GiB, expert
streaming is inactive, and no numeric `AX_MLX_WIRED_LIMIT_SCALE` override is set.
Unknown metadata or hardware retain the previous wiring policy. The startup
trace event identifies application as `tiel-auto-no-wire-v1`.

This reduces idle-to-submit waiting. It does not add GPU keepalive work or
change sampling, buffer-cache limits, allocation limits, or MTP certification.
To retain the previous wiring, set `AX_MLX_WIRED_LIMIT_SCALE=0.9` before process
startup. `0` explicitly disables wiring. Unwired buffers can be evicted under
competing memory pressure; the evidence covers one resident model in isolation,
not contention or mixed-model co-residency. MLX wired limits have process scope.

Re-exported or modified metadata will not match the audited fingerprints and
will keep previous wiring until separately evaluated. The metadata check is a
performance-policy selector, not authentication of every weight byte.
