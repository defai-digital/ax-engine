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

This hardware guard limits automatic performance tuning, not the availability
of wired memory. AX Engine's native host requirement remains Apple M2 or newer
on macOS 26 or newer. MLX's `set_wired_limit` is a general Metal memory control,
not an M5-only or Tiel-only API; its [documented OS requirement](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)
is macOS 15 or newer. Other supported Macs retain AX's existing wiring policy
and can use the explicit override. Whether unwiring improves latency needs
separate evidence for the hardware, memory capacity and model workload. API
availability does not establish that every model fits or benefits.

This reduces idle-to-submit waiting. It does not add GPU keepalive work or
change sampling, buffer-cache limits, allocation limits, or MTP certification.
To retain the previous wiring, set `AX_MLX_WIRED_LIMIT_SCALE=0.9` before process
startup. `0` explicitly disables wiring. Unwired buffers can be evicted under
memory pressure; setting zero does not disable GPU execution or move inference
to the CPU. The setting controls residency, not the execution backend.
The speed comparison covers one resident model in
isolation. A separate [co-residency probe](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19-coexistence/README.md)
checks these two packs together with up to 32 GiB of retained competing memory.
Its 48 GiB attempt triggers a compressor-growth guard and is not a passed
sustained-load result. Mixed families and high-pressure/endurance behavior
remain unqualified. MLX wired limits have process scope.

Re-exported or modified metadata will not match the audited fingerprints and
will keep previous wiring until separately evaluated. The metadata check is a
performance-policy selector, not authentication of every weight byte.

The [four-host resident comparison](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-20-peer/RESULTS.md)
adds M2 Ultra, M3 Ultra and M4 Pro controls. Those measurements remain separate
from the automatic M5 policy and do not qualify high-pressure residency.

## Model residency is different from wired memory

| Control | What it changes |
| --- | --- |
| Expert streaming Auto/On | AX can load expert layers on demand instead of retaining every expert tensor in the MLX allocator. |
| Expert streaming Off | AX loads and retains the full model for repeated requests; required-stream packs still reject Off. |
| Wired limit zero | Removes the OS residency lock. It does not unload the model, enable expert streaming, or switch inference to the CPU. |

In the matched resident benchmark, **both AX and MTPLX load once per process**
and keep the model loaded across warmups and measured requests. Model loading
is outside the timer. Cold KV means no reuse of earlier prompt state; it does
not mean cold or reloaded model weights. Unwired allocations remain eligible
for OS eviction under pressure, independently of engine-managed expert paging.

## Expert streaming on smaller hosts

The 48 GiB Auto reserve is an admission accounting allowance, not a separate
48 GiB allocation or an enforced empty-memory reservation. It is separate from
wired-memory control. These Tiel
exports estimate about 20.51 GiB for complete residency. On a 64 GiB Mac mini,
20.51 + 48 exceeds capacity, so the peer-campaign Auto build paged experts
even though one model's observed resident allocation was about 21.25 GiB. That was the measured admission
policy, not evidence that the GPU cannot execute the model. The bounded
default-session exception below addresses that case.

The earlier full-resident comparison explicitly selected `--stream-experts off` in
the server, or `Session(..., mlx_stream_experts="off")` in Python. Python now
honors `AX_STREAM_EXPERTS` when the argument is omitted; an explicit argument
wins. Earlier Python builds always installed Auto and masked the environment.
`off` still fails closed for packs with `required=true`. Model capacity,
context and competing allocations must fit; no Auto reserve or required-pack
guard was changed to produce those peer benchmark results.

Hardware probes fall back to `/usr/sbin/sysctl` when the command cannot be
found through `PATH`, fails, or returns empty output. This keeps the memory
guard effective in service environments with a restricted `PATH`. Earlier
builds used a PATH-only memory probe, which could silently disable this
residency optimization. The [PATH regression evidence](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19-path/README.md)
compares both launch environments on the two audited packs. Unknown or
unparseable memory still does not qualify for automatic unwiring.

To inspect skipped application in native CLI/server logs, enable
`RUST_LOG=ax_engine_mlx::runner=debug`. Startup debug events distinguish an
operator override or active expert streaming, and unknown/unreadable export
metadata. When the audited export matches but the host is outside Apple M5
Max with at least 128 GiB, the skip is a warning: existing wiring stays in
place, and Apple M4 Pro with exactly 64 GiB uses `tiel-session-resident-v1`
for expert residency. Hardware events include detected memory and CPU brand
and contain no model paths or metadata hashes.
They describe the existing guards and do not change admission. Non-Tiel loads
can also emit the metadata-skip event at debug level. Embedded library users
need a tracing subscriber; the environment variable alone does not install one.


## Inspect the load decision without loading weights

Run `ax-engine doctor --mlx-model-artifacts-dir /path/to/model --verbose --json` to
inspect `model_artifacts.expert_stream`. Doctor resolves the default SDK/session
Auto decision at the time of the probe, including optional plans inferred from native
expert tensor roles when `ax_expert_stream.json` is absent. `enabled` means
paging would be selected; it does not mean weights have been loaded.

The report includes the selected mode, plan source, decision reason, full
weight estimate, physical RAM and the 48 GiB Auto accounting allowance.
Physical RAM is not live free memory; the allowance is not an allocation.
Unknown host capacity is labelled unknown rather than described as a fit.
Required packs still reject explicit Off.

`resident_estimate` applies the shared server footprint formula to the plan's
full weight estimate and the **default session KV pool**. It does not inspect
a running server's custom pool. `assumed_prefill_chunk` identifies the default
prefill bound used by the session exception. `kv_pool_bytes: null` means unknown geometry,
not zero KV memory; the legacy fallback floor is retained. This baseline
excludes other loaded models, current host pressure and independently measured
prefill peaks. It is not a whole-system fit guarantee or a replacement for
server admission. The bounded exception below uses the same known-KV formula;
the general baseline alone does not grant residency.


## Idle GPU-touch screening

The [four-host native idle screening](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-20-idle/README.md)
compares wired, explicitly unwired and disclosed same-thread GPU-touch arms.
All arms retain full model weights. Touch lowers median TTFT in the eight
short coding cells, but M2 Cyber-Tiel decode regresses about 3.7% versus
unwired, exceeding the 3% regression screen. No server keepalive or automatic
policy expansion was introduced. Server latency, longer idle, long context,
pressure and lifecycle qualification remain separate work. Energy saving is
not the acceptance gate; latency, sustained throughput and stability are.


## Bounded default-session residency

On **M4 Pro exactly 64 GiB**, SDK, Python and server sessions can retain the
two audited Tiel export configurations automatically. This exception requires:

- The same exact config and export metadata fingerprints as the audited packs;
  family/name matching alone does not qualify. These fingerprints identify
  the export configuration; they do not authenticate every weight byte.
- A known session KV pool of 1..16,384 tokens and prefill chunk of 1..2,048.
- Normal OS memory pressure and no more than 512 MiB current MLX active memory.
- Known KV geometry. Full weights plus the shared KV/runtime estimate and
  existing active memory must fit the lower of the Metal recommended working
  set and 48 GiB (75% of physical capacity).

Here **48 GiB is the model budget cap**, leaving at least 16 GiB outside that
budget. It is distinct from the general Auto rule's **48 GiB added allowance**.
Neither number is a live free-memory reading or a hard allocator guarantee.
The decision is made before loading weights; it does not change residency
mid-generation. Existing server add/replace preflight remains enforced.

Required packs and explicit On/Off keep their precedence. Unknown probes,
geometry, larger/custom session bounds and other hardware keep the general
Auto decision. Raw `load_weights` and runner constructors without an admitted
session budget also retain that general rule. Doctor describes the default
session, not every low-level caller or an already-running server.

The [default-server acceptance record](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-20-default/README.md)
is separate from the earlier explicit-MTP MTPLX comparison. This change does
not promote MTP, extend automatic unwiring beyond its M5 target, or add an idle
keepalive. Those settings require their own evidence.
