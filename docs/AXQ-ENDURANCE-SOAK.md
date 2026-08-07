# AXQ 72-hour Endurance Soak

This procedure checks whether a single AX Engine process can keep the pinned
Qwen 3.6 27B AXQ 6-bit model serving for 72 hours without hiding failures by
restarting it. It is an endurance test, not a maximum-throughput benchmark.

The test target is:

| Item | Value |
| --- | --- |
| Model | `AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP` |
| Revision | `8c37715c7b5f5ebca00eda6f73be47116a3e4ebc` |
| AX Engine mode | Native MLX server, one loaded model |
| Request concurrency | 1 |
| Logical-KV geometry | Explicitly pinned at 1,024 blocks × 16 tokens/block |
| Scheduler prefill budget | Explicitly pinned at 2,048 tokens per step; longer text probes are chunked |
| Target duration | 72 hours |
| Status cadence | 4 hours |

This does not promote the current candidate checkpoint. In particular, record
the result of `ax-engine doctor` alongside the soak artifacts; the candidate's
manifest lineage issue described in
[Qwen 3.6 27B AXQ Certification](model-certifications/qwen3.6-27b-axq.md)
remains a separate artifact-integrity gate.

## Decision and test boundary

The decision is deliberately narrow: can one exact AX Engine binary keep one
exact AXQ package loaded and answer a representative low-rate workload for 72
measured hours without a restart or evidence of resource-retention or
performance degradation? It is **not** a maximum-throughput, multi-user,
crash-recovery, or fault-injection test. Keeping those questions separate
prevents queueing pressure or a recovery policy from hiding the endurance
signal.

## Design basis

The procedure follows the general performance-testing sequence of a small
smoke/integration gate, a stable baseline, and then a prolonged soak; this
keeps a test-script or observability defect from consuming a multi-day run.
Grafana's [test taxonomy](https://grafana.com/docs/k6/latest/testing-guides/automated-performance-testing/)
similarly distinguishes smoke, average, stress, and soak workloads, and its
[threshold guidance](https://grafana.com/docs/k6/latest/using-k6/thresholds/)
uses explicit error and percentile criteria rather than a subjective final
readout. The runner applies that idea to client-visible TTFT/decode/prefill,
per-shape samples, lifecycle state, and memory trends.

MLX uses a [unified-memory model](https://ml-explore.github.io/mlx/build/html/)
on Apple silicon, so host wired and IOGPU counters cannot be treated as an AX
process allocation in isolation. This is why the procedure correlates those
host counters with the owned server's RSS and AX/MLX/KV metrics instead of
calling a nonzero or warm cache a leak by itself.

Run it on AC power with no competing AX Engine job and preserve the host state
in the run manifest. The runner captures a host snapshot before model launch,
records host and server snapshots for each cache-capacity probe, and takes its
first resource sample immediately before measured time. This prevents
model-load or warm-up allocation from being mistaken for a later leak.

## Test contract

The runner starts one owned `ax-engine-server` process, records its PID, and
never restarts it. A server exit, PID disappearance, or three consecutive
stream failures ends the run and preserves the evidence. A completed 72-hour
result therefore means that the same loaded server instance survived the whole
measured interval.

It uses a completion-paced, single-client workload: after every request has
fully streamed and the native lifecycle has drained, it waits at least 60
seconds before the next request. This intentionally leaves resource headroom;
it avoids a request-rate catch-up burst after an unusually slow prefill or
decode. This is a reliability experiment under a stable, low duty-cycle load,
not a throughput or saturation experiment. The native MLX streaming route on
this server accepts pre-tokenized input, so the runner uses the model's local
`tokenizer.json` with `add_special_tokens=false`, sends `input_tokens`, and
reconciles that exact client count with the native request event. Client-side
tokenization is deliberately outside the TTFT/prefill timing boundary.

Each 20-request cycle contains:

| Requests | Shape | Purpose |
| ---: | --- | --- |
| 14 | 128-word unique prompt, 96 output tokens | Steady decode and release behavior |
| 3 | 1,024-word unique prompt, 128 output tokens | Routine prefill and allocator churn |
| 2 | 1,024-word shared prefix plus unique tail, 96 output tokens | Intentional prefix-cache/KV reuse |
| 1 | 4,096-word unique prompt, 128 output tokens | Bounded long-prefill coverage |

Warm-up prompts use a disjoint nonce range, so the first measured requests
cannot obtain a false cache hit from warm-up. By default, warm-up is one full
20-request workload cycle, including the long-prefill and shared-prefix
probes; this settles expected first-use KV, allocator, and cache allocations
before measured time begins.

Warm-up alone is not enough to validate a reusable KV cache: it can fill the
logical block ledger while every warm-up request still succeeds. The runner
therefore performs a **cache-capacity rehearsal** after the full warm-up and
before the measured clock starts. It first records an anchor with core
logical-KV occupancy (`ax_engine_step_kv_usage_blocks`,
`ax_runtime_kv_pages_total`, and `ax_runtime_kv_utilization`) and calculates
whether the fresh long prompt must exceed available logical blocks. If it does
not, the runner adds up to four bounded fresh long prompts to fill the cache,
then rechecks the anchor. It fails as inconclusive if it still cannot prove
that reclamation will be exercised. Only then does it send one fresh
long-unique request and one fresh medium-unique request from a second nonce
range. High cache utilization is permitted; the gate is that a fresh prompt
must still make productive prefill progress. By default, more than 64 prefill
steps or cache-only continuations per 1,000 input tokens is a preflight
failure. That limit is deliberately far above normal chunking and is aimed at
a one-token scheduler loop. Each anchor, filler, and probe retains its
host/server snapshot, pass/fail verdict, and concern list. A rejected
preflight writes the ordinary final summary and checkpoint with
`measurement_started=false`; it is not reported as a short endurance run.

Only after this rehearsal passes do the first four measured hours form the
baseline. No performance regression decision is made until that baseline is
complete and it has sufficient same-shape measurements. The runner also
checks whether the first and last quartiles show material growth *and* the
latter half is still rising. A baseline that climbs by at least 1 GiB while
its latter-half slope remains at least 256 MiB/hour is a `watch`, rather than
a reference that could hide a leak.

## Execution gates

Use a short pilot to validate the exact host, artifact, metrics, and lifecycle
contract before starting the irreversible 72-hour evidence run. The pilot is
an integration gate, not a performance certification; its shortened windows
will not necessarily have enough long-prompt samples for a clean baseline.

1. Run `ax-engine doctor --verbose --mlx-model-artifacts-dir "$stage/models/axq"`
   and retain its output. Do not relabel a manifest-lineage warning as a passed
   model certification.
2. Run a 20–30 minute pilot in its own fresh output directory. Before its
   measured interval begins, the runner executes one complete warm-up cycle,
   then the anchored fresh long/medium cache-capacity rehearsal. Each anchor,
   filler, and probe must have a fully drained lifecycle and core
   KV-occupancy telemetry; each request must additionally have a terminal
   response, native prompt length, every target-model KV-memory gauge, and
   bounded prefill fragmentation. It stops before the long run if that
   instrumentation, capacity-pressure proof, or progress contract is absent.
3. Inspect the pilot `summary.json`, `cache_capacity_probe` JSONL events, and
   server log. If the capacity rehearsal and normal workload are clean, start
   the final run in a new empty directory. The final runner owns one PID and
   must never be restarted.

For a no-measurement implementation check, run the same launch contract in a
fresh output directory with `--preflight-only`. It runs the full warm-up and
cache-capacity rehearsal, then stops the owned server cleanly. Its
`preflight_passed` status validates only the gate; it is not a 72-hour result.

For example, a pilot can use short cadence and short reporting without
increasing concurrency:

```bash
pilot_dir="$stage/endurance/axq-6bit-pilot-$(date -u +%Y%m%dT%H%M%SZ)"
env VIRTUAL_ENV="$stage/.venv" PYO3_PYTHON="$stage/.venv/bin/python" \
  PATH="$stage/source/target/release:$stage/.venv/bin:/opt/homebrew/bin:/usr/bin:/bin" \
  caffeinate -dimsu "$stage/.venv/bin/python" \
  "$stage/source/scripts/run_axq_endurance.py" \
  --server "$stage/source/target/release/ax-engine-server" \
  --model-dir "$stage/models/axq" \
  --model-id qwen3.6-27b-axq-6bit \
  --output-dir "$pilot_dir" \
  --duration-hours 0.35 --baseline-hours 0.10 --report-interval-hours 0.05 \
  --request-interval-seconds 30 --resource-interval-seconds 15
```

## What is measured

The runner emits `events.jsonl` continuously and samples the server and host
independently every minute. It retains both wall-clock and monotonic sample
times: a prolonged sampling gap or sleep-like clock divergence makes the
result a `watch`, rather than allowing paused time to look like uninterrupted
endurance. Every four hours it atomically updates
`summary.json` and adds immutable JSON and Markdown checkpoints.

| Concern | Evidence |
| --- | --- |
| Process lifetime | Owned PID, server exit code, server log |
| Client-visible latency | Same-shape client p95 TTFT, p05 decode token/s, p05 effective prefill token/s, E2E latency and output token count |
| Server corroboration | `ax_runtime_ttft_p95_ms`, `ax_runtime_decode_tok_per_sec`, `ax_runtime_error_rate`, queue depth and saturation counters |
| Errors | Client failures, health failures, HTTP 5xx/saturation/backlog counter deltas |
| Native drain/KV | Post-response jobs, pending commands, active streams and buffered events must be zero; target-model logical/physical KV and prefix-cache gauges are retained |
| Logical-KV capacity | Post-warm-up anchor proves the next fresh long prompt exceeds free logical blocks; bounded fillers make pressure explicit if needed; fresh long/medium probes report used/total logical blocks, utilization, native prefill steps, and cache-only continuations |
| Prefix-cache exercise | The dedicated shared-prefix requests retain their physical-cache hit/miss/blocked/store/eviction route decisions separately in each checkpoint |
| AX process memory | Server RSS, MLX active/cache/peak metrics and target-model logical/physical KV and prefix-cache memory |
| macOS memory and confounders | `vm_stat` wired, compressor and active pages; swap; IOGPU driver alloc/in-use memory when exposed; disk, load, and `pmset` thermal state |

`ax_engine_http_requests_in_flight` is recorded but excluded from the drain
gate: `/metrics` correctly observes its own scrape as an in-flight HTTP
request. It would otherwise create a false lifecycle leak signal.

## Interpreting cache and memory

A nonzero KV capacity, paged-pool slab, MLX cache, or prefix-cache payload is
not by itself a memory leak. These allocations may intentionally remain warm.
A memory concern requires evidence such as a post-baseline growth of at least
4 GiB *and* a sustained slope of at least 64 MiB/hour. The lower long-run
slope is intentional: a 4 GiB leak spread through the 68 measured hours after
the baseline is only about 60 MiB/hour. Separately, a host
swap increase of 512 MiB after baseline is a watch condition. A post-response
lifecycle drain timeout or unexpectedly large logical KV after the lifecycle
is drained is separately reported as a KV-retirement concern.

Host wired memory is important but is a host-wide value. A wired-memory trend
is strongest evidence of an AX problem when it correlates with growth in the
owned server RSS or AX/MLX/logical-KV memory metrics; otherwise it is reported
as a host-level observation rather than attributed automatically to AX Engine.
Compressor and swap establish whether host pressure is growing even if an AX
allocation is not visible in RSS. Where macOS exposes IOGPU
`PerformanceStatistics`, the runner also retains driver-wide unified-memory
alloc/in-use counters. They are corroborating host evidence, not a per-process
leak attribution.

The runner additionally compares retained memory using resource samples whose
native lifecycle gauges are all zero. On this Apple unified-memory host,
`vm_stat` wired pages can rise materially while a request is active and fall
once model pages are no longer pinned. The report preserves those active peaks
for diagnosis, but it uses like-for-like quiescent samples for leak slopes and
growth guardrails when enough such samples exist. This avoids calling normal
request-phase pinning a slow leak while still detecting memory that remains
after the lifecycle drains.

## Default guardrails

- Hard failure: server exits/PID disappears, or three consecutive stream
  failures.
- Preflight failure: the fresh cache-capacity rehearsal lacks its native
  lifecycle/KV/occupancy telemetry, cannot prove that the fresh long prompt
  exceeds available logical blocks after its bounded fill phase, or exceeds
  64 prefill steps or cache-only continuations per 1,000 input tokens. This is
  intentionally before measured time: a 72-hour run cannot establish a
  meaningful baseline after cache reclamation has already collapsed into
  token-by-token prefill.
- Preflight-only pass: the full warm-up and cache-capacity rehearsal passed,
  then the process stopped before measured time. This validates the launch
  contract only; it is explicitly not a soak pass.
- Watch: any observed error, failed/inconclusive lifecycle drain, error rate
  over 0.1%, server 5xx/saturation/backlog counter growth, or a configured
  memory/KV guardrail breach. A resource-observation gap beyond the
  cadence-derived threshold or a wall/monotonic divergence is also a watch:
  the request results remain useful, but cannot prove a continuous run.
- Baseline watch: fewer than eight successful same-shape samples for TTFT,
  decode, or effective-prefill evidence; missing native prompt length; or an
  unsettled baseline as defined above. This makes the run **inconclusive**,
  not a clean pass.
- Performance watch: with at least eight baseline and eight current samples of
  the same shape, p95 TTFT over 1.5× baseline, p05 decode below 0.75×
  baseline, or p05 effective prefill below 0.75× baseline. Effective prefill
  is exact native prompt tokens divided by TTFT, so it is an end-to-end
  serving measure rather than a pure kernel microbenchmark.
- Pass: the server completed 72 hours without hard failure and no configured
  watch condition remained. A missing lifecycle, KV, or required client-metric
  series is **inconclusive**, not a clean pass. A thermal warning is retained
  as a performance confounder and must be considered when interpreting a
  token/s or TTFT change.

The values are configurable command-line arguments and are written verbatim to
the run manifest, so later review can distinguish a real regression from a
changed policy.

## Pilot finding on `df-macmini-03`

The short pilot completed its bounded workload and shut down its owned server
cleanly; it did **not** start a 72-hour run. It found a repeatable cache-pressure
path at the pinned 1,024-block configuration. A fresh 4,281-token long prompt
used 3,261 native prefill steps, had 238.8 s TTFT, and achieved 17.9 effective
prefill token/s. A later fresh 1,091-token medium prompt used 836 steps, had
59.8 s TTFT, and achieved 18.3 effective prefill token/s. In contrast, normal
nearby requests used a few prefill steps and roughly 100+ effective prefill
token/s.

The host did not show swap growth or a thermal/performance warning during
those events, and the server stayed healthy. The evidence instead points to
the logical-KV cache being full of reclaimable warm entries and the scheduler
entering a one-token prefill policy before each allocation can reclaim enough
space. The accompanying runner change makes that exact transition a
pre-measurement gate. It is not a memory-leak conclusion, and it is not a
72-hour pass. Do not launch the final endurance run with this configuration
until the new capacity rehearsal passes or the runtime behavior is explicitly
changed and revalidated.

A subsequent `--preflight-only` replay exercised the same transition more
directly. Its bounded filler reached 992/1,024 logical blocks, after which a
fresh 4,280-token long probe failed the guard with 3,770 prefill steps and
3,769 cache-only continuations (880.8 and 880.6 per 1,000 input tokens),
258.998 s TTFT, and 16.5 effective-prefill token/s. It never started the
measured interval. At the failed probe's drained snapshot, AX reported about
20.1 GiB MLX active memory, 20.6 GiB host-resident memory, and a 21.5 GiB MLX
peak; target-model logical and physical KV gauges were about 275 MiB and
435 MiB. Swap stayed at zero. In this isolated run, macOS wired pages were
about 22.9 GiB while the model was loaded and about 2.1 GiB after the owned
server stopped, with no surviving server process. This distinguishes the
configured logical-KV exhaustion from physical exhaustion of the 64 GiB host;
the bounded replay does not establish either a 72-hour memory leak or a
72-hour pass.

## Launch and status

Run on AC power with no other intentional AX Engine workload, and retain the
host's power settings in the manifest. Use `caffeinate` so normal macOS sleep
does not invalidate the server-lifetime result:

```bash
stage=/Users/devop/ax-engine-qwen36-27b-axq-6bit-20260807T015129Z
run_id=$(date -u +%Y%m%dT%H%M%SZ)
output_dir="$stage/endurance/axq-6bit-72h-$run_id"
mkdir -p "$stage/endurance"

nohup env VIRTUAL_ENV="$stage/.venv" PYO3_PYTHON="$stage/.venv/bin/python" \
  PATH="$stage/source/target/release:$stage/.venv/bin:/opt/homebrew/bin:/usr/bin:/bin" \
  caffeinate -dimsu "$stage/.venv/bin/python" \
  "$stage/source/scripts/run_axq_endurance.py" \
  --server "$stage/source/target/release/ax-engine-server" \
  --model-dir "$stage/models/axq" \
  --model-id qwen3.6-27b-axq-6bit \
  --output-dir "$output_dir" \
  --block-size-tokens 16 \
  --total-blocks 1024 \
  --duration-hours 72 \
  --report-interval-hours 4 \
  > "$stage/endurance/launcher-$run_id.log" 2>&1 &
```

The current status is always available without stopping the test:

```bash
python3 -m json.tool "$output_dir/summary.json"
```

The operator should report the four-hour checkpoint's status, elapsed time,
same PID result, request/error counts, **per-shape** p95 TTFT and p05
decode/effective-prefill rates, shared-prefix cache-route evidence, baseline
quality, cache-capacity rehearsal step/continuation ratios and logical-KV
occupancy, drain/KV findings, and resource growth/slope. The raw
`events.jsonl`, `server.log`, and immutable checkpoint files remain the source
of truth for final analysis.
