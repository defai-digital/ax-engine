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

Run it on AC power with no competing AX Engine job and preserve the host state
in the run manifest. The runner captures a host snapshot before model launch,
then another immediately after warm-up, so model-load allocation is not
mistaken for a later leak.

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
not a throughput or saturation experiment.

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
before measured time begins. The first four hours then form the baseline; no
performance regression decision is made until that baseline is complete and
it has sufficient same-shape measurements. The runner also checks whether the
first and last quartiles show material growth *and* the latter half is still
rising. A baseline that climbs by at least 1 GiB while its latter-half slope
remains at least 256 MiB/hour is a `watch`, rather than a reference that could
hide a leak.

## Execution gates

Use a short pilot to validate the exact host, artifact, metrics, and lifecycle
contract before starting the irreversible 72-hour evidence run. The pilot is
an integration gate, not a performance certification; its shortened windows
will not necessarily have enough long-prompt samples for a clean baseline.

1. Run `ax-engine doctor --verbose --mlx-model-artifacts-dir "$stage/models/axq"`
   and retain its output. Do not relabel a manifest-lineage warning as a passed
   model certification.
2. Run a 20–30 minute pilot in its own fresh output directory. Confirm that the
   owned server becomes ready, every response has a terminal event and native
   prompt length, the lifecycle gauges drain, and model-KV gauges are present.
3. Inspect the pilot `summary.json` and server log. If its evidence is clean
   enough to exercise all probes, start the final run in a new empty directory.
   The final runner owns one PID and must never be restarted.

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

## Default guardrails

- Hard failure: server exits/PID disappears, or three consecutive stream
  failures.
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
quality, drain/KV findings, and resource growth/slope. The raw `events.jsonl`,
`server.log`, and immutable checkpoint files remain the source of truth for
final analysis.
