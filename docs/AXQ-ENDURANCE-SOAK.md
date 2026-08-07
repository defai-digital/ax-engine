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
| Target duration | 72 hours |
| Status cadence | 4 hours |

This does not promote the current candidate checkpoint. In particular, record
the result of `ax-engine doctor` alongside the soak artifacts; the candidate's
manifest lineage issue described in
[Qwen 3.6 27B AXQ Certification](model-certifications/qwen3.6-27b-axq.md)
remains a separate artifact-integrity gate.

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
decode.

Each 20-request cycle contains:

| Requests | Shape | Purpose |
| ---: | --- | --- |
| 14 | 128-word unique prompt, 96 output tokens | Steady decode and release behavior |
| 3 | 1,024-word unique prompt, 128 output tokens | Routine prefill and allocator churn |
| 2 | 1,024-word shared prefix plus unique tail, 96 output tokens | Intentional prefix-cache/KV reuse |
| 1 | 4,096-word unique prompt, 128 output tokens | Bounded long-prefill coverage |

Warm-up prompts use a disjoint nonce range, so the first measured requests
cannot obtain a false cache hit from warm-up. The first four hours form the
baseline; no performance regression decision is made until that baseline is
complete.

## What is measured

The runner emits `events.jsonl` continuously and samples the server and host
independently every minute. Every four hours it atomically updates
`summary.json` and adds immutable JSON and Markdown checkpoints.

| Concern | Evidence |
| --- | --- |
| Process lifetime | Owned PID, server exit code, server log |
| Client-visible latency | Same-shape client p95 TTFT, p05 decode token/s, p05 effective prefill token/s, E2E latency and output token count |
| Server corroboration | `ax_runtime_ttft_p95_ms`, `ax_runtime_decode_tok_per_sec`, `ax_runtime_error_rate`, queue depth and saturation counters |
| Errors | Client failures, health failures, HTTP 5xx/saturation/backlog counter deltas |
| Native drain/KV | Post-response jobs, pending commands, active streams and buffered events must be zero; target-model logical/physical KV and prefix-cache gauges are retained |
| AX process memory | Server RSS, MLX active/cache/peak metrics and model KV/prefix-cache memory |
| macOS memory | `vm_stat` wired pages, compressor/active pages, swap, IOGPU wired limit, disk and load |

`ax_engine_http_requests_in_flight` is recorded but excluded from the drain
gate: `/metrics` correctly observes its own scrape as an in-flight HTTP
request. It would otherwise create a false lifecycle leak signal.

## Interpreting cache and memory

A nonzero KV capacity, paged-pool slab, MLX cache, or prefix-cache payload is
not by itself a memory leak. These allocations may intentionally remain warm.
A memory concern requires evidence such as a post-baseline growth of at least
4 GiB *and* a sustained slope of at least 256 MiB/hour. Separately, a host
swap increase of 512 MiB after baseline is a watch condition. A post-response
lifecycle drain timeout or unexpectedly large logical KV after the lifecycle
is drained is separately reported as a KV-retirement concern.

Host wired memory is important but is a host-wide value. A wired-memory trend
is strongest evidence of an AX problem when it correlates with growth in the
owned server RSS or AX/MLX memory metrics; otherwise it is reported as a host
level observation rather than attributed automatically to AX Engine.

## Default guardrails

- Hard failure: server exits/PID disappears, or three consecutive stream
  failures.
- Watch: any observed error, failed/inconclusive lifecycle drain, error rate
  over 0.1%, server 5xx/saturation/backlog counter growth, or a configured
  memory/KV guardrail breach.
- Performance watch: with at least eight baseline and eight current samples of
  the same shape, p95 TTFT over 1.5× baseline, p05 decode below 0.75×
  baseline, or p05 effective prefill below 0.75× baseline. Effective prefill
  is exact native prompt tokens divided by TTFT, so it is an end-to-end
  serving measure rather than a pure kernel microbenchmark.
- Pass: the server completed 72 hours without hard failure and no configured
  watch condition remained. A missing lifecycle/metric series is
  **inconclusive**, not a clean pass.

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

nohup caffeinate -dimsu "$stage/.venv/bin/python" \
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
same PID result, request/error counts, p95 TTFT, p05 decode rate, drain/KV
findings, and resource growth/slope. The raw `events.jsonl`, `server.log`, and
immutable checkpoint files remain the source of truth for final analysis.
