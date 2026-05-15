# Serving Benchmarks

AX Engine uses two benchmark layers:

- `scripts/bench_mlx_inference_stack.py` is the model-inference comparison
  layer. It uses fixed token shapes and a required `mlx_lm.benchmark` baseline
  to compare raw MLX-family runtime behavior.
- `scripts/bench_ax_serving.py` is the online serving layer. It uses a prompt
  corpus and `/v1/generate/stream` to measure user-visible latency,
  throughput, request scheduling, and SLO goodput.

Do not merge these outputs into one unlabeled table. A fast prefill/decode row
does not prove production serving behavior, and a serving latency row does not
replace same-shape model-runtime comparison.

## Market-Standard Shape

Serving benchmarks should report:

- prompt corpus identity and hash
- prompt category mix and input/output token distributions
- concurrency and request-rate policy
- TTFT percentiles
- client TPOT percentiles
- streaming step-interval percentiles
- end-to-end latency percentiles
- request throughput
- output-token throughput
- queue delay
- goodput against explicit TTFT, TPOT, and E2E SLO thresholds
- error count and route/runtime identity when available
- runtime route-decision counters for promotion gates that must prove a
  specific path was exercised

This matches the direction used by serving-oriented tools such as vLLM serving
benchmarks, GenAI-Perf, and MLPerf-style load tests: the benchmark must measure
request mixes and latency distributions, not only prefill/decode throughput.

## Running A Serving Benchmark

Start an AX server first, then run:

```text
python3 scripts/bench_ax_serving.py \
  --base-url http://127.0.0.1:8080 \
  --model-id qwen3_dense \
  --corpus benchmarks/corpora/serving/smoke.jsonl \
  --input-kind tokens \
  --requests 12 \
  --warmup-requests 2 \
  --concurrency 2 \
  --slo-ttft-ms 2000 \
  --slo-tpot-ms 100 \
  --slo-e2e-ms 15000 \
  --output benchmarks/results/serving/$(date +%F)-qwen3-dense-smoke.json
```

Validate the artifact before citing it:

```text
python3 scripts/check_ax_serving_benchmark_artifact.py \
  benchmarks/results/serving/<artifact>.json \
  --min-requests 12 \
  --min-concurrency 2 \
  --require-slo \
  --min-goodput-ratio 0.95
```

For long-prompt serving claims, also require the corpus shape explicitly:

```text
python3 scripts/check_ax_serving_benchmark_artifact.py \
  benchmarks/results/serving/<artifact>.json \
  --min-input-tokens-p95 8192 \
  --require-slo
```

For disk-durable prefix-cache promotion claims, require a route counter in
addition to latency and corpus-shape gates:

```text
python3 scripts/check_ax_serving_benchmark_artifact.py \
  benchmarks/results/serving/<artifact>.json \
  --min-input-tokens-p95 8192 \
  --require-route-decision-min ax_mlx_prefix_cache_disk_hits=1 \
  --require-slo
```

This gate proves the serving artifact exercised the disk-cache path instead of
only measuring a generic long-prompt run.

Render a review report after validation:

```text
python3 scripts/render_ax_serving_benchmark_report.py \
  benchmarks/results/serving/<artifact>.json \
  --min-input-tokens-p95 8192 \
  --require-route-decision-min ax_mlx_prefix_cache_disk_hits=1 \
  --require-slo \
  --output benchmarks/results/serving/<artifact>.md
```

Use `--request-rate-rps` when the claim is open-loop serving behavior. Without
it, the harness uses closed-loop concurrency.

The checked-in smoke corpus is intentionally small:

- `short_chat`
- `coding_medium`
- `rag_long`
- `structured_output`

It is a harness smoke and review template, not a final market-representative
claim. A public serving claim should use a larger corpus with representative
question categories, short and long prompts, fixed token provenance, and a
published prompt-mix table.

## Artifact Contract

`bench_ax_serving.py` writes `ax.serving_benchmark.v1` JSON:

```json
{
  "schema_version": "ax.serving_benchmark.v1",
  "methodology": {
    "scope": "online_serving_streaming_latency",
    "endpoint": "/v1/generate/stream",
    "timing_scope": "client_observed_sse"
  },
  "load": {
    "concurrency": 2,
    "request_rate_rps": null,
    "warmup_requests": 2,
    "measured_requests": 12
  },
  "summary": {
    "request_throughput_rps": 0.0,
    "output_token_throughput_tok_s": 0.0,
    "ttft_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
    "client_tpot_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
    "e2e_latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
    "route_decisions": {
      "ax_mlx_prefix_cache_disk_hits": 0
    },
    "goodput": {
      "ratio": 0.0,
      "ttft_slo_ms": 2000.0,
      "client_tpot_slo_ms": 100.0,
      "e2e_slo_ms": 15000.0
    }
  }
}
```

Definitions:

- `ttft_ms`: client request start to the first non-empty `step` SSE event.
- `client_tpot_ms`: stream completion minus TTFT, divided by output tokens
  after the first token.
- `stream_step_interval_ms`: observed intervals between non-empty SSE `step`
  events. If one server step emits multiple tokens, the harness does not invent
  synthetic per-token timestamps.
- `route_decisions`: numeric route/runtime counters aggregated from final
  `response.route.crossover_decisions` SSE payloads.
- `goodput`: measured requests that succeeded and met every configured SLO.

## Recommended Rollout

Use this sequence before publishing production serving performance:

1. Run the smoke corpus on one model and inspect the artifact manually.
2. Expand the corpus to at least short chat, long answer, coding, RAG,
   structured JSON/tool output, and shared-prefix prompts.
3. Run closed-loop concurrency at 1/2/4/8 requests.
4. Run open-loop request-rate sweeps until goodput drops below the chosen SLO.
5. Repeat on at least one smaller Apple Silicon host before making broad
   performance claims.
