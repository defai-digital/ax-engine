# Fair chunked-prefill interleave — serving A/B (Phase 3)

Wall-clock serving A/B quantifying the fair-prefill trade-off, to inform (not
decide) the default-on question. Host: Apple M3 Max, MLX 0.32.0.

## Method

- Model: `mlx-community/Qwen3-4B-4bit` (dense), one server process per arm.
- Harness: `scripts/bench_ax_serving.py` over `/v1/generate/stream`.
- Workload: 32 requests (4 warmup), **closed-loop concurrency 8**, `input_tokens`
  prompts cycling lengths `[128, 512, 1024, 2048, 256, 768, 1536, 384]` (long
  prompts create prefill contention against active decode), `max_output_tokens=96`.
- **OFF** arm: server defaults (a lone prefill may take up to `--max-batch-tokens`
  = 2048 tokens in one step).
- **ON** arm: `--multi-prefill-fair --max-prefill-tokens-per-request-per-step 256`
  (a prefill is chunked to ≤256 tokens/step even uncontended).
- Artifacts: `benchmarks/results/serving/2026-07-18-fair-prefill-ab/`.

## Result

| metric | OFF | ON | Δ |
| --- | ---: | ---: | ---: |
| **decode-step interval p99 (ms)** | 1846 | 709 | **−61.6%** |
| **decode-step interval max (ms)** | 7180 | 2979 | **−58.5%** |
| decode-step interval p95 (ms) | 108 | 119 | +10.2% |
| decode-step interval p50 (ms) | 31.5 | 33.5 | +6.5% |
| ttft p50 (ms) | 37860 | 30761 | −18.8% |
| ttft p95 (ms) | 113753 | 128476 | +12.9% |
| ttft p99 (ms) | 131560 | 139269 | +5.9% |
| queue delay p50 (ms) | 35177 | 26532 | −24.6% |
| e2e p50 (ms) | 39717 | 34971 | −11.9% |
| output tok/s (aggregate) | 10.35 | 10.46 | +1.1% |
| errors | 0/32 | 0/32 | — |

## Interpretation

Fair-prefill does exactly what the mechanism predicts, and the balance is better
than the naive "TPOT better, TTFT worse" framing:

- **Decode smoothness — big win.** The decode-step tail (p99 **−62%**, max **−59%**)
  collapses: capping a prefill to 256 tokens/step stops a long prompt from bolting
  ~2000 tokens of prefill onto a step the decode cohort is sharing. Median step
  interval is ~flat (+6%, fair-mode bookkeeping).
- **TTFT median + queueing — also better** (p50 −19%, queue delay −25%): fairer
  step composition reduces head-of-line blocking, so most requests start sooner.
- **TTFT tail — the expected cost** (p95/p99 +6–13%): the longest prompts take more
  steps to finish prefill under the cap.
- **Throughput neutral** (+1%), zero errors.

## Caveats

- **One workload / model / concurrency.** Directional evidence, not a full sweep.
- **The workload is deliberately heavy** (closed-loop c=8, long prompts, small
  model → ~35 s queue delay). Overload *amplifies* the fair-prefill effect; on
  light load the decode-step-tail win shrinks. The absolute latencies here are not
  representative of a well-provisioned deployment — only the **relative** A/B is.

## Verdict

The evidence **supports fair-prefill for decode-smoothness-priority serving**
(interactive / streaming SLOs on inter-token latency): a ~60% cut in decode-step
tail latency, better median TTFT, throughput-neutral, for a modest TTFT-tail cost.
Whether to flip the default is still a **product SLO call** — a batch/offline
deployment optimizing TTFT-tail or total throughput might keep it off. This A/B
turns that call from a guess into a data-backed decision. Recommended next step if
default-on is pursued: confirm on 2–3 realistic load points (not just this
overload case) and pick the default `max_prefill_tokens_per_request_per_step` from
the sweep.

## Reproduce

```bash
Q4=~/models/models--mlx-community--Qwen3-4B-4bit/snapshots/*/
DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/mlx/0.32.0/lib \
  ./target/release/ax-engine-server --port 8080 --model-id qwen3-4b \
  --mlx --mlx-model-artifacts-dir "$Q4" \
  [--multi-prefill-fair --max-prefill-tokens-per-request-per-step 256]   # ON arm only
python3 scripts/bench_ax_serving.py --base-url http://127.0.0.1:8080 \
  --model-id qwen3-4b --corpus corpus.jsonl --input-kind tokens \
  --requests 32 --warmup-requests 4 --concurrency 8 --output arm.json
```
