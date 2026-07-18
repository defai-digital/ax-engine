# Fair chunked-prefill interleave — serving A/B (Phase 3)

Wall-clock serving A/B quantifying the fair-prefill trade-off, to inform (not
decide) the default-on question. Host: Apple M3 Max, MLX 0.32.0.

## Method

- Model: `mlx-community/Qwen3-4B-4bit` (dense), one server process per arm.
- Harness: `scripts/bench_ax_serving.py` over `/v1/generate/stream`.
- Two load points, same corpus (`input_tokens` prompts cycling lengths
  `[128, 512, 1024, 2048, 256, 768, 1536, 384]` — long prompts create prefill
  contention against active decode), `max_output_tokens=96`:
  - **Overloaded:** 32 requests, closed-loop concurrency 8 (p50 queue delay ~35 s).
  - **Realistic:** 24 requests, open-loop **0.3 rps** arrivals (p50 queue delay
    ~4 ms — non-saturated). This point tests whether the win is a saturation
    artifact.
- **OFF** arm: server defaults (a lone prefill may take up to `--max-batch-tokens`
  = 2048 tokens in one step).
- **ON** arm: `--multi-prefill-fair --max-prefill-tokens-per-request-per-step 256`
  (a prefill is chunked to ≤256 tokens/step even uncontended).
- Artifacts: `benchmarks/results/serving/2026-07-18-fair-prefill-ab/`.

## Result

**Overloaded (c=8 closed-loop):**

| metric | OFF | ON | Δ |
| --- | ---: | ---: | ---: |
| **decode-step interval p99 (ms)** | 1846 | 709 | **−61.6%** |
| **decode-step interval max (ms)** | 7180 | 2979 | **−58.5%** |
| decode-step interval p95 (ms) | 108 | 119 | +10.2% |
| decode-step interval p50 (ms) | 31.5 | 33.5 | +6.5% |
| ttft p50 (ms) | 37860 | 30761 | −18.8% |
| ttft p95 / p99 (ms) | 113753 / 131560 | 128476 / 139269 | +12.9% / +5.9% |
| queue delay p50 (ms) | 35177 | 26532 | −24.6% |
| output tok/s | 10.35 | 10.46 | +1.1% |

**Realistic (0.3 rps open-loop, non-saturated — the caveat test):**

| metric | OFF | ON | Δ |
| --- | ---: | ---: | ---: |
| **decode-step interval p99 (ms)** | 2331 | 42 | **−98.2%** |
| **decode-step interval max (ms)** | 11490 | 582 | **−94.9%** |
| decode-step interval p95 (ms) | 125.7 | 33.1 | −73.6% |
| decode-step interval p50 (ms) | 23.0 | 20.4 | −11.3% |
| ttft p50 (ms) | 10757 | 4083 | −62.0% |
| ttft p95 / p99 (ms) | 86276 / 89190 | 82406 / 89105 | −4.5% / −0.1% |
| e2e p50 (ms) | 30629 | 5633 | −81.6% |
| output tok/s | 11.94 | 12.41 | +3.9% |
| errors (both points) | 0/32, 0/24 | 0/32, 0/24 | — |

## Interpretation

Fair-prefill does what the mechanism predicts, and **the realistic point is
stronger than the overloaded one — the opposite of a saturation artifact**:

- **Decode smoothness — the win, and it grows at realistic load.** Capping a
  prefill to 256 tokens/step stops a long prompt from bolting ~2000 tokens of
  prefill onto a step the decode cohort is sharing. Decode-step tail: overloaded
  p99 −62% / max −59%; realistic **p99 −98% / max −95%** (a 2.3 s worst decode
  step becomes 42 ms). At realistic load even p50 improves (−11%).
- **TTFT median — better at both** (overloaded −19%, realistic **−62%**): fairer
  step composition removes head-of-line blocking behind a monopolizing prefill.
- **TTFT tail — the cost only appears under overload** (+6–13% at c=8) and
  **vanishes at realistic load** (−4.5% / −0.1%). So the trade-off the overloaded
  run suggested is largely a saturation effect, not intrinsic.
- **Throughput neutral-to-positive** (+1% / +4%), zero errors across both points.

## Caveats

- **Overload-artifact objection: tested and refuted.** The non-saturated point
  (p50 queue delay ~4 ms) shows the decode-tail win is real and *larger*, not an
  artifact of the deliberately-overloaded c=8 run.
- **Still one model, one prompt mix.** Qwen3-4B-4bit only. A production default
  flip should confirm on 1–2 more families (a dense mid-size + a hybrid) and a
  lighter prompt-length mix.
- **Absolute tail TTFT stays high** (~89 s p99) in both arms — a property of the
  heavy 2048-token prompts in this corpus, which fair mode neither fixes nor
  worsens. Only the **relative** A/B is the claim.

## Verdict

The evidence now **strongly supports default-on for decode-smoothness / streaming
SLOs** (inter-token latency): decode-step tail cut 62–98%, median TTFT better,
TTFT-tail neutral at realistic load, throughput neutral-to-positive, zero errors —
across both an overloaded and a non-saturated load point. The residual reason it
is not auto-flipped here is scope, not doubt: it remains a **product SLO decision**
(a batch/throughput-first deployment could still prefer off) and wants a
1–2-model confirmation before the shipped default changes. The knob is a
kill-switchable config flag (`--multi-prefill-fair`), so flipping the default is
low-risk once signed off. Recommended default cap: start at
`max_prefill_tokens_per_request_per_step = 256` (this A/B), tune per the
confirmation models.

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
