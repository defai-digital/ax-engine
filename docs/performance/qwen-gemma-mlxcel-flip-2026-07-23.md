# Qwen 3 + Gemma 4 vs mlxcel v0.4.2 flip decision

Date: 2026-07-23

Status: implementation complete; competitive flip rejected (`not_yet`)

Primary host: `AKMBPM5MAXx` (`applegpu_g17s`, 128 GB unified memory)

## Decision

Do not claim or perform the Qwen 3 + Gemma 4 competitive flip yet.

The implementation plan is complete: model-scoped lifecycle and admission,
adaptive long-prefill isolation, exact Qwen/Gemma decode coalescing, memory
accounting, focused MTP coalescing, and the repeatable peer harness are all in
place. The final 24-trial campaign passed every comparison-contract and
availability check, but AX missed the locked performance gates in all four
scenarios.

The machine-readable result is
[the flip decision](../../benchmarks/results/profiling/qwen-gemma-mlxcel-flip/2026-07-23-primary-m5max-v2/flip-decision.json).
This is a completed `not_yet` decision, not a public performance win.

## Locked comparison

The campaign compared:

- AX Engine runtime commit
  `c02dd7e90ede6d21afe382ab28cc49d76c4f8482`, serving both models from one
  process;
- official mlxcel v0.4.2 commit
  `1b9a0018d01d40ba7314c80060d8758533ed6b0e`, serving one model per process;
- the same Qwen 3.5 9B 4-bit and Gemma 4 12B 4-bit package paths and package
  identity hashes;
- a 96 GiB aggregate memory cap: 96 GiB for the AX process and 48 GiB for each
  of the two mlxcel processes;
- raw `/v1/completions` SSE requests with
  `temperature=0`, `top_p=1`, `top_k=0`, and `seed=0`;
- three fresh-process, cache-isolated repetitions per target and scenario,
  alternating target order with a 15-second cooldown.

Raw completions avoid chat-template differences. The harness also requires
the paired requests to report identical prompt-token counts. In particular,
the final contract measured 34 Qwen tokens and 13,826 Gemma long-prefill
tokens on both targets.

The threshold manifest was locked before the final campaign and is identified
by SHA-256
`9a6be4274c7ebf152e5d757aa2f765f63b8c42aa6aa84ad55e2c4b722c76b192`:
[Qwen/Gemma flip gates](../../benchmarks/manifests/qwen_gemma_flip_gates.v1.json).

| Required gate | Threshold |
| --- | ---: |
| Median aggregate output throughput | AX / mlxcel >= 1.15x |
| Median p95 TTFT | AX / mlxcel <= 0.90x |
| Median interactive p95 stream gap | AX / mlxcel <= 0.90x |
| Absolute median interactive p95 stream gap | AX <= 50 ms |
| AX request, HTTP 503, and lifecycle errors | zero |

## Scenario matrix

| Scenario | Workload |
| --- | --- |
| S0 | One 192-token Qwen interactive stream |
| S1 | The Qwen stream plus a sibling 13,826-token Gemma prefill |
| S2 | The Qwen stream while Gemma unloads and reloads |
| S3 | Two Qwen and two Gemma interactive streams, staggered by 50 ms |

Each value below is the median of the three per-trial values. TTFT and stream
gap are the median per-trial p95.

| Scenario | AX tok/s | mlxcel tok/s | Throughput | AX TTFT ms | mlxcel TTFT ms | TTFT | AX gap ms | mlxcel gap ms | Gap | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 72.37 | 94.71 | 0.764x | 268.83 | 66.93 | 4.016x | 12.57 | 10.96 | 1.147x | fail |
| S1 | 6.46 | 19.65 | 0.329x | 28,145.04 | 9,094.58 | 3.095x | 35.13 | 35.77 | 0.982x | fail |
| S2 | 68.50 | 74.19 | 0.923x | 269.84 | 67.34 | 4.007x | 15.43 | 11.41 | 1.352x | fail |
| S3 | 55.35 | 110.81 | 0.499x | 2,020.60 | 163.22 | 12.380x | 67.30 | 27.67 | 2.432x | fail |

All 24 trials returned zero request errors and zero HTTP 503 responses. AX
also completed every S2 lifecycle operation without an error. S0, S1, and S2
remained below the absolute 50 ms stream-gap cap; S3 did not.

Evidence:

- [campaign ledger](../../benchmarks/results/profiling/qwen-gemma-mlxcel-flip/2026-07-23-primary-m5max-v2/campaign.json)
- [decision report](../../benchmarks/results/profiling/qwen-gemma-mlxcel-flip/2026-07-23-primary-m5max-v2/flip-decision.md)
- [all 24 raw artifacts](../../benchmarks/results/profiling/qwen-gemma-mlxcel-flip/2026-07-23-primary-m5max-v2/artifacts)

## What the implementation established

The runtime work remains useful despite the peer decision:

1. Model-scoped admission and lifecycle keep sibling models available during
   load, unload, drain, and replacement. S2 confirms that behavior under the
   peer harness.
2. Adaptive one-token prefill turns remove the historical second-scale Qwen
   streaming stall. The focused M5 replay reduced the maximum gap from
   1,570.11 ms to 30.03 ms with token-identical output.
3. Production row-exact decode coalescing is exact across the Qwen and Gemma
   certification matrices. It improved the internal sequential baseline by
   1.12-1.33x for Qwen and 1.06-1.19x for Gemma.
4. Memory metrics now separate process measurements from per-model
   attribution and expose KV topology and rollback behavior. The experimental
   physical KV pool stayed off because its M5 results were neutral or slower.
5. Gemma assistant-MTP coalesced verification passed its independent route and
   output-identity checks, but MTP was not part of this peer workload and is
   not credited to the comparison.

The detailed implementation and internal evidence are in
[Multi-model execution priorities](../designs/multimodel-execution-priorities-2026-07-23.md).

## Interpretation

The peer campaign separates correctness progress from competitive
performance:

- S0 is already 24% behind mlxcel in throughput and about 4x slower in TTFT.
  The next competitive lever therefore has to improve the base request path;
  multi-model scheduling alone cannot produce a flip.
- S1 shows the current latency-throughput trade. Adaptive isolation keeps the
  stream gap bounded and nearly matches mlxcel's gap, but the one-token
  prefill quantum leaves AX at one third of mlxcel's aggregate throughput.
- S2 proves lifecycle isolation, but its throughput, TTFT, and stream gap still
  miss their relative gates.
- S3 is the largest serving gap: roughly half the aggregate throughput,
  12.38x TTFT, and a 67.30 ms interactive p95 gap. Internal row-exact wins do
  not yet translate into sustained mixed-family serving wins.

## Next evidence-gated work

Do not relax the checked-in gates. The next iteration should:

1. Split first-request TTFT into tokenization, prefill, first decode,
   materialization, and stream-write time on the same fresh-process contract.
   This will distinguish load-time graph preparation from a steady-state
   execution deficit.
2. Profile execution-arbiter hold time and actual row-exact cohort engagement
   in S3. Improve batch formation or cross-family turn policy only after that
   trace identifies the dominant loss.
3. Replace the fixed one-token long-prefill policy only with a mechanism that
   preserves the absolute 50 ms stream-gap cap. The retained 4-, 8-, and
   16-token quantum experiments are negative evidence against merely raising
   the chunk size.
4. Rerun the same three-repetition S0-S3 campaign on the M5 Max and require
   every locked gate to pass before changing `not_yet` to `flip`.

## Reproduction

With the target manifests' binary and model environment variables set on the
M5 Max:

```bash
python3 scripts/run_qwen_gemma_flip_campaign.py \
  --target benchmarks/manifests/targets/ax-qwen-gemma-m5max.json \
  --target benchmarks/manifests/targets/mlxcel-v0.4.2-qwen-gemma-m5max.json \
  --repetitions 3 \
  --cooldown 15 \
  --workers 16 \
  --timeout 600 \
  --output-dir benchmarks/results/profiling/qwen-gemma-mlxcel-flip/primary \
  --keep-going

python3 scripts/summarize_qwen_gemma_flip_campaign.py \
  benchmarks/results/profiling/qwen-gemma-mlxcel-flip/primary/campaign.json \
  --gates benchmarks/manifests/qwen_gemma_flip_gates.v1.json \
  --output benchmarks/results/profiling/qwen-gemma-mlxcel-flip/primary/flip-decision.json \
  --markdown benchmarks/results/profiling/qwen-gemma-mlxcel-flip/primary/flip-decision.md \
  --report-only
```
