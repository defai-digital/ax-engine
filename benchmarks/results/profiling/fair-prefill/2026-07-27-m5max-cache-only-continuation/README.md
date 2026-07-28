# Fair-prefill cache-only continuation A/B

This is a targeted engineering A/B for scheduler-split text prefill. It is not
a publication benchmark and does not compare AX Engine with another runtime.

## Environment

- Host: `mbp-m5` / Apple M5 Max / macOS 26.5.2
- Source baseline: `40743ada4a466c075e0c5db6feb5ec535e9c4ce9`
- MLX: 0.32.0 from the repository PyPI wheel
- Prefix cache: disabled (`AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0`)
- MTP: disabled
- Prompt: deterministic synthetic valid token IDs, 2,048 tokens
- Warmup/repetitions: 1 / 5
- Cooldown: 15 seconds between measured processes
- Surface: production `MlxRunner`, without HTTP or tokenizer overhead

The baseline and candidate were built in the same isolated clone and Cargo
target directory. The candidate was the baseline plus the cache-only
continuation patch. Binary SHA-256:

- Baseline:
  `a05e7fedd90d246bcf905d4b92a2b11491201f70f05ec83b8631531b07ff0c91`
- Candidate:
  `11252957ea8cc9549ff377f38a10a640e7388addecb35a133cc9f225f1fece98`

## Results

| Model / sampling | Quantum | Baseline TTFT | Candidate TTFT | TTFT delta | Prefill delta | Token check |
|---|---:|---:|---:|---:|---:|---|
| Qwen3.5 9B / greedy | 64 | 2306.53 ms | 2249.16 ms | −2.49% | +2.55% | 5/5 exact |
| Qwen3.5 9B / greedy | 256 | 992.92 ms | 986.51 ms | −0.65% | +0.65% | 5/5 exact |
| Qwen3.5 9B / top-k 20 | 64 | 1730.19 ms | 1666.36 ms | −3.69% | +3.83% | candidate repeat exact |
| Gemma 4 12B / greedy | 64 | 2790.02 ms | 2749.33 ms | −1.46% | +1.48% | 5/5 exact |
| Qwen3.5 9B / greedy, unsplit control | full | 695.13 ms | 695.01 ms | −0.02% | +0.02% | 5/5 exact |

The candidate reported 31 cache-only continuations for quantum 64 and seven
for quantum 256. The unsplit control reported zero. This proves the new route
engaged and that direct single-item prefill stayed neutral.

For sampled requests the old implementation sampled and discarded one token
per non-terminal execution item, advancing request RNG state. Candidate output
therefore is not expected to equal the incorrect baseline. Two independent
candidate runs produced the same five tokens:
`[605, 360, 5166, 516, 62]`.

Raw process output is retained beside this file. Load time is reported in each
file but excluded from TTFT and throughput comparisons.
