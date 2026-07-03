# MTP Benchmark Summary

Date: 2026-05-25  
Model: `ce011316c6a23387800e825657c9c4df7b9f9435`  
Sampling: temperature=0.6, top_p=0.95, top_k=20  
Generation tokens: 1000  
Repetitions: 5 + 1 warmup  

MTPLX reference: version=0.3.7, hardware=Apple M5 Max 128GB

## Decode throughput (tok/s, median across cases)

| Suite | AX direct | AX MTP | AX MTP accept rate | MTPLX reference | MTPLX accept rate | MTPLX depth |
|---|---:|---:|---:|---:|---:|---:|
| flappy | — | 37.0 | 83.5% | 47.7 | 69.4% | 3 |

## Artifact provenance

- `benchmarks/results/mtp-compare/2026-05-25-rejection-sampling-v1/flappy/flappy.json` — 4 prompt cases, AX direct + n-gram rows
- MTPLX reference: `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-compare/2026-05-23-mtplx-ref/mtplx.json`

## Reproduction

```bash
python3 scripts/bench_mtp_compare.py \
  --model-dir /Users/akiralam/.cache/huggingface/hub/models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed/snapshots/ce011316c6a23387800e825657c9c4df7b9f9435 \
  --mtplx-results <path-to-mtplx-results.json> \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all
```

