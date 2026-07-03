# MTP Benchmark Summary

Date: 2026-05-25  
Model: `ce011316c6a23387800e825657c9c4df7b9f9435`  
Sampling: temperature=0.6, top_p=0.95, top_k=20  
Generation tokens: 1000  
Repetitions: 5 + 1 warmup  

## Decode throughput (tok/s, median across cases)

| Suite | AX direct | AX MTP | AX MTP accept rate |
|---|---:|---:|---:|
| flappy | — | 41.7 | 91.4% |

## Artifact provenance

- `benchmarks/results/mtp-compare/2026-05-25-speed-depth2-smoke/flappy/flappy.json` — 4 prompt cases, AX direct + n-gram rows

## Reproduction

```bash
python3 scripts/bench_mtp_compare.py \
  --model-dir /Users/akiralam/.cache/huggingface/hub/models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed/snapshots/ce011316c6a23387800e825657c9c4df7b9f9435 \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all
```

