# MTP Benchmark Summary

Date: 2026-05-27  
Model: `2293023686ca76e7f4a08220ea08f34251dd9c2d`  
Sampling: temperature=0.6, top_p=0.95, top_k=20  
Generation tokens: 1000  
Repetitions: 5 + 1 warmup  

## Decode throughput (tok/s, median across cases)

| Suite | AX depth | AX direct | AX MTP | AX MTP accept rate |
|---|---:|---:|---:|---:|
| flappy | 2 | — | 36.2 | 96.0% |

## Artifact provenance

- `benchmarks/results/mtp-compare/2026-05-27-ax-mtp-quality-depth2-flappy-clean/flappy/flappy.json` — 4 prompt cases, AX MTP-only rows

## Reproduction

```bash
python3 scripts/bench_mtp_compare.py \
  --model-dir /Users/akiralam/.cache/huggingface/hub/models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality/snapshots/2293023686ca76e7f4a08220ea08f34251dd9c2d \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all
```
