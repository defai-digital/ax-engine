# MTP Benchmark Summary

Date: 2026-05-27  
Model: `2293023686ca76e7f4a08220ea08f34251dd9c2d`  
Sampling: temperature=0.6, top_p=0.95, top_k=20  
Generation tokens: 1000  
Repetitions: 1 + 1 warmup  

> **WARNING: dirty build** — this run used uncommitted source changes (base commit `5b8650a00e83`).  
> Numbers are not reproducible from any tagged commit. Do not promote to PERFORMANCE.md  
> or README until a clean build is confirmed.  

MTPLX reference: version=0.3.7, hardware=Apple M5 Max 128GB

## Decode throughput (tok/s, median across cases)

| Suite | AX depth | AX direct | AX MTP | AX MTP accept rate | MTPLX reference | MTPLX accept rate | MTPLX depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| flappy | 3 | — | 39.0 | 98.1% | 43.0 | 99.4% | 3 |

## Artifact provenance

- `benchmarks/results/mtp-compare/2026-05-27-ax-mtp-quality-depth3-flappy-skip-snapshot-d3-smoke/flappy/flappy.json` — 4 prompt cases, AX direct + n-gram rows
- MTPLX reference: `benchmarks/results/mtp-compare/2026-05-27-mtplx-apple-to-apple-d3/mtplx.json`

## Reproduction

```bash
python3 scripts/bench_mtp_compare.py \
  --model-dir /Users/akiralam/.cache/huggingface/hub/models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality/snapshots/2293023686ca76e7f4a08220ea08f34251dd9c2d \
  --mtplx-results <path-to-mtplx-results.json> \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all
```

