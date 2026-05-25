# MTP Benchmark Summary

Date: 2026-05-25
Model: `2293023686ca76e7f4a08220ea08f34251dd9c2d`
Sampling: temperature=0.6, top_p=0.95, top_k=20
Generation tokens: 1000
Repetitions: 1 + 1 warmup

## Decode throughput (tok/s, median across cases)

| Suite | AX direct | AX MTP | AX MTP accept rate |
|---|---:|---:|---:|
| flappy | — | 35.2 | 90.4% |

## Artifact provenance

- `benchmarks/results/mtp-compare/2026-05-25-quality-depth3-no-full-logits-eval-smoke/flappy/flappy.json` — 4 prompt cases, AX direct + n-gram rows

## Reproduction

```bash
python3 scripts/bench_mtp_compare.py \
  --model-dir /Users/akiralam/.cache/huggingface/hub/models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality/snapshots/2293023686ca76e7f4a08220ea08f34251dd9c2d \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all
```
