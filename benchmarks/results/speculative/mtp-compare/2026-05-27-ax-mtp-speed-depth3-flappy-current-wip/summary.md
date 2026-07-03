# MTP Benchmark Summary

Date: 2026-05-27  
Model: `ce011316c6a23387800e825657c9c4df7b9f9435`  
Sampling: temperature=0.6, top_p=0.95, top_k=20  
Generation tokens: 1000  
Repetitions: 1 + 1 warmup  

> **WARNING: dirty build** — this run used uncommitted source changes (base commit `5b8650a00e83`).  
> Numbers are not reproducible from any tagged commit. Do not promote to PERFORMANCE.md  
> or README until a clean build is confirmed.  

## Decode throughput (tok/s, median across cases)

| Suite | AX depth | AX direct | AX MTP | AX MTP accept rate |
|---|---:|---:|---:|---:|
| flappy | 3 | — | 62.8 | 90.6% |

## Artifact provenance

- `benchmarks/results/mtp-compare/2026-05-27-ax-mtp-speed-depth3-flappy-current-wip/flappy/flappy.json` — 4 prompt cases, AX direct + n-gram rows

## Reproduction

```bash
python3 scripts/bench_mtp_compare.py \
  --model-dir /Users/akiralam/.cache/huggingface/hub/models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed/snapshots/ce011316c6a23387800e825657c9c4df7b9f9435 \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all
```

