# MTP Benchmark Summary

Date: 2026-05-27  
Model: `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`  
Sampling: temperature=0.6, top_p=0.95, top_k=20  
Generation tokens: 1000  
Repetitions: 2 + 1 warmup  

> **WARNING: dirty build** — this run used uncommitted source changes (base commit `4063079a3026`).  
> Numbers are not reproducible from any tagged commit. Do not promote to PERFORMANCE.md  
> or README until a clean build is confirmed.  

## Decode throughput (tok/s, median across cases)

| Suite | AX depth | AX direct | AX MTP | AX MTP accept rate |
|---|---:|---:|---:|---:|
| flappy | default | — | 83.9 | 81.9% |

## Artifact provenance

- `benchmarks/results/mtp-compare/2026-05-27-ax-ngram-think-gate-qwen3-4b-smoke/flappy/flappy.json` — 4 prompt cases, AX direct + n-gram rows

## Reproduction

```bash
python3 scripts/bench_mtp_compare.py \
  --model-dir /Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all
```

