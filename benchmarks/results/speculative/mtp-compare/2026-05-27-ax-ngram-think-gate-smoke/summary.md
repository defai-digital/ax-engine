# MTP Benchmark Summary

Date: 2026-05-27  
Model: `c000ac2c2057d94be3fa931000c31723aac53282`  
Sampling: temperature=0.6, top_p=0.95, top_k=20  
Generation tokens: 1000  
Repetitions: 2 + 1 warmup  

## Decode throughput (tok/s, median across cases)

| Suite | AX depth | AX direct | AX MTP | AX MTP accept rate |
|---|---:|---:|---:|---:|
| flappy | default | — | 22.2 | — |

## Artifact provenance

- `benchmarks/results/mtp-compare/2026-05-27-ax-ngram-think-gate-smoke/flappy/flappy.json` — 4 prompt cases, AX direct + n-gram rows

## Reproduction

```bash
python3 scripts/bench_mtp_compare.py \
  --model-dir /Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/c000ac2c2057d94be3fa931000c31723aac53282 \
  --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all
```

