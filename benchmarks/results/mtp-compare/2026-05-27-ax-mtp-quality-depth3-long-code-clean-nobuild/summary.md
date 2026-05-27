# MTP Benchmark Summary

Date: 2026-05-27  
Model: `2293023686ca76e7f4a08220ea08f34251dd9c2d`  
Model bundle: `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality`  
Sampling: temperature=0.6, top_p=0.95, top_k=20  
Generation tokens: 1000  
Repetitions: 5 + 1 warmup  
AX depth: 3  
Build: release server binary from `/private/tmp/ax-engine-v5-mtp-clean`, `git_tracked_dirty: false`  

MTPLX reference: version=0.3.7, hardware=Apple M5 Max 128GB  

## Decode throughput (tok/s, median across cases)

| Suite | AX depth | AX direct | AX MTP | AX MTP accept rate | MTPLX reference | MTPLX accept rate | MTPLX depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| long_code | 3 | - | 34.7 | 91.5% | 43.2 | 99.7% | 3 |

## Per-case AX rows

| Case | Prompt tokens | AX MTP tok/s | Accepted / drafted |
|---|---:|---:|---:|
| long_code_c_audit | 721 | 36.6 | 3610 / 3845 |
| long_code_api_audit | 852 | 33.7 | 3507 / 3908 |
| long_code_test_stubs | 714 | 34.4 | 3522 / 3901 |
| long_code_sql_schema | 451 | 35.0 | 3560 / 3870 |

## Artifact provenance

- `benchmarks/results/mtp-compare/2026-05-27-ax-mtp-quality-depth3-long-code-clean-nobuild/long_code/long_code.json` - 4 prompt cases, AX MTP-only row, depth=3
- MTPLX reference: `benchmarks/results/mtp-compare/2026-05-27-mtplx-apple-to-apple-d3/mtplx.json`

## Reproduction

```bash
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /Users/akiralam/.cache/huggingface/hub/models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality/snapshots/2293023686ca76e7f4a08220ea08f34251dd9c2d \
  --prompt-source real \
  --real-prompt-suite benchmarks/prompts/mtp-suites/long_code.jsonl \
  --generation-tokens 1000 \
  --repetitions 5 \
  --cooldown 15 \
  --ax-ngram-accel \
  --ax-sampling '{"temperature": 0.6, "top_p": 0.95, "top_k": 20}' \
  --skip-mlx-lm \
  --ax-mtp-max-depth 3 \
  --no-thinking \
  --no-build-ax-engine \
  --output benchmarks/results/mtp-compare/2026-05-27-ax-mtp-quality-depth3-long-code-clean-nobuild/long_code/long_code.json
```
