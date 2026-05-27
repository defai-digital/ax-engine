# MTP Benchmark Summary

Date: 2026-05-27  
Model: `ce011316c6a23387800e825657c9c4df7b9f9435`  
Model bundle: `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`  
Sampling: temperature=0.6, top_p=0.95, top_k=20  
Generation tokens: 1000  
Repetitions: 5 + 1 warmup  
AX depth: 3  
Build: release server binary from `/private/tmp/ax-engine-v5-mtp-clean`, `git_tracked_dirty: false`  

MTPLX reference: version=0.3.7, hardware=Apple M5 Max 128GB  

## Decode throughput (tok/s, median across cases)

| Suite | AX depth | AX direct | AX MTP | AX MTP accept rate | MTPLX reference | MTPLX accept rate | MTPLX depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| long_code | 3 | - | 52.5 | 90.4% | 59.8 | 99.6% | 3 |

## Per-case AX rows

| Case | Prompt tokens | AX MTP tok/s | Accepted / drafted |
|---|---:|---:|---:|
| long_code_c_audit | 721 | 52.3 | 3545 / 3860 |
| long_code_api_audit | 852 | 51.6 | 3499 / 3912 |
| long_code_test_stubs | 714 | 52.7 | 3516 / 3899 |
| long_code_sql_schema | 451 | 53.3 | 3525 / 3915 |

## Artifact provenance

- `benchmarks/results/mtp-compare/2026-05-27-ax-mtp-speed-depth3-long-code-clean-nobuild/long_code/long_code.json` - 4 prompt cases, AX MTP-only row, depth=3
- MTPLX reference: `benchmarks/results/mtp-compare/2026-05-27-mtplx-apple-to-apple-d3/mtplx.json`

## Reproduction

```bash
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir /Users/akiralam/.cache/huggingface/hub/models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed/snapshots/ce011316c6a23387800e825657c9c4df7b9f9435 \
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
  --output benchmarks/results/mtp-compare/2026-05-27-ax-mtp-speed-depth3-long-code-clean-nobuild/long_code/long_code.json
```
