# AX Engine Benchmark (AX-only)

- Label: `qwen35-27b-q4km`
- Model: `models/Qwen3.5-27B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `5s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 163.3 |
| Decode | 15.2 |

- Prefill plan: `mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned pipelined=off`
- Decode plan: `sync=single_cb scratch=hybrid_backend barriers=implicit qkv=split kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.ilp4/profile_preferred q5k=q5_k.ilp4/stable q6k=q6_k.base/profile_preferred`
- Prefill CBs: `65.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260404-212504-001/ax.json`
