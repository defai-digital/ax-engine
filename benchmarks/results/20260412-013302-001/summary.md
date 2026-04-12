# AX Engine Benchmark (AX-only)

- Label: `Qwen3.5-35B-A3B-Q4_K_M`
- Model: `models/Qwen3.5-35B-A3B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 541.3 |
| Decode | 42.9 |

- Prefill plan: `mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned pipelined=off`
- Decode plan: `sync=pipelined scratch=hybrid_backend barriers=implicit qkv=split kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/stable q5k=q5_k.ilp4/stable q6k=q6_k.nr2/stable`
- Prefill CBs: `59.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-013302-001/ax.json`
