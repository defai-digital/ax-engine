# AX Engine Benchmark (AX-only)

- Label: `final`
- Model: `models/Qwen3.5-27B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `30s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 192.7 |
| Decode | 16.4 |

- Prefill plan: `mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned pipelined=on`
- Decode plan: `sync=pipelined scratch=hybrid_backend barriers=smart qkv=split kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/profile_preferred q5k=q5_k.base/stable q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `65.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260331-033122-001/ax.json`
