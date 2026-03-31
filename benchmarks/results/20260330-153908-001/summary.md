# AX Engine Benchmark (AX-only)

- Label: `f32-gpu-fix`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 615.4 |
| Decode | 57.0 |

- Prefill plan: `mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned pipelined=on`
- Decode plan: `sync=pipelined scratch=hybrid_backend barriers=smart qkv=split kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/profile_preferred q5k=q5_k.base/stable q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `33.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-153908-001/ax.json`
