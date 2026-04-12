# AX Engine Benchmark (AX-only)

- Label: `Qwen3-Coder-30B-A3B-Q4_K_M`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 1089.2 |
| Decode | 76.2 |

- Prefill plan: `mode=gpu_batch`
- Decode plan: `sync=pipelined scratch=gpu_shared barriers=smart qkv=split kv=f16 attn=splitk_hd128/profile_preferred q4k=q4_k.nr2/stable q5k=q5_k.ilp4/stable q6k=q6_k.nr2/stable`
- Prefill CBs: `2.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260411-203523-001/ax.json`
