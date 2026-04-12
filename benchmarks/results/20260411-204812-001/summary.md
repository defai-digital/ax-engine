# AX Engine Benchmark (AX-only)

- Label: `Qwen3-Coder-30B-A3B-Q8_0`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 490.5 |
| Decode | 42.5 |

- Prefill plan: `mode=gpu_batch`
- Decode plan: `sync=pipelined scratch=gpu_shared barriers=explicit qkv=split kv=f16 attn=splitk_hd128/profile_preferred q4k=q4_k.nr2/stable q5k=q5_k.ilp4/stable q6k=q6_k.nr2/stable`
- Prefill CBs: `1.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260411-204812-001/ax.json`
