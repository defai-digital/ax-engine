# AX Engine Benchmark (AX-only)

- Label: `Gemma4-26B-A4B-Q5_K_M`
- Model: `models/gemma-4-26B-A4B-it-Q5_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 904.4 |
| Decode | 73.3 |

- Prefill plan: `mode=gpu_batch`
- Decode plan: `sync=single_cb scratch=gpu_shared barriers=implicit qkv=fused kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/stable q5k=q5_k.ilp4/stable q6k=q6_k.nr2/stable`
- Prefill CBs: `1.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-004209-001/ax.json`
