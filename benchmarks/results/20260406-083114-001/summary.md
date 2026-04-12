# AX Engine Benchmark (AX-only)

- Label: `gemma-4-31B-it-Q4_K_M`
- Model: `models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 9.2 |
| Decode | 9.9 |

- Prefill plan: `mode=gpu_batch`
- Decode plan: `sync=single_cb scratch=gpu_shared barriers=implicit qkv=split kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/profile_preferred q5k=q5_k.ilp4/profile_preferred q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `513.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260406-083114-001/ax.json`
