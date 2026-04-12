# AX Engine Benchmark (AX-only)

- Label: `Qwen3.5-4B-Q8_0`
- Model: `models/Qwen3.5-4B-Q8_0.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 48.7 |
| Decode | 48.3 |

- Prefill plan: `mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned pipelined=on`
- Decode plan: `sync=single_cb scratch=hybrid_backend barriers=implicit qkv=split kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/profile_preferred q5k=q5_k.base/profile_preferred q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `512.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260404-124420-001/ax.json`
