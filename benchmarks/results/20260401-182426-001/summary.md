# AX Engine Benchmark (AX-only)

- Label: `qwen35-4b-dim-aware`
- Model: `models/Qwen3.5-4B-Q8_0.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `30s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 1057.2 |
| Decode | 50.9 |

- Prefill plan: `mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned pipelined=on`
- Decode plan: `sync=pipelined scratch=hybrid_backend barriers=smart qkv=split kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/profile_preferred q5k=q5_k.nr2/stable q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `33.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260401-182426-001/ax.json`
