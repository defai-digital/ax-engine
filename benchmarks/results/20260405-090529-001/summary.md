# AX Engine Benchmark (AX-only)

- Label: `Qwen3.5-4B-Q8_0`
- Model: `models/Qwen3.5-4B-Q8_0.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 778.1 |
| Decode | 35.8 |

- Prefill plan: `mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned pipelined=off`
- Decode plan: `sync=single_cb scratch=hybrid_backend barriers=implicit qkv=split kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/stable q5k=q5_k.nr2/stable q6k=q6_k.nr2/stable`
- Prefill CBs: `57.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-090529-001/ax.json`
