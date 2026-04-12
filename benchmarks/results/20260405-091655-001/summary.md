# AX Engine Benchmark (AX-only)

- Label: `Qwen3.5-9B-Q4_K_M`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `30s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 550.4 |
| Decode | 51.1 |

- Prefill plan: `mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned pipelined=off`
- Decode plan: `sync=single_cb scratch=hybrid_backend barriers=implicit qkv=split kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/profile_preferred q5k=q5_k.nr2/stable q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `57.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-091655-001/ax.json`
