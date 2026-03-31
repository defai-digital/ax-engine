# AX Engine Benchmark (AX-only)

- Label: `qwen35-9b-final`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `15s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 322.3 |
| Decode | 54.5 |

- Prefill plan: `mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned`
- Decode plan: `sync=pipelined scratch=hybrid_backend barriers=implicit qkv=split kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/profile_preferred q5k=q5_k.base/stable q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `105.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-195724-001/ax.json`
