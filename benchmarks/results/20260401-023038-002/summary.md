# AX Engine Benchmark (AX-only)

- Label: `gemma-3-27b-it-Q4_K_M`
- Model: `models/gemma-3-27b-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `15s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 188.8 |
| Decode | 21.1 |

- Prefill plan: `mode=gpu_batch kv=f16 f16_io=off pair=on qkv=fused batch_simd=off split_rope=off attn=cached window=0 wo_in=attn_f32 attn_route=cache_fa2_simd_hd128/profile_preferred`
- Decode plan: `sync=pipelined scratch=gpu_shared barriers=smart qkv=fused kv=f16 attn=f16kv_hd128_n2/profile_preferred q4k=q4_k.nr2/profile_preferred q5k=q5_k.base/stable q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `1.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260401-023038-002/ax.json`
