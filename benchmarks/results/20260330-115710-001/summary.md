# AX Engine Benchmark (AX-only)

- Label: `final`
- Model: `models/meta-llama-3-70b-instruct.Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 23.3 |
| Decode | 4.1 |

- Prefill plan: `mode=gpu_batch kv=f16 f16_io=on pair=off qkv=fused batch_simd=off split_rope=on attn=local window=0 wo_in=attn_f32 attn_route=fa2_simd_hd128/experimental q5k_prefill=base`
- Decode plan: `sync=pipelined scratch=gpu_shared barriers=smart qkv=fused kv=f16 attn=f16kv_hd128/stable q4k=q4_k.nr2/profile_preferred q5k=q5_k.base/stable q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `2.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-115710-001/ax.json`
