# AX Engine Benchmark (AX-only)

- Label: `meta-llama-3.1-8b-instruct-q5_k_m`
- Model: `models/meta-llama-3.1-8b-instruct-q5_k_m.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 623.3 |
| Decode | 53.3 |

- Prefill plan: `mode=gpu_batch kv=f16 f16_io=off pair=off qkv=fused batch_simd=off split_rope=on attn=local window=0 wo_in=attn_f32 attn_route=ax_bc64/experimental q5k_prefill=base`
- Decode plan: `sync=pipelined scratch=gpu_shared barriers=implicit qkv=fused kv=f16 attn=f16kv_hd128/stable q4k=q4_k.nr2/stable q5k=q5_k.ilp4/profile_preferred q6k=q6_k.nr2/stable`
- Prefill CBs: `2.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-083626-001/ax.json`
