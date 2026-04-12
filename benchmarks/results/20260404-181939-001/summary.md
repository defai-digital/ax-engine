# AX Engine Benchmark (AX-only)

- Label: `llama31-8b-q5km`
- Model: `models/meta-llama-3.1-8b-instruct-q5_k_m.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `5s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 465.3 |
| Decode | 38.9 |

- Prefill plan: `mode=gpu_batch kv=f16 f16_io=off pair=off qkv=fused batch_simd=off split_rope=on attn=local window=0 wo_in=attn_f32 attn_route=ax_bc64/experimental q5k_prefill=base`
- Decode plan: `sync=pipelined scratch=gpu_shared barriers=implicit qkv=fused kv=f16 attn=f16kv_hd128/stable q4k=q4_k.nr2/stable q5k=q5_k.ilp4/profile_preferred q6k=q6_k.nr2/stable`
- Prefill CBs: `2.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260404-181939-001/ax.json`
