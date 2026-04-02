# AX Engine Benchmark (AX-only)

- Label: `gemma3-12b-rerun`
- Model: `models/gemma-3-12b-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `30s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 388.0 |
| Decode | 34.7 |

- Prefill plan: `mode=gpu_batch kv=f32 f16_io=off pair=on qkv=fused batch_simd=off split_rope=off attn=cached window=0 wo_in=attn_f32 attn_route=cache/stable`
- Decode plan: `sync=pipelined scratch=gpu_shared barriers=smart qkv=fused kv=f32 attn=hd256/stable q4k=q4_k.nr2/profile_preferred q5k=q5_k.nr2/stable q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `1.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260402-080357-001/ax.json`
