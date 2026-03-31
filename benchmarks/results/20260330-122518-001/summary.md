# AX Engine Benchmark (AX-only)

- Label: `final-cooled`
- Model: `models/gemma-3-12b-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 380.9 |
| Decode | 35.9 |

- Prefill plan: `mode=gpu_batch kv=f16 f16_io=off pair=on qkv=fused batch_simd=off split_rope=off attn=cached window=0 wo_in=attn_f32 attn_route=cache/stable`
- Decode plan: `sync=pipelined scratch=gpu_shared barriers=smart qkv=fused kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/profile_preferred q5k=q5_k.base/stable q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `1.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-122518-001/ax.json`
