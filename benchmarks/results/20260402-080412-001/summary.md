# AX Engine Benchmark (AX-only)

- Label: `codex-qwen35-shared-gate-fix`
- Model: `models/Qwen3.5-35B-A3B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `1`
- Cooldown: `0s`

| Phase | Median tok/s |
|---|---:|
| Prefill | 51.0 |
| Decode | 5.9 |

- Prefill plan: `mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned pipelined=on`
- Decode plan: `sync=single_cb scratch=hybrid_backend barriers=smart qkv=split kv=f16 attn=splitk_hd256/profile_preferred q4k=q4_k.nr2/profile_preferred q5k=q5_k.base/profile_preferred q6k=q6_k.nr2/profile_preferred`
- Prefill CBs: `24389.0`
- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260402-080412-001/ax.json`
