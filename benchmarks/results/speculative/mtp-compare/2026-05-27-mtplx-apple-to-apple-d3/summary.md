# MTPLX Prompt-Parity MTP Benchmark Summary

Date: 2026-05-27  
Runner: `scripts/bench_mtplx_prompt_suites.py`  
MTPLX: 0.3.7  
Profile: `sustained`  
Depth: 3  
Sampling: temperature=0.6, top_p=0.95, top_k=20  
Generation tokens: 1000  
Repetitions: 5 + 1 warmup  
Cooldown: 15s  

This run uses the AX Engine repo prompt suites under
`benchmarks/prompts/mtp-suites/`, not MTPLX's built-in `flappy` or `long_code`
suites. The goal is prompt-parity comparison with AX Engine MTP artifacts.
AX rows are measured through the AX Engine server SSE runner; these MTPLX rows
are measured through the local MTPLX runtime depth-sweep runner.

| Model bundle | Suite | MTPLX decode tok/s | MTPLX accept rate | Validations | Artifact |
|---|---|---:|---:|---:|---|
| Speed | flappy | 59.2 | 99.5% | 4 / 4 | [mtplx.json](speed-flappy/mtplx.json) |
| Speed | long_code | 59.8 | 99.6% | 4 / 8 | [mtplx.json](speed-long-code/mtplx.json) |
| Quality | flappy | 43.0 | 99.4% | 4 / 4 | [mtplx.json](quality-flappy/mtplx.json) |
| Quality | long_code | 43.2 | 99.7% | 4 / 8 | [mtplx.json](quality-long-code/mtplx.json) |

Combined reference JSON: [mtplx.json](mtplx.json)
