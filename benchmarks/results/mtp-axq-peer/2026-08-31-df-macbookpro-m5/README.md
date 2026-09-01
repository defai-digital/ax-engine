# AXQ MTP peer campaign

This directory contains the 2026-08-31 MTP benchmark campaign run on
`df-macbookpro-m5` (Apple M5 Max, 128 GB, macOS 26.6.2).

The campaign uses the repository `flappy` prompt suite, four prompt cases,
256 generated tokens, greedy sampling, two warmups, five measured repetitions,
and a three-second cooldown between measurements. Decode throughput is the
median over 20 measured runs. Prefix-cache and n-gram stacking were disabled.

AX Engine used version 7.2.0 with Qwen linear-MTP exact mode and Gemma4
assistant-MTP depth 2. MTPLX used version 2.9.0. OMLX used the latest release
available for this campaign, version 0.6.4, with `mtp_enabled` and one draft
token.

The requested Qwen3.6 25B and Gemma4 35B labels do not match published
AutomatosX AXQ packs. The campaign therefore uses Qwen3.6 27B and Gemma4 31B,
respectively, and preserves those mappings in `summary.json`.

The `raw/` files are the runtime artifacts. OMLX Qwen rows are text-only:
the AXQ vision sidecar was excluded from temporary staging because OMLX's VLM
loader rejected the AXQ vision-key layout. MTPLX and OMLX Gemma assistant-MTP
rows are reported as unsupported rather than substituted with direct decode.

The OMLX lane can be reproduced with `scripts/bench_omlx_prompt_suites.py` by
installing OMLX 0.6.4, pointing `--model` at the prepared text-only AXQ stage,
and passing `--prompt-token-dir` from the matching AX Engine run. The runner
requires no OMLX dependency for repository test or lint execution.
