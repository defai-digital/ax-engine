# Qwen 3.8 cross-runtime benchmark — M5 Max — 2026-08-21

## Result

Median client-observed throughput from three fresh-process repetitions:

| Runtime | Measured mode | Prefill input | Prefill | Decode input/output | Decode |
| --- | --- | ---: | ---: | ---: | ---: |
| MTPLX 2.9.0 | Turbo, MTP depth 3 | 2,117 tokens | 881.8 tok/s | 152 / 256 | 54.82 tok/s |
| oMLX 0.6.2 | MTP enabled, depth 3 | 2,117 tokens | 843.2 tok/s | 152 / 256 | 53.15 tok/s |
| AX Engine 7.1.5 + MLX 0.32.1 | Production-safe direct | 2,117 tokens | 820.7 tok/s | 152 / 256 | 29.27 tok/s |

Inputs are median authoritative server token counts. Across the three prompts,
prefill input ranged from 2,114 to 2,156 tokens and decode input from 142 to
154 tokens. Every request completed exactly 256 output tokens.

MTPLX led long-prompt prefill by 7.5% and short-prompt decode by 87.3% versus
AX. oMLX led AX by 2.7% and 81.6%, respectively. The decode comparison is a
mode comparison as well as a runtime comparison: the peers used their native
MTP paths, while AX failed closed to direct decode for the uncertified
third-party MTPLX sidecar. AX's production-default speculation probe was also
29.29 tok/s, so it did not improve this workload.

## Runtime and model identity

- Apple M5 Max MacBook Pro, 128 GB unified memory, macOS 26.6.1, AC power.
- Swap was zero before and after the run. Load average was 1.00 at the start
  and 1.36 at the end.
- Model: `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`, pinned Hugging Face
  revision `123db8bcc7101455b00d9aad36c0e760c6e7de02`.
- The oMLX view derives from the same pinned source pack, with a compatible
  manifest/config and sidecar tensor-name remap for its MTP loader.
- AX Engine: 7.1.5, linked against MLX 0.32.1.
- oMLX: exact tag 0.6.2 (`f2d36f3d25a7e7a2401a92eecafc28b8f8968ec7`),
  with its environment's MLX 0.32.0 dependency and native kernels available.
- MTPLX: 2.9.0, with MLX 0.32.1.

Upstream references: [MLX 0.32.1](https://github.com/ml-explore/mlx/releases/tag/v0.32.1),
[oMLX 0.6.2](https://github.com/jundot/omlx/releases/tag/v0.6.2), and
[MTPLX 2.9.0](https://mtplx.com/releases/notes/v2.9.0).

## Method

- Streaming OpenAI chat endpoint, temperature 0, top-p 1, top-k 0, fixed seed,
  and thinking disabled.
- One active request, a fresh server process for each runtime/repetition, a
  32-token warmup, and a 15-second cooldown.
- Runtime order rotated by repetition.
- Prefill throughput is authoritative prompt tokens divided by client TTFT.
- Decode throughput is `(completion_tokens - 1) / (last visible chunk - first
  visible chunk)` so all runtimes use the same client boundary.
- Dispersion (sample CV): long-prompt prefill was 1.0% MTPLX, 0.9% oMLX, and
  7.0% AX; short-prompt decode was 5.2%, 4.1%, and 0.3%, respectively.

The artifact is complete: 18 of 18 expected measurements, no request errors,
and no forced kills. oMLX exited on the harness SIGTERM. MTPLX completed every
measurement but reported exit `-11` during post-measurement teardown in all
three repetitions; this does not invalidate the timed windows but is a runtime
lifecycle issue worth reporting upstream.

## Output-equivalence limitation

This is performance evidence, not a bit-equivalence certification. The three
runtimes did not produce the same 256-token output hash in any of the six
prompt/seed cells; oMLX and MTPLX matched each other in one cell. Prompt and
completion token counts matched, but the artifact retains hashes rather than
full text, so it cannot support a semantic-quality conclusion.

## What AX learned from oMLX

The DeepSeek V4 Pro and Qwen 3.8 Max reviews independently identified AX's
partial-accept handling and verifier projection traffic as the highest-value
areas. AX previously cloned verifier state and could replay the full
transformer backbone after a partial Qwen gated-delta MTP accept. The retained
implementation now:

1. stashes verifier QKV/A/B projections only for the transient MTP capture;
2. slices the accepted prefix and replays only convolution/recurrent updates;
3. trims ordinary attention KV and adopts the rebuilt verifier cache; and
4. falls back to the existing full replay on any unsupported shape or state.

The route is explicit and default-off via
`AX_MLX_MTP_LINEAR_PROJECTED_REPLAY=1`. The exact one-sample diagnostic matched
oMLX's output hash but reached only 23.54 tok/s versus oMLX's 56.20 tok/s. A
relaxed target-verified diagnostic reached 29.23 tok/s but changed the output
hash. Neither result clears promotion, so the feature remains experimental.

An oMLX-style QMM verifier port was also tested and rejected: the broad route
fell to 18.00 tok/s and the vocabulary-head-only route to 20.04 tok/s. Those
code paths were removed. The remaining decode gap points to draft-head and
verification projection/memory traffic rather than rollback alone; the next
useful experiment is an AX-native fused verifier projection path with a strict
same-output and M5 performance gate.

## Evidence

- [`benchmark.json`](benchmark.json): canonical 18-measurement result.
- [`logs/`](logs/): all nine fresh-process server logs.
- [`probe-ax-projected-replay.json`](probe-ax-projected-replay.json): retained
  exact projected-state replay diagnostic.
- [`probe-ax-production-spec.json`](probe-ax-production-spec.json): AX's
  unforced production-default speculation check.
- [`probe-ax-relaxed-replay.json`](probe-ax-relaxed-replay.json): non-promoted
  relaxed arithmetic diagnostic.
- [`rejected-ax-verify-qmm.json`](rejected-ax-verify-qmm.json) and
  [`rejected-ax-head-qmm.json`](rejected-ax-head-qmm.json): rejected QMM trials.
