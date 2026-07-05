# Gemma 4 E2B / E4B 6-bit AX direct refresh (r2)

AX-engine-only refresh of the README Gemma 4 E2B/E4B 6-bit direct rows on a
clean `ax-engine-server` release build at commit `6f2e6cd7`
(`git_tracked_dirty: false`), replacing the prior cells measured on build
`01976818` (2026-06-27, `v6.5.2`). Contract: `mlx_lm.benchmark` seed-0 random
prompts at 128/512/2048 prompt tokens, generation 128, 5 repetitions,
1 warmup, 15 s cooldown, prefix cache disabled, `--ax-direct`
(n-gram acceleration, MTP, and assistant drafting disabled).

## Why AX-only

`mlx-lm` 0.31.3 cannot strict-load either E-series 6-bit checkpoint: the
snapshots ship K/V tensors for the shared-KV layers that the upstream Gemma4
text model does not declare — `gemma-4-e2b-it-6bit` fails with
`Received 140 parameters not in model` (shared-KV layers 15..34) and
`gemma-4-e4b-it-6bit` with `Received 126 parameters not in model`
(layers 24..41). The E2B 6-bit `mlx_lm` README reference cells therefore
remain the retained 2026-05-26 measurements; E4B 6-bit `mlx_lm` cells stay
blank.

## Why r2

The first run of this refresh (same command, same build, r1) executed while
three repo-analysis subagents were running concurrently on the host; the
E2B prompt=512 prefill cell measured 13,408.8 tok/s there versus
14,791.3 tok/s on this quiet-machine rerun (+10.3%), while decode cells moved
< 1.5%. Short-prompt prefill on the dispatch-bound E2B is the most
CPU-contention-sensitive cell, so the contaminated r1 was discarded and this
quiet rerun is the published artifact.

## Result vs the replaced 01976818 cells

Decode and short-prompt prefill are flat within run-to-run drift (±4%);
prompt=2048 prefill improves (E2B +17.2% → 22,157.8 tok/s, E4B +4.6% →
7,913.3 tok/s) with TTFT down 14.7% / 4.4% at 2048. The earlier hypothesis
that these rows were understated the way the 26B/31B 6-bit rows were
(`2026-07-02-gemma4-6bit-direct-refresh`) did not hold: the decode-path gains
that lifted the bandwidth-bound large models between `01976818` and
`d4c59ffc` do not apply to the dispatch-bound E-series.
