# Flash Next MXFP4 MTP: un-sticky fallback A/B (2026-10-08)

Mac Studio, Apple M2 Ultra, 192 GiB unified memory, internal NVMe SSD —
matches the ADR-037 Flash Next target spec (functional test host; campaign
evidence, not certification). macOS 27.0.1.

Pack: `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` @
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` (49-file pack, inventory-bound at
run time).

Binaries (both release profile, rustup 1.97.1 pin, pip-wheel MLX 0.32.3):

- New: workspace 7.6.1, commit `ad9dfb1f` plus the 2026-10-08 MTP change set
  (uncommitted patch; Phase 3 = paused-cursor fallback with eager capture +
  resume-time batch absorb, greedy fallback routed to the double-buffer
  direct pipeline, correction/bonus sync coalescing; kill switch
  `AX_MLX_FLASH_NEXT_STICKY_FALLBACK=1`), `ax-engine-server` sha256
  `4f2a957a8bdc7b8b3e46247d5b30ce06aeeb109165ed1177d877bb11e1f409b3`.
- Baseline: pristine `ad9dfb1f`, sha256
  `46d939fdf61c96affe3b1e7047dd1b3c52706631f77f9bf23e50e5abb0d3eb16`.

Functional gate: `scripts/check_flash_next_native_support.py` passed on the
new binary (required arm records verified steps on every control, all three
error counters zero, `AX_NO_SPEC` rejects activation). Raw:
`native-support/result.json`.

## Generate matrix (canonical verifier; modes disabled + required)

`scripts/run_flash_next_generate_matrix.py`, two independent server processes
per arm; the un-sticky run is the new-code default, the sticky run adds
`--ax-server-env AX_MLX_FLASH_NEXT_STICKY_FALLBACK=1`. Ratios are
required-MTP over disabled-direct decode tok/s, median of measured runs.

| Input tokens | 2026-10-05 record | new code, sticky | new code, un-sticky |
| ---: | ---: | ---: | ---: |
| 512 | 0.917x | 0.937x | 0.931x |
| 2048 | 0.934x | 0.936x | 0.932x |
| 8192 | 0.901x | 0.930x | 0.931x |

Token identity: 16/16 trajectory prompts and 9/9 fixed-workload cells
identical between modes in both runs; route errors 0. Sticky and un-sticky
are at parity here: this workload (thinking disabled) produces no transient
fallback blocks, so the canonical two-singleton verifier ceiling (~1.00x by
construction, see the 2026-10-06 profile) dominates. The 8192-token cell
improved from 0.901x (2026-10-05) to ~0.930x on both new-code runs.

Raw: `matrix-unsticky/`, `matrix-sticky/` (result.json + raw-runs.json).

## Think-budget workload (the fallback-heavy case)

Six reasoning prompts on `/v1/chat/completions` with
`ax_max_think_tokens=220` (arms the think soft-close probe window every
request), greedy, up to 768 output tokens, thinking enabled. Client-timed
decode rate; counters diffed from `/metrics` per request.

| Arm | median decode tok/s | verified steps | cursor drops | paused steps | resumes |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline binary (legacy sticky + legacy routing) | 30.56 | 0 | 6/6 | n/a | n/a |
| new binary + sticky kill switch | 31.03 | 0 | 6/6 | 0 | 0 |
| new binary, un-sticky default | 29.41 | 1424 | 0/6 | 961 | 6/6 |

Output text is identical across all three arms on all six prompts.

Interpretation, narrowly scoped:

- The mechanism works: with a think budget, the legacy behavior drops the
  draft cursor at the first soft-close probe step and never verifies again
  (0 verified steps per request); the new paused-cursor path suspends
  drafting for exactly the probe window (~220 steps per request), resumes
  afterwards (6/6), and keeps verifying (1424 steps total).
- On throughput, un-sticky is ~4% slower than drop-to-direct on this
  workload (29.41 vs 30.56–31.03). Keeping the cursor alive preserves the
  canonical MTP schedule, which is still ~0.93x direct, so preserving it
  costs slightly more than abandoning it. The pause/resume machinery is
  verifier-schedule-neutral infrastructure: its throughput payoff is
  contingent on a future verifier decision (the ADR-041 opt-in batched
  verifier measured 1.20x on this pack, with near-tie divergence), not on
  this change set. Until then its value is behavioral: opted-in MTP requests
  keep collecting verified-step evidence instead of silently degrading to
  direct for the rest of the request.

## Caveats

Single host, single day; the think-budget workload is one small prompt set
with client-timed rates; MTP remains opt-in for Flash Next and this is not
MTP-S/P/D evidence. The canonical verifier schedule is unchanged. Bench and
campaign numbers never qualify a SKU.
