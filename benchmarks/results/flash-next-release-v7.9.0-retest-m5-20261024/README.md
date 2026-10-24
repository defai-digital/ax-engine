# Published v7.9.0 retest on MacBook Pro M5 Max 128 GB — 2026-10-24

The signed v7.9.0 release binaries (see
[the release-gate record](../qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/))
on the pinned Flash Next MXFP4 MTP pack (`ff2a2848`), compared with the 2026-10-23 dev
build that the release was cut from.

| Workload | v7.9.0 release | 2026-10-23 dev build | Text |
| --- | --- | --- | --- |
| Long-context decode, 2,207 / 8,976 / 39,225 tokens (`fn_ab.py`, purged cache, two rounds) | 47.4 / 45.9 / 44.7 tok/s, TTFT 2.4 / 13.6 / 74.7 s (better round) | 47.3 / 45.7 / 44.5 tok/s | identical hashes (`6189ecc2`, `c14a8512`, `a3ac5cb9`) |
| Flappy suite, direct (20 samples) | 51.39 decode / 850.5 prefill | 51.41 / 852.6 | — |
| Flappy suite, MTP required (20 samples) | 47.43 / 785.4 | 47.45 / 798.9 | — |

The release reproduces the dev-build numbers within 1% and the same generated text.
Files: `rel790-longctx-{1,2}.json`, `rel790-flappy-direct.json`, `flappy-mtp/`,
`retest_m5_b.sh` (the driver).
