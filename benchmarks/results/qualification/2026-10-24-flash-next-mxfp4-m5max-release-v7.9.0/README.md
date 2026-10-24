# Flash Next MXFP4 MTP direct-default release gates — published v7.9.0 on MacBook Pro M5 Max 128 GB, 2026-10-24

Release scope: **Direct decoding is the default product path; model MTP stays an explicit, experimental opt-in (--mlx-mtp-policy required). MTP-S, MTP-P and MTP-D are not promoted and no MTP speedup is claimed.**

Verdict: **`release_ready = true`** ([`release-verdict.json`](release-verdict.json)) for the
**published v7.9.0 binaries**: the signed release tarball
`ax-engine-v7.9.0-macos-arm64.tar.gz` (sha256 `bc41b5d3…`, minisign verified), unpacked
unchanged, with the harness of the `v7.9.0` tag. This repeats the 2026-10-23
[dev-build record](../2026-10-23-flash-next-mxfp4-m5max-release-gates/) with the shipped
artifacts; every server ran with `--max-concurrent-requests 1`.

| Gate | Result | Detail |
| --- | --- | --- |
| `delivery` | pass | 49 members, 132,261,877,478 bytes from an empty cache; one Hub-timeout attempt resumed (`AX_ENGINE_HF_MAX_WORKERS=2`, `HF_HUB_DOWNLOAD_TIMEOUT=60`) |
| `integrity` | pass | one server binary (`2b0aacdf…`) and one pack inventory across all results; standard host |
| `quality` | pass | 101/105 in disabled, default and required modes, the same four retained failures, identical text across modes; every AX failure also fails on the reference |
| `lifecycle` | pass | 7 actions in each mode with drained counters |
| `memory` | pass | peak server RSS 78.1 (disabled) and 78.4 (default) GiB within 96 GiB |
| `readiness` | pass | first load after delivery 44.5 s; staged-pack worst 49.5 s |
| `long_lookup` | pass | 29,774-token lookup 56.6 s (disabled) and 55.0 s (default), limit 400 s |
| `reference_relative` | pass | all 6 AX-arm cells within decode >= 0.9 and TTFT <= 1.15 of the reference |
| `mtp_policy` | pass | the default mode never activated the drafter |

Fixed workload (median of three runs, lower-latency of two processes): decode 50.7 / 47.0 /
45.8 tok/s and TTFT 0.48 / 1.58 / 12.04 s at 512 / 2,048 / 8,192 tokens; token identity in
all 9 cells and 16 trajectories. The pinned reference runs memory-constrained on 128 GB
(15.6 / 14.2 / 12.7 tok/s), so the reference-relative floor passes with a margin that is
not a speedup claim. Single host; MTP-S/P/D not assessed.
