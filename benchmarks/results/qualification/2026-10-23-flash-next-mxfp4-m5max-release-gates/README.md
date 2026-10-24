# Flash Next MXFP4 MTP direct-default release gates on MacBook Pro M5 Max 128 GB — 2026-10-23

Release scope: **Direct decoding is the default product path; model MTP stays an explicit, experimental opt-in (--mlx-mtp-policy required). MTP-S, MTP-P and MTP-D are not promoted and no MTP speedup is claimed.**

Verdict: **`release_ready = true`** for that scope on the ADR-049 standard host, by
[`release-verdict.json`](release-verdict.json), which the evaluator
(`scripts/check_flash_next_release_gates.py`) produced from the results below and the
thresholds in `scripts/flash_next_release_thresholds.json`. The evaluator accepts only a
MacBook Pro with an Apple M5 Max and 128 GiB; it rejects the 2026-10-05 Mac Studio M2 Ultra
results, which stay in [their own record](../2026-10-05-flash-next-mxfp4-release-gates/) as
larger-host evidence. MTP-S, MTP-P and MTP-D remain `not_assessed`; nothing here promotes
model MTP, tags a release or changes a public package.

| | |
| --- | --- |
| Pack | `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` (49 files, 132,261,877,478 bytes) |
| Host | MacBook Pro, Apple M5 Max, 128 GB, internal NVMe SSD (Apple Fabric), macOS 27.0.1 |
| Envelope | single-stream resident: every server ran with `--max-concurrent-requests 1`; the default path keeps MTP off; the longest prompt is 29,774 tokens |
| Server source | `a3655fd0` (clean tree, profile `release-server`); binary sha256 `8b19d1f32225b29d` |
| Reference | pinned MLX-VLM 0.7.0rc0 (commit `6102cb4a`, MLX 0.32.2, Transformers 5.18.0.dev0); labelled non-primary because `mlx_lm` cannot load `qwen4_exp` |

## Gates

| Gate | Result | Detail |
| --- | --- | --- |
| `delivery` | pass | 49 members, 132,261,877,478 bytes verified from an empty cache after one attempt stopped on Hub read timeouts and was resumed (transport override `AX_ENGINE_HF_MAX_WORKERS=2`, `HF_HUB_DOWNLOAD_TIMEOUT=60`) |
| `integrity` | pass | one server binary and one pack inventory across all results; hardware is the standard host |
| `quality` | pass | 105/105 complete in every mode, default and required text equal direct text, every AX failure also fails on the reference |
| `lifecycle` | pass | 7 actions in each mode with drained counters |
| `memory` | pass | peak server RSS 78.1 (disabled) and 78.0 (default) GiB within 96 GiB |
| `readiness` | pass | first load after delivery 48.5 s; staged-pack worst 50.5 s |
| `long_lookup` | pass | 29,774-token lookup 55.9 s (disabled) and 55.4 s (default), limit 400 s |
| `reference_relative` | pass | all 6 AX-arm cells within decode >= 0.9 and TTFT <= 1.15 of the reference |
| `mtp_policy` | pass | the default mode never activated the drafter; required MTP stays an explicit opt-in |

## Fixed workload (uniform random token IDs, 128 greedy tokens)

Two warmups and three measured runs; each AX arm is the lower-latency of two independent
server processes after a discarded warm pass. AX is timed over the HTTP event stream, the
reference in process.

| Input tokens | AX decode (tok/s) | AX TTFT (s) | Reference decode | Reference TTFT (s) |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 50.8 | 0.48 | 13.8 | 5.80 |
| 2,048 | 47.3 | 1.57 | 12.9 | 15.64 |
| 8,192 | 45.7 (45.9 default) | 12.2 (12.0 default) | 7.4 | 95.92 |

Greedy tokens match between explicit direct and the product default in all 16
natural-prompt trajectories and all 9 fixed cells; repeats of an arm emit identical tokens
(31/31 in each mode).

## Quality, memory and lifecycle (frozen 105-item cohort)

| Mode | Passed | Retained failures | Verified MTP steps | Peak RSS (GiB) | Ready (s) | 29,774-token lookup (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| disabled | 101/105 | format_status_colon, instruction_alphabet_first, knowledge_water_formula, science_gravity_earth | 0 | 78.1 | 39 | 56 |
| default | 101/105 | same four | 0 | 78.0 | 47 | 55 |
| required | 101/105 | same four | 444 | 74.8 | 48 | 60 |

Text is identical across the three modes for all 105 items, and the four retained failures
are the same four as on the 2026-10-05 larger host. Every retained failure also fails on the
reference runtime on the same pack; none is reclassified as a pass.

## Disclosures

- **The reference runtime is memory-constrained on this host.** The pinned MLX-VLM graph
  materializes the whole pack, including the 53.6 GiB n-gram table that the AX server reads
  from disk, so on 128 GB it ran at 13.8 / 12.9 / 7.4 tok/s where it ran at 22.9 / 17.4 /
  20.4 tok/s on the 192 GB host. The `reference_relative` gate is a floor (AX at least 0.9x
  the reference decode, at most 1.15x its TTFT); it passes with a wide margin here, and the
  ratios are not a speedup claim.
- The binaries were built with the Homebrew Rust 1.99.1 that the build host's PATH resolved,
  not the repository's 1.97.1 pin (`build-receipt.json`).
- The memory, readiness and long-lookup bounds were set on the earlier host's single
  measurements; ADR-049 kept the 96 GiB memory bound for this host (75% of 128 GiB).
- The first delivery attempt stopped on Hub read timeouts at about 101 GB and is kept in
  `delivery-attempt-1/`; this is not a default-transport reliability rate.
- One host, one pack revision, one frozen cohort; no endurance, multi-model, multimodal or
  NAS evidence, and nothing outside the single-stream envelope.

## Files

`delivery*/` (download record, verification, controls on the delivered copy),
`generate-matrix/` (fixed workload and natural-prompt identity), `qa/`, `lifecycle/`,
`reference/` (QA and matrix on the pinned MLX-VLM), `build-receipt.json`,
`release-verdict.json`. Raw results carry `qualification=false` by design; only the verdict
states the release scope outcome.
