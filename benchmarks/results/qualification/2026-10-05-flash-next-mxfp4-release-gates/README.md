# Flash Next MXFP4 MTP direct-default release gates — 2026-10-05

Release scope: **Direct decoding is the default product path; model MTP stays an explicit, experimental opt-in (--mlx-mtp-policy required). MTP-S, MTP-P and MTP-D are not promoted and no MTP speedup is claimed.**

Verdict: **`release_ready = true`** for that scope, by
[`release-verdict.json`](release-verdict.json), which the frozen evaluator
(`scripts/check_flash_next_release_gates.py`) produced from the results below and
the thresholds frozen in `scripts/flash_next_release_thresholds.json`. MTP-S, MTP-P
and MTP-D remain `not_assessed`; nothing here promotes model MTP, tags a release or
changes a public package.

| | |
| --- | --- |
| Pack | `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` (49 files, 132,261,877,478 bytes) |
| Host | Mac Studio, Apple M2 Ultra, 192 GiB, internal NVMe SSD (Apple Fabric), macOS 27.0.1 |
| Server source | `e41d9646` (clean tree, profile `release-server`, rustc 1.97.1 (8bab26f4f 2026-07-14)); binary sha256 `d5254f62adbfa476` |
| Verdict source | `a790a52d`, tracked tree dirty: False |
| Reference | pinned MLX-VLM 0.7.0rc0 (commit `6102cb4a`, MLX 0.32.2, Transformers 5.18.0.dev0); labelled non-primary because `mlx_lm` cannot load `qwen4_exp` |

## Gates

| Gate | Result | Detail |
| --- | --- | --- |
| `delivery` | pass | 49 members, 132261877478 bytes verified from an empty cache after 4 failed attempt(s) that were resumed (transport override {'AX_ENGINE_HF_MAX_WORKERS': '2', 'HF_HUB_DOWNLOAD_TIMEOUT': '60'}) |
| `integrity` | pass | one server binary and one pack inventory across all results; hardware is the target SKU |
| `quality` | pass | 105/105 complete in both arms, default text equals direct text, every AX failure also fails on the reference (['format_csv_pair', 'format_status_colon', 'instruction_alphabet_first', 'knowledge_water_formula', 'science_gravity_earth']) |
| `lifecycle` | pass | 7 actions in each of ['disabled', 'default'] with drained counters |
| `memory` | pass | peak server RSS {disabled: 76.3, default: 76.2} GiB within 96 GiB |
| `readiness` | pass | first load after delivery 44.215s; staged-pack worst 75.543s |
| `long_lookup` | pass | 29774-token lookup took {'disabled': 103.637, 'default': 103.65} s (limit 400 s) |
| `reference_relative` | pass | all 6 AX-arm cells within decode >= 0.9 and TTFT <= 1.15 of the reference |
| `mtp_policy` | pass | default mode never activated the drafter in any harness; required MTP stays an explicit opt-in and MTP-S/P/D remain not_assessed |

## Speed against the reference (fixed workload)

Uniform random token IDs, 128 greedy tokens (`ignore_eos`), two warmups and three
measured runs, each AX arm the lower-latency of two independent server processes
after a discarded warm pass. AX is timed over the HTTP event stream, the reference in
process. Bounds: decode at least 0.9x, TTFT at most
1.15x of the reference.

| Input tokens | AX arm | TTFT (s) | Reference TTFT (s) | TTFT ratio | Decode (tok/s) | Reference decode | Decode ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | product default | 1.22 | 1.13 | 1.08 | 23.8 | 22.9 | 1.04 |
| 512 | explicit direct | 1.22 | 1.13 | 1.08 | 23.7 | 22.9 | 1.03 |
| 2,048 | product default | 4.31 | 6.53 | 0.66 | 22.5 | 17.4 | 1.30 |
| 2,048 | explicit direct | 4.32 | 6.53 | 0.66 | 22.4 | 17.4 | 1.29 |
| 8,192 | product default | 23.94 | 24.55 | 0.97 | 21.9 | 20.4 | 1.07 |
| 8,192 | explicit direct | 24.07 | 24.55 | 0.98 | 21.7 | 20.4 | 1.06 |

The same harness on the pre-change server (`7efa5516`,
[`generate-matrix-before/`](generate-matrix-before/)) measured TTFT of
4.3, 16.4 and 69.7 s and decode of
18.0, 17.5 and 17.0 tok/s at 512, 2,048 and 8,192
tokens; see [the performance note](../../../../docs/performance/flash-next-prefill-decode-2026-10-05.md).
Greedy tokens match between explicit direct and the product default in all
16 natural-prompt trajectories and all 9 fixed cells, and repeats of an
arm emit identical tokens (31/31 and 31/31).

## Quality, memory and lifecycle (frozen 105-item cohort)

| Mode | Passed | Retained failures | Verified MTP steps | Peak RSS (GiB) | Ready (s) | 29,774-token lookup (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| disabled | 101/105 | format_status_colon, instruction_alphabet_first, knowledge_water_formula, science_gravity_earth | 0 | 76.3 | 27 | 104 |
| default | 101/105 | format_status_colon, instruction_alphabet_first, knowledge_water_formula, science_gravity_earth | 0 | 76.2 | 64 | 104 |
| required | 101/105 | format_status_colon, instruction_alphabet_first, knowledge_water_formula, science_gravity_earth | 490 | 73.1 | 67 | 112 |

Text is identical across the three modes for all 105 items. Every retained failure
also fails on the reference runtime on the same pack (format_csv_pair, format_status_colon, instruction_alphabet_first, knowledge_water_formula, science_gravity_earth); none
is reclassified as a pass. All 7 lifecycle actions (baseline, SSE, one- and
two-token budgets, stop, disconnect, recovery) pass in each of 3 modes with drained
counters and clean exits; the product default never verified an MTP step.

## MTP stays opt-in

With `--mlx-mtp-policy required`, 16/16 natural-prompt trajectories and 9/9 fixed cells
are identical to direct, but decode runs at 0.93, 0.92 and 0.91 of direct speed
at 512, 2,048 and 8,192 tokens ([`generate-matrix-mtp/`](generate-matrix-mtp/)).
That fails the MTP-P bounds, so no MTP speedup is claimed and the default is unchanged.

## Fresh delivery

The product downloader (`ax-engine download qwen3.8-flash-next:mxfp4`, source build
`a790a52d`) fetched the pinned revision into an empty cache on the
internal SSD; all 49 members match the inventory by size and SHA-256, the generated native
manifest is runtime ready, and the disabled, required and product-default functional
controls pass on the delivered copy (first server start 44 s; page cache not controlled).

The delivery took 5 attempts and the record keeps all of them
([`delivery-attempt-1/`](delivery-attempt-1/) to `delivery-attempt-4/`, then
[`delivery/`](delivery/)): 4 attempts stopped on Hub read timeouts, one of them
(1 s) on a defect this attempt exposed in the downloader, which
mistook a partial pinned snapshot for a finished one and is fixed in `cf48c216`. The
final attempt ran with the documented transport knobs `{'AX_ENGINE_HF_MAX_WORKERS': '2', 'HF_HUB_DOWNLOAD_TIMEOUT': '60'}`, so
this is not a default-transport reliability rate. A further attempt that was stopped
by hand while the override was added left no record.

## Disclosures

- The memory, readiness and long-lookup bounds were set after the single measurements
  they cite; the reference-relative, quality and identity bounds were frozen before any
  reference or product-default data. The two-process aggregation and the idle-host
  requirement were added before the final evidence and change no bound
  (`measurement_amendment` and `timing_host` in the thresholds file).
- The first server process after a cache change is 30-45% slower; the warm pass of the
  matrix keeps that observation (`warm_pass`). The host also showed time-varying
  interference, which is why each arm keeps the lower-latency of two processes; the
  other repeat of each arm is in `raw-runs.json`.
- Prefill arithmetic for long prompts changed (see the performance note); this is not
  a teacher-forced logit comparison against the official graphs, which remains open.
- Single host, single pack revision, one frozen cohort; no endurance, multi-model,
  multimodal or NAS evidence. The 16,384-token default KV pool is unchanged
  (`--total-blocks 4096` was used for the long lookup).
- The MTP-S evidence recorded earlier (private-cursor transactions, numerical
  diagnostic) was collected before the prefill change and is not repeated here.

## Files

`reference/` (QA and matrix on the pinned MLX-VLM), `delivery*/` (download record,
verification, controls on the delivered copy), `generate-matrix*/` (fixed workload and
natural-prompt identity), `qa/`, `lifecycle/`, `build-receipt.json`,
`release-verdict.json`. Raw results carry `qualification=false` by design; only the
verdict states the release scope outcome.
