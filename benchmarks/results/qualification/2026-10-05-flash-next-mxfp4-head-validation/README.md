# Flash Next MXFP4 current-HEAD validation

Date: 2026-10-05 (local date). Functional, closed-answer QA, API lifecycle,
numerical-identity and server-path direct/MTP comparison runs pass or are
disclosed below; `qualification` and `release_ready`
remain false. MTP-S, MTP-P and MTP-D are not assessed. No default MTP
promotion, speed claim or quality certification follows from this record.

The server is built from clean source `7efa5516` (`release-server`, Rust
1.97.1) and executed against the pinned
`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` (49 files, 132,261,877,478 bytes)
on a Mac Studio, Apple M2 Ultra, 192 GiB, internal NVMe SSD, macOS 27.0.1.
No Rust source changed between that commit and the harness commit
(see [build-receipt.json](build-receipt.json)), so all three runs bind the
same server binary. This closes the gap in the 2026-10-04 receipt, whose
isolated build excluded then-pending edits. The pack, derived native manifest
and server binary are hash-checked before and after every run.

MLX note: `mlx.core` and `libmlx.dylib` report 0.32.3 and match the repo pin,
while the pip metadata of the `mlx` and `mlx-metal` distributions still reads
0.32.2. The library hashes are in the receipt.

## Functional controls ([native-support](native-support/result.json))

The existing seven-case direct/required-MTP HTTP/SSE contract passes
unchanged: all seven choice pairs match, count and JSON checks pass,
`required` plus `AX_NO_SPEC` rejects activation, and attach, prefill and step
errors are zero. The harness now derives the form factor from `hw.model`
instead of hard-coding it.

## Closed-answer QA cohort ([qa](qa/result.json))

The frozen 105-item cohort of the 2026-09-19 installed QA (104 short cases at
a 256-token cap, one deterministic 29,774-token lookup at 64) was replayed with
the frozen checkers, greedy decoding and thinking disabled, MTP disabled and
required. The server runs with `--total-blocks 4096`: the default pool of 1,024
blocks of 16 tokens caps context at 16,384 tokens and rejects the lookup with
`context_length_exceeded`, which is configuration, not a defect.

| | Direct | Required MTP |
| --- | ---: | ---: |
| Items passed | 101/105 | 101/105 |
| Normal stops | 105/105 | 105/105 |
| Route errors | 0 | 0 |
| Verifier steps | 0 | 442 |
| Lookup answer (`1734`) | pass | pass |
| Server ready | 33 s | 80 s |
| Peak server RSS | 76.2 GiB | 73.1 GiB |
| Lookup latency | 268 s | 272 s |

All 105 direct/MTP text pairs are identical. The four retained failures are
the same in both modes and are model outputs, not engine behavior:
`format_csv_pair` (`pair,1`), `format_status_colon` (`ready`),
`instruction_alphabet_first` (`A, B, C, D, E`) and `knowledge_water_formula`
(`H₂O`). The water answer passes under the NFKC matcher added to `qa/checkers.py`
(unit-tested); the frozen checker keeps its original verdict and is not
rewritten. `science_gravity_earth` answers `9.8` and passes, where the older
pack answered `9`. This pack changed 15 trunk shards, so the failure set is
not expected to match the historical one. Latencies and RSS are single
observations with different cache histories, not a performance or memory
qualification.

## API lifecycle ([lifecycle](lifecycle/result.json))

Seven actions per mode, 14 in total, on `/v1/chat/completions`: baseline, full
SSE, one- and two-token budgets, a stop sequence, an unfinished-stream
disconnect and an identical recovery. The disconnect action receives a real
data event, observes one active stream, drops the connection, and the five
in-flight counters drain to zero. Recovery text and usage equal the baseline.
Both servers exit zero. Required-MTP baseline shows 19 verifier steps.

## MTP numerical diagnostic ([mtp-diagnostic](mtp-diagnostic/result.json))

The real-pack `flash_next_first_token_divergence` test (canonical singleton
schedule, default arithmetic, no environment overrides) runs the four frozen
holdout prompts at 64 output tokens. Every prompt completes with no token
difference and no state difference: 63 consumed positions each, exact compared
full state, stream hidden and canonical logits, proposed/accepted of 34/29,
31/31, 33/29 and 33/30. Scope limits: it runs on a native test thread in a
dev-profile build with the runner's fast-path guards not entered, so it is not
the server path; and the canonical schedule has no legacy-batched step, so
the batched-state control added in `858fc4f8` is not exercised here.

## MTP private-cursor transactions ([mtp-transactions](mtp-transactions))

The real-pack `flash_next_real_private_cursor_evidence` test (canonical singleton
schedule, Auto expert paging, one 66-token holdout prompt, eight trajectory
steps) ran once with the trained draft head and once with the output projection
permuted. The independent reader `scripts/verify_flash_next_mtp_transactions.py`
recomputes both from the raw private records: the trained arm has 15 cases plus
six forced verifier controls and covers accepted and failing paths
(`accepted_failure_coverage: true`); the permuted arm has 13 cases plus six
controls and, as expected for a wrong head, no accepted path. Both report
`collection_valid` and `diagnostic_passed`. The reader itself reports MTP-S,
MTP-P and MTP-D as `not_assessed`, and this is one prompt, so it is bounded
control evidence only. The raw state blobs (114 files, about 2.9 GB) are not
committed; only each arm's `result.json` and the reader output are, so the
reader cannot be rerun from this directory.

## Server-path direct vs required-MTP ([generate-matrix](generate-matrix/result.json))

Through `/v1/generate/stream`, greedy, MTP disabled and required, generic
n-gram acceleration disabled, `--total-blocks 4096`.

* **Token identity:** all 16 chat-templated natural-language prompts produce
  identical 128-token output sequences in both modes (1,118 verifier steps in
  required mode), and all 9 measured workload cells are identical. Prompt token
  IDs are equal across modes. Route errors are zero.
* **Decode rate:** required MTP is slower than direct in every cell. Timings
  are client-side (the server leaves its own timing fields at zero on this
  path): TTFT, then tokens after the first token event over their time window,
  loopback, median of three measured runs after two warmups, 128 output tokens.

| Input tokens | Direct tok/s | Required-MTP tok/s | MTP / direct (range) | TTFT direct / MTP | Draft acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 17.93 | 16.45 | 0.917 (0.911-0.921) | 4.0 s / 4.1 s | 182/196 |
| 2,048 | 17.48 | 16.24 | 0.934 (0.907-0.934) | 16.3 s / 16.5 s | 72/86 |
| 8,192 | 17.13 | 15.46 | 0.901 (0.844-0.908) | 69.3 s / 70.5 s | 91/121 |

Acceptance is high, so the shortfall is not a drafting problem. It is
consistent with the ADR-032 design, which runs ordinary single-token target
forwards for each MTP decision to keep exact parity; no profile was taken, so
the cause is not established. All direct cells ran before all required cells,
so cache histories differ. There is no reference-runtime baseline, no frozen
pair of authorizing workloads and no negative control, so MTP-P stays
`not_assessed` and no speedup is claimed; the observed ratios are below 1.0,
so this record gives no support for promoting MTP to the default.

## Not established

Model quality beyond this cohort, contexts beyond 8,192 input tokens in the
timed matrix, memory and cold-start acceptance, fresh-cache delivery, an
independent reconstruction of verifier decisions on the server path (MTP-S),
MTP-P and MTP-D, default promotion and the required `mlx_lm` primary baseline
remain open. The older
47-file pack receipts stay historical.

## Reproduction

```sh
python3 scripts/check_flash_next_native_support.py --dry-run
python3 scripts/run_flash_next_qa_cohort.py --dry-run
python3 scripts/run_flash_next_lifecycle.py --dry-run
python3 scripts/run_flash_next_generate_matrix.py --dry-run
```

Live runs take `--model-dir`, `--inventory`, `--server-bin` and a new
`--output`, and need the pinned pack on an internal SSD of an Ultra-class
192 GiB host.
