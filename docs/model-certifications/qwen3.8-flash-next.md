# Qwen 3.8 Flash Next

Status: **Candidate; release qualification open**

Second SKU. M2 evidence only; checkpoint qualification pending. MTP Tier 2 pending. AX certification record: Candidate (gates open).

Target SKU: **Mac Studio M5 Ultra, 256 GB**. Current real-pack evidence is
from **Apple M2 Ultra, 192 GB**. Last reviewed: **2026-09-17**.

## Identity and implemented execution

HF family `qwen4_exp` identifies the 125B-A6B hybrid Gated-DeltaNet / sparse
attention MoE with its 51B n-gram table. It is distinct from Qwen 3.8 27B
and Super-class 2.4T. The dedicated AX graph implements GDN, QSA, gated
residual streams, PLE disk-row gathers, MoE, and the final mixer. MLX owns
quantized matrix multiplication. This is not a `qwen3_5` remap or an adapter.

Request-owned state, prefix serialization/restore, expert paging and native
HTTP/SSE are implemented. N-gram table payloads are excluded from weight-load
evaluation. Selected-expert and selected-prefill paths remain opt-in.

The CLI maps `qwen3.8-flash-next:axq` and `qwen3.8-flash-next:axq-6bit` to
`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-4bit-MTP` and
`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP`, respectively. An alias is
not proof of published-pack availability or successful download qualification.
Audited legacy manifests identify source `Qwen/Qwen3.8-Flash-Next` revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`.

## Current gates

| Gate | Current result | Remaining requirement |
| --- | --- | --- |
| Support tier | Experimental graph; checkpoint Candidate | Complete reproducible checkpoint qualification |
| Pack delivery | Public packs and immutable CLI revisions verified; four metadata files match native test packs | Verify full payload identity, fresh download, doctor, and installed-runtime admission |
| Numerical | Eight prompts, 3,260 aligned positions; aggregate statistical rule passes | Independent holdout verification; retain 22 high-margin disagreements and the revised 1% rule |
| Functional QA | Completed 105 items per route: direct 102, required MTP 102, reference 101 hard passes | Direct/MTP text differs on two reasoning items; encoded QA acceptance is false |
| Long context / NLL | Long-context lookup completed across all three routes; 3,999 scored tokens, AX mean NLL 1.96226 versus reference 1.96609 | Broader contexts; recover matching historical harness or rerun with frozen provenance |
| Trained head | Recorded real acceptance 95/114 (83.3%), permuted 0/207 | Only 50 of 104 requests contribute to acceptance; 54 short cases are excluded. This is a bounded falsification control, not Tier 2 |
| MTP integration | All 12 state/runner controls completed: 9 pass, 3 fail unchanged bounds | Investigate 2-bit tie state and 4-bit tie state/runner before promotion |
| HTTP / SSE | Six modes and 12 requests completed; 4 pass, 2 fail direct/MTP text identity | Investigate 4/6-bit required MTP identity; extend beyond four-token requests |
| Throughput | Fresh fixed-output matrix: 11/18 complete; two failed cells and five without results; 128 tokens in every measured sample | Resolve long-context budget and reference GPU timeout; collect five missing results and target-SKU memory evidence |
| Target hardware | No M5 Ultra 256 GB result | Run target-SKU qualification |
| Release | Not release-ready | Close the numerical, QA, MTP, throughput, delivery and target-hardware gates above; merged validation alone is insufficient |

The statistical threshold was adjusted on the collected sample: the earlier
zero-high-margin-disagreement rule failed, while the later at-most-1% rule
passes at 22/3,260 (0.67%). The aggregate mean KL is 0.0860 against a 0.1011
limit, and top-1 disagreement is 2.85% against 3.22%. Do not describe this as
an independent confirmation or exact full-model parity.

## Immutable public pack metadata

The 4-bit alias is pinned to `680573112360bfd3f71556082f875c907c21a6e7`;
the 6-bit alias to `d514dcebf3086068ed7968caf395083c95ebcfca`. Both repositories
are publicly accessible. Their config, AXQuant manifest, tensor index and expert
stream manifest match the native test packs byte-for-byte. See the
[metadata identity record](../../benchmarks/results/flash-next-public-pack-metadata-20260917.json).
This does not verify every weight byte, a fresh download, or installed-runtime
qualification.

## MTP state and runner coverage

The historical cells below use test binary `79f30efe` (SHA-256 prefix), built from commit
`1819e4bb`. A pass applies to the recorded prompt and tolerance contract.

| Pack | Primary state | Primary runner | Tie state | Tie runner |
| --- | --- | --- | --- | --- |
| 2-bit | Pass | Pass | Fail: bonus margin 0.9375 > 0.5 | Pass |
| 4-bit | Pass | Pass | Fail: logit relative error 0.100864 > 0.1 | Fail: margin 1.3125 > 0.5 |
| 6-bit | Pass | Pass | Pass | Pass |

A [corrected 4-bit tie runner replay](../../benchmarks/results/flash-next-runner-margin-replay-m2-20260917.json)
measures a direct-to-MTP token gap of 0.125 at position 2, below the unchanged
0.5 limit. The historical 1.3125 margin above duplicated the pipeline bootstrap
token. The output sequences still differ; this is a bounded tie pass, not text
identity or a fix for the separate 4-bit state failure. Other historical runner
cells have not been regenerated with the corrected diagnostic.

The pre-merge native API matrix uses server binary `f61f46a0` (SHA-256 prefix), also
from `1819e4bb`. Each mode runs completion and SSE with the same five input
tokens and four output tokens. All six modes return usage and a terminal SSE
marker, repeat their own text, preserve pack metadata, and exit cleanly.
Default-on MTP remains zero. Required MTP produces different text from direct
for 4-bit and 6-bit; 2-bit agrees for this prompt. These strict identity
failures remain visible even though the bounded 6-bit runner controls pass.
This short API control does not establish long-request quality or stability.

The merged source `ea4eb15b` was rebuilt on M2 Ultra with Rust 1.97.1
(`release-server`, binary SHA-256 prefix `6c5c0188`). Its separate six-mode,
12-request rerun again passes four modes and fails 4/6-bit required MTP
text identity. All transport, repeat identity, usage, metadata and clean-exit
checks pass. The 4/6-bit runs remove both experimental family/2-bit opt-ins;
2-bit retains both. Selected-expert/prefill opt-ins remain enabled.
Local merged validation passes 3,681 Rust tests (46 ignored), pinned Clippy,
formatting, script gates and 50 Python CLI tests with 92 subtests.
These bounded checks do not close the release gates above.

## Evidence and provenance

These artifacts preserve development outcomes, including failures and missing
cells. `qualification=false` and `release_ready=false` are intentional.
Recorded binary/harness hashes are never replaced with hashes of newer files.
A mismatching or unavailable harness is an open reproducibility gate.

- [Statistical acceptance](../../benchmarks/results/flash-next-statistical-acceptance-m2-20260916.json)
- [Native build identity and checked source hashes](../../benchmarks/results/flash-next-native-build-m2-20260916.json)
- [Completed QA, long context and NLL](../../benchmarks/results/flash-next-extended-qa-v3-m2-20260916.json)
- [Trained-head falsification control](../../benchmarks/results/flash-next-mtp-head-oracle-m2-20260916.json)
- [Merged native build identity](../../benchmarks/results/flash-next-native-build-merged-m2-20260917.json)
- [Merged native HTTP / SSE matrix](../../benchmarks/results/flash-next-http-merged-m2-20260917.json)
- [Pre-merge native HTTP / SSE matrix](../../benchmarks/results/flash-next-http-m2-20260916.json)
- [Batched MTP matrix](../../benchmarks/results/flash-next-mtp-batched-verify-m2-20260916.json)
- [Observed six-bit Auto paging](../../benchmarks/results/flash-next-sixbit-paging-m2-20260917.json)
- [Throughput audit with explicit failure and missing counts](../../benchmarks/results/flash-next-throughput-audit-m2-20260917.json)
- [Throughput matrix, including incomplete cells](../../benchmarks/results/flash-next-throughput-ab-m2-20260916.json)
- [Earlier affine 2/4/6-bit execution controls](../../benchmarks/results/flash-next-affine-formats-m2-20260915.json)
- [Earlier native selected-prefill controls](../../benchmarks/results/flash-next-selected-prefill-m2-20260915.json)

Earlier operator-level, cache, QSA-boundary and numerical comparisons remain in
`benchmarks/results/flash-next-*.json`. They describe their recorded snapshots;
they do not supersede the current gate table or validate a later merged binary.

The fresh comparison fixes the earlier EOS mismatch: AX uses the native
fixed-output endpoint with `ignore_eos=true`, and both routes emit 128 tokens.
The post-first-token decode interval covers 127 tokens. AX client-wall timing
and reference in-process timing remain distinct; native runner timings are
retained separately. The baseline is pinned MLX-VLM because `mlx_lm` has no
`qwen4_exp` graph; this is not an `mlx_lm.benchmark` result.

The host had background indexing/sync activity. AX RSS is an end-of-request
snapshot, and its MLX peak covers the server lifetime; the reference resets
MLX peak per request. No controlled performance or equivalent memory-peak
claim follows from these fields. Although the historical configuration label
says `auto_resident_on_192gib`, observed 6-bit Auto execution uses expert
paging. A stack sample in the first prefill reaches
`ExpertStackPager::ensure_layer` and `load_safetensors_mmap_filtered`.
The first 512-token prefill takes about 448 seconds; subsequent samples take
about 14 seconds. A one-second diagnostic sample was taken during this
unmeasured first warmup. Read actual behavior rather than that stale
residency label.

## Admission and operator contract

Audited affine 4-bit/group64 and 6-bit/group64 manifests can be admitted
without an environment variable. `runtime_status.ready` expresses loader
admission, not checkpoint certification. Unknown exporter layouts, invalid
geometry, mixed expert layouts and MXFP4 remain rejected by the native loader.
2-bit/group32 requires both `AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1` and
`AX_ENGINE_2BIT_EXPERIMENTAL=1`. Auto/On/Off expert residency is unchanged.

```bash
python3 scripts/qualify_qwen38_flash_next.py --dry-run --json
python3 scripts/qualify_qwen38_flash_next.py --model-dir /path/to/converted-pack
```

The second command is a metadata preflight only. It does not load tensors,
validate exporter identity/geometry, exercise generation, or certify the pack.
Native `ax-engine doctor` and actual server loading remain required. A legacy
manifest may retain its old trunk blocker; this conservative preflight rejects
it even when the native loader can re-audit and admit the same artifact.

```bash
ax-engine-server --mlx \
  --mlx-model-artifacts-dir /path/to/converted-pack \
  --host 127.0.0.1 --port 31418
```

An available MTP sidecar attaches automatically, but default-on MTP remains
disabled. Explicit `--mlx-mtp-policy required` exercises the experimental
verified path. The test contract records identity until an observed tie and
bounded numerical divergence; it does not promise universal text identity.
The QA mismatches remain failures under the original all-items identity rule.
The historical 4-bit tie runner margin of 1.3125 was measured after replaying
the pipeline bootstrap token twice. It is not a valid margin for the divergent
position. The corrected diagnostic skips tokens already represented by the
snapshot, verifies the direct prediction, and measures the gap to the actual
MTP token. Its state control still exceeds the logit bound (0.100864 versus
0.1); the runner diagnostic correction does not resolve that failure.

The primary default remains [Qwen 3.8 27B AXQ](qwen3.8-27b-axq.md) on
Mac mini M5 64 GB. See [Supported Models](../SUPPORTED-MODELS.md) and
[Testing](../TESTING.md) for the wider operator contract.

The throughput campaign stopped after entering 6-bit reference / 2,048 tokens.
The SSH command exited 255; the process was absent on inspection, and its
termination cause is unconfirmed. The five missing rows are not successful
measurements. The 6-bit direct / 8,192-token server was deliberately terminated
by the 900-second progress supervisor; the reference / 512-token Metal GPU
timeout was a separate recorded failure. See the
[supervisor record](../../benchmarks/results/flash-next-throughput-budget-m2-20260917.json).

## Verifier diagnostic controls

Test binaries support `AX_FLASH_NEXT_VERIFY_DIAGNOSTICS=1`. Each verify call
reports `batched_row0` separately from a `singleton` decision whose source is
`rejection_replay` or `one_slot_budget`. The existing correction margin retains
its meaning. Missing comparisons are null; an accepted window does not trigger
an extra singleton forward for diagnostics. These synchronized records cannot
be used for throughput claims and are absent from production builds.

The [tiny-model control](../../benchmarks/results/flash-next-verify-diagnostics-m3-20260917.json)
checks accepted, rejected, terminal and one-slot cases on M3 Max. Enabling the
diagnostic preserves committed tokens and state hashes. It does not resolve the
real-pack numerical failures above. The newer throughput audit re-curates the
same immutable M2 input: 11 complete, two failed, five missing; it is not a new
hardware run.
