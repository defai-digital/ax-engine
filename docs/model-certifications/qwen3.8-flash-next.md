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
| Pack delivery | CLI aliases exist; published pack and immutable download revision are unverified | Verify published artifacts, download, doctor, and installed-runtime admission |
| Numerical | Eight prompts, 3,260 aligned positions; aggregate statistical rule passes | Independent holdout verification; retain 22 high-margin disagreements and the revised 1% rule |
| Functional QA | Completed 105 items per route: direct 102, required MTP 102, reference 101 hard passes | Direct/MTP text differs on two reasoning items; encoded QA acceptance is false |
| Long context / NLL | Long-context lookup completed across all three routes; 3,999 scored tokens, AX mean NLL 1.96226 versus reference 1.96609 | Broader contexts; recover matching historical harness or rerun with frozen provenance |
| Trained head | Recorded real acceptance 95/114 (83.3%), permuted 0/207 | Only 50 of 104 requests contribute to acceptance; 54 short cases are excluded. This is a bounded falsification control, not Tier 2 |
| MTP integration | All 12 state/runner controls completed: 9 pass, 3 fail unchanged bounds | Investigate 2-bit tie state and 4-bit tie state/runner before promotion |
| HTTP / SSE | Six modes and 12 requests completed; 4 pass, 2 fail direct/MTP text identity | Investigate 4/6-bit required MTP identity; extend beyond four-token requests |
| Throughput | Fresh fixed-output matrix: 10/18 complete; 128 tokens in every measured sample | Finish remaining eight cells; verify memory behavior on target SKU |
| Target hardware | No M5 Ultra 256 GB result | Run target-SKU qualification |
| Release | Not release-ready | Final merged-tree gates and native validation |

The statistical threshold was adjusted on the collected sample: the earlier
zero-high-margin-disagreement rule failed, while the later at-most-1% rule
passes at 22/3,260 (0.67%). The aggregate mean KL is 0.0860 against a 0.1011
limit, and top-1 disagreement is 2.85% against 3.22%. Do not describe this as
an independent confirmation or exact full-model parity.

## MTP state and runner coverage

All cells use test binary `79f30efe` (SHA-256 prefix), built from commit
`1819e4bb`. A pass applies to the recorded prompt and tolerance contract.

| Pack | Primary state | Primary runner | Tie state | Tie runner |
| --- | --- | --- | --- | --- |
| 2-bit | Pass | Pass | Fail: bonus margin 0.9375 > 0.5 | Pass |
| 4-bit | Pass | Pass | Fail: logit relative error 0.100864 > 0.1 | Fail: margin 1.3125 > 0.5 |
| 6-bit | Pass | Pass | Pass | Pass |

The native API matrix uses server binary `f61f46a0` (SHA-256 prefix), also
from `1819e4bb`. Each mode runs completion and SSE with the same five input
tokens and four output tokens. All six modes return usage and a terminal SSE
marker, repeat their own text, preserve pack metadata, and exit cleanly.
Default-on MTP remains zero. Required MTP produces different text from direct
for 4-bit and 6-bit; 2-bit agrees for this prompt. These strict identity
failures remain visible even though the bounded 6-bit runner controls pass.
This short API control does not establish long-request quality or stability.

## Evidence and provenance

These artifacts preserve development outcomes, including failures and missing
cells. `qualification=false` and `release_ready=false` are intentional.
Recorded binary/harness hashes are never replaced with hashes of newer files.
A mismatching or unavailable harness is an open reproducibility gate.

- [Statistical acceptance](../../benchmarks/results/flash-next-statistical-acceptance-m2-20260916.json)
- [Native build identity and checked source hashes](../../benchmarks/results/flash-next-native-build-m2-20260916.json)
- [Completed QA, long context and NLL](../../benchmarks/results/flash-next-extended-qa-v3-m2-20260916.json)
- [Trained-head falsification control](../../benchmarks/results/flash-next-mtp-head-oracle-m2-20260916.json)
- [Native HTTP / SSE matrix](../../benchmarks/results/flash-next-http-m2-20260916.json)
- [Batched MTP matrix](../../benchmarks/results/flash-next-mtp-batched-verify-m2-20260916.json)
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
about 14 seconds. Read actual behavior rather than that stale residency label.

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
The 4-bit tie runner also diverges outside the allowed tie margin (1.3125
versus 0.5), and its state control exceeds the logit bound (0.100864 versus
0.1). These are unresolved correctness failures, not accepted tie exceptions.

The primary default remains [Qwen 3.8 27B AXQ](qwen3.8-27b-axq.md) on
Mac mini M5 64 GB. See [Supported Models](../SUPPORTED-MODELS.md) and
[Testing](../TESTING.md) for the wider operator contract.
