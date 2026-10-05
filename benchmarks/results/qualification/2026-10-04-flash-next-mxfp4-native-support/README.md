# Flash Next MXFP4 native functional support

Date: 2026-10-04 (local date). Functional controls pass; qualification and
release_ready remain false. MTP-S/P/D, model-quality certification and
performance are not assessed. No default MTP promotion or speed claim.

The dedicated qwen4_exp native server runs the current
AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP pack at
ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3 with MLX 0.32.3, Rust 1.97.1,
Mac Studio M2 Ultra, 192 GiB, macOS 27.0.1 and internal SSD. diskutil reports
Internal=true, SolidState=true, BusProtocol=Apple Fabric. The staging record
identifies the device as internal NVMe; no NAS, external SSD or other SKU
inference follows from this run.

All 49 published files (132,261,877,478 bytes) are hash-bound in
[pack-inventory.json](pack-inventory.json), including publisher LFS payload
hashes. The 15 changed trunk shards distinguish this revision from the
older 47-file 0b0bf6c1 pack; the BF16 31-tensor MTP payload is unchanged.
The Qwen upstream source revision remains de4b8e4d43b917e7706784d8bb445c9af86a3540.
The model copy is read-only to the probes except native manifest derivation;
full file hashes and the derived manifest hash are checked before and after.
Only the harness's child servers are terminated.

[result.json](result.json) contains raw requests, responses and metrics.
[build-receipt.json](build-receipt.json) binds the executed scripts, native
binary, runtime source hashes and MLX libraries. The isolated runtime build
excludes unrelated pending diagnostics, mmap, QA and documentation edits.
No model weights or executable binaries are committed. Logs replace account
home paths and listen addresses with placeholders; timestamps are retained.
The one initial negative-control oracle expected SDK Display text, while the
CLI prints Engine(MlxMtpRequiredButUnavailable); the session already rejected
activation correctly. The final oracle accepts the actual typed diagnostic.

## Functional matrix

Greedy text, temperature 0, thinking disabled, no logits processors, generic
n-gram acceleration disabled, expert streaming Auto. Direct and required MTP
run in separate processes. Requests repeat the same prefix and extend it;
the repeat increments the logical prefix-hit counter, but native-FA hit
counters remain zero. These controls do not certify every prefix/cache path.

| Case | Direct output tokens | Required output tokens | Required verifier steps |
| --- | ---: | ---: | ---: |
| count | 37 | 37 | 19 |
| repeat | 37 | 37 | 19 |
| extend | 38 | 38 | 19 |
| json | 24 | 24 | 12 |
| stream | 37 | 37 | 19 |
| after_stream | 37 | 37 | 19 |
| one_token | 1 | 1 | 0 |

All seven choice/text pairs match and count/JSON content checks pass. Direct
requests execute zero MTP steps. The six eligible required requests execute
107 verifier steps and accept 103 draft tokens. These request deltas exclude
the server's startup warmup counters. Attach, prefill-absorb and step-error
deltas are zero for every request. The one-token control completes its budget
with zero verifier steps, as expected. SSE includes finish, usage and DONE;
a subsequent non-stream request continues to verify through MTP.

The negative control sets AX_NO_SPEC=1 and requires MTP. Startup exits 1 with
MlxMtpRequiredButUnavailable, before serving. The positive modes clear inherited
AX_* and MLX_* overrides and keep default promotion false. The server still
reports MlxPreview. Non-greedy, logits-processor, multimodal or thinking controls
may use direct fallback and are outside this functional receipt.

## Reproduction

Use the native executable built with the pinned toolchain, the matching
MLX runtime, a verified current pack on the target internal SSD, and its
staging inventory. The output directory must not already exist.

```bash
python scripts/check_flash_next_native_support.py --dry-run
python scripts/check_flash_next_native_support.py \
  --model-dir /path/to/current-pack \
  --inventory /path/to/staging-inventory.json \
  --server-bin /path/to/native/ax-engine-server \
  --output /path/to/new-functional-receipt
```

The metadata-only qualification script remains non-promoting. The current
49-file pack and this source require new broad numerical, long-context,
quality, installed delivery, lifecycle, memory and performance evidence
before product qualification. Historical affine and older MXFP4 receipts
remain unchanged; MXFP8 admission remains pending.

## Software validation

[software-gates.json](software-gates.json) records pinned formatting, strict
all-target/all-feature clippy, 4,068 Rust test passes (62 ignored), the final
233-test catalog follow-up, the release-pyext maturin install, 285 Python tests
and 165 subtests (22 skipped), deterministic script gates and both qualification
dry runs. The script gate runs in an isolated candidate because unrelated
local wiki exports contain retired references; those exports are preserved.
All three requested read-only CLI reviewers completed: Claude Opus 5.5,
Muse Spark 1.3 contributor and Grok 4.7 Build. Their concrete manifest/storage
and process-refusal oracle feedback was applied and tested.
