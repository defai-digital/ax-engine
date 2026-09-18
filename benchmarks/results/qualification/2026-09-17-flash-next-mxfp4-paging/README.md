# Flash Next MXFP4 diagnostic paging and pack audit

This tranche enables explicit MXFP4 expert paging behind the existing family
opt-in. It does not qualify the full model, MTP, performance or release.

The pinned target pack audit records all 47 files, 26 safetensor headers,
2690 tensors, 2659 index entries and 288 stream entries. It preserves the
528 MXFP4 projections, 227 affine8/group32 overrides, affine8/group64 output
head and 31 BF16 MTP sidecar tensors. The first Hub attempt failed decoding a
CAS response; the second resumed attempt used unchanged defaults and passed
all file hashes. Storage was NAS over SMB on M5 Max 128 GiB.

Small native controls run on an M3 Max build host. They cover known MXFP4
codes and E8M0 exponents, generated F32/BF16 projections, routing order and
duplicates, selected/full-layer prefill and decode, byte caps, malformed
sidecars, truncation and recovery. These results are not M5 measurements.

The complete target matrix remains open: installed delivery and QA,
independent numerical and MTP state/runner controls, long context, nine
throughput cells, cold latency and peak memory. Six-bit is out of scope.

Validation: 3710 Rust tests passed, 46 ignored; Python 199 passed,
36 skipped. Formatting, script gates and CI-policy Clippy passed. Strict
restriction-lint Clippy failed on existing test uses of expect; this is not
reported as strict all-lints green. Logs retain expected negative-control
MLX errors; process exit and final test summaries determine the result.

output-head-repro.txt is the expected failing regression before the
role-specific affine8/group64 fix; output-head-fixed.txt passes afterward.
The final workspace log includes the corrected regression. Source file
hashes and exact commands are recorded in validation.json.

The installed candidate wheel passes 81 packaging/API tests on the M3 Max.
On M5 Max, bundled runtime doctor, metadata generation, default MXFP4 rejection
and explicit-opt-in metadata validation pass. installed-preflight.json records
the wheel, native binaries, model inputs and exact positive/negative commands.
The pinned stock mlx-lm 0.31.3 class lookup rejects qwen4_exp; its primary
reference gap remains open. The preflight artifact establishes admission;
native-selected-r2.json separately records the completed selected-route control.

The first M5 selected-paging direct request completed 32 output tokens with
zero MTP draft tokens and normal server exit. The paired required-MTP request
did not start: the harness reused the same listening port after shutdown and
its bind probe failed. native-selected-attempt-1.json preserves the partial
result and error. The retry uses distinct ports and a separate output tree;
it does not replace or relabel this failed paired control. Full payload
hashing before startup means this is not cold-storage latency evidence.

The observed 85a2bab0 health metadata still contained the former Studio SKU
note. Current source changes that generated note to the M5 Max MXFP4 MTP
qualification target, explicitly pending. The running retry keeps the frozen
85a2bab0 binary and manifest unchanged; it does not claim the note fix is
already installed on that target.

The selected-route retry completed and passed both requests plus full post-run
file hashes. Direct and required MTP produced identical text and usage (46 input,
32 output tokens). Direct drafted zero tokens; required MTP drafted 16 and
accepted 14. Both servers exited normally. This is one fixed-input diagnostic,
not trained-head certification, full QA, or a throughput comparison.

The default Auto route also passed both requests, identical text/usage and
complete post-run file hashes (native-default-r2.json). It used no selected-row
overrides and recorded zero selected gathers. Both servers exited normally;
MTP certification remained off. The required request was observed with a
three-second live stack sample, so its elapsed time is not benchmark evidence.
No performance ratio is inferred across these single-request controls.

native-memory-samples-r2.jsonl retains all172 process/system observations;
native-memory-observation.json summarizes their scope and sampled RSS maxima.
System swap used remained0.25MiB across these samples. This neither attributes
swap to the model nor proves a zero-swap guarantee for longer workloads.
Ten-second process samples can miss peaks. MLX allocator peaks in the native
results are separate from process RSS and system filesystem-cache accounting.

mxfp4-state-controls.json adds four tiny full-trunk controls on the M3 Max build
host: F32/BF16 fixtures, each with selected decode and selected prefill. Only
synthetic expert matrices are quantized to MXFP4; other tensors are unchanged.
All four controls require exact resident/selected logits and serialized state,
unchanged committed state after an injected expert-file read failure, and exact
recovery after restoring that file. Fixture and log hashes are retained. These
are storage/state unit controls, not full-checkpoint M5 numerical qualification.
