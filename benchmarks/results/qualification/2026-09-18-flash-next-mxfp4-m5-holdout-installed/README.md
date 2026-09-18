# Canonical M5 holdout and candidate installation

Four-prompt native holdout: collection PASS and numerical identity PASS.
Candidate installation: PASS. Product qualification remains open.

Source `bf06cbb22c93d3a52b12cac4c31f396a36066da2`; native binary `9b78d73551fed8462c0bc9d557c593d785fe56408db547d81e802d95962837bc`;
proper candidate wheel `cc37411b0c5f7e24deb662c21f225441c1f8d2eccc8e9bb983d88eb094abe59a`. The target is Apple M5 Max
128 GiB with the pinned mixed-format MXFP4 MTP pack on NAS/SMB storage.

The four previously frozen prompts contain 66/72/76/70 input IDs. Each produces
64 equal output IDs under the original 64-token cap, with no padding, expected
output forcing or tolerance change. Each records 109 exact prefill state arrays
and compares 63 consumed positions; the final emitted token is not consumed by a
further forward pass. All actual primary and aligned final state/hidden/full-logit
witnesses are exact. All 47 payload pre/post hashes and clean process ownership
pass. This is forced-streaming native evidence with one cached expert layer,
not installed Auto/default or performance qualification. The root performed a
second-method reconstruction from raw token rows; no external reviewer verdict
is claimed for that reconstruction.

The exact proper wheel was installed in a separate target environment. All 21 AX
package members match; 28 tokenizer members and 52 runtime files are bound.
Isolated import and bundled-runtime doctor pass. Default MXFP4 remains rejected;
explicit experimental opt-in validates the existing manifest without rewriting it.
Installed MLX/JACCL/metallib bytes match the diagnostic runtime. All installer
commands exited as expected and their owned groups are gone. No model request
was executed by this installation check. Dependencies are version-pinned and
actual tokenizer bytes are captured; no prior immutable third-party wheel build
is asserted.

`native-diagnostic.json.gz` retains the complete native numerical record unchanged.
`integrity.json.gz` retains the holdout collection, source/runtime/freeze, preflight,
transfer, 47 pre/post hashes, ownership and raw logs. `installation.json.gz` retains
actual command outputs, installed roles/files, installation freeze, transfer and
root review. All numeric/null/boolean types and values are preserved. Private
paths, hosts, URLs, addresses and credential strings are redacted; unrelated
whole-host process snapshots retain their hashes. `summary.json` binds compressed
and decompressed bytes plus original private input hashes. Gzip timestamps are zero.

The fourteen-request lifecycle, full 105-per-mode QA/long context, target MTP
controls, selected/default comparisons, memory/cold latency, nine-cell performance
matrix, required primary mlx_lm baseline and fresh-cache delivery remain separate
open gates. Historical numerical, QA, timeout and download failures remain intact.
No throughput or MTP speedup, release readiness or default promotion is claimed.
