# Flash Next MXFP4 M5 short native control

**Collection completed valid; numerical identity failed.** This is Apple M5 Max
128 GiB evidence for source `cd207324af669f37d5ebfe29430c0e4ea41f9a81`, native
binary `c7cceff3aba9163ce5444c63481dd6cf208210e777743bf8526b1ac70295b457`, and
pack revision `0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35`.
Qualification and release readiness remain false.

The frozen roses prompt contains **41 input IDs**, processed as a whole
**40-ID prefix plus one singleton**, with a five-output-token cap. Direct and
MTP both produce `[2665, 13, 271, 332, 67407]`. Two proposals yield one
acceptance; consumed window lengths are `[1, 2, 1]`, followed by the final
pending output. No token difference or early EOS is reported. Repeated prefill
and all 109 prefill state comparisons are exact.

Decode state is compared after consumed generated prefixes **1, 3 and 4**;
the final emitted token is not consumed by another forward pass. Prefix 1 is
exact. The first mismatch is at prefix **3**, zero-based layer **2**,
`gdn.conv`, BF16 shape `[1, 3, 10240]`, max_abs **0.03125**. At that prefix
**104 of 109 state arrays differ**. The maximum-relative row is layer 15
`qsa.k` (max_abs 0.234375, relative 0.03750000149011612); the greatest absolute
error across that inventory is 0.421875 at layer 44 `gdn.conv`. Prefix 4 also
has a nonzero aggregate state difference. Earliest stored state does not prove
exact incoming hidden inputs or identify a particular faulty operator.

All 47 original model members, totaling **132,261,853,669 bytes**, match their
full SHA-256 identities before and after execution. The generated manifest is
an additional 48th snapshot entry and matches
`4252bb87db718f88e7e2eae3a08abe7e118999ffe8b664e301574348b5195512`.
The source-bound preflight, interpreter, bundled MLX/JACCL/Metal identities,
native environment and cleanup receipts are retained. This uses the original
forced-streaming native control with a one-layer expert cache; it is not a default
installed HTTP workload. All owned processes/groups were observed gone with
no timeout, signal or cleanup error. The native test's successful exit means
collection completed, not that numerical parity passed.

`native-diagnostic.json.gz` preserves the complete original numerical record,
including tokens, margins, all state descriptors/scalars and tensor metadata.
`integrity.json.gz` preserves complete terminal/provenance and all 47 pre/post
identities. Both gzip files use mtime zero and have compressed/uncompressed
SHA-256 bindings in `summary.json`; original private-input hashes are also
retained. Private paths are replaced with stable opaque labels; UUIDs and
private host/credential/URL strings are redacted. No model weights, captured
activation payloads, executable or raw crash report is included.

The frozen worker's line-anchored dyld parser records 549 paths. Independent
parsing finds 550: one allowed system AGX driver entry follows Rust test text
on the same line. The preserved raw log is sanitized but this observation
omission is not rewritten. The only non-system images are the admitted native
binary and bundled MLX/JACCL.

The retained completed M2 short control has different token/prefill observations;
that comparison is descriptive and grants no cross-hardware identity waiver.
`separate-m2-context.json` records the separately stopped M2 full diagnostic:
47 pre-hashes, zero post-hashes, incomplete collection, and native
`EXC_BAD_ACCESS / SIGBUS / KERN_MEMORY_ERROR`. It has **no numerical result**.
The underlying mapping/storage cause and missing supervisor terminal remain
unresolved; no raw crash identifiers are published.

This five-token record does not establish full native parity, installed
lifecycle/QA, paging performance, throughput or release readiness. Earlier
[shared-gate evidence](../2026-09-18-flash-next-mxfp4-shared-gate/README.md)
remains unchanged.
