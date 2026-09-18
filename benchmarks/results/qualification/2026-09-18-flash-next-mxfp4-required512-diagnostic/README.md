# Flash Next MXFP4 required-512 progress diagnostic

One installed Apple M5 Max 128 GiB required-MTP request completed all 128
outputs, but greedy identity failed against every one of the five original
direct trials: 37 initial IDs match; position 37 is direct `3070` versus
required `7701`, with 91 different aligned positions. Qualification, release
readiness and performance claims remain false.

This is installed source `85a2bab0dd72c9e2076f455237dca052ff8d12b7`, binary
`9d6725d82da906ef324d1601a939405b64779d4251ec3bf7ae5435e80d3cb4ed`, and pack
`0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35`, with NAS-backed model storage.
It is not a test of later verifier corrections. The original 512 input IDs,
128-output budget, seed 0, temperature 0, ignore-EOS, required MTP, Auto
expert streaming and disabled prefix reuse/ngram stacking are retained.
EOS token `248046` occurs at output index 1 and generation continues because
ignore-EOS is true; the output is not a quality qualification.

The complete terminal, client and trial records agree with all 76 SSE events
(74 steps), their token deltas and cumulative progress. Both decoded SSE
lines and event objects are preserved with private strings sanitized; they
still parse into matching events. Decoded strings are not byte-exact network
packets. All numbers, token IDs, event order and observations are retained.

The 76 resource samples include one retained metrics connection-refused error
during readiness and 75 HTTP-200 observations. RSS is sampled, not a measured
peak. The canonical trial's `peak_memory_gb` field is one post-request RSS
sample despite its name. Host VM/swap counters cannot isolate this request,
NAS traffic, GPU occupancy or a causal bottleneck. Physical NAS I/O is
explicitly unavailable.

The frozen helper validates all 68 installed/model files before and after
execution, against a pinned inventory (20 installed/runtime and 48 model
snapshot files). The terminal retains source-bound completion receipts,
not per-file observed pre/post digest maps. The curator verifies those
bindings; it does not reread remote payloads. All 21 transferred raw files
match remote-before, remote-after and local hashes; owned children exit 0
and retained process observations show all owned PIDs absent.

The request finishes within unchanged 1200-second readiness, 3600-second
request and 1800-second socket-operation bounds. It omits the five preceding
direct requests and adds log flushing/metrics sampling. Payload prehash
warms storage caches; request/cache order is explicitly not equivalent.
The [original matrix timeout](../2026-09-18-flash-next-mxfp4-workload/README.md)
remains preserved and unlocalized: its empty client log and absent completed
required trial do not prove zero generated tokens or identify the stalled
phase. This isolated completion does not resolve that failure, establish
repeatability, provide a throughput comparison or close a release gate.

Seven deterministic gzip artifacts retain terminal/client/trial data, both
journals, integrity and the local audit. Integrity includes the original
failed terminal and five direct trials. summary.json records compressed and
uncompressed hashes and numeric-preservation counts. Private paths, users,
hosts and network-share identities are hashed; public loopback/port and API
endpoint contracts remain. No model weights or activation payloads are included.
