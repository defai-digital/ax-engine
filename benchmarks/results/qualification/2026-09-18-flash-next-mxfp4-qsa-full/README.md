# Flash Next MXFP4 full QSA diagnostic

Supplementary Apple M2 Ultra 192 GiB evidence for source `358bac91d5340e2b45b55a14e6cfb13b5fb5c63f`,
pack `0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35`. Collection completed valid with all 47
pack files unchanged before/after. Qualification and release readiness remain false.

Both requests have 109 exact prefill state arrays and exact repeated prefill.
With a 256-token budget, `reasoning_cause_effect` matches 89 token positions
including terminal EOS 248046, but state first differs after generated prefix
2: zero-based layer 24, `gdn.conv`, with 54 differing state arrays.
`reasoning_syllogism_roses` matches 13 positions, then differs at zero-based
position 13 (bonus token): direct 3750 versus MTP 15394. Its state first
differs after generated prefix 4: zero-based layer 8, `gdn.conv`, with 90
differing arrays. The direct top-two margin at the token mismatch is zero;
this remains a failed identity comparison, with no near-tie waiver.

The complete sanitized native records retain numeric state/logit summaries,
checkpoint replays, token IDs, and tensor geometry, but no model weights or
captured activation payloads. `summary.json` binds deterministic gzip
artifacts to compressed and uncompressed SHA-256 values. Original private
input hashes and frozen source/binary/library identities are in `integrity.json.gz`.

This native diagnostic does not establish installed HTTP trajectory identity,
M5 qualification, quality, or throughput. The earlier
[three-token QSA evidence](../2026-09-18-flash-next-mxfp4-qsa/README.md)
remains unchanged and does not establish longer-trajectory state parity.
