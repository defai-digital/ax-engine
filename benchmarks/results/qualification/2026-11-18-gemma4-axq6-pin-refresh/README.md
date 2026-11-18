# Gemma 4 AXQ 6-bit pin refresh

The previous immutable 26B-A4B and 31B catalog revisions returned HTTP 404
for authenticated metadata lookups. The new snapshots each contain 32
verified files, including a depth-one assistant contract, assistant config
and assistant weights. Their target and assistant tokenizers are
byte-identical. Both snapshots ship without native manifests.

| Pack | Verified revision | Publisher bytes |
| --- | --- | ---: |
| [26B-A4B AXQ 6-bit MTP](26b-a4b-inventory.json) | `a772139a221fbf6f2fdc962ebb0747d8961c1a67` | 24,324,125,953 |
| [31B AXQ 6-bit MTP](31b-inventory.json) | `de5906d049f00b464673f5f48dfb2b473df0ff2d` | 29,384,488,645 |

On the 26B-A4B publisher snapshot, the previous loader completed the direct
request with MTP unavailable. The corrected loader creates and validates
the absent assistant manifest during attachment. It preserves the rejection
of malformed existing manifests and mismatched model families. The
[pre-fix 26B-A4B response](m2-26b-before-attach.json) binds the old executable.

The revised executables use Rust 1.99.1, MLX 0.32.4 and the dev profile.
They were built on `cd3fbea1b91bc0da2800598c7d7110cdc55e1bc6`
with patch SHA-256
`ee7241e7990a53dd48365a86dd20533b940b6c3aafe27a1d18177a293d815f00`.
Each real-weight pair submitted the same token IDs to loopback
`/v1/generate` with a 24-token budget, greedy sampling, EOS ignored and
the assistant depth capped at one. It ran explicit disabled and required
MTP arms with one concurrent request. The M4 Pro conversion regression
passed the absent-manifest and corrupt existing manifest controls.

| Pack and hardware | Verified drafts | Accepted | Output IDs match |
| --- | ---: | ---: | --- |
| [26B-A4B, M5 Max 128 GB](m5-26b-a4b-functional.json) | 11 | 11 | 24/24 |
| [26B-A4B, M2 Ultra 192 GB](m2-26b-a4b-functional.json) | 11 | 11 | 24/24 |
| [31B, M5 Max 128 GB](m5-31b-functional.json) | 11 | 11 | 24/24 |
| [31B, M2 Ultra 192 GB](m2-31b-functional.json) | 3 | 2 | 24/24 |
| [12B local pair, M2 Ultra 192 GB](m2-12b-local-pair-functional.json) | 3 | 2 | 24/24 |

The 12B fixture pairs the [published AXQ target](12b-target-inventory.json)
with the official assistant pinned at
`3b0b0a4a3b31b5556516329227fb55f275c23c9f`. It uses the local preparation
helper and the target tokenizer. With the publisher chat template, the
required arm exercised acceptance and rejection. A separate
[raw-prompt probe](m2-12b-raw-zero-draft.json) finished with the assistant
attached and zero drafts. The catalog 12B assistant-MTP revision remains
unavailable from the publisher.

The two refreshed catalog entries are Candidates. These bounded controls
cover attachment, draft verification and paired token IDs. Phase-1
qualification, throughput and default promotion remain open.
