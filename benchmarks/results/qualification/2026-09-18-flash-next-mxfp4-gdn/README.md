# Flash Next MXFP4 GDN projection attribution

Status: component correction verified; whole-model qualification remains open.
Hardware: supplementary Apple M2 Ultra, 192 GiB. This is not M5 target or
throughput evidence. Pack revision: `0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35`.

## Observations and change

With the original 41-token roses input, the first accepted two-token verifier
window changes stored state before any generated token differs. The original
layer-zero capture reproduces the uninstrumented request, first step and state
summaries exactly. It does not prove unrecorded tensor equality.

Actual BF16 inputs and original MXFP4 weights reproduce one changed QKV value
and one independently changed GDN output-projection value, each with absolute
difference `4.76837158203125e-7`. Both execution schedules repeat exactly.
Per-row projection matches independent singleton projection in both probes.
The candidate uses existing per-row MLX projection for these two verifier
operators. Direct/prefill, affine routing, expert paging and admission policies
retain their existing behavior.

The same three-token native experiment gives:

| Variant | First changed stored state | Nonzero state arrays |
| --- | --- | --- |
| Original | Layer 0 GDN convolution | 76 |
| QKV correction only | Layer 18 GDN convolution | 68 |
| QKV and output correction | Layer 43 QSA keys | 11 |

All three generated-token prefixes match their direct controls for these three
positions. Every candidate completes all 47 pre/post model-file hashes.
**State differences remain.** These observations do not close longer generation,
M5 direct/MTP identity, installed QA, numerical, latency or throughput gates.
No tolerance, answer checker, default admission or certification is promoted.

## Records and provenance

- `summary.json`: scope, first differences and artifact hashes.
- `native-controls.json.gz`: original and both candidate native records, including
  prompt IDs, proposals, logits summaries and every measured state difference.
- `components.json.gz`: captured component comparisons, tensor hash inventory,
  independently replayed projection results and original integrity hashes.
- `integrity.json.gz`: both candidate source-base/patch/file and binary identities,
  plus complete pre/post model hashes. Candidates were built from recorded
  worktree patches; the base commit alone is not their source identity.

Private filesystem paths are replaced with placeholders. Numeric evidence,
model identifiers, source hashes and recorded failures are retained. Gzip uses
mtime zero. Raw weight and activation payloads are not distributed here.
The earlier unfiltered capture exceeded its 2 GiB limit and is not qualifying
evidence; the filtered capture preserves that limit and excludes expert weights.

Local verification: 3,716 Rust tests pass, 48 remain ignored; Python 209 pass,
26 skipped; six focused GDN tests, F32/BF16 MTP budget/terminal controls,
formatting, CI-policy Clippy and extension rebuild pass. Strict restriction
Clippy retains baseline failures. These software gates do not replace target
model qualification. The model-size generated controls require rowwise identity
but do not assume every backend exhibits the original batch discrepancy.
