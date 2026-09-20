# Flash Next MXFP4 QSA projection evidence

Supplementary Apple M2 Ultra, 192 GiB. Not MacBook Pro M5 Max qualification.
The target pack is AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP at
`0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35`. Qualification and release readiness
remain false. No throughput or memory-performance claim is made.

## Reproduced component difference

The original 41-token roses prompt and first accepted two-token verifier
window have exact BF16 attention inputs and inherited K/V cache. Native
main-key projection differs at two values between batch and singleton schedules
(maximum absolute difference 0.00390625). Original-weight MLX 0.32.2 replay
reproduces both native output schedules exactly. Per-row projections match
independent singletons; repeated schedules are exact.

The raw-index projection replay differs at one value by 0.00048828125 in a
raw-key channel. It also matches independent singletons when run per row.
Its native raw projection was not captured, so no native projection identity
is claimed for that probe. Query/gate, value and ordered gathers were exact
in this particular baseline window, not universally across inputs.

## Bounded candidate control

| Candidate | Equal generated positions | First differing stored state | Differing arrays at that boundary |
| --- | --- | --- | --- |
| GDN-corrected baseline | 3 | Layer 43 QSA K after the first accepted pair | 11 |
| QSA main-K/index correction | 3 | None in the compared accepted pair | No differing boundary recorded |

The correction changes only the verifier policy for MXFP4 main-key and indexer
projections. Ordinary Shared, non-MXFP4, existing RowExact head, normalization,
rotary, selection order, cache publication and expert paging remain unchanged.
All 47 model-file pre/post hashes match in both collections. Repeated prefill
is exact. No tolerances, sampling policy, token coverage or QA checker changed.

The baseline binary is bound by the previous GDN correction's source/patch
identity. The new binary is bound to base `293c6094` plus the recorded source
patch and four compiled-source hashes in integrity.json.gz. Build identity
must include that patch, not only the base commit. Raw failed or earlier
candidate observations are retained by the adjacent GDN/MTP evidence bundles.

Three generated BF16 controls exercise cache and mode boundaries; they are
not claimed to reproduce the actual model error on every device. The real-pack
baseline, exact-input replay and candidate native comparison are the concrete
regression evidence. The longer original two-prompt diagnostic and full M5/NAS
qualification remain separate gates; a three-token control is not release proof.

## Artifacts

- `summary.json`: bounded result and compressed/decompressed artifact hashes.
- `native-controls.json.gz`: complete baseline/candidate native JSON, including
  token accounting and state comparisons.
- `components.json.gz`: component summaries, both probes and captured-file
  identities; no model weights or raw activation arrays.
- `integrity.json.gz`: terminal records, pre/post model hashes, freeze and build
  provenance. Private absolute paths and host aliases are sanitized.

Gzip timestamps are zero. Numeric observations are preserved; hashes of original
private records identify those originals, not their sanitized public encodings.
The previous GDN candidate and target MTP/QA failures remain historical evidence
and are not relabeled as passing by this component correction.
