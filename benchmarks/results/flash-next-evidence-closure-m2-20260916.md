# Flash Next development evidence closure

These recovered M2 Ultra 192 GB records do not establish release readiness.
Each JSON records the original binary hash, the raw input hash, the recorded
harness hash and (where available) the surviving harness hash separately.
A summary pass does not override `release_ready=false`.

| Artifact | Interpretation |
| --- | --- |
| `flash-next-native-build-m2-20260916.json` | Original build record for `1819e4bb`; all four recorded source hashes match Git, and both test/server hashes match their campaign artifacts. |
| `flash-next-extended-qa-v3-m2-20260916.json` | All three routes completed 105 items plus the NLL phase. Original QA acceptance fails on two direct/MTP text mismatches. The surviving harness differs from the recorded hash. |
| `flash-next-mtp-head-oracle-m2-20260916.json` | Real head 95/114 acceptance, permuted 0/207. Only 50 requests per head contribute; 54 short requests are excluded. The harness hash verifies. The recorded 70% threshold is retained, not replaced by a new certification rule. |
| `flash-next-mtp-batched-verify-m2-20260916.json` | All 12 primary/tie state and runner controls for 2/4/6-bit completed: 9 pass and 3 fail. Completion does not override the failed verdict. |
| `flash-next-http-m2-20260916.json` | Six modes and 12 requests completed: four pass, two fail direct/MTP text identity (4/6-bit required). SSE, usage, repeat identity, unchanged metadata, default MTP off, and clean shutdown pass throughout. |
| `flash-next-throughput-ab-m2-20260916.json` | Preserves partial cells and short outputs. Fixed decode requires all measured samples to emit the requested token count; warmups and early EOS cannot substitute. |

The original 105-item QA reports direct/MTP hard passes 102/105 and reference
101/105. Direct and MTP differ in `reasoning_cause_effect` and
`reasoning_syllogism_roses`. The original scorer and failures remain unchanged.
AX NLL is 1.96225664 versus reference 1.96609234 on 3,999 scored tokens;
MTP uses the same prefill logits and is not an independent NLL measurement.

The 2-bit near-tie control on binary
`79f30efefe5c2e889944f2efa3b4bb53f194d01c6820ef3314f3db3b78328060`
(commit `1819e4bb`) failed with sequential token 271 versus batched token 198,
margin 0.9375 exceeding the fixed 0.5 tie margin. Both primary controls and the tie runner passed. All four 6-bit controls passed.
The failure is retained; no tolerance or production model math was changed.
The 4-bit primary controls also pass, but its tie state control has relative
logit error 0.10086382 above 0.1 and its tie runner differs at position 2:
direct 271 versus MTP 561, margin 1.3125 above the unchanged 0.5 bound.

The throughput host was not isolated: system indexing and a background file
sync were active during collection. These measurements remain development
diagnostics and cannot support a controlled public performance claim.

## Re-curation

Use the raw campaign JSON and its historical harness, not a newer script that
happens to occupy the same path. The utility retains an unverified hash as an
explicit reproducibility gap rather than overwriting the recorded identity.

```bash
python3 scripts/curate_flash_next_evidence.py qa \
  --input /path/to/raw/extended-qa-v3.json \
  --harness /path/to/raw/run-extended-qa-v3.py \
  --output /path/to/output/qa.json
```

The other kinds are `head`, `mtp`, `http`, and `throughput`. Native completion and SSE
controls, target-SKU qualification, final merged-binary verification and broad
quality acceptance are separate gates. See the
[qualification record](../../docs/model-certifications/qwen3.8-flash-next.md).
