# Qwen GDN activation rounding repair

Status: **Candidate; not ship-ready**.

`aa38f18bd4c7f6894cbd346372f42dd1717d1bf0` restores the activation-dtype
boundaries omitted by AX's fused conv/SiLU/QK-normalization prework. The
default prework, optional SIMD32 prework, and both fused verifier variants
share the correction. Recurrent-state arithmetic, MTP admission, fastpath
defaults, sampling, and acceptance rules are unchanged.

Selected SKU: **Mac mini M4 Pro 64 GB (Mac16,11)**. Checkpoint:
`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` at
`3e290738e96972307c6aeb9934ab170ca0eae1c1`.

## Reference-guided attribution

The local reference comparison identified the stored activation boundaries in
the composed GDN path: convolution, sigmoid, SiLU product, RMS normalization,
and scaling. AX's fused implementation retained float32 intermediates across
several of these boundaries. Restoring only the final cast is insufficient.
FP16 also requires precise float32 exponential/division followed by the
corresponding activation-dtype rounding, as the standalone primitive check
demonstrates. References were design input; no reference code was copied.

[Instrumentation and reference provenance](instrumentation.md) records
same-cache, same-input controls and the exact reference revisions. The observed
pre-fix layer-zero Q/K/V differences disappear in the instrumented intermediate
fix, and same-input recurrent-state differences change from 48/48 layers to
zero. Tape versus ordinary recurrence with identical Q/K/V is exact, as are
all 240 same-body compiled/eager leaf comparisons in each captured run.
The evidence does not implicate the acceptance rule or outer compile boundary
for that captured split.

Those instrumented fixed observations **precede the final FP16 correction**.
They are not measurements of the final wheel. Full-layer hidden differences
remain, including two differing layer-zero elements, and each run has its own
preceding state history. The source-bound diagnostic patches are preserved as
[pre-fix](post-input-stage.patch) and [intermediate fixed](stage.patch); apply
either to its recorded `d0fbb3a7` base, never to the final release source.

## Final-commit cold token controls

The clean `aa38f18b` native server was built separately, with its binary hash
recorded in [cold-controls.json](cold-controls.json). Every arm starts a fresh
process and uses identical input token IDs, greedy seed 0, and a 192-token cap.
Direct disables n-gram acceleration; MTP disables n-gram stacking. No numerical
override is set. This native-server control uses the previous installed
wheel's MLX libraries; it is separate from bundled-wheel qualification.

First differing output indices are zero-based; a dash means all 192 tokens
match. Reference columns use the unchanged pinned MLX 0.32.2 / mlx-lm 0.31.3
[reference records](../2026-09-18-qwen38-27b-residual-state/independent-reference.json).
For compsec-079 only, the reference ran with a separate 512-token budget,
so its comparison here is prefix-only.

| Case | Old direct/MTP split | Final direct/MTP split | Final direct/reference | Final MTP/reference |
| --- | --- | --- | --- | --- |
| compsec-077 | 161 | — | — | — |
| compsec-079 | 46 | 116 | 116 | 137 |
| compsec-087 | 122 | 155 | 155 | 177 |
| compsec-092 | 182 | — | 182 | 182 |

Paired agreement improves from 0/4 to 2/4 on these selected cases. This is
not a general fidelity or quality improvement claim: old MTP matched the
reference for compsec-092, whereas both final AX routes now differ from it.
Direct is not an independent ground truth. Broad route consistency and
reference agreement remain open.

## Final bundled-wheel qualification

The clean `aa38f18b` wheel has SHA-256
`8177d8a18b986dfe6d0997db7802feb8f908397e067be3e71f731cdea99d9cbb`.
It passes the executable qualification on the selected SKU, macOS 26.6.2,
in a new isolated environment containing only AX Engine and pip:

- doctor ready, clean matching source, pinned model inventory, installed
  wheel-member identity, and native import pass;
- direct and MTP each pass 32/32 hard QA and 7/7 surface checks, with zero
  soft failures or skipped surface checks;
- direct reports zero MTP counters; MTP records 60 draft and 82 verification
  tokens, confirming active execution rather than a silent direct fallback;
- the paired 64-token greedy probe matches, with no runtime overrides.

[Qualification](qualification.json) preserves exact route probes, counts,
artifact hashes, and packaging evidence. The loader resolves MLX and JACCL
from `ax_engine/.dylibs`. This validates this bundled artifact on the selected
SKU; it does not establish portability to every supported host or a hosted
release. The original `release-pyext` load failure is not present in this
artifact.

The [first qualification attempt](qualification-attempts.json) rejected a
preparation mistake: the receipt used the bundled native CLI hash for the
installed Python launcher. The launcher was then verified against the wheel's
console-script entry point, its hash was recorded separately, and the complete
qualification was rerun successfully. Neither the wheel nor qualifier source
was changed to bypass the check.

This scoped pass does not override the broader token and quality failures.
MTP Tier 2, endurance, long-context decode-at-depth, and throughput campaigns
remain separate; no performance improvement is claimed here.

## Unchanged bounded quality diagnostic

The final bundled wheel runs the same twelve failure-selected questions per
route with unchanged requests, gold sets, strict grader, greedy sampling,
thinking disabled, and a 512-token answer-only cap. Request and gold identity
are checked against the previous `4cbda8b5` diagnostic.
[Preserved grades and responses](quality-diagnostics.json) show:

| Route | Correct | Wrong | Truncated |
| --- | --- | --- | --- |
| Direct | 0/12 | 7 | 5 |
| MTP | 0/12 | 7 | 5 |

Only 6/12 response texts match between routes, versus 8/12 previously.
Truncations increase from four direct / two MTP to five per route. These are
observed regressions in this bounded diagnostic; the rounding repair does not
close quality or imply better answers. The selected failure subset is not an
overall accuracy estimate. This answer-only replay does not replace the
original 32k thinking campaign or saved long-context recovery. The prompt/scorer
contract issue described below remains unresolved; no gold or grade is relaxed.

## Regression and source validation

The BF16 boundary regression fails before repair for sequence lengths 1 and
4, both v3/SIMD32 and ordinary/target scopes. After repair, both BF16 and FP16
pass exact Q/K/V and convolution-state comparisons. The BF16 D128 fused
verifier tests also pass exact output, recurrent-state, convolution-state,
and checkpoint comparisons for sequence lengths 2 and 4 and both checkpoint
variants. These deterministic dyadic fixtures isolate rounding; they do not
prove arbitrary convolution accumulation orders or every FP16 fused recurrence.
[Regression receipts](regression.json) and the preserved logs contain the
before/after results. No tolerance was relaxed.
Published logs redact absolute local build paths; their hashes are recorded
separately from the retained raw-log hashes.

Final source passes formatting, full Rust tests, Python (209 passed,
26 skipped, 140 subtests), default `maturin develop` using `release-pyext`,
complete script tests, both Qwen dry-run contracts, and primary-claim checks.
[Source validation](source-validation.json) records exact commands and log
hashes. CI-policy Clippy passes with the existing force-warn restriction
exceptions. The strict command ending only in `-D warnings` exits 101 on
unchanged core restriction-lint failures; that failure is not waived or labeled
green. Dry-run success is not hardware qualification.

## Reproduction

At `aa38f18b`, run the focused native regressions with Rust 1.97.1 and the pinned
MLX runtime on Apple Silicon:

```sh
AX_MLX_MTP_GDN_PREWORK_SIMD32=0 rustup run 1.97.1 cargo test -p ax-engine-mlx --lib preserves_bf16_rounding_boundaries
AX_MLX_MTP_GDN_PREWORK_SIMD32=0 rustup run 1.97.1 cargo test -p ax-engine-mlx --lib preserves_fp16_rounding_boundaries
AX_MLX_MTP_GDN_PREWORK_SIMD32=1 rustup run 1.97.1 cargo test -p ax-engine-mlx --lib preserves_bf16_rounding_boundaries
AX_MLX_MTP_GDN_PREWORK_SIMD32=1 rustup run 1.97.1 cargo test -p ax-engine-mlx --lib preserves_fp16_rounding_boundaries
rustup run 1.97.1 cargo test -p ax-engine-mlx --lib fused_gated_delta_verify_bf16_128d_matches_portable_prework
bash scripts/build-pypi-wheel.sh
```

Replay each recorded raw request against a fresh server with the recorded route
flags. Instrumented reproduction additionally requires its exact patch, base,
and captured window; successful execution alone is not observer fidelity.
The observer must preserve the original logits and output tokens as documented.

Original long-thinking quality, saved recovery, broader numerical consistency,
and MTP Tier 2 remain open. Endurance and throughput require separate campaigns;
these diagnostic timings are not performance evidence. The earlier report's
compsec-086 gold typo is corrected to `3,13-15` while preserving raw grades.
Its contiguous-span wording control is incompatible with that non-contiguous
gold and cannot resolve the prompt/scorer mismatch.
