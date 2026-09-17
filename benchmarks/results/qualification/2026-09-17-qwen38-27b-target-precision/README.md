# Qwen 3.8 27B target precision and paired-route qualification

Status: **Candidate; not ship-ready.**

The target output-head correction preserves the pinned checkpoint's BF16 head
for both singleton decode and multi-row verification. Previously, production
loading unconditionally created a lossy 2-bit cache used only for singleton
rows. This was a target-distribution change, not just a storage optimization.
Commit `1b15fdf0` removes that automatic substitution from normal and pipeline
loaders. Explicit draft-head quantization remains separate.

The regression failed before correction (`max_abs=0.124999985`) and passed
afterward. Preserving weights does not establish bitwise kernel equivalence.

## Fixed-input comparison

The selected SKU is Mac mini M4 Pro 64 GB (Mac16,11). All runs use checkpoint
`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` at
`3e290738e96972307c6aeb9934ab170ca0eae1c1`. The separate reference environment
uses MLX 0.32.2 and mlx-lm 0.31.3. Nine predeclared token sequences, sampling
settings, outputs and build hashes are saved in [token-comparison.json](token-comparison.json).
The eight chat inputs were tokenized once and then reused as raw IDs, alongside
one synthetic raw-ID probe. This small diagnostic set is not overall model
accuracy or a qualified performance comparison.

| Build / diagnostic profile | Direct equals reference | MTP equals reference | Direct equals MTP |
| --- | ---: | ---: | ---: |
| Before (`e2b3e354`), defaults | 3/9 | 8/9 | 3/9 |
| Corrected (`1b15fdf0`), defaults | 7/9 | 8/9 | 8/9 |
| Corrected, FFN compile off | 7/9 | 8/9 | 8/9 |
| Corrected, gate/up matvec off | 9/9 | 8/9 | 8/9 |
| Corrected, both off | 9/9 | 8/9 | 8/9 |

After correction, default AX routes agree on all eight chat cases. The raw-ID
probe still differs, now at zero-based output index 14 (previously 25).
The retrieval reference differs from both default AX routes in list formatting;
all answer Ridge at 08:30. Token inequality is not itself a semantic failure.
Diagnostic overrides do not qualify product defaults. No fastpath default was
changed to make this selected probe pass.

The pack is mixed precision: its config defaults to 4-bit/group-size 32 with
per-tensor overrides, so the custom 4-bit FFN matvec is eligible despite the
6-bit pack name. Its fused intermediate arithmetic differs from stock split
operations. Isolated overrides narrow the remaining investigation, but do not
prove an equivalent replacement profile across both routes.

A [reference-only head-layout control](head-layout-reference.json) produced
identical outputs before/after a contiguous transpose on synthetic and retrieval
inputs. At synthetic index 14 the reference's two leading BF16 log probabilities
are tied (-0.875). This is reference evidence, not a measurement of AX's logit
margin and not an exception to the paired gate.

## Release gate and scope

Commits `c73dc6a8` and `875aa5da` require identical paired greedy output in live
qualification, save explicit greedy requests and hash both requests/responses,
and distinguish the new contract with result schema 2. A standalone QA pass,
preflight or dry-run is insufficient. The earlier health-only qualification at
`e2b3e354` is historical and fails the new paired validator at index 25.

Final SKU run on clean `875aa5da`: **failed** at paired output index 14,
process exit 1. Direct and MTP each passed 32/32 hard QA with zero soft
failures and 7/7 surface probes without skips. Each route uses the same
16 questions in two streaming modes, not 32 independent questions. Actual
server counters were direct 0 draft / 0 verify and MTP 63 draft / 86 verify.
Doctor was ready. The isolated environment contained only ax-engine and pip;
native import loaded MLX and JACCL from the installed wheel's `.dylibs`.

[qualification.json](qualification.json) preserves the failed verdict, clean
source and bundled-wheel hashes, route evidence, and probe hashes. Wheel SHA-256:
`c844d6c790d6ca393e5a5eedb80c90982407962a83ae92d1fbb435fd95918ebe`.
This establishes selected-SKU packaging and product-health evidence, not a
successful release qualification. Reproduce using the clean source and installed wheel with
`scripts/qualify_qwen38_27b.py --run` as described in
[Testing](../../../../docs/TESTING.md#primary-qualification-mac-mini-m4-pro-64-gb).
Raw internal logs retain machine paths; public artifacts omit them.

The prior LINE_SET/recovery campaign has not been rerun under this correction.
Its results remain historical; these token comparisons do not close broad
quality, MTP Tier 2, long-context or endurance gates. Historical throughput
numbers from the lossy direct head must not describe the corrected runtime.
