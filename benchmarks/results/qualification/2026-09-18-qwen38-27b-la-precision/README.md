# Qwen linear-attention gate precision

Selected SKU: Mac mini M4 Pro 64 GB (Mac16,11). Pinned checkpoint:
`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` at
`3e290738e96972307c6aeb9934ab170ca0eae1c1`.

## Root cause

The MTP-only linear-attention output projection used a generic low-precision
SiLU/multiply composition. Ordinary gated normalization instead computes its
gate in float32, casts before projection, and selects full/narrow Metal gating
according to the layer policy. These are distinct from FFN SwiGLU's required
low-precision intermediate tensor boundaries.

An isolated BF16 identity-projection test failed with maximum absolute error
0.015625. `21687d71` restored float32 gating but retained a portable-only graph.
It passed that test and the scoped qualification, yet the longer counterexample
still diverged at output index 5. A second regression proved that portable-only
gating also differs from ordinary relaxed full-Metal gating.

`892c2fc9` shares the existing gated-normalization helper and layer-specific
kernel policy with ordinary target execution. It retains MTP and existing
flag defaults. Regression coverage includes BF16/FP16, S=2/4/8, exact scope and
both relaxed full/narrow gate policies. No reference implementation was copied.

## Fixed-token controls

Two original LINE_SET prompts were tokenized once with the pinned tokenizer,
thinking disabled. Their fixed `/v1/generate` requests use temperature 0,
seed 0, top-k 0, top-p 1, repetition penalty 1 and a 128-token cap. This raw
protocol is distinct from the earlier chat endpoint's 512-token diagnostic.

The pre-fix `ad999f3f` wheel splits compsec-092 at output index 5; turning off
only `AX_MLX_MTP_LA_OUT_PROJ_SILU_MUL_QMM` restores equality. compsec-086 already
matches within this shorter raw protocol. The intermediate `21687d71` wheel
has the same outcome. These controls identify the projection entry as a cause,
not every possible MTP divergence. [Token controls](token-controls.json) retain
all three source versions and the diagnostic override separately. Final
`892c2fc9` matches both cases on the default direct/MTP routes and the
diagnostic LA-off control; the index-5 split is closed within this protocol.

## Limits

Finite token agreement does not establish population accuracy, universal
sequence equivalence, or MTP Tier 2. Original strict LINE_SET gold remains
unchanged even where prompt/gold wording is ambiguous. This run does not
replace the original 32000-token thinking campaign or endurance testing.
The saved 32851-token wrong-answer recovery was not rerun on this final wheel;
its earlier Answer B against gold C remains unresolved historical evidence.
Prior throughput measurements are version-bound and do not qualify this repair.

## Source validation

Final `892c2fc9` passes formatting, full Rust tests, repository CI-policy
Clippy, both Qwen dry-run contracts, primary-claim checks and Python. The
rebuilt/installed extension also passes Python: 209 passed, 26 skipped and
140 subtests. The script suite passed on the preceding `21687d71`; the final
follow-up changes only the Rust gate policy and its regression coverage.
[Validation record](source-validation.json) includes log hashes and the
before/after precision regressions. CI-policy Clippy retains the repository's
five existing force-warn restrictions; the lint policy was not changed.

## Installed-wheel qualification and quality

Clean `892c2fc9` on the selected SKU passes the executable schema-2 gate:
direct/MTP each 32/32 hard QA, zero soft failures and 7/7 surface probes without
skips; doctor ready; installed wheel members and pinned inventory match.
The 64-token paired probe matches. Direct draft/verify counters are 0/0;
MTP counters are 63/86, proving active verification. The isolated environment
has ax-engine and pip only; native loading resolves MLX/JACCL from the package.
[Qualification](qualification.json) includes build and raw-artifact hashes.

The unchanged selected twelve-case LINE_SET diagnostic retains original gold,
original wording, answer-only system instruction, thinking disabled and a
512-token output cap:

| Route | Correct | Wrong | Truncated |
| --- | --- | --- | --- |
| Direct | 1/12 | 2/12 | 9/12 |
| MTP | 1/12 | 3/12 | 8/12 |

Only **3/12** response texts match. The new raw-control pass therefore does not
close broader route consistency or quality. [Strict grades](quality-diagnostics.json)
are selected failure evidence, not an estimate of overall model accuracy.
All twelve server-rendered chat prompts and token arrays exactly match the
pinned tokenizer; [template checks](template-comparison.json) rule out framing
mismatch for these particular requests.

## Reproduction

Replay the exact requests in `token-controls.json` against separately started
direct (`--disable-ngram-acceleration`) and MTP
(`--mlx-mtp-disable-ngram-stacking`) product servers. The `mtp-la-off` arm adds
only its explicitly recorded diagnostic environment override. Use the recorded
wheel/source version and pinned pack; compare output token arrays, not decoded
text. For the live release gate, follow the
[installed-wheel procedure](../2026-09-17-qwen38-27b-swiglu-consistency/README.md#reproduction)
with this build manifest and a fresh output directory. The quality diagnostic
uses chat requests and a different output budget, so it remains a separate
protocol and must not be merged with the two raw-token controls.

## Subsequent cache isolation

An identical raw compsec-086 request produces four tokens in a fresh direct
server, but reaches the 128-token cap after seven other prompts with one output
token each. The warm request records one scheduler retained-cache hit and one
branch prefill, while runner prefix-cache hits and warmup tokens remain zero.
This identifies a separate context-restoration investigation; these quality
results must not be attributed solely to model capability. The grid-trim path
can omit the scheduler-claimed prefix without requesting full recomputation.
A subsequent fix requires its own source-bound wheel and cold/warm validation.
