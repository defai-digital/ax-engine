# Low-precision activation and route consistency

This evidence covers the selected Mac mini M4 Pro 64 GB (Mac16,11),
macOS 26.6.2, and the pinned Qwen 3.8 27B AXQ checkpoint at
`3e290738e96972307c6aeb9934ab170ca0eae1c1`.
**Candidate; MTP Tier 2 and release-ready gates remain open.**

## Root cause and correction

The singleton dual gate/up projection kernel fused SiLU and the up multiply
in float before a single low-precision cast. The packed SwiGLU path had the
same float-only activation behavior. These operations do not preserve the
stock MLX BF16/FP16 tensor boundaries and activation semantics. An isolated
BF16 test with binary-exact quantization scales and one-hot input failed on
the original kernel with maximum absolute difference 0.001953125, eliminating
projection reduction-order differences from that reproduction. The old packed
BF16 test also failed when its 0.02 allowance was replaced with exact equality.

`891385f8` emits gate/up tensors in the input dtype and applies the existing
MLX `silu_mul`; packed BF16/FP16 rows use the same activation. Float32 packed
optimization remains. On this 6-bit dense `qwen3_5` checkpoint the default
runtime exercises only the packed dense SwiGLU path
(`AX_MLX_DENSE_SWIGLU_PACKED_METAL`, default on); the corrected singleton
gate/up matvec kernel admits 4-bit weights only unless
`AX_MLX_QWEN_DENSE_FFN_MATVEC_EXT_BITS` is set, so it did not contribute to
the evidence below. The opt-in prefill dual-QMM MMA kernel and the opt-in
fused MoE expert block still use the float-only activation and stay off by
default. `ad999f3f` additionally rejects invalid dimensions before
slicing. No optimization environment-variable defaults changed. Tests cover
exact BF16/FP16 singleton activation, dense S=2/S=4 and MoE packed rows, and
invalid widths. The float32 projection tolerance is unchanged.

## Paired raw-token checks

The clean installed `891385f8` bundled wheel produced:

| Input set | Direct/MTP identical | Same-pack reference agreement |
| --- | --- | --- |
| Nine original diagnostics | 9/9 | 8/9 for each AX route |
| Twelve predeclared holdout cases | 12/12 | Not measured |

[Original diagnostics](diagnostic-tokens.json) and
[holdout records](holdout-tokens.json) contain fixed token inputs, sampling,
outputs, route counters, wheel/source hashes, and raw-artifact hashes. Both
routes use temperature 0, seed 0, top-k 0, top-p 1 and repetition penalty 1.
The holdout covers arithmetic, JSON, Python, logic, formatting, retrieval,
security, and a 2107-token inventory input; its cap is 128 output tokens.

These are finite token-consistency checks, not task-accuracy scores or a
proof of universal equivalence. The separate 512-token LINE_SET replay
retains direct/MTP content divergences; that broader route-consistency gate
remains open. The correction closes the reproduced activation defect, not
every numerical difference in the runtime. The synthetic diagnostic deliberately reaches
its 64-token cap; the Python tie holdout reaches its 128-token cap on both
routes. Two one-token holdout completions do not exercise MTP verification.
The nine-case reference mismatch is retrieval wording: both AX routes retain
the answer Ridge at 08:30. No post-fix throughput claim is made; preserving
activation semantics adds operations, so old speed measurements do not qualify
this implementation.

## Campaign-host throughput A/B

[m5-throughput-ab](m5-throughput-ab/) holds a same-session before/after run
on the README peer-table campaign host (Apple M5 Max, 128 GB, macOS 27.0),
not the qualification SKU. Both runs use the peer-campaign contract
(`flappy` suite, four cases, 256 generated tokens, greedy, non-thinking chat
template, two warmups, five measured repetitions, three-second cooldown,
prefix cache off, MTP head only with n-gram stacking off, exact Qwen linear
MTP verifier) and identical prompt token hashes. "Before" is the retained
2026-09-15 campaign binary (build commit unknown, as in the published
campaign); "after" is a clean release build of `d60f9895` against MLX
0.32.2 from the same venv.

| Case | Decode before | Decode after | Delta | Prefill before | Prefill after | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flappy_pipes | 77.11 | 76.94 | -0.2% | 717.7 | 707.7 | -1.4% |
| flappy_score_gates | 74.97 | 75.19 | +0.3% | 841.3 | 837.5 | -0.5% |
| flappy_collision_checks | 74.76 | 74.96 | +0.3% | 846.5 | 842.3 | -0.5% |
| flappy_sound_channels | 77.11 | 77.03 | -0.1% | 706.5 | 702.3 | -0.6% |

Values are per-case medians of five repetitions in tok/s; every row is
classified `stable_enough` by the harness. Decode is unchanged within noise
and prefill is about 0.5-1.4% lower, consistent with the extra split
activation dispatches at multi-token shapes. This is a regression check for
the activation correction, not a refreshed peer table: the "before" binary
has no recorded commit, and the published README numbers remain the
2026-09-15 campaign values.

## Source validation

Final runtime source `ad999f3f` passed formatting, the full Rust suite,
repository CI-policy Clippy, primary-claim checks and both Qwen dry-run
contracts. The full script suite passed on the activation commit `891385f8`;
the final guard-only follow-up passed the full Rust suite again. Rebuilding/installing the Python extension passed;
Python reports 208 passed, 26 skipped and 136 subtests passed. The 22 SwiGLU
regressions pass. CI-policy Clippy retains the repository's existing five
`--force-warn` restrictions; no lint policy was weakened by this change.
Dry-run checks are contract checks, not hardware qualification.

## Final bundled-wheel qualification

The clean `ad999f3f` wheel passed the executable schema-2 selected-SKU gate:
direct and MTP each **32/32 hard QA**, zero soft failures and **7/7 surface
probes**, with no skips. Doctor reported ready. Direct draft/verify counters
were 0/0; MTP counters were 63/86 on the 64-token probe, proving active
verification. The paired probe now matches all 64 output tokens.
[Qualification record](qualification.json) preserves source, wheel, binary
and CLI hashes plus request/response artifact hashes.

The isolated Python environment contains ax-engine 7.4.0 and pip only; native
import succeeded and dyld loaded MLX/JACCL from `ax_engine/.dylibs`. Installed
wheel members and the pinned model inventory were verified. This establishes
packaging operation on the selected SKU, not every supported Mac or a public
release attestation. The final wheel SHA-256 is
`2bc3f5e1a1bc94d346a5e1bc1aa732840466825b347dead1b05f137be9784409`.

The twelve-case quality replay below used the preceding `891385f8` wheel;
`ad999f3f` only adds an invalid-dimension guard. Its results must remain bound
to the build actually tested. A scoped qualification pass is not a release
promotion and does not supersede the remaining counterexamples.

## Quality remains open

[Quality diagnostics](quality-diagnostics.json) preserve the original gold and
strict grades for twelve selected LINE_SET failures, original question wording,
a fixed answer-only system instruction, thinking disabled and a 512-token cap.
They ran on the clean installed `891385f8` activation build.

| Route | Correct | Wrong | Truncated |
| --- | --- | --- | --- |
| Direct | 1/12 | 2/12 | 9/12 |
| MTP | 2/12 | 4/12 | 6/12 |

Only **4/12** paired chat response texts are identical. These counterexamples
keep broader direct/MTP consistency open despite the 21 matching shorter raw
token cases. Default throughput-MTP uses relaxed target arithmetic; this
experiment does not isolate every remaining operation that changes an argmax.
The single synthetic qualification probe cannot close that broader gate.

The truncations are a harness effect, not silent output. Every truncated
direct response contains a line-by-line analysis that ignores the answer-only
system instruction (non-thinking template, `enable_thinking=false`) and runs
into the 512-token cap before emitting the `Answer:` line.
[Output-cap ablation](quality-ablation-maxtok2048-direct.json) replays the
same twelve direct requests on the final `ad999f3f` wheel with a 2048-token
cap and otherwise unchanged protocol:

| Direct route | Correct | Wrong | Truncated | Detection flag set |
| --- | --- | --- | --- | --- |
| 512-token cap | 1/12 | 2/12 | 9/12 | 2/12 |
| 2048-token cap | 3/12 | 8/12 | 1/12 | 9/12 |

Six of the eight remaining wrong answers carry the grader's detection flag
(reported lines fall inside the gold span but the span is incomplete or
over-extended); the other two (compsec-079: 18, 19, 20 against 18-19, compsec-081: 0 against 10-15) do not. This ablation is direct-only, single-cap, and not a
promotion input; it shifts the open question from "the model produces
nothing" to "answer-format compliance and span-boundary precision under
strict grading". The original 32k thinking campaign is still the quality
gate of record.

There is also a concrete prompt/gold ambiguity: compsec-076 requests a single
best bug line or smallest adjacent set, while gold 17-20 includes a guard,
return, blank line and unsafe memcpy. The model reports line 20. Gold and
strict grading remain unchanged. This does not explain every wrong or truncated
answer and does not justify promoting the checkpoint.

Both routes completed the saved **32851-token** recovery in approximately
290 seconds and returned identical `Answer: B` against gold C. This is a
replay of a saved continuation, not a fresh-question evaluation. The earlier
same-pack reference also answered B. Neither that observation nor these
selected failure scores establish general model accuracy. The original full
32000-token thinking campaign, representative quality, MTP Tier 2 performance,
and endurance gates remain open.

## Reproduction

Build/install the bundled wheel from the recorded clean runtime commit and
verify its manifest against the wheel, installed binaries, CLI and pinned pack.
On the selected SKU, with runtime overrides unset, run:

```sh
python scripts/qualify_qwen38_27b.py --run \
  --model-dir /path/to/pinned-pack --output /path/to/new-evidence \
  --build-manifest /path/to/build-manifest.json --wheel /path/to/bundled.whl \
  --cli /path/to/venv/bin/ax-engine \
  --server-bin /path/to/venv/lib/python3.12/site-packages/ax_engine/_bin/ax-engine-server \
  --bench-bin /path/to/venv/lib/python3.12/site-packages/ax_engine/_bin/ax-engine-bench
```

For the finite route comparisons, start each recorded route separately and
replay each artifact's exact request to `/v1/generate`; compare output token
arrays. Do not substitute retokenized chat text or change the sampling fields.
A passing scoped qualification does not override the quality and broader
consistency failures above.
