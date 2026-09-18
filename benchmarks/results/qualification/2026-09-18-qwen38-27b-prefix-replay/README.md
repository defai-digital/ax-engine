# Qwen scheduler prefix replay

Selected SKU: Mac mini M4 Pro 64 GB (Mac16,11). Checkpoint:
`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` at
`3e290738e96972307c6aeb9934ab170ca0eae1c1`.

## Root cause and repair

The scheduler sends a reused prefix separately and removes it from the input
slice. The runner aligns a cached prefix to the cold prefill chunk grid. When
that alignment shortened the scheduler claim, the runner could restore no
context or an incomplete context without asking the prefill fallback to replay
the omitted tokens. Its extension-length calculation also subtracted the
prefix again even though the scheduler input was already a suffix.

`4cbda8b5` uses the actual suffix length for scheduler-claimed prefixes. If the
aligned restore point cannot cover the full claim, it defers full prefix-plus-
suffix recomputation through the existing fallback. Aligned restores, runner
probes without a scheduler claim, existing KV state, and decode retain their
existing behavior. Cache and MTP defaults remain enabled.

The regression fails with the old selection logic: a four-token prefix and
eight-token suffix select a zero-length restore instead of full replay.
Coverage includes zero/partial trim, a two-token suffix, aligned prefixes,
one-token tails, and runner-only probes.

## Cold/warm reproduction

The same raw compsec-086 request is run in a fresh direct server, then in
another direct server after seven recorded prompts with one output token each.
Both use temperature 0, seed 0, top-k 0, top-p 1, repetition penalty 1,
a 128-token target cap, and n-gram acceleration disabled. Warmup order and
all input/output token arrays are in [cold/warm controls](cold-warm-controls.json).

| Source | Cold output tokens | Warm output tokens | Exact agreement | Warm prefix replay |
| --- | --- | --- | --- | --- |
| `892c2fc9` | 4 | 128 (cap) | No | 0 |
| `4cbda8b5` | 4 | 4 | Yes | 96 |

Both warm requests retain one scheduler cache hit and one branch prefill.
The repaired request replays the missing 96-token prefix. Its output matches
the cold output `[15666, 25, 220, 18]`. This is evidence of context restoration,
not a claim that the answer itself meets strict LINE_SET gold.

Reproduce by replaying the recorded requests, in order, against separately
started product servers. Use `--disable-ngram-acceleration`,
`--total-blocks 4096` and `--max-output-tokens 8192`. Restart between cold and
warm arms, keep the pinned pack and recorded wheel identity, and compare token
arrays and prefix counters rather than decoded text alone.

## Source and packaging validation

[Source validation](source-validation.json) binds the before/after regression
and log hashes to the source commit. Formatting, full Rust tests, CI-policy
Clippy, the complete script suite, both Qwen dry-run contracts, primary-claim
checks, and rebuilt-extension Python tests pass. Python reports 209 passed,
26 skipped and 140 subtests. Existing Clippy force-warn restrictions remain
unchanged. The bundled release-pyext wheel passes isolated native loading.

## Acceptance boundaries

This repair does not establish population accuracy, universal route equality,
MTP Tier 2, or release readiness. Earlier quality failures ran with this
context-omission defect and cannot be attributed solely to model capability.
Their original records remain intact. The unchanged selected twelve-case
512-token answer-only diagnostic is separate from the original 32000-token
thinking campaign, saved recovery continuation, and endurance acceptance.

## Installed-wheel and unchanged quality results

[Qualification](qualification.json) passes on clean `4cbda8b5`: direct/MTP
each 32/32 hard QA, zero soft failures, 7/7 surface probes without skips,
doctor ready, package member hashes and pinned model inventory verified.
The paired 64-token probe matches. Direct draft/verify counters are 0/0;
MTP counters are 63/86. The isolated environment contains only ax-engine and
pip; MLX/JACCL load from the package dylib directory.

The [strict diagnostic](quality-diagnostics.json) preserves original wording,
gold, answer-only system instruction, thinking disabled, greedy seed 0 and
a 512-token output cap. All 24 requests completed without transport errors.

| Route | Strict correct | Wrong | Truncated |
| --- | --- | --- | --- |
| Direct | 0/12 | 8/12 | 4/12 |
| MTP | 0/12 | 10/12 | 2/12 |

Direct/MTP response content matches **8/12**, versus 3/12 on `892c2fc9`.
Truncations decrease from 9 to 4 (direct) and 8 to 2 (MTP), but strict quality
does not pass. Differences remain on compsec-077, compsec-079, compsec-087
and compsec-092. Matching responses include failed/truncated answers and are
not evidence of task correctness.

Many completed answers report a single relevant line while strict gold
requires the complete span. Neither gold nor grading was changed. This
selected failure subset is not an estimate of overall model accuracy. The
original long-thinking campaign, saved wrong-answer continuation and endurance
were not rerun on this build. Status remains **Candidate; not ship-ready**.

No new throughput claim is made for this repair. Recomputing previously
omitted context changes the work performed; older throughput measurements
remain bound to their original source, cache state and workload.
