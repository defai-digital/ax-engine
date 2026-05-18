# N-gram Acceleration

AX Engine ships a deterministic n-gram drafter that proposes continuation
tokens from observed repetitions, then verifies each draft against the
target model's argmax output. The accepted prefix advances the decode
state; rejected drafts fall back to a single target-model step. The path
lives at `crates/ax-engine-mlx/src/ngram_accel.rs` and is enabled by
default for greedy decode.

## Correctness contract

The current verifier accepts a drafted token iff the target model's
**argmax** over the next-token logits matches the draft. That is a
deterministic, distribution-exact contract under greedy sampling:

- `temperature == 0`
- `top_p == 1.0` (or unset)
- `top_k == 0` (or unset)
- `repetition_penalty == 1.0` (or unset)

Outside that envelope, argmax verification is **not** distribution-exact.
A sampling distribution with `temperature > 0` can prefer a different
token than the argmax for any given step; the drafter cannot prove that
the chosen-token probability under the draft matches the target without a
probability-ratio acceptance test plus residual correction. AX Engine
does not currently implement either.

## Claim taxonomy

Every benchmark row emits two separate fields, deliberately kept apart so
fallback state and correctness mode cannot be conflated.

### `ax_decode_claim_mode`

Records what correctness claim the row can support at all. Defined in
`scripts/bench_mlx_inference_stack.py::ax_decode_claim_mode`.

| Value | Meaning |
|---|---|
| `direct_greedy_exact_baseline` | Direct decode, greedy sampling. The canonical same-policy baseline for n-gram promotion. |
| `ngram_greedy_exact_candidate` | N-gram-accelerated decode, greedy sampling. May be promoted as same-policy-baseline-equivalent when paired with a matching baseline row and identical generated tokens. |
| `direct_sampling_not_distribution_exact` | Direct decode with any sampling knob active. |
| `ngram_sampling_not_distribution_exact` | N-gram-accelerated decode with any sampling knob active. **Forbidden** from distribution-exact promotion under the current verifier. |

### `ax_decode_claim_status`

Records the fallback / promotion state independent of correctness mode.
Defined in `scripts/bench_mlx_inference_stack.py::ax_decode_claim_status`.

| Value | Meaning |
|---|---|
| `direct_same_policy_baseline` | Row was produced with n-gram disabled and serves as the baseline. |
| `ngram_no_observed_draft_path` | N-gram was enabled but the drafter never proposed nor rejected drafts (effectively unused). |
| `ngram_no_draft_direct_fallback` | N-gram path was hit but every step fell through to direct decode (no draft attempts). |
| `ngram_no_accept_fallback` | Drafter proposed but the verifier never accepted (no acceleration realized). |
| `ngram_acceleration_effective_throughput` | Drafter proposed and at least one token was accepted. |
| `direct_same_policy_baseline_degenerate_decode` | Direct row whose output collapsed into a repeated n-gram loop (see [Decode degeneracy](#decode-degeneracy-on-synthetic-benchmark-prompts)). |
| `ngram_acceleration_degenerate_decode` | N-gram row whose output collapsed into a repeated n-gram loop. The throughput is real but achieved on degenerate output, not healthy decode. |

## Decode degeneracy on synthetic benchmark prompts

The standard AX benchmark prompts are uniform-random token IDs sampled
from `mx.random.randint(0, vocab_size, ...)` with `seed=0`, matching
`mlx_lm.benchmark` exactly so prompt-hash parity holds across rows.
Random-token prompts are the worst case for n-gram acceleration in
terms of *prompt structure* — every 4-gram is unique — so the drafter
gets no help from the prompt itself.

The harder question is what the model *outputs* when fed random tokens
at greedy decode (`temperature=0`). With no semantic anchor in the
input, models commonly collapse into a repeated-n-gram loop: the same
short token sequence is emitted over and over. When that happens, two
things become true at once:

1. The n-gram drafter sees its own repeating output, predicts the next
   chunk near-perfectly, and the row reports a large effective decode
   speedup.
2. The matching direct row (n-gram disabled) hits the same loop without
   speculative work, so its reported throughput also reflects a
   degenerate output stream rather than healthy decode.

These rows still measure real wall-clock throughput, but they do not
measure what most users want to compare against — coherent generation.
The bench harness flags them so README readers and downstream tooling
can keep the two regimes apart.

### How the validator works

Every AX row carries a `decode_degeneracy` field built by
`scripts/bench_mlx_inference_stack.py::summarize_decode_degeneracy`.
The validator slides an 8-token window over each trial's decode token
IDs and counts the most common n-gram. If any trial's max repetition
count exceeds the threshold (default 3), `detected_in_any_trial` is set
and `ax_decode_claim_status` promotes to a `_degenerate_decode`
variant. The check is borrowed in spirit from MTPLX's
`validate_no_degenerate_loop` but operates directly on token IDs so it
needs no detokenization and is unambiguous across tokenizers.

Schema fields on each AX row:

| Field | Meaning |
|---|---|
| `decode_degeneracy.schema_version` | `ax.decode_degeneracy.v1` |
| `decode_degeneracy.validator` | `no_repeated_ngram_loop_v1` |
| `decode_degeneracy.ngram_size` | Window size (default 8) |
| `decode_degeneracy.max_repeats_threshold` | Threshold for `detected` (default 3) |
| `decode_degeneracy.per_trial[i]` | `{trial, token_count, max_ngram_repeats, detected}` or `{trial, skipped}` |
| `decode_degeneracy.detected_in_any_trial` | True if any trial crossed the threshold |
| `decode_degeneracy.max_repeats_observed` | Worst-trial repeat count |
| `decode_degeneracy.mean_repeats_observed` | Average across trials |
| `decode_degeneracy.skipped` | Set when no trial captured token IDs |
| `decode_degeneracy.partial_evidence` | Set when only some trials captured IDs |

The harness always captures token IDs internally so the validator can
run on every AX row. The `--capture-output-token-ids` CLI flag only
controls whether the raw IDs are persisted in the trial artifacts
themselves; the `decode_degeneracy` summary is always emitted.

### Reference evidence

The first end-to-end run is at
`benchmarks/results/mlx-inference/2026-05-18-decode-degeneracy-validator/gemma-4-e2b-it-4bit.json`.
All six AX rows on `mlx-community/gemma-4-e2b-it-4bit`
(direct + n-gram, prompt={128,512,2048}, generation=128, 3 trials each)
flag as `*_degenerate_decode` with `max_ngram_repeats = 121` in every
trial — essentially a single 8-token block repeated across the entire
128-token decode window. This is the regime in which random-token
benchmark prompts magnify n-gram speedup numbers; it is not the regime
most user workloads operate in.

### Reading a `_degenerate_decode` row

- The reported `decode_tok/s` is real wall-clock throughput. The label
  does not say "the number is wrong"; it says "the output the number
  was measured on was a repetition loop."
- The matching direct row, if present, is likely to carry the same
  label for the same prompt because both rows decode from the same
  initial state under greedy sampling.
- For workload-shape claims (coding completion, structured diffs,
  JSON/tool payloads) prefer non-random prompts. The decode-degeneracy
  validator is a transparency gate, not a verdict on the n-gram
  drafter's value on real workloads.

## Same-policy greedy promotion gate

The Rust gate at `crates/ax-engine-bench/src/harness/ngram_claim_gate.rs`
decides whether a baseline-candidate pair can be promoted to a single
greedy-exact accelerated claim. Inputs:

- `RowIdentity` for both rows: `model_id`, `prompt_hash`, `seed`,
  `max_output_tokens`, `sampler_signature`.
- The full generated token sequences from each row.
- An `is_sampling_mode` flag.

Outcomes:

| Outcome | Meaning |
|---|---|
| `Promoted` | All identity fields equal; generated token IDs match byte-for-byte; greedy mode. |
| `IdentityMismatch` | One or more identity fields differ. The artifact records which field. |
| `TokensDiverged` | Identity matched but token streams differ. The artifact records `first_mismatch_index`, `baseline_len`, `candidate_len`. |
| `SamplingModeRefused` | `is_sampling_mode == true`. Promotion refused unconditionally. |

The gate is **fail-closed**: any input ambiguity defaults to non-promotion.

## Sampling-mode promotion is forbidden

The companion Python helper
`assert_no_distribution_exact_promotion_under_sampling` lives next to the
bench-row aggregator. It raises `ValueError` if a row attempts to set
`ax_decode_distribution_exact_claim: True` while
`ax_decode_claim_mode` is `*sampling_not_distribution_exact`.

A future probability-ratio-acceptance verifier could close this gap;
until that ships and an ADR justifies it, sampling-mode acceleration is
reported but not claimed as distribution-exact.

## Required artifact fields for release claims

Every n-gram row used for release claims must include:

- `ax_decode_claim_mode`
- `ax_decode_claim_status`
- Sampler settings (`temperature`, `top_p`, `top_k`, `repetition_penalty`)
- Prompt hash and random seed
- Direct same-policy baseline row identity (when claiming acceleration)
- `ax_ngram_draft_attempts`, `ax_ngram_draft_tokens`,
  `ax_ngram_accepted_tokens`, fallback reason counts
- `decode_degeneracy.detected_in_any_trial` and per-trial repeat counts,
  so a release claim cannot silently rest on a degenerate output stream

The bench script enforces presence of `ax_decode_claim_status` and
`ax_decode_claim_mode` on every emitted row; the same-policy gate
enforces presence of the rest at promotion time.

## Reproducing the gate locally

```bash
# Run the Python claim-taxonomy tests.
python -m unittest scripts.test_bench_mlx_inference_stack -v -k claim

# Run the decode-degeneracy validator tests.
python -m unittest scripts.test_bench_mlx_inference_stack -v -k decode_degeneracy

# Run the Rust same-policy gate tests.
cargo test -p ax-engine-bench --quiet ngram_claim_gate
```

Both surfaces stay in different runtimes by design — neither side can
silently relax the other's checks.
