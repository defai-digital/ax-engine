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

The bench script enforces presence of `ax_decode_claim_status` and
`ax_decode_claim_mode` on every emitted row; the same-policy gate
enforces presence of the rest at promotion time.

## Reproducing the gate locally

```bash
# Run the Python claim-taxonomy tests.
python -m unittest scripts.test_bench_mlx_inference_stack -v -k claim

# Run the Rust same-policy gate tests.
cargo test -p ax-engine-bench --quiet ngram_claim_gate
```

Both surfaces stay in different runtimes by design — neither side can
silently relax the other's checks.
