# N-gram Acceleration

AX Engine ships a deterministic n-gram drafter that proposes continuation
tokens from observed repetitions, then verifies each draft against the
target model's argmax output. The accepted prefix advances the decode
state; rejected drafts fall back to a single target-model step. The path
lives at `crates/ax-engine-mlx/src/ngram_accel.rs` and is enabled by
default for greedy decode.

## When n-gram acceleration helps

The README decode-throughput tables show the n-gram column at large
percentages above `mlx_lm`. Those numbers are correctly measured on the
mlx_lm.benchmark random-token contract (seed=0 uniform random IDs over
the full vocab, greedy decode, batch=1, gen=128). They are the same
prompt contract every published row uses, which is why prompt-hash
parity holds across the table.

What that contract *cannot* tell you is which real workloads will see
the same uplift. The literature is unambiguous about this — n-gram
speculative decoding is an **input-output overlap** technique, not a
"coding" technique. Whenever the model's output reuses chunks of the
prompt (or chunks of its own earlier decoded output), the drafter
proposes long correct sequences and the row throughput climbs. When the
output is fresh content unrelated to the prompt, the drafter's accepts
are short and the verifier overhead can outweigh the gains.

The split, from published benchmarks:

| Workload shape | Typical n-gram speedup | Source |
|---|---:|---|
| Summarization (CNN/DailyMail) | **2.4×–2.8×** | Saxena 2024; vLLM blog |
| Context-QA over a retrieved doc | **2.4×** | Saxena 2024 |
| Code **editing / refactor** (output reuses input) | **2.1×** (n-gram alone, InstructCoder batch=1) — even beats EAGLE / EAGLE-3 on 8B models when BLEU-4 overlap ≥ 0.6 | SpecDecode-Bench 2025; SGLang |
| Code editing with reuse-aware drafting | up to **8.26×** (Qwen2.5-Coder), **13.09×** (DeepSeek-Coder) | EfficientEdit (arXiv 2506.02780) |
| Code **generation / completion** (output is fresh code) | **~1.10×** | vLLM blog |
| Multi-turn chat, turn 1 (some prior content to copy) | "very high gain" | Saxena 2024 |
| Multi-turn chat, turn 0 (no prior content) | "much smaller gain" | Saxena 2024 |
| Roleplay / open-ended generation | worst case, near-baseline or slower | Saxena 2024 |

The pattern is consistent across sources: **input-output overlap is the
single variable that predicts n-gram speedup**, not "is it code". Code
editing is great for n-gram because ~70% of the post-edit code is
unchanged from the pre-edit code, so the drafter can directly copy
multi-token spans. Code generation from a brief is not great for
n-gram because the model is producing new content.

### What our own measurements show

Two evidence sets, both Gemma 4 E2B 4-bit, batch=1, gen=128, 3 trials
per row, greedy decode:

| Prompt source | Direct decode | N-gram decode | N-gram uplift | Accept rate | Synthetic loop caveat? |
|---|---:|---:|---:|---:|:---:|
| Random tokens (mlx_lm contract), p=128 | 205.4 | 609.1 | **+196%** | ~100% | **Yes** — repeated 8-token block observed in legacy token-output review |
| Random tokens, p=512 | 196.8 | 599.5 | **+205%** | ~100% | **Yes** |
| Random tokens, p=2048 | 190.1 | 550.0 | **+189%** | ~100% | **Yes** |
| Real coding *generation*, lru_cache (90t prompt) | 203.5 | 173.1 | **−15%** | 11.9% | no |
| Real coding *generation*, axum_handler (184t) | 203.5 | 183.5 | **−9.8%** | 37.1% | no |
| Real coding *generation*, repo_refactor (447t) | 193.5 | 189.0 | **−2.3%** | 51.4% | no |

Evidence directories:
- 2026-05-18 legacy random-token output-loop review
- `benchmarks/results/mlx-inference/2026-05-18-real-prompt-coding/` — real coding **generation** prompts (not editing); every AX row labeled `ngram_acceleration_effective_throughput` with healthy decode

The direct-decode column is stable (~190–205 tok/s) across both
sources, so the model and the kernel are not the variable. The n-gram
column collapses from +196% on repeated random-token output to a small
negative on real generation prompts. Neither evidence set has covered
the workload where n-gram is known to dominate (code editing); a
follow-up suite is the next planned bench.

### Practical guidance

If the workload is editing, refactoring, structured diffs, JSON/tool
payload completion, or any chat turn where the output naturally echoes
the input or echoes prior turns, expect the n-gram path to help and
the README-style headline numbers to be representative direction
(even if the absolute multiple differs by model and host).

If the workload is fresh generation from a brief — "write a function
that …", "summarize what we should do", first-turn answers, novel
prose — expect the n-gram path to perform near direct decode, possibly
slightly slower under verifier overhead. The fallback statuses
documented below (`ngram_no_draft_direct_fallback`,
`ngram_no_accept_fallback`) explicitly record this regime so it does
not silently degrade.

If the workload is open-ended creative / roleplay, prefer direct
decode by passing `--disable-ngram-acceleration` to the server. The
n-gram drafter is correctness-safe on rejection (the verifier owns
the final token decision) but it costs draft work that the workload
will not reward.

### Prompt seeding

The drafter table is fed the prompt tokens during prefill commit
(`runner.rs::extend_prompt_prefix_tokens`). Without this, the table
starts decode empty and can only learn from self-emitted tokens, which
collapses to zero acceleration on models that produce coherent
(non-repeating) decode from synthetic random prompts — e.g. Qwen 3.6
27B reported `ngram_no_draft_direct_fallback` on every random-token
benchmark row. The synthetic random-token benchmark suite still shows
no uplift on coherent-decoding models (each random 4-gram is unique,
so the seeded table holds no matchable continuations), but real
input-output-overlap workloads benefit because the post-prefill decode
context now has prompt-derived precedent in the table.

Validation artifact:
`benchmarks/results/mlx-inference/2026-05-18-ngram-real-prompt-validation/qwen3_6-27b-4bit.json`
runs the three-case input-output-overlap suite at
`benchmarks/manifests/real_prompts/input_output_overlap.jsonl` on Qwen
3.6 27B 4-bit (a model that produced zero draft attempts on every
prior synthetic-prompt measurement). The `reformat_json` case fires
36 draft attempts with 87 accepted tokens and a +7% decode-throughput
gain over direct decode; the other two cases (`rename_identifier`,
`summarize_then_echo`) stay at zero attempts because the model
paraphrases rather than echoing the input tokens verbatim — the
expected fall-through behavior when input-output overlap is low.

### Linear-attention no-draft fallback

Linear-attention models pay more for failed speculative probes than dense models:
partial rejection needs branch verification and recurrent-state recompute rather
than a cheap KV trim. AX therefore separates two no-draft cases.

`LinearInitialNoDraft` is assigned before decode when the prompt is classified as
non-repeating. For that request, AX stays on the direct double-buffer pipeline
and does not feed generated fallback tokens back into the n-gram table. This is
intentional for random-token / sparse prompts: the prompt provided no overlap
signal, and repeated per-token re-enable checks showed up as overhead without
producing accepted drafts.

`LinearNoDraft` is assigned after the runtime actually tried to draft and found
no candidate. That path can still feed generated output into the n-gram table and
periodically re-open n-gram if repeated evidence appears later. In other words,
prompt-level no-draft is fixed direct fallback; runtime no-draft remains
recoverable.

The disabled-request direct fallback fast path is limited to greedy decode with
no MTP head and no repetition penalty. Sampling and repetition-penalty requests
fall back through the original single-step path so the optimization does not
change token-selection semantics.

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

## Synthetic repeated-output loops

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
The benchmark harness keeps `ax_decode_claim_status` limited to
throughput and fallback state, so it no longer folds this output-quality
classification into each performance row. If a release claim needs to say
the output was coherent, attach a separate token-output validation artifact
or an application-shaped prompt suite. The `--capture-output-token-ids`
flag can still persist generated token IDs for that kind of external review,
but the default throughput artifact stays size-conscious and claim-scoped.

### Reading random-token n-gram rows

- The reported `decode_tok/s` is real wall-clock throughput. It does not
  prove that the decoded text resembles a normal user workload.
- The matching direct row, if present, may hit the same repeated-output
  loop for the same prompt because both rows decode from the same initial
  state under greedy sampling.
- For workload-shape claims (coding completion, structured diffs,
  JSON/tool payloads) prefer non-random prompts. Random-token throughput is
  useful for engine comparison, not for user-workload generalization.

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
- A separate token-output validation artifact when the claim depends on
  coherent generated content rather than throughput alone

The bench script enforces presence of `ax_decode_claim_status` and
`ax_decode_claim_mode` on every emitted row; the same-policy gate
enforces presence of the rest at promotion time.

## Cross-engine context: Lightning MLX MTP+n-gram layering

Lightning MLX (≥ 0.6.10) supports layering n-gram prompt-lookup drafting before the
MTP head in a single verify pass. The architecture (`scheduler.py::_install_mtp()`):

1. At each decode step, n-gram looks up the current history for candidate continuations
   (up to K=6 tokens, min-occurrences=2).
2. If n-gram finds a hit and `ngram_hybrid_verify=True`, one MTP head draft is appended
   as the tail of the n-gram candidates.
3. A single forward pass verifies all candidates: n-gram positions accept on greedy
   argmax; the hybrid MTP tail uses probability-ratio acceptance.
4. If n-gram misses, the step falls through to pure MTP.

This is architecturally different from AX Engine's n-gram stacking (ADR-008), which
uses a cost-gated n-gram-first path with KV trim on rejection. AX Engine n-gram
verification is argmax-exact (distribution-exact under greedy), while Lightning MLX
n-gram uses argmax accept for its positions and probability-ratio for the MTP tail.

### Benchmarking Lightning MLX MTP+n-gram

Use `bench_rapid_mlx_prompt_suites.py --lightning-mode --enable-ngram` to run a single
suite, or `bench_qwen36_mtp_fair.py --lightning-ngram` to compare all four engines
(MTPLX, Lightning MLX, Lightning MLX MTP+ngram, AX Engine) on the full fair benchmark:

```text
python3 scripts/bench_rapid_mlx_prompt_suites.py \
  --model /path/to/model \
  --suite flappy \
  --prompts benchmarks/prompts/mtp-suites/flappy.jsonl \
  --lightning-mode \
  --enable-ngram \
  --output benchmarks/results/mtp-fair/lightning-ngram-flappy.json
```

The `--enable-ngram` flag applies the production benchmark preset:

| Flag | Value | Reason |
|---|---|---|
| `--enable-ngram` | on | activates prompt-lookup drafter |
| `--ngram-num-draft-tokens` | 6 | wide K for long-overlap workloads |
| `--ngram-min-occurrences` | 2 | require bigram to appear twice before drafting |
| `--ngram-acceptance-mode` | greedy | argmax accept for n-gram positions |
| `--ngram-hybrid-verify` | on | append one MTP head draft as tail when n-gram hits |
| `--ngram-everywhere` | on | required: `--no-thinking` suppresses `<think>` blocks entirely, so the default `ngram_only_in_think=True` would produce zero drafts |
| `--ngram-self-tune` | on | disable n-gram for the rest of a request when per-request running accept falls below 0.30 after 32-token warmup |
| `--ngram-auto-disable-mtp-threshold` | 0.0 | threshold=0 disables auto-disable; n-gram is always active for clean benchmark coverage |

Per-request n-gram accept ratio is fetched from `/v1/requests?limit=1` and stored
as `ngram_acceptance_ratio` alongside the existing `mtp_acceptance_ratio`. The
`bench_qwen36_mtp_fair.py` markdown output shows this as "Lightning ngram+MTP accept"
for the `lightning_mtp_ngram` ("Lightning ngram+MTP") engine row.

### Claim boundary

Lightning MLX MTP+n-gram results are cross-engine comparison evidence, not
repo-owned MLX throughput claims. They must not be cited in the same table as
`bench_mlx_inference_stack.py` random-token baseline rows or AX n-gram
`ngram_acceleration_effective_throughput` rows. The claim taxonomies and
correctness contracts differ.

## Reproducing the gate locally

```bash
# Run the Python claim-taxonomy tests.
python -m unittest scripts.test_bench_mlx_inference_stack -v -k claim

# Run the Rust same-policy gate tests.
cargo test -p ax-engine-bench --quiet ngram_claim_gate
```

Both surfaces stay in different runtimes by design — neither side can
silently relax the other's checks.
