# ADR: MTP/N-gram Acceptance Rate Optimization Design Decisions

## Status

**Implemented.** All five decisions shipped (2026-06-02). Auto-optimistic hysteresis tightened from 0.99/0.95 to 0.98/0.96 post-implementation to increase activation rate on harder prompts (python_modules_long: 10.4% → expected ≥80%). Per-depth accept rate telemetry added.

## Context

AX Engine's MTP speculative decode path achieves high acceptance rates on Qwen3.6-27B-MTP (depth-0 ≈ 99.9%, depth-1 ≈ 99.8%, depth-2 ≈ 98.9%), but several design decisions limit further improvement. The latest benchmark (`2026-06-02-qwen36-fair-ax-opt1`) shows 3.7x tokens per forward; target is 4.0x+. This ADR records the design decisions for five acceptance rate optimizations.

## Decision 1: Greedy mode draft log-probs at temperature=1.0

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **A1: GPU log-probs at T=1.0** (chosen) | Enables rejection sampling in greedy mode; correct draft probability (model's own confidence in argmax); lazy graph, single eval | Adds one softmax+take+log per depth level |
| A2: GPU log-probs at request temperature | Matches target temperature directly | Conflates draft and target distributions; draft model doesn't know request temperature |
| A3: CPU-side log-prob computation | No GPU work | Requires logits transfer to CPU; defeats the purpose of lazy GPU graph |
| A4: Keep argmax-only (status quo) | No code change | Caps acceptance at argmax match rate; misses high-probability draft tokens that disagree with argmax |

### Decision

**A1 is chosen.** Temperature=1.0 gives `p_draft = softmax(logits)[argmax_token]`, which is the draft model's own confidence in its argmax choice. This is the correct probability for rejection sampling: we're comparing the target model's probability for the draft token against the draft model's own confidence in that token.

The additional GPU work (one softmax+take+log per depth) is already lazy and batched into the single `eval`, so there is no additional GPU sync barrier.

### Tradeoffs

- Greedy mode now has the same GPU work as sampled mode for log-prob computation. On models where greedy mode is the common case, this is a small GPU cost for potentially significant acceptance rate improvement.
- The draft log-probs at T=1.0 are not the same as the request's sampling distribution. This is intentional — the draft distribution should be independent of the target distribution for correct rejection sampling.

## Decision 2: Draft-target temperature alignment via log-prob rescaling

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **B1: Rescale log-probs in acceptance loop** (chosen) | Simple; O(1) per position; no GPU work; correct temperature adjustment | Requires passing two temperatures to `mtp_accept_count` |
| B2: Rescale draft logits before log-prob computation | Correct at source | Requires changing `mtp_draft_tokens_sampled` and `mtp_draft_tokens_greedy`; more invasive |
| B3: Force draft temperature to match request temperature | Eliminates mismatch entirely | Changes draft behavior; may reduce draft quality if head was trained with specific temperature |
| B4: No alignment (status quo) | No code change | Incorrect acceptance probabilities when T_draft != T_target; potential distribution drift |

### Decision

**B1 is chosen.** Rescaling in log space: `log(p_scaled) = log(p_draft) * (T_draft / T_target)` is mathematically equivalent to `p_scaled = p_draft ^ (T_draft / T_target)`, which is the correct temperature adjustment for softmax probabilities. This is O(1) per position, requires no GPU work, and is applied only when `T_draft != T_target`.

### Tradeoffs

- The rescaling assumes the draft log-probs are from a softmax distribution. This is true for both greedy mode (T=1.0 softmax) and sampled mode (T=draft_sampling.temperature softmax).
- When `T_target == 0` (greedy target), no rejection sampling applies, so rescaling is irrelevant.

## Decision 3: N-gram pseudo log-probs from confidence

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **C1: `ln(confidence)` as pseudo log-prob** (chosen) | Simple; uses existing n-gram confidence; gives rejection sampling a signal | Heuristic, not a true draft model probability |
| C2: Train a separate n-gram probability model | True probability distribution | Massive engineering effort; defeats the purpose of lightweight n-gram |
| C3: Keep argmax-only for n-gram tokens (status quo) | No code change; correct (no draft model) | Misses acceptance opportunities when target is uncertain |
| C4: Use n-gram support count as probability | Simple | Support count is not normalized; can exceed 1.0 |

### Decision

**C1 is chosen.** `ln(confidence)` where `confidence = support/total` is the n-gram's own estimate of the probability of the next token given the context. Interpreting this as a draft probability and taking the log gives a pseudo log-prob that is compatible with the rejection sampling formula.

Clamping to `[-30, 0]` matches the floor in `gpu_draft_log_prob_lazy`, ensuring numerical stability.

### Tradeoffs

- This is a heuristic. The n-gram confidence is not a true probability distribution over the vocabulary (it doesn't sum to 1.0 across all possible tokens). But it provides a useful signal: high-confidence n-gram tokens have `log_prob ≈ 0` (p_draft ≈ 1.0), making acceptance depend on the target model's probability alone.
- Low-confidence n-gram tokens have `log_prob < 0` (p_draft < 1.0), making acceptance more likely (since `accept_prob = p_target / p_draft` is larger). This is correct behavior: if the n-gram is uncertain, the draft is "weak," and the target model is more likely to accept.

## Decision 4: N-gram saturation gate threshold at 0.97

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **D1: 0.97 for depth≥3** (chosen) | Keeps n-gram active when MTP acceptance is 97-99%; n-gram still adds value in this range | May add CPU overhead when n-gram acceptance is low |
| D2: 0.99 (status quo) | Minimal CPU overhead | Disables n-gram too early; misses hybrid tail opportunities |
| D3: 0.95 | Maximizes n-gram usage | May keep n-gram active when it's clearly worse than MTP; more CPU overhead |
| D4: Adaptive threshold based on n-gram hit rate | Self-tuning | Complex; requires additional state tracking |

### Decision

**D1 is chosen.** At 97% MTP-only acceptance, ~3% of drafts are rejected. N-gram can fill these gaps with high-confidence predictions from repeating patterns. The gate still disables n-gram when MTP is truly saturated (≥99%), but keeps it active in the 97-99% range.

Depth 0/1 keep the gate disabled (threshold = ∞) because per-step rate is binary and EWMA reaches high values on random streaks, causing false gating.

### Tradeoffs

- The threshold is a tunable parameter. If benchmarking shows n-gram adds more overhead than value at 97%, it can be raised. If n-gram adds value below 97%, it can be lowered.
- The gate uses `mtp_only_accept_rate_ewma` (excluding n-gram sources) so that n-gram rejections cannot suppress gating when the model itself accepts ≥97%.

## Decision 5: Adaptive depth floor on consecutive misses

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **E1: Progressive floor (2→1→0)** (chosen) | Fast recovery from low-acceptance regions; fewer wasted forwards | Requires tracking consecutive miss count |
| E2: Constant floor of 2 (status quo) | Simple; no additional state | Wastes forwards on low-acceptance prompts |
| E3: Drop to 0 on first miss | Fastest recovery | Too aggressive; single miss may be noise |
| E4: Beta-Bernoulli posterior for depth decision | Statistically principled | Already used for n-gram disable; adds complexity to MTP path |

### Decision

**E1 is chosen.** After the first complete miss (`accept_count == 0`), floor remains at 2 (same as current behaviour — no change on a single miss). After the second consecutive miss, floor drops to 1. After the third+, floor drops to 0 (pure n-gram or direct pipeline). This balances responsiveness with noise tolerance: a single miss may be a transient outlier, so one step of observation before acting reduces unnecessary depth reduction.

The consecutive miss counter resets on any accept (`accept_count > 0`), ensuring that transient misses don't permanently reduce depth.

### Tradeoffs

- On prompts with intermittent low-acceptance regions (e.g., code switching from repeating to creative), the floor drops and recovers quickly. This is the desired behavior.
- On prompts with consistently low acceptance, the floor drops to 0, falling back to n-gram or direct pipeline. This avoids wasting forwards on MTP drafts that are consistently rejected.

## Consequences

### Positive
- Higher acceptance rate in greedy mode (Decision 1)
- Better-calibrated acceptance probabilities (Decision 2)
- N-gram tokens participate in rejection sampling in hybrid mode (Decision 3)
- N-gram stays active longer when MTP acceptance is high but not saturated (Decision 4)
- Faster recovery from low-acceptance regions (Decision 5)

### Negative
- Additional GPU work in greedy mode (one softmax+take+log per depth)
- Additional state tracking (consecutive miss counter)
- More tunable parameters (gate threshold, miss floor progression)

### Risks
- Greedy mode log-probs may change output distribution slightly (rejection sampling accepts different tokens than argmax-only). Mitigated by temperature=1.0 draft distribution, which is the model's own confidence.
- N-gram pseudo log-probs are heuristic, not true probabilities. Mitigated by clamping to [-30, 0] and using confidence (a well-calibrated signal).
- Lower gate threshold may add CPU overhead when n-gram acceptance is low. Mitigated by the gate still requiring `mtp_only_accept_rate_ewma_samples >= 4` before activating.
