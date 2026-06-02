# MTP/N-gram Acceptance Rate Optimization — Technical Specification

## Overview

This spec details the implementation plan for five MTP/n-gram acceptance rate optimizations targeting the draft generation, verification, and adaptive policy paths.

## Phase 1 — Greedy Mode Draft Log-Probs

### 1.1 Problem

`mtp_draft_tokens_greedy` (`mtp.rs:452-513`) returns empty `draft_log_probs`:

```rust
let draft_log_probs: Vec<f32> = vec![];
```

This causes `mtp_accept_count` (`runner.rs:6977-6981`) to fall through to the greedy argmax fallback for every draft position:

```rust
let can_rejection_sample = !matches!(source, MtpDraftSource::Ngram)
    && pending_log_probs.get(i).is_some_and(|log_prob| log_prob.is_finite())
    && target_probs_cpu.is_some();
```

When `pending_log_probs` is empty, `can_rejection_sample` is always false, so acceptance requires `predicted[i] == pending[i]` — argmax match only.

### 1.2 Solution

Modify `mtp_draft_tokens_greedy` to compute draft log-probs using `gpu_draft_log_prob_lazy` with temperature=1.0 (uniform draft distribution). The token selection remains argmax; only the log-prob computation is added.

```rust
fn mtp_draft_tokens_greedy(
    // ... existing params ...
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);  // NEW
    // ... existing loop ...
    for _ in 0..max_depth {
        let new_hidden = mtp_head_forward(/* ... */);
        let post_norm_hidden = mtp_hidden_post_norm(&new_hidden, head, cfg);
        let logits = mtp_post_norm_to_logits(&post_norm_hidden, weights, cfg);

        let lazy_tok = lazy_argmax_logits(&logits);
        lazy_tokens.push(lazy_tok.clone());

        // NEW: compute draft log-prob at temperature=1.0
        let lazy_lp = gpu_draft_log_prob_lazy(&logits, &lazy_tok, 1.0, vocab);
        lazy_log_probs.push(lazy_lp);

        prev_hidden = post_norm_hidden;
        prev_token_arr = lazy_tok;
    }

    let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2);
    for t in &lazy_tokens { all_refs.push(t); }
    for lp in &lazy_log_probs { all_refs.push(lp); }
    eval(&all_refs);

    let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
    let draft_log_probs: Vec<f32> = lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();
    // ... rest unchanged ...
}
```

### 1.3 Why temperature=1.0 for greedy draft log-probs

The draft distribution in greedy mode is implicitly uniform over the argmax token (probability 1.0 for the selected token, 0 for all others). Using temperature=1.0 gives `p_draft = softmax(logits)[argmax_token]`, which is the model's own confidence in its argmax choice. This is the correct draft probability for rejection sampling: we're asking "how probable is this draft token under the draft model's own distribution?"

Using the request temperature would conflate draft and target distributions. Temperature=1.0 keeps them independent.

### 1.4 Impact on existing behavior

- When `target_probs_cpu.is_none()` (temperature=0 request), the greedy fallback still applies — no change.
- When `target_probs_cpu.is_some()` (temperature>0 request), rejection sampling now engages for greedy-mode MTP drafts.
- The additional GPU work is one `softmax + take + log` per depth level, already lazy and batched into the single `eval`.

## Phase 2 — Temperature Alignment

### 2.1 Problem

The rejection sampling formula `accept_prob = min(1, p_target/p_draft)` assumes both distributions use the same temperature. When `T_draft != T_target`:

- `p_draft = softmax(logits_draft / T_draft)[token]`
- `p_target = softmax(logits_target / T_target)[token]`

If `T_draft > T_target`, the draft distribution is smoother, making `p_draft` smaller for peak tokens and `accept_prob` larger than it should be. This can cause false accepts that drift the output distribution.

### 2.2 Solution

In `mtp_accept_count`, rescale the draft probability to match the target temperature:

```rust
fn mtp_accept_count(
    // ... existing params ...
    draft_temperature: f32,      // NEW: from head.draft_sampling.temperature
    target_temperature: f32,     // NEW: from sampling.temperature
) -> MtpAcceptOutcome {
    // ... in the rejection sampling loop ...
    let p_draft = pending_log_probs[i].exp().max(1e-37_f32);
    
    // Rescale draft probability to target temperature
    let p_draft_scaled = if draft_temperature != target_temperature && draft_temperature > 0.0 {
        // p_scaled = p_draft ^ (T_draft / T_target)
        // In log space: log(p_scaled) = log(p_draft) * (T_draft / T_target)
        let log_p_draft = pending_log_probs[i];
        let scaled_log_prob = log_p_draft * (draft_temperature / target_temperature);
        scaled_log_prob.exp().max(1e-37_f32)
    } else {
        p_draft
    };
    
    let accept_prob = (p_target_d / p_draft_scaled).min(1.0_f32);
    // ... rest unchanged ...
}
```

### 2.3 Call site changes

In `run_mtp_decode` (`runner.rs:5727`), pass temperatures to `mtp_accept_count`:

```rust
let accept = mtp_accept_count(
    &pending,
    &state.mtp_pending_draft_log_probs,
    &state.mtp_pending_draft_distributions,
    &state.mtp_pending_draft_sources,
    target_probs_cpu,
    target_distributions_cpu,
    &predicted,
    &mut state.rng,
    self.weights.mtp.as_ref().map(|h| h.draft_sampling.temperature).unwrap_or(1.0),
    sampling.temperature,
);
```

### 2.4 Edge cases

- `target_temperature == 0`: No rejection sampling (greedy target), temperature alignment is irrelevant.
- `draft_temperature == 0`: Should not happen (draft always has temperature > 0 in sampled mode). Guard with `draft_temperature > 0.0` check.
- `draft_temperature == target_temperature`: No rescaling needed, fast path skips the computation.
- Very small `target_temperature` (e.g., 0.1 with `draft_temperature` 0.7): ratio is 7×, compressing `log_p_draft` toward −∞. This is mathematically correct — at near-greedy target temperatures the distribution is highly peaked, so only draft tokens near the target argmax survive; acceptance converges toward argmax matching. No numerical guard is needed because `log_p_draft ≤ 0` and ratio > 0 guarantee `scaled_log_prob ≤ 0` (i.e., `p_draft_scaled ≤ 1.0`).

## Phase 3 — N-gram Pseudo Log-Probs

### 3.1 Problem

In `mtp_accept_count`, n-gram-sourced tokens (`MtpDraftSource::Ngram`) always use greedy argmax comparison:

```rust
let can_rejection_sample = !matches!(source, MtpDraftSource::Ngram)
    && pending_log_probs.get(i).is_some_and(|log_prob| log_prob.is_finite())
    && target_probs_cpu.is_some();
```

This is correct — n-gram has no draft model, so there are no draft log-probs. But in hybrid mode (n-gram prefix + MTP tail), the n-gram tokens are often high-confidence (from repeating patterns). We can derive a pseudo draft probability from the n-gram confidence score and remove the `MtpDraftSource::Ngram` exclusion so those positions enter the rejection sampling path.

### 3.2 Solution

Add a new field to the n-gram draft outcome that carries confidence scores:

```rust
// In ngram_accel.rs
pub struct NgramDraftOutcome {
    pub draft: Vec<u32>,
    pub rejection: Option<NgramDraftRejection>,
    pub requested_max_len: usize,
    pub confidence: Vec<f32>,  // NEW: per-token confidence from n-gram table
}
```

In `predict_with_policy`, populate confidence from the prediction's `confidence()` method:

```rust
// In select_draft_step
DraftStepSelection::Selected(step) => {
    draft.push(step.token);
    confidence.push(step.confidence);  // from NgramPrediction::confidence()
    // ...
}
```

In `run_mtp_decode`, convert n-gram confidence to pseudo log-probs and drop the n-gram source exclusion from `can_rejection_sample`:

```rust
// After building new_draft from n-gram + MTP:
let mut aligned_log_probs = vec![f32::NAN; ngram_len];
for (i, &conf) in ngram_confidence.iter().enumerate() {
    // pseudo_log_prob = ln(confidence), clamped to [-30, 0]
    let log_conf = conf.max(1e-37_f32).ln().max(-30.0);
    aligned_log_probs[i] = log_conf;
}
aligned_log_probs.extend(log_probs);  // MTP tail log-probs
```

In `mtp_accept_count`, remove the `MtpDraftSource::Ngram` exclusion from the rejection sampling guard so that n-gram positions with a finite pseudo log-prob enter the rejection sampling path:

```rust
// BEFORE:
let can_rejection_sample = !matches!(source, MtpDraftSource::Ngram)
    && pending_log_probs.get(i).is_some_and(|log_prob| log_prob.is_finite())
    && target_probs_cpu.is_some();

// AFTER:
let can_rejection_sample = pending_log_probs.get(i).is_some_and(|log_prob| log_prob.is_finite())
    && target_probs_cpu.is_some();
```

The n-gram exclusion was the only thing blocking rejection sampling for n-gram tokens; now that `aligned_log_probs` carries finite pseudo log-probs for those positions, the existing `is_finite()` check is sufficient to gate the path.

### 3.3 Why `ln(confidence)` as pseudo log-prob

The n-gram confidence is `support/total` — the fraction of observations that produced this token. Interpreting this as a probability, `ln(confidence)` is the log-probability. Clamping to `[-30, 0]` matches the floor in `gpu_draft_log_prob_lazy`.

This is a heuristic, not a true draft model probability. But it gives rejection sampling a signal: high-confidence n-gram tokens (confidence ≈ 1.0) have `log_prob ≈ 0`, so `p_draft ≈ 1.0`, making `accept_prob = p_target/1.0 = p_target` — the target model's own probability. Low-confidence n-gram tokens (confidence ≈ 0.1) have `log_prob ≈ -2.3`, so `p_draft ≈ 0.1`, making `accept_prob = p_target/0.1 = 10 * p_target` (capped at 1.0) — more likely to accept.

### 3.4 Impact on existing behavior

- Pure n-gram mode (no MTP): No change — `mtp_accept_count` is not called.
- Pure MTP mode (n-gram disabled): No change — no n-gram tokens.
- Hybrid mode: N-gram prefix tokens now participate in rejection sampling with pseudo log-probs.

## Phase 4 — Adaptive Gate and Depth

### 4.1 Lower n-gram saturation gate threshold

Current threshold (`runner.rs:6269-6270`):

```rust
state.mtp_telemetry.mtp_only_accept_rate_ewma
    >= adaptive_ngram_saturation_threshold(mtp_max_depth)
```

Where `adaptive_ngram_saturation_threshold` returns 0.99 for depth≥3.

Change to (preserving the existing `AX_MLX_MTP_NGRAM_GATE_THRESHOLD` env var override path):

```rust
fn adaptive_ngram_saturation_threshold(mtp_depth: usize) -> f32 {
    if mtp_depth <= 1 {
        return 2.0;  // effectively ∞ — gate disabled for depth 0/1
    }
    std::env::var("AX_MLX_MTP_NGRAM_GATE_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(if mtp_depth >= 3 { 0.97 } else { 0.98 })
        //                              ^^^^ lowered from 0.99
}
```

Rationale: At 97% MTP-only acceptance, ~3% of drafts are rejected. N-gram can fill these gaps. The gate still disables n-gram when MTP is truly saturated (≥99%), but keeps it active in the 97-99% range where n-gram still adds value. The env var override is preserved so the rollback kill switch (`AX_MLX_MTP_NGRAM_GATE_THRESHOLD=0.99`) continues to work.

### 4.2 Adaptive depth floor on complete miss

Current behavior (`runner.rs:6799-6800`):

```rust
let floor = 2.min(max_depth);
accept_count.clamp(floor, max_depth)
```

After `accept_count == 0`, depth is clamped to 2. This means the next step still attempts depth-2 drafts, which are likely to be rejected again.

Change to track consecutive misses and adjust floor:

```rust
// In RequestState
mtp_consecutive_misses: u32,

// In mtp_next_adaptive_depth
fn mtp_next_adaptive_depth(
    current_depth: usize,
    max_depth: usize,
    pending_len: usize,
    accept_count: usize,
    consecutive_misses: u32,  // NEW
) -> usize {
    // ... existing logic ...
    
    if accept_count == 0 {
        // Complete miss: drop floor based on consecutive miss count
        let floor = match consecutive_misses {
            0 => 2.min(max_depth),   // first miss: floor 2
            1 => 1.min(max_depth),   // second consecutive miss: floor 1
            _ => 0,                   // third+ consecutive miss: floor 0 (pure n-gram)
        };
        return floor;
    }
    
    // Reset consecutive miss counter on any accept
    // (caller handles this)
    // ...
}
```

The caller (`run_mtp_decode`) updates `mtp_consecutive_misses`:

```rust
if accept_count == 0 {
    state.mtp_consecutive_misses += 1;
} else {
    state.mtp_consecutive_misses = 0;
}
```

## Testing Strategy

### Unit tests

- `mtp.rs`: Test `mtp_draft_tokens_greedy` returns non-empty `draft_log_probs` with correct values (softmax at T=1.0 for argmax token).
- `runner.rs`: Test `mtp_accept_count` with temperature rescaling produces correct `accept_prob` for known `p_target`, `p_draft`, `T_draft`, `T_target`.
- `ngram_accel.rs`: Test pseudo log-prob computation from confidence: `ln(0.9) ≈ -0.105`, `ln(0.1) ≈ -2.30`.
- `runner.rs`: Test `mtp_next_adaptive_depth` with consecutive misses: 0→2, 1→1, 2→0.

### Integration tests

- `cargo run --release --bin decode-trace` on Qwen3.6-27B-MTP 4-bit with greedy: verify `ax_mtp_accepted_depth2` increases vs baseline.
- `cargo run --release --bin decode-trace` on Qwen3.6-27B-MTP 4-bit with T=0.6, top-k=20, top-p=0.95: verify `ax_mlx_bonus_tokens` increases vs baseline (3723 → target 4000+).
- Hybrid n-gram+MTP test: run with `AX_MLX_MTP_DISABLE_NGRAM_STACKING=0` on repeating prompt, verify `ax_mtp_ngram_hybrid_tail_steps > 0` and `ax_mtp_ngram_hybrid_tail_tokens > 0`.

### Benchmark gates

- `benchmarks/results/mtp-fair/` A/B on Qwen3.6-27B-MTP 4-bit:
  - `ax_engine.json` (pure MTP) vs baseline: depth-2 accept rate ≥ 99.5%, bonus tokens ≥ 4000.
  - `ax_engine_ngram.json` (hybrid) vs baseline: hybrid tail steps > 0, no regression on cycle acceptance.
- No regression on Qwen3.6-35B-A3B 4-bit (MoE model, different acceptance profile).

## Rollback Plan

All changes are gated behind existing patterns:
- Phase 1: New log-prob computation in greedy mode only engages when `pending_log_probs` is non-empty. Existing empty-vector path is preserved for backward compatibility.
- Phase 2: Temperature rescaling only applies when `draft_temperature != target_temperature`. Fast path skips computation when equal.
- Phase 3: Pseudo log-probs only used for `MtpDraftSource::Ngram` tokens. MTP tokens use real log-probs unchanged.
- Phase 4: Gate threshold and depth floor changes are parameterized; can be reverted by restoring constants.

Kill switches via env var:
- `AX_MLX_MTP_GREEDY_LOGPROBS=0` — disable greedy mode draft log-probs, fall back to argmax-only.
- `AX_MLX_MTP_TEMP_ALIGN=0` — disable temperature alignment, use raw draft log-probs.
- `AX_MLX_MTP_NGRAM_PSEUDO_LOGPROB=0` — disable n-gram pseudo log-probs, fall back to argmax comparison.
- `AX_MLX_MTP_NGRAM_GATE_THRESHOLD=0.99` — override gate threshold back to 0.99.
- `AX_MLX_MTP_ADAPTIVE_MISS_FLOOR=2` — override adaptive miss floor to constant 2.
