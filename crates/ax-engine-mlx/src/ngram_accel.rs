use std::collections::{HashMap, VecDeque};

use mlx_sys::{MlxArray, argmax, eval};

use crate::sampling::{Xorshift64, sample_categorical};

use crate::kv_cache::MlxKVCache;
use crate::model::{
    ModelConfig, TurboQuantModelDecodeContext, forward_all_positions,
    forward_with_turboquant_context,
};
use crate::weights::ModelWeights;

/// Default number of draft tokens to attempt per n-gram acceleration step.
pub const DEFAULT_DRAFT_LEN: usize = 4;

/// Extended draft ceiling for dense models when confidence is high.
///
/// Dense models pay O(1) rollback (just a seq_len pointer move), so longer
/// drafts at high confidence are cheap to attempt.  Linear-attention models
/// keep `DEFAULT_DRAFT_LEN` because partial-reject triggers branch/recompute.
pub const MAX_DRAFT_LEN: usize = 6;

/// Minimum confidence (support/total) for a bigram/trigram to be drafted.
///
/// A prediction with confidence below this threshold stops the draft chain.
/// Calibration: conf=0.4 filters contexts where at most 2 out of 5 observed
/// continuations matched the current best — reliable enough to attempt.
pub const DRAFT_CONFIDENCE_THRESHOLD: f32 = 0.4;

/// Linear-attention draft verification is expensive on partial reject
/// because recurrent state is not O(1)-trimmable. Require repeated n-gram
/// evidence before probing that path.
pub const LINEAR_MIN_NGRAM_SUPPORT: u32 = 2;

#[derive(Clone, Copy)]
struct NgramPrediction {
    token: u32,
    /// Observations where `token` was the continuation for this context key.
    support: u32,
    /// Total observations for this context key across all continuations.
    total: u32,
}

impl NgramPrediction {
    /// Fraction of observations that produced `token`.
    fn confidence(self) -> f32 {
        self.support as f32 / self.total as f32
    }
}

/// N-gram lookup table for self-drafting decoding.
///
/// Tracks bigrams and trigrams observed in the prompt and generated tokens.
/// `predict()` chains lookups to produce a draft sequence.
pub struct NgramTable {
    bigrams: HashMap<(u32, u32), NgramPrediction>,
    trigrams: HashMap<(u32, u32, u32), NgramPrediction>,
    /// Last 3 observed tokens — context window for next prediction.
    tail: VecDeque<u32>,
}

impl NgramTable {
    pub fn new() -> Self {
        Self {
            bigrams: HashMap::new(),
            trigrams: HashMap::new(),
            tail: VecDeque::with_capacity(4),
        }
    }

    /// Ingest a slice of tokens (call with the prompt, then with each batch
    /// of accepted tokens as generation proceeds).
    pub fn feed(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.observe(t);
        }
    }

    /// Record one token and update the n-gram table.
    fn observe(&mut self, t: u32) {
        let n = self.tail.len();
        if n >= 2 {
            let a = self.tail[n - 2];
            let b = self.tail[n - 1];
            update_prediction(&mut self.bigrams, (a, b), t);
            if n >= 3 {
                update_prediction(&mut self.trigrams, (self.tail[n - 3], a, b), t);
            }
        }
        self.tail.push_back(t);
        if self.tail.len() > 3 {
            self.tail.pop_front();
        }
    }

    /// Predict up to `max_len` draft tokens by chaining n-gram lookups.
    /// Returns an empty vec when no matching n-gram exists yet.
    pub fn predict(&self, max_len: usize) -> Vec<u32> {
        self.predict_with_confidence(max_len, 1, 0.0)
    }

    /// Predict only from n-grams that have observed the same continuation at
    /// least `min_support` times. Useful for expensive verification policies
    /// where one-off prompt n-grams are more likely to be harmful probes.
    pub fn predict_with_min_support(&self, max_len: usize, min_support: u32) -> Vec<u32> {
        self.predict_with_confidence(max_len, min_support, 0.0)
    }

    /// Predict up to `max_len` draft tokens, stopping when a step's n-gram
    /// confidence (support/total) drops below `conf_threshold`.
    ///
    /// Trigrams are preferred; if a trigram fails either filter the predictor
    /// falls back to the bigram for the same terminal pair.  This mirrors the
    /// chain-fallback in `predict_with_min_support` and is important when a
    /// trigram is contested but the corresponding bigram is dominant.
    ///
    /// `conf_threshold = 0.0` makes this equivalent to `predict_with_min_support`.
    pub fn predict_with_confidence(
        &self,
        max_len: usize,
        min_support: u32,
        conf_threshold: f32,
    ) -> Vec<u32> {
        let mut draft = Vec::with_capacity(max_len);
        // Fixed-size ring; self.tail has at most 3 elements.
        let mut buf = [0u32; 3];
        let mut len = self.tail.len().min(3);
        for (i, &t) in self.tail.iter().take(len).enumerate() {
            buf[i] = t;
        }

        let passes =
            |p: &&NgramPrediction| p.support >= min_support && p.confidence() >= conf_threshold;

        for _ in 0..max_len {
            let next = if len >= 3 {
                self.trigrams
                    .get(&(buf[0], buf[1], buf[2]))
                    .filter(passes)
                    .map(|p| p.token)
                    .or_else(|| {
                        self.bigrams
                            .get(&(buf[1], buf[2]))
                            .filter(passes)
                            .map(|p| p.token)
                    })
            } else if len == 2 {
                self.bigrams
                    .get(&(buf[0], buf[1]))
                    .filter(passes)
                    .map(|p| p.token)
            } else {
                None
            };

            match next {
                Some(t) => {
                    draft.push(t);
                    if len < 3 {
                        buf[len] = t;
                        len += 1;
                    } else {
                        buf[0] = buf[1];
                        buf[1] = buf[2];
                        buf[2] = t;
                    }
                }
                None => break,
            }
        }
        draft
    }
}

fn update_prediction<K>(map: &mut HashMap<K, NgramPrediction>, key: K, token: u32)
where
    K: Eq + std::hash::Hash,
{
    match map.get_mut(&key) {
        Some(p) if p.token == token => {
            p.support = p.support.saturating_add(1);
            p.total = p.total.saturating_add(1);
        }
        Some(p) => {
            // A different continuation was observed: the new token takes over as
            // the best prediction but total accumulates across all continuations,
            // which reduces confidence and prevents drafting contested contexts.
            p.token = token;
            p.support = 1;
            p.total = p.total.saturating_add(1);
        }
        None => {
            map.insert(
                key,
                NgramPrediction {
                    token,
                    support: 1,
                    total: 1,
                },
            );
        }
    }
}

impl Default for NgramTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Run one n-gram acceleration decode step and return a batch of verified tokens.
///
/// ## Algorithm
///
/// 1. **Draft** — look up `max_draft` candidates from the n-gram table.
/// 2. **Verify** — one `forward_all_positions` pass over `[last_token] ++ draft`.
/// 3. **Accept/reject** — accept draft tokens where the model's
///    argmax prediction matches, stopping at the first mismatch.
/// 4. **Trim** — roll back the KV cache to remove rejected positions.
/// 5. **Update** — feed accepted tokens into the n-gram table.
///
/// ## Returns
///
/// A `Vec<u32>` with **at least one element**:
/// - `result[0]` — output token for the current step.
/// - `result[1..]` — bonus tokens already verified; caller should queue them.
///
/// The bonus tokens already have their KV entries in the cache.  The caller
/// must NOT re-run the model for them; just pop them as subsequent outputs.
/// The LAST element of `result` is the starting `last_token` for the next
/// n-gram acceleration step.
#[allow(clippy::too_many_arguments)]
pub fn ngram_accel_decode_step(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    ngram: &mut NgramTable,
    last_token: u32,
    draft: &[u32],
    temperature: f32,
    rng: &mut Xorshift64,
) -> Vec<u32> {
    if draft.is_empty() {
        return single_decode(cfg, weights, cache, ngram, last_token, temperature, rng);
    }

    if cfg.linear_attention.is_some() {
        return ngram_accel_decode_step_linear_safe(
            cfg,
            weights,
            cache,
            ngram,
            last_token,
            draft,
            temperature,
            rng,
        );
    }

    let token_offset = cache.seq_len;
    let verification = verify_draft(
        cfg,
        weights,
        cache,
        last_token,
        draft,
        token_offset,
        temperature,
        rng,
    );

    // Trim KV cache: keep only [last_token + accepted_drafts].
    // The correction/bonus token at result.last() is NOT yet in the cache.
    // KV buffers were already materialised inside verify_draft's combined eval.
    let trimmed = cache.trim_to(verification.committed_len);
    debug_assert!(
        trimmed,
        "n-gram verification committed_len must not exceed cache seq_len"
    );

    // Update n-gram table.
    ngram.feed(&draft[..verification.accept_count]);
    ngram.feed(&verification.result[verification.accept_count..]); // correction or bonus

    verification.result
}

#[allow(clippy::too_many_arguments)]
fn ngram_accel_decode_step_linear_safe(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    ngram: &mut NgramTable,
    last_token: u32,
    draft: &[u32],
    temperature: f32,
    rng: &mut Xorshift64,
) -> Vec<u32> {
    let token_offset = cache.seq_len;
    let mut verify_cache = cache.clone();
    let verification = verify_draft(
        cfg,
        weights,
        &mut verify_cache,
        last_token,
        draft,
        token_offset,
        temperature,
        rng,
    );

    if verification.accept_count == draft.len() {
        // verify_cache's KV buffers were already materialised inside verify_draft's
        // combined eval — no separate materialize_cache call needed.
        let trimmed = verify_cache.trim_to(verification.committed_len);
        debug_assert!(
            trimmed,
            "linear-safe verification committed_len must not exceed cache seq_len"
        );
        *cache = verify_cache;
    } else {
        recompute_committed_prefix(
            cfg,
            weights,
            cache,
            last_token,
            &draft[..verification.accept_count],
            token_offset,
        );
    }

    ngram.feed(&draft[..verification.accept_count]);
    ngram.feed(&verification.result[verification.accept_count..]); // correction or bonus

    verification.result
}

struct DraftVerification {
    accept_count: usize,
    committed_len: usize,
    result: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn verify_draft(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    last_token: u32,
    draft: &[u32],
    token_offset: usize,
    temperature: f32,
    rng: &mut Xorshift64,
) -> DraftVerification {
    // Verification sequence: [last_token, D1, D2, ... D_n].
    let mut verify_input = Vec::with_capacity(1 + draft.len());
    verify_input.push(last_token);
    verify_input.extend_from_slice(draft);
    let verify_len = verify_input.len();

    // One causal forward pass -> [verify_len, vocab_size] f32 logits.
    let logits_all = forward_all_positions(cfg, weights, &verify_input, cache, token_offset);
    cache.seq_len += verify_len;

    // Build the argmax graph node before any GPU sync.
    // Evaluating predicted_arr transitively evaluates logits_all, so we can
    // combine what was three separate evals (logits_all, predicted_arr, KV refs)
    // into a single blocking call.
    let predicted_arr = argmax(&logits_all, None);
    let kv_refs = cache.collect_eval_refs();
    let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
    targets.push(&predicted_arr);
    targets.extend(kv_refs);
    eval(&targets);

    // Both logits_all (transitive dep of predicted_arr) and predicted_arr are
    // now materialised. KV backing buffers are also flat — no separate
    // materialize_cache call is needed in the caller.
    //
    // For temperature > 0, read all verify-step logits to CPU once so we can
    // sample the correction / bonus token from the full distribution.
    // Draft acceptance always uses argmax comparison (n-gram drafts are
    // deterministic, so there is no draft distribution to resample from).
    let cpu_logits: Vec<f32> = if temperature > 0.0 {
        logits_all.data_f32().to_vec()
    } else {
        Vec::new()
    };
    let predicted: Vec<u32> = predicted_arr.data_u32().to_vec();

    let vocab = cfg.vocab_size;

    // Accept/reject.
    // predicted[i] = model's argmax prediction for the token AFTER verify_input[i].
    // draft[i]     = verify_input[i+1].
    let mut result: Vec<u32> = Vec::new();
    let mut accept_count = 0usize;

    for i in 0..draft.len() {
        if predicted[i] == draft[i] {
            result.push(draft[i]);
            accept_count += 1;
        } else {
            // Correction token: sample at position i with temperature.
            let tok = sample_pos(&cpu_logits, predicted[i], i, vocab, temperature, rng);
            result.push(tok);
            break;
        }
    }

    // Bonus: if ALL draft tokens were accepted, sample the next token for free.
    if accept_count == draft.len() {
        let pos = draft.len();
        let tok = sample_pos(&cpu_logits, predicted[pos], pos, vocab, temperature, rng);
        result.push(tok);
    }

    DraftVerification {
        accept_count,
        committed_len: token_offset + 1 + accept_count,
        result,
    }
}

fn recompute_committed_prefix(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    last_token: u32,
    accepted_draft: &[u32],
    token_offset: usize,
) {
    let mut commit_input = Vec::with_capacity(1 + accepted_draft.len());
    commit_input.push(last_token);
    commit_input.extend_from_slice(accepted_draft);

    let logits = forward_all_positions(cfg, weights, &commit_input, cache, token_offset);
    cache.seq_len += commit_input.len();
    let kv_refs = cache.collect_eval_refs();
    let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
    targets.push(&logits);
    targets.extend(kv_refs);
    eval(&targets);
}

/// Sample token at `pos` in the flattened `[verify_len, vocab]` logit buffer.
/// Falls back to `argmax_tok` when temperature is 0 or the buffer is empty.
fn sample_pos(
    cpu_logits: &[f32],
    argmax_tok: u32,
    pos: usize,
    vocab: usize,
    temperature: f32,
    rng: &mut Xorshift64,
) -> u32 {
    if temperature <= 0.0 || cpu_logits.is_empty() {
        return argmax_tok;
    }
    let start = pos * vocab;
    let end = start + vocab;
    if end > cpu_logits.len() {
        return argmax_tok;
    }
    sample_categorical(&cpu_logits[start..end], temperature, rng)
}

/// Single-token decode fallback (used when n-gram table has no prediction).
///
/// Respects `temperature`: 0.0 → argmax, > 0.0 → categorical sampling.
pub fn single_decode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    ngram: &mut NgramTable,
    last_token: u32,
    temperature: f32,
    rng: &mut Xorshift64,
) -> Vec<u32> {
    single_decode_with_turboquant_context(
        cfg,
        weights,
        cache,
        ngram,
        last_token,
        temperature,
        rng,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn single_decode_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    ngram: &mut NgramTable,
    last_token: u32,
    temperature: f32,
    rng: &mut Xorshift64,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> Vec<u32> {
    let token_offset = cache.seq_len;
    let logits = forward_with_turboquant_context(
        cfg,
        weights,
        &[last_token],
        cache,
        token_offset,
        turboquant_context,
    );
    cache.seq_len += 1;

    let kv_refs = cache.collect_eval_refs();
    let tok = if temperature > 0.0 {
        let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
        targets.push(&logits);
        targets.extend(kv_refs);
        eval(&targets);
        sample_categorical(logits.data_f32(), temperature, rng)
    } else {
        let token_arr = argmax(&logits, None);
        let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
        targets.push(&token_arr);
        targets.extend(kv_refs);
        eval(&targets);
        token_arr.data_u32().first().copied().unwrap_or(0)
    };

    ngram.observe(tok);
    vec![tok]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_from_sequence(tokens: &[u32]) -> NgramTable {
        let mut t = NgramTable::new();
        t.feed(tokens);
        t
    }

    #[test]
    fn predict_returns_empty_for_empty_table() {
        let t = NgramTable::new();
        assert!(t.predict(4).is_empty());
    }

    #[test]
    fn predict_returns_empty_when_no_matching_ngram_at_tail() {
        // After feeding a linear sequence, the tail ends with the last 3 tokens
        // but the bigram for the LAST pair was never recorded (recording happens
        // when a subsequent token is observed).  So predict returns empty.
        let t = table_from_sequence(&[1, 2, 3, 4, 5]);
        // tail=[3,4,5]; bigrams recorded: (1,2)→3, (2,3)→4, (3,4)→5
        // predict looks up bigram(4,5) which was not recorded → []
        assert!(t.predict(4).is_empty());
    }

    #[test]
    fn predict_chains_trigrams_over_bigrams() {
        // Repeated pattern builds trigrams; predict reconstructs the cycle.
        let t = table_from_sequence(&[1, 2, 3, 1, 2, 3, 1, 2, 3]);
        // tail=[1,2,3]; trigrams: (1,2,3)→1, (2,3,1)→2, (3,1,2)→3
        let draft = t.predict(4);
        assert_eq!(draft, vec![1, 2, 3, 1]);
    }

    #[test]
    fn predict_with_min_support_filters_one_off_prompt_matches() {
        let t = table_from_sequence(&[9, 8, 7, 9, 8, 7]);

        assert_eq!(t.predict(3), vec![9, 8, 7]);
        assert!(
            t.predict_with_min_support(3, 2).is_empty(),
            "the tail trigram has only one observed continuation"
        );

        let t = table_from_sequence(&[9, 8, 7, 9, 8, 7, 9, 8, 7]);
        assert_eq!(t.predict_with_min_support(3, 2), vec![9, 8, 7]);
    }

    #[test]
    fn predict_deterministic_across_calls() {
        // predict() must not mutate the table; two calls return identical drafts.
        let tokens: Vec<u32> = (1u32..=5).cycle().take(20).collect();
        let t = table_from_sequence(&tokens);
        // Repeating 1..5 cycle: tail ends at the 3-token suffix of the last triplet.
        // Trigrams are established so prediction is non-empty.
        let d1 = t.predict(6);
        let d2 = t.predict(6);
        assert_eq!(
            d1, d2,
            "predict must be deterministic and must not mutate the table"
        );
        assert!(
            !d1.is_empty(),
            "repeating sequence should produce a non-empty draft"
        );
    }

    #[test]
    fn predict_stops_at_max_len() {
        // With a fully cycling pattern, predict should respect the length cap.
        let tokens: Vec<u32> = (1u32..=3).cycle().take(12).collect();
        let t = table_from_sequence(&tokens);
        assert!(t.predict(2).len() <= 2);
        assert!(t.predict(0).is_empty());
    }

    #[test]
    fn predict_with_confidence_zero_threshold_matches_wrappers() {
        // conf_threshold=0.0 must be identical to predict() / predict_with_min_support().
        let tokens: Vec<u32> = (1u32..=5).cycle().take(20).collect();
        let t = table_from_sequence(&tokens);
        assert_eq!(t.predict_with_confidence(6, 1, 0.0), t.predict(6));
        assert_eq!(
            t.predict_with_confidence(6, 2, 0.0),
            t.predict_with_min_support(6, 2)
        );
    }

    #[test]
    fn predict_with_confidence_high_conf_chain_matches_predict() {
        // All bigrams in a uniform cycle have confidence = 1.0 — a high threshold
        // must not filter anything out.
        let t = table_from_sequence(&[1, 2, 3, 1, 2, 3, 1, 2, 3]);
        assert_eq!(
            t.predict_with_confidence(4, 1, 0.9),
            t.predict(4),
            "uniform cycle should pass any confidence threshold"
        );
    }

    #[test]
    fn predict_with_confidence_stops_at_contested_bigram() {
        // Build bigram(3,1) with support=1, total=3 → confidence=0.33.
        // Arrange the tail to end with a context that looks up bigram(3,1).
        //
        // Feed [3,1,2, 3,1,4, 3,1,5] to observe bigram(3,1) three times with
        // three different continuations (2→4→5), leaving the last best=5, conf=0.33.
        // Then feed [9,3,1] to place the tail at [9,3,1] so the next predict call
        // checks trigram(9,3,1) (absent) then falls back to bigram(3,1).
        let mut t = NgramTable::new();
        t.feed(&[3, 1, 2, 3, 1, 4, 3, 1, 5]);
        t.feed(&[9, 3, 1]);

        // Without confidence gating: bigram(3,1) is found and the chain continues.
        let draft_unfiltered = t.predict(4);
        assert!(
            !draft_unfiltered.is_empty(),
            "predict() should find bigram(3,1) regardless of confidence"
        );

        // With confidence gating at 0.4: conf=0.33 < 0.4 → chain stops immediately.
        let draft_filtered = t.predict_with_confidence(4, 1, 0.4);
        assert!(
            draft_filtered.is_empty(),
            "bigram with confidence 0.33 should be filtered at threshold 0.4"
        );
    }

    #[test]
    fn predict_with_confidence_total_accumulates_across_champion_changes() {
        // After the best token for a context changes twice, total > support,
        // which should lower confidence and filter drafts at a moderate threshold.
        let mut t = NgramTable::new();
        // bigram(1,2): observe → A (sup=1,tot=1), then B (sup=1,tot=2), then B (sup=2,tot=3)
        // Tail after this sequence ends at [...,1,2,B] → not the test context; we need
        // another feed to put the tail back at a position that looks up bigram(1,2).
        t.feed(&[1, 2, 10, 1, 2, 20, 1, 2, 20]);
        // At this point bigram(1,2): token=20, support=2, total=3, confidence=0.67
        // Tail = [1, 2, 20] — predict would check trigram(1,2,20) / bigram(2,20).
        // Extend the tail so bigram(1,2) is the lookup target:
        t.feed(&[5, 1, 2]);
        // After observe(5): bigram(2,20)→5; tail=[2,20,5]
        // After observe(1): bigram(20,5)→1; tail=[20,5,1]
        // After observe(2): bigram(5,1)→2; tail=[5,1,2]
        // Now predict checks trigram(5,1,2) [absent] → bigram(1,2): token=20, sup=2, tot=3, conf=0.67.
        // conf=0.67 ≥ 0.4 → should pass
        assert!(
            !t.predict_with_confidence(2, 1, 0.4).is_empty(),
            "confidence 0.67 should pass threshold 0.4"
        );
        // conf=0.67 < 0.75 → should fail
        assert!(
            t.predict_with_confidence(2, 1, 0.75).is_empty(),
            "confidence 0.67 should not pass threshold 0.75"
        );
    }
}
