use std::collections::{HashMap, VecDeque};

use mlx_sys::{argmax, eval};

use crate::kv_cache::MlxKVCache;
use crate::model::{ModelConfig, forward_all_positions};
use crate::weights::ModelWeights;

/// Default number of draft tokens to attempt per speculative step.
pub const DEFAULT_DRAFT_LEN: usize = 4;

/// N-gram lookup table for self-speculative (draft-free) decoding.
///
/// Tracks bigrams and trigrams observed in the prompt and generated tokens.
/// `predict()` chains lookups to produce a draft sequence.
pub struct NgramTable {
    bigrams: HashMap<(u32, u32), u32>,
    trigrams: HashMap<(u32, u32, u32), u32>,
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
            self.bigrams.insert((a, b), t);
            if n >= 3 {
                self.trigrams.insert((self.tail[n - 3], a, b), t);
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
        let mut draft = Vec::with_capacity(max_len);
        let mut tail: Vec<u32> = self.tail.iter().copied().collect();

        for _ in 0..max_len {
            let n = tail.len();
            let next = if n >= 3 {
                self.trigrams
                    .get(&(tail[n - 3], tail[n - 2], tail[n - 1]))
                    .copied()
                    .or_else(|| self.bigrams.get(&(tail[n - 2], tail[n - 1])).copied())
            } else if n >= 2 {
                self.bigrams.get(&(tail[n - 2], tail[n - 1])).copied()
            } else {
                None
            };

            match next {
                Some(t) => {
                    draft.push(t);
                    tail.push(t);
                    if tail.len() > 3 {
                        tail.remove(0);
                    }
                }
                None => break,
            }
        }
        draft
    }
}

impl Default for NgramTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Run one speculative decode step and return a batch of verified tokens.
///
/// ## Algorithm
///
/// 1. **Draft** — look up `max_draft` candidates from the n-gram table.
/// 2. **Verify** — one `forward_all_positions` pass over `[last_token] ++ draft`.
/// 3. **Accept/reject** — greedily accept draft tokens where the model's
///    greedy prediction matches, stopping at the first mismatch.
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
/// speculative step.
pub fn speculative_decode_step(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    ngram: &mut NgramTable,
    last_token: u32,
    max_draft: usize,
) -> Vec<u32> {
    let draft = ngram.predict(max_draft);

    if draft.is_empty() {
        return single_decode(cfg, weights, cache, ngram, last_token);
    }

    let token_offset = cache.seq_len;

    // Verification sequence: [last_token, D1, D2, …, D_n].
    let mut verify_input = Vec::with_capacity(1 + draft.len());
    verify_input.push(last_token);
    verify_input.extend_from_slice(&draft);
    let verify_len = verify_input.len();

    // One causal forward pass → [verify_len, vocab_size] f32 logits.
    let logits_all = forward_all_positions(cfg, weights, &verify_input, cache, token_offset);
    eval(&[&logits_all]);
    cache.seq_len += verify_len;

    // argmax over vocab axis (last axis of 2-D tensor) → [verify_len] u32.
    let predicted_arr = argmax(&logits_all, None);
    eval(&[&predicted_arr]);
    let predicted: Vec<u32> = predicted_arr.data_u32().to_vec();

    // Accept/reject.
    // predicted[i] = model's greedy prediction for the token AFTER verify_input[i].
    // draft[i]     = verify_input[i+1].
    let mut result: Vec<u32> = Vec::new();
    let mut accept_count = 0usize;

    for i in 0..draft.len() {
        if predicted[i] == draft[i] {
            result.push(draft[i]);
            accept_count += 1;
        } else {
            // Correction: take the model's prediction at this position.
            result.push(predicted[i]);
            break;
        }
    }

    // Bonus: if ALL draft tokens were accepted, the model's next prediction
    // (after the last draft) is also valid — append it for free.
    if accept_count == draft.len() {
        result.push(predicted[draft.len()]);
    }

    // Trim KV cache: keep only [last_token + accepted_drafts].
    // The correction/bonus token at result.last() is NOT yet in the cache.
    cache.trim_to(token_offset + 1 + accept_count);

    // Update n-gram table.
    ngram.feed(&draft[..accept_count]);
    ngram.feed(&result[accept_count..]); // correction or bonus

    result
}

/// Single-token greedy decode (fallback when n-gram table has no prediction).
pub fn single_decode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    ngram: &mut NgramTable,
    last_token: u32,
) -> Vec<u32> {
    let token_offset = cache.seq_len;
    let logits = crate::model::forward(cfg, weights, &[last_token], cache, token_offset);
    cache.seq_len += 1;
    let token_arr = argmax(&logits, None);
    eval(&[&token_arr]);
    let tok = token_arr.data_u32().first().copied().unwrap_or(0);
    ngram.observe(tok);
    vec![tok]
}
