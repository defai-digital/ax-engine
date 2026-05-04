use std::collections::{HashMap, VecDeque};

use mlx_sys::{MlxArray, argmax, eval};

use crate::sampling::{Xorshift64, sample_categorical};

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
        // Fixed-size ring to avoid heap allocation on every decode step.
        // self.tail has at most 3 elements; we maintain the last ≤3 tokens here.
        let mut buf = [0u32; 3];
        let mut len = self.tail.len().min(3);
        for (i, &t) in self.tail.iter().take(len).enumerate() {
            buf[i] = t;
        }
        for _ in 0..max_len {
            let next = if len >= 3 {
                self.trigrams
                    .get(&(buf[0], buf[1], buf[2]))
                    .copied()
                    .or_else(|| self.bigrams.get(&(buf[1], buf[2])).copied())
            } else if len == 2 {
                self.bigrams.get(&(buf[0], buf[1])).copied()
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
#[allow(clippy::too_many_arguments)]
pub fn speculative_decode_step(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    ngram: &mut NgramTable,
    last_token: u32,
    max_draft: usize,
    temperature: f32,
    rng: &mut Xorshift64,
) -> Vec<u32> {
    let draft = ngram.predict(max_draft);

    if draft.is_empty() {
        return single_decode(cfg, weights, cache, ngram, last_token, temperature, rng);
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

    // For temperature > 0, read all verify-step logits to CPU once so we can
    // sample the correction / bonus token from the full distribution.
    // Draft acceptance always uses greedy comparison (n-gram drafts are
    // deterministic, so there is no draft distribution to resample from).
    let cpu_logits: Vec<f32> = if temperature > 0.0 {
        logits_all.data_f32().to_vec()
    } else {
        Vec::new()
    };

    // argmax over vocab axis (last axis of 2-D tensor) → [verify_len] u32.
    let predicted_arr = argmax(&logits_all, None);
    eval(&[&predicted_arr]);
    let predicted: Vec<u32> = predicted_arr.data_u32().to_vec();

    let vocab = cfg.vocab_size;

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

    // Trim KV cache: keep only [last_token + accepted_drafts].
    // The correction/bonus token at result.last() is NOT yet in the cache.
    cache.trim_to(token_offset + 1 + accept_count);

    // Update n-gram table.
    ngram.feed(&draft[..accept_count]);
    ngram.feed(&result[accept_count..]); // correction or bonus

    // Materialise the KV backing buffers to break the slice_update graph chain.
    // forward_all_positions wrote speculative positions into the buffers; the
    // trim_to above moved the logical boundary back, but the chain still grows
    // unless we eval here.  Bundled with the already-paid eval(&[&predicted_arr])
    // sync so no extra GPU round-trip is introduced.
    let kv_refs = cache.collect_eval_refs();
    if !kv_refs.is_empty() {
        eval(&kv_refs);
    }

    result
}

/// Sample token at `pos` in the flattened `[verify_len, vocab]` logit buffer.
/// Falls back to `greedy_tok` when temperature is 0 or the buffer is empty.
fn sample_pos(
    cpu_logits: &[f32],
    greedy_tok: u32,
    pos: usize,
    vocab: usize,
    temperature: f32,
    rng: &mut Xorshift64,
) -> u32 {
    if temperature <= 0.0 || cpu_logits.is_empty() {
        return greedy_tok;
    }
    let start = pos * vocab;
    let end = start + vocab;
    if end > cpu_logits.len() {
        return greedy_tok;
    }
    sample_categorical(&cpu_logits[start..end], temperature, rng)
}

/// Single-token decode fallback (used when n-gram table has no prediction).
///
/// Respects `temperature`: 0.0 → greedy argmax, > 0.0 → categorical sampling.
pub fn single_decode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    ngram: &mut NgramTable,
    last_token: u32,
    temperature: f32,
    rng: &mut Xorshift64,
) -> Vec<u32> {
    let token_offset = cache.seq_len;
    let logits = crate::model::forward(cfg, weights, &[last_token], cache, token_offset);
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
}
