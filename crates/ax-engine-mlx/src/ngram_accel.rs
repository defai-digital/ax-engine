use std::collections::{HashMap, HashSet, VecDeque};

use mlx_sys::{MlxArray, MlxDtype, argmax, eval, multiply, reshape, slice, softmax, take};

use crate::sampling::{
    MlxSamplingParams, MlxSamplingRequest, Xorshift64, sample_categorical_gpu,
    sample_categorical_into, sample_categorical_with_topk_gpu, sample_categorical_with_topp_gpu,
};

use crate::kv_cache::MlxKVCache;
use crate::model::{
    ModelConfig, TurboQuantModelDecodeContext, forward_all_positions,
    forward_all_positions_update_cache, forward_with_turboquant_context,
};
use crate::weights::ModelWeights;

/// Default number of draft tokens to attempt per n-gram acceleration step.
pub const DEFAULT_DRAFT_LEN: usize = 4;

/// Prompt class route codes. Stored as u32 because the route decision sink is
/// `BTreeMap<String, u32>`. Ordered so `max()` merge promotes the louder signal:
/// `UNSET < NON_REPEATING < REPEATING`.
pub const PROMPT_CLASS_UNSET: u32 = 0;
pub const PROMPT_CLASS_NON_REPEATING: u32 = 1;
pub const PROMPT_CLASS_REPEATING: u32 = 2;

/// Classify a prompt by structural repetition.
///
/// Returns one of `PROMPT_CLASS_NON_REPEATING` or `PROMPT_CLASS_REPEATING`.
/// Heuristic: if at most half of the 4-grams in the prompt are unique, the
/// prompt is classified as repeating. This is the regime where the shipped
/// n-gram speculative path excels; non-repeating prompts are where MTP-style
/// speculation (ADR 0022 D3) would plausibly help.
///
/// Prompts with fewer than 4 tokens are classified as non-repeating (no
/// 4-gram structure to measure).
pub fn classify_prompt_class(tokens: &[u32]) -> u32 {
    if tokens.len() < 4 {
        return PROMPT_CLASS_NON_REPEATING;
    }
    let total = tokens.len() - 3;
    let mut seen: HashSet<(u32, u32, u32, u32)> = HashSet::with_capacity(total);
    for w in tokens.windows(4) {
        seen.insert((w[0], w[1], w[2], w[3]));
    }
    if seen.len().saturating_mul(2) <= total {
        PROMPT_CLASS_REPEATING
    } else {
        PROMPT_CLASS_NON_REPEATING
    }
}

/// Extended draft ceiling for dense models when confidence is high.
///
/// Dense models pay O(1) rollback (just a seq_len pointer move), so longer
/// drafts at high confidence are cheap to attempt.  Linear-attention models
/// keep `DEFAULT_DRAFT_LEN` because partial-reject triggers branch/recompute.
pub const MAX_DRAFT_LEN: usize = 6;

/// Minimum confidence (support/total) for an n-gram to be drafted.
///
/// A prediction with confidence below this threshold stops the draft chain.
/// Calibration: conf=0.4 filters contexts where at most 2 out of 5 observed
/// continuations matched the current best — reliable enough to attempt.
pub const DRAFT_CONFIDENCE_THRESHOLD: f32 = 0.4;

/// Environment variable that overrides `DRAFT_CONFIDENCE_THRESHOLD` at runtime.
/// Lets per-family tuning be observed against `ax.bw_profile.v1` artifacts
/// without recompilation.
pub const DRAFT_CONFIDENCE_THRESHOLD_ENV: &str = "AX_NGRAM_CONFIDENCE_THRESHOLD";

/// Parse a candidate confidence threshold. Returns the default when `raw` is
/// `None`; panics on invalid syntax or out-of-range values. Split out from
/// the env-reading wrapper so the validation logic can be unit-tested without
/// process-global env state.
pub fn parse_confidence_threshold(raw: Option<&str>) -> f32 {
    let Some(value) = raw else {
        return DRAFT_CONFIDENCE_THRESHOLD;
    };
    let parsed: f32 = value.parse().unwrap_or_else(|_| {
        panic!("{DRAFT_CONFIDENCE_THRESHOLD_ENV} must be a float in [0.0, 1.0]; got {value:?}")
    });
    if parsed.is_nan() || !(0.0..=1.0).contains(&parsed) {
        panic!("{DRAFT_CONFIDENCE_THRESHOLD_ENV} must be in [0.0, 1.0]; got {parsed}");
    }
    parsed
}

/// Resolve the effective draft confidence threshold for the current process.
/// Reads `AX_NGRAM_CONFIDENCE_THRESHOLD` once and caches the result. Invalid
/// values fail fast on first call rather than silently clamping.
pub fn effective_draft_confidence_threshold() -> f32 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_confidence_threshold(
            std::env::var(DRAFT_CONFIDENCE_THRESHOLD_ENV)
                .ok()
                .as_deref(),
        )
    })
}

/// Minimum softmax probability for a draft token to be accepted without matching
/// the model's argmax.  When `p_target(draft) >= NGRAM_SPECULATIVE_ACCEPT_THRESHOLD`
/// the draft is accepted even if the argmax predicted a different token.
///
/// Set to 0.0 to disable (pure argmax comparison, original behaviour).
/// Default 0.30: accept if draft token holds at least 30% probability mass.
pub const NGRAM_SPECULATIVE_ACCEPT_THRESHOLD: f32 = 0.30;

/// Environment variable that overrides `NGRAM_SPECULATIVE_ACCEPT_THRESHOLD`.
pub const NGRAM_SPECULATIVE_ACCEPT_THRESHOLD_ENV: &str = "AX_NGRAM_SPECULATIVE_ACCEPT_THRESHOLD";

pub fn parse_speculative_accept_threshold(raw: Option<&str>) -> f32 {
    let Some(value) = raw else {
        return NGRAM_SPECULATIVE_ACCEPT_THRESHOLD;
    };
    let parsed: f32 = value.parse().unwrap_or_else(|_| {
        panic!(
            "{NGRAM_SPECULATIVE_ACCEPT_THRESHOLD_ENV} must be a float in [0.0, 1.0]; got {value:?}"
        )
    });
    if parsed.is_nan() || !(0.0..=1.0).contains(&parsed) {
        panic!("{NGRAM_SPECULATIVE_ACCEPT_THRESHOLD_ENV} must be in [0.0, 1.0]; got {parsed}");
    }
    parsed
}

/// Resolve the effective speculative accept threshold for the current process.
pub fn effective_speculative_accept_threshold() -> f32 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_speculative_accept_threshold(
            std::env::var(NGRAM_SPECULATIVE_ACCEPT_THRESHOLD_ENV)
                .ok()
                .as_deref(),
        )
    })
}

/// Linear-attention draft verification is expensive on partial reject
/// because recurrent state is not O(1)-trimmable. Require repeated n-gram
/// evidence before probing that path.
pub const LINEAR_MIN_NGRAM_SUPPORT: u32 = 2;

/// Maximum number of context keys to retain per n-gram order.
///
/// llama.cpp's draftless n-gram implementations keep memory bounded; AX keeps a
/// richer per-key continuation table, so cap each order and evict the least
/// recently observed context when long generations exceed the budget.
const MAX_CONTEXTS_PER_ORDER: usize = 4096;

/// Number of oldest context keys to evict when one n-gram order crosses cap.
///
/// Pruning a hash map requires scanning context metadata.  Evicting a small
/// batch at once keeps the table under cap while avoiding one full scan per
/// token after long generations reach steady state.
const CONTEXT_PRUNE_BATCH: usize = 64;

/// Maximum continuation candidates tracked for one context key.
///
/// This mirrors the practical shape of llama.cpp's bounded n-gram map variants:
/// keep the most recent few alternatives, not an unbounded histogram.
const MAX_CONTINUATIONS_PER_CONTEXT: usize = 4;

#[derive(Clone)]
struct ContinuationStats {
    token: u32,
    count: u32,
    /// Number of times this continuation was observed in the prompt context
    /// (rather than in generated output). Prompt-derived evidence allows
    /// linear-attention models to draft with `min_support=1` instead of the
    /// output-only threshold, enabling immediate speculation on the first
    /// decode step for repeating real-workload prompts.
    prompt_count: u32,
    last_seen: u64,
    accepted: u32,
    rejected: u32,
}

#[derive(Clone)]
struct NgramPrediction {
    token: u32,
    /// Observations where `token` was the continuation for this context key.
    support: u32,
    /// Prompt-sourced observations for the currently selected continuation.
    /// Prompt-derived bigrams bypass the `LINEAR_MIN_NGRAM_SUPPORT=2` threshold,
    /// allowing linear-attention models to draft immediately on repeating prompts.
    selected_prompt_count: u32,
    /// Total observations for this context key across all continuations.
    total: u32,
    /// Observation index for the selected continuation.  Used to break equal
    /// support ties toward the newest local pattern.
    last_seen: u64,
    /// Observation index for this context key, regardless of which continuation
    /// is currently selected. Used for bounded-table eviction.
    key_last_seen: u64,
    /// Per-continuation counts for this context key.
    ///
    /// The previous implementation let the latest observed continuation replace
    /// the prediction, even when an older continuation was clearly dominant.
    /// Keeping counts makes the draft source stable for repeated prompts with
    /// occasional outliers.
    continuations: Vec<ContinuationStats>,
}

impl NgramPrediction {
    fn new(token: u32, last_seen: u64, from_prompt: bool) -> Self {
        let prompt_count = if from_prompt { 1 } else { 0 };
        let continuations = vec![ContinuationStats {
            token,
            count: 1,
            prompt_count,
            last_seen,
            accepted: 0,
            rejected: 0,
        }];
        Self {
            token,
            support: 1,
            selected_prompt_count: prompt_count,
            total: 1,
            last_seen,
            key_last_seen: last_seen,
            continuations,
        }
    }

    /// Fraction of observations that produced `token`.
    fn confidence(&self) -> f32 {
        self.support as f32 / self.total as f32
    }

    fn record(&mut self, token: u32, last_seen: u64, from_prompt: bool) {
        self.total = self.total.saturating_add(1);
        self.key_last_seen = last_seen;
        if let Some(stats) = self
            .continuations
            .iter_mut()
            .find(|stats| stats.token == token)
        {
            stats.count = stats.count.saturating_add(1);
            if from_prompt {
                stats.prompt_count = stats.prompt_count.saturating_add(1);
            }
            stats.last_seen = last_seen;
        } else {
            self.continuations.push(ContinuationStats {
                token,
                count: 1,
                prompt_count: if from_prompt { 1 } else { 0 },
                last_seen,
                accepted: 0,
                rejected: 0,
            });
        }
        self.prune_oldest_continuations(MAX_CONTINUATIONS_PER_CONTEXT);
        self.recompute_champion();
    }

    fn effective_confidence(&self) -> f32 {
        let lexical_confidence = self.confidence();
        let Some(stats) = self.selected_continuation() else {
            return lexical_confidence;
        };
        let feedback_count = stats.accepted.saturating_add(stats.rejected);
        if feedback_count == 0 {
            return lexical_confidence;
        }
        let acceptance_confidence =
            (stats.accepted.saturating_add(1)) as f32 / (feedback_count.saturating_add(2)) as f32;
        lexical_confidence * acceptance_confidence
    }

    fn record_feedback(&mut self, token: u32, accepted: bool) -> bool {
        let Some(stats) = self
            .continuations
            .iter_mut()
            .find(|stats| stats.token == token)
        else {
            return false;
        };
        if accepted {
            stats.accepted = stats.accepted.saturating_add(1);
        } else {
            stats.rejected = stats.rejected.saturating_add(1);
        }
        true
    }

    fn prune_oldest_continuations(&mut self, max_entries: usize) {
        while self.continuations.len() > max_entries {
            let Some(oldest_index) = self
                .continuations
                .iter()
                .enumerate()
                .min_by_key(|(_, stats)| stats.last_seen)
                .map(|(index, _)| index)
            else {
                break;
            };
            let removed = self.continuations.remove(oldest_index);
            self.total = self.total.saturating_sub(removed.count);
        }
    }

    fn recompute_champion(&mut self) {
        if let Some(stats) = self
            .continuations
            .iter()
            .max_by_key(|stats| (stats.count, stats.last_seen))
        {
            self.token = stats.token;
            self.support = stats.count;
            self.selected_prompt_count = stats.prompt_count;
            self.last_seen = stats.last_seen;
        } else {
            self.support = 0;
            self.selected_prompt_count = 0;
            self.last_seen = self.key_last_seen;
        }
    }

    fn selected_continuation(&self) -> Option<&ContinuationStats> {
        self.continuations
            .iter()
            .find(|stats| stats.token == self.token)
    }

    fn latest_continuation(&self) -> Option<&ContinuationStats> {
        self.continuations
            .iter()
            .max_by_key(|stats| stats.last_seen)
    }
}

#[derive(Clone, Copy)]
enum NgramContextKey {
    Bigram((u32, u32)),
    Trigram((u32, u32, u32)),
    Fourgram((u32, u32, u32, u32)),
}

#[derive(Clone, Copy)]
struct DraftStep {
    token: u32,
    support: u32,
    confidence: f32,
    source: NgramContextKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NgramPolicyVariant {
    MajorityRecency,
    LlamaMapLatest,
    SharedPoolMajority,
}

impl NgramPolicyVariant {
    pub fn route_code(self) -> u32 {
        match self {
            Self::MajorityRecency => 1,
            Self::LlamaMapLatest => 2,
            Self::SharedPoolMajority => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NgramDraftRejection {
    NoCandidate,
    ConfidenceFiltered,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NgramDraftOutcome {
    pub draft: Vec<u32>,
    pub rejection: Option<NgramDraftRejection>,
    pub requested_max_len: usize,
    /// Per-token effective confidence scores from the n-gram table (`effective_confidence()`),
    /// parallel to `draft`. Used by the MTP hybrid path to derive pseudo log-probs for
    /// rejection sampling on n-gram-sourced positions.
    pub confidence: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NgramDraftPolicy {
    pub variant: NgramPolicyVariant,
    pub max_len: usize,
    pub min_support: u32,
    pub confidence_threshold: f32,
    /// When true, cap the total draft length by the first selected match's
    /// occurrence count (`support + 1`, bounded by `max_len`). This follows the
    /// lightning-mlx adaptive-K rule: one-off matches may probe narrowly, while
    /// repeated contexts can justify wider verifier batches.
    pub adaptive_match_len: bool,
    /// When true, bigrams whose `selected_prompt_count >= 1` are allowed to
    /// draft even when their output-derived `count < min_support`. This lowers
    /// the effective minimum to 1 for prompt-seeded continuations, enabling
    /// linear-attention models to draft from the first decode step on repeating
    /// real-workload prompts. Must be `false` for the initial-disable check
    /// (see `linear_ngram_initial_prompt_should_disable_request`) so that
    /// non-repeating random-token prompts remain correctly classified.
    pub bypass_prompt_min_support: bool,
    /// Minimum context length (in tokens) a match may use: `2` allows the full
    /// bigram→trigram→fourgram fallback (default), `3` forbids bigram matches,
    /// `4` requires a full fourgram context. Longer contexts are far less
    /// ambiguous, so raising this raises the accept rate of the drafts that do
    /// fire — at the cost of firing less often. Used by the MTP-stacked path to
    /// keep n-gram accept high next to a strong MTP/assistant drafter.
    pub min_context_len: usize,
}

impl NgramDraftPolicy {
    pub fn majority(max_len: usize, min_support: u32, confidence_threshold: f32) -> Self {
        Self {
            variant: NgramPolicyVariant::MajorityRecency,
            max_len,
            min_support,
            confidence_threshold,
            adaptive_match_len: false,
            bypass_prompt_min_support: false,
            min_context_len: 2,
        }
    }
}

/// Snapshot of the n-gram proposer table.
///
/// This intentionally reports aggregate pressure only.  It is safe to expose in
/// tests or future route metadata without leaking generated token content.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NgramTableStats {
    pub bigram_contexts: usize,
    pub trigram_contexts: usize,
    pub fourgram_contexts: usize,
    pub total_contexts: usize,
    pub total_continuations: usize,
    pub accepted_feedback: u64,
    pub rejected_feedback: u64,
    pub max_contexts_per_order: usize,
    pub max_continuations_per_context: usize,
    pub context_prune_batch: usize,
}

/// N-gram lookup table for self-drafting decoding.
///
/// Tracks 2/3/4-token contexts observed in the prompt and generated tokens.
/// `predict()` chains lookups to produce a draft sequence.
pub struct NgramTable {
    bigrams: HashMap<(u32, u32), NgramPrediction>,
    trigrams: HashMap<(u32, u32, u32), NgramPrediction>,
    fourgrams: HashMap<(u32, u32, u32, u32), NgramPrediction>,
    /// Last 4 observed tokens — context window for next prediction.
    tail: VecDeque<u32>,
    observation_index: u64,
}

impl NgramTable {
    pub fn new() -> Self {
        Self {
            bigrams: HashMap::new(),
            trigrams: HashMap::new(),
            fourgrams: HashMap::new(),
            tail: VecDeque::with_capacity(5),
            observation_index: 0,
        }
    }

    /// Ingest a slice of tokens (call with each batch of accepted output tokens
    /// as generation proceeds).
    pub fn feed(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.observe(t, false);
        }
    }

    /// Ingest prompt tokens, marking each bigram as prompt-sourced.
    ///
    /// Prompt-sourced bigrams satisfy the `LINEAR_MIN_NGRAM_SUPPORT` threshold
    /// with a single observation, enabling linear-attention models to draft
    /// immediately on the first decode step for repeating real-workload prompts
    /// without waiting for two output-derived observations.
    pub fn feed_from_prompt(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.observe(t, true);
        }
    }

    /// Return aggregate table pressure and verifier-feedback counters.
    pub fn stats(&self) -> NgramTableStats {
        let bigram_stats = ngram_prediction_map_stats(&self.bigrams);
        let trigram_stats = ngram_prediction_map_stats(&self.trigrams);
        let fourgram_stats = ngram_prediction_map_stats(&self.fourgrams);

        let bigram_contexts = self.bigrams.len();
        let trigram_contexts = self.trigrams.len();
        let fourgram_contexts = self.fourgrams.len();

        NgramTableStats {
            bigram_contexts,
            trigram_contexts,
            fourgram_contexts,
            total_contexts: bigram_contexts
                .saturating_add(trigram_contexts)
                .saturating_add(fourgram_contexts),
            total_continuations: bigram_stats
                .continuations
                .saturating_add(trigram_stats.continuations)
                .saturating_add(fourgram_stats.continuations),
            accepted_feedback: bigram_stats
                .accepted_feedback
                .saturating_add(trigram_stats.accepted_feedback)
                .saturating_add(fourgram_stats.accepted_feedback),
            rejected_feedback: bigram_stats
                .rejected_feedback
                .saturating_add(trigram_stats.rejected_feedback)
                .saturating_add(fourgram_stats.rejected_feedback),
            max_contexts_per_order: MAX_CONTEXTS_PER_ORDER,
            max_continuations_per_context: MAX_CONTINUATIONS_PER_CONTEXT,
            context_prune_batch: CONTEXT_PRUNE_BATCH,
        }
    }

    /// Record one token and update the n-gram table.
    fn observe(&mut self, t: u32, from_prompt: bool) {
        self.observation_index = self.observation_index.saturating_add(1);
        let observation_index = self.observation_index;
        let n = self.tail.len();
        if n >= 2 {
            let a = self.tail[n - 2];
            let b = self.tail[n - 1];
            update_prediction_and_prune(
                &mut self.bigrams,
                (a, b),
                t,
                observation_index,
                MAX_CONTEXTS_PER_ORDER,
                CONTEXT_PRUNE_BATCH,
                from_prompt,
            );
            if n >= 3 {
                update_prediction_and_prune(
                    &mut self.trigrams,
                    (self.tail[n - 3], a, b),
                    t,
                    observation_index,
                    MAX_CONTEXTS_PER_ORDER,
                    CONTEXT_PRUNE_BATCH,
                    from_prompt,
                );
                if n >= 4 {
                    update_prediction_and_prune(
                        &mut self.fourgrams,
                        (self.tail[n - 4], self.tail[n - 3], a, b),
                        t,
                        observation_index,
                        MAX_CONTEXTS_PER_ORDER,
                        CONTEXT_PRUNE_BATCH,
                        from_prompt,
                    );
                }
            }
        }
        self.tail.push_back(t);
        if self.tail.len() > 4 {
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
    /// Longer contexts are preferred; if a 4-gram or trigram fails either
    /// filter, the predictor falls back to the shorter suffix context.  This
    /// mirrors llama.cpp/vLLM-style longer lookup windows while preserving a
    /// useful fallback when a longer context is sparse or contested.
    ///
    /// `conf_threshold = 0.0` makes this equivalent to `predict_with_min_support`.
    pub fn predict_with_confidence(
        &self,
        max_len: usize,
        min_support: u32,
        conf_threshold: f32,
    ) -> Vec<u32> {
        self.predict_with_policy(NgramDraftPolicy::majority(
            max_len,
            min_support,
            conf_threshold,
        ))
        .draft
    }

    pub fn predict_with_policy(&self, policy: NgramDraftPolicy) -> NgramDraftOutcome {
        let mut draft = Vec::with_capacity(policy.max_len);
        let mut confidence = Vec::with_capacity(policy.max_len);
        // Fixed-size ring; self.tail has at most 4 elements.
        let mut buf = [0u32; 4];
        let mut len = self.tail.len().min(4);
        for (i, &t) in self.tail.iter().take(len).enumerate() {
            buf[i] = t;
        }

        let mut rejection = None;
        let mut allowed_len = policy.max_len;
        while draft.len() < allowed_len {
            match self.select_draft_step(&buf, len, policy) {
                DraftStepSelection::Selected(step) => {
                    if policy.adaptive_match_len {
                        // Tighten the ceiling at every step, not just the first.
                        // A sparse match mid-chain (support=1) stops the chain
                        // early without discarding the already-confident prefix.
                        let step_budget = draft
                            .len()
                            .saturating_add(step.support as usize)
                            .saturating_add(1);
                        allowed_len = allowed_len.min(step_budget);
                        if allowed_len <= draft.len() {
                            rejection = Some(NgramDraftRejection::ConfidenceFiltered);
                            break;
                        }
                    }
                    draft.push(step.token);
                    confidence.push(step.confidence);
                    push_prediction_context_token(&mut buf, &mut len, step.token);
                }
                DraftStepSelection::Rejected(reason) => {
                    rejection = Some(reason);
                    break;
                }
            }
        }
        NgramDraftOutcome {
            draft,
            confidence,
            rejection,
            requested_max_len: policy.max_len,
        }
    }

    /// Record verifier feedback for a draft generated from the current tail.
    ///
    /// Feedback is applied to accepted draft positions and the first rejected
    /// position only. Later draft tokens were verified under an already-wrong
    /// speculative context, so treating them as rejected would over-penalize
    /// unrelated continuations.
    ///
    /// `policy` MUST be the exact policy that produced `draft` (typically via
    /// `predict_with_policy`). Replaying a *different* policy here — e.g. a
    /// reconstructed `NgramDraftPolicy::majority(..)` when the draft was
    /// actually produced with `bypass_prompt_min_support: true`, a non-default
    /// `variant`, or a different `min_context_len` — makes `select_draft_step`
    /// walk a different context/candidate than the one that actually produced
    /// `token`. That silently drops feedback (the `step.token == token` check
    /// fails and `record_feedback_for_context` is never called) or, worse,
    /// attributes feedback to the wrong context key entirely.
    pub(crate) fn record_draft_feedback(
        &mut self,
        draft: &[u32],
        accept_count: usize,
        policy: NgramDraftPolicy,
    ) {
        if draft.is_empty() {
            return;
        }

        let mut buf = [0u32; 4];
        let mut len = self.tail.len().min(4);
        for (i, &t) in self.tail.iter().take(len).enumerate() {
            buf[i] = t;
        }

        let feedback_len = if accept_count < draft.len() {
            accept_count + 1
        } else {
            accept_count
        };

        for (index, &token) in draft.iter().take(feedback_len).enumerate() {
            let accepted = index < accept_count;
            if let DraftStepSelection::Selected(step) = self.select_draft_step(&buf, len, policy)
                && step.token == token
            {
                self.record_feedback_for_context(step.source, token, accepted);
            }
            push_prediction_context_token(&mut buf, &mut len, token);
        }
    }

    fn select_draft_step(
        &self,
        buf: &[u32; 4],
        len: usize,
        policy: NgramDraftPolicy,
    ) -> DraftStepSelection {
        let mut filtered = false;
        let allow_trigram = policy.min_context_len <= 3;
        let allow_bigram = policy.min_context_len <= 2;
        if len >= 4 {
            let key = (buf[0], buf[1], buf[2], buf[3]);
            if let Some(prediction) = self.fourgrams.get(&key) {
                match draft_step_from_prediction(prediction, policy) {
                    Some(candidate) => {
                        return DraftStepSelection::Selected(DraftStep {
                            token: candidate.token,
                            support: candidate.support,
                            confidence: candidate.confidence,
                            source: NgramContextKey::Fourgram(key),
                        });
                    }
                    None => filtered = true,
                }
            }
            let key = (buf[1], buf[2], buf[3]);
            if let Some(prediction) = allow_trigram.then(|| self.trigrams.get(&key)).flatten() {
                match draft_step_from_prediction(prediction, policy) {
                    Some(candidate) => {
                        return DraftStepSelection::Selected(DraftStep {
                            token: candidate.token,
                            support: candidate.support,
                            confidence: candidate.confidence,
                            source: NgramContextKey::Trigram(key),
                        });
                    }
                    None => filtered = true,
                }
            }
            let key = (buf[2], buf[3]);
            if let Some(prediction) = allow_bigram.then(|| self.bigrams.get(&key)).flatten() {
                match draft_step_from_prediction(prediction, policy) {
                    Some(candidate) => {
                        return DraftStepSelection::Selected(DraftStep {
                            token: candidate.token,
                            support: candidate.support,
                            confidence: candidate.confidence,
                            source: NgramContextKey::Bigram(key),
                        });
                    }
                    None => filtered = true,
                }
            }
        } else if len == 3 {
            let key = (buf[0], buf[1], buf[2]);
            if let Some(prediction) = allow_trigram.then(|| self.trigrams.get(&key)).flatten() {
                match draft_step_from_prediction(prediction, policy) {
                    Some(candidate) => {
                        return DraftStepSelection::Selected(DraftStep {
                            token: candidate.token,
                            support: candidate.support,
                            confidence: candidate.confidence,
                            source: NgramContextKey::Trigram(key),
                        });
                    }
                    None => filtered = true,
                }
            }
            let key = (buf[1], buf[2]);
            if let Some(prediction) = allow_bigram.then(|| self.bigrams.get(&key)).flatten() {
                match draft_step_from_prediction(prediction, policy) {
                    Some(candidate) => {
                        return DraftStepSelection::Selected(DraftStep {
                            token: candidate.token,
                            support: candidate.support,
                            confidence: candidate.confidence,
                            source: NgramContextKey::Bigram(key),
                        });
                    }
                    None => filtered = true,
                }
            }
        } else if len == 2 {
            let key = (buf[0], buf[1]);
            if let Some(prediction) = allow_bigram.then(|| self.bigrams.get(&key)).flatten() {
                match draft_step_from_prediction(prediction, policy) {
                    Some(candidate) => {
                        return DraftStepSelection::Selected(DraftStep {
                            token: candidate.token,
                            support: candidate.support,
                            confidence: candidate.confidence,
                            source: NgramContextKey::Bigram(key),
                        });
                    }
                    None => filtered = true,
                }
            }
        }
        DraftStepSelection::Rejected(if filtered {
            NgramDraftRejection::ConfidenceFiltered
        } else {
            NgramDraftRejection::NoCandidate
        })
    }

    fn record_feedback_for_context(
        &mut self,
        source: NgramContextKey,
        token: u32,
        accepted: bool,
    ) -> bool {
        match source {
            NgramContextKey::Bigram(key) => self
                .bigrams
                .get_mut(&key)
                .is_some_and(|prediction| prediction.record_feedback(token, accepted)),
            NgramContextKey::Trigram(key) => self
                .trigrams
                .get_mut(&key)
                .is_some_and(|prediction| prediction.record_feedback(token, accepted)),
            NgramContextKey::Fourgram(key) => self
                .fourgrams
                .get_mut(&key)
                .is_some_and(|prediction| prediction.record_feedback(token, accepted)),
        }
    }
}

enum DraftStepSelection {
    Selected(DraftStep),
    Rejected(NgramDraftRejection),
}

fn draft_step_from_prediction(
    prediction: &NgramPrediction,
    policy: NgramDraftPolicy,
) -> Option<DraftCandidate> {
    // Prompt-derived bigrams bypass min_support when `bypass_prompt_min_support`
    // is set. This is enabled during actual decode but NOT during the initial-
    // disable check (`linear_ngram_initial_prompt_should_disable_request`) so
    // that non-repeating random-token prompts are still classified correctly and
    // do not trigger unwanted speculation on their random prompt bigrams.
    let effective_min_support =
        if policy.bypass_prompt_min_support && prediction.selected_prompt_count >= 1 {
            1
        } else {
            policy.min_support
        };
    match policy.variant {
        NgramPolicyVariant::MajorityRecency | NgramPolicyVariant::SharedPoolMajority => {
            let conf = prediction.effective_confidence();
            prediction_passes(
                prediction.support,
                conf,
                effective_min_support,
                policy.confidence_threshold,
            )
            .then_some(DraftCandidate {
                token: prediction.token,
                support: prediction.support,
                confidence: conf,
            })
        }
        NgramPolicyVariant::LlamaMapLatest => {
            let latest = prediction.latest_continuation()?;
            let confidence = latest.count as f32 / prediction.total as f32;
            prediction_passes(
                latest.count,
                confidence,
                effective_min_support,
                policy.confidence_threshold,
            )
            .then_some(DraftCandidate {
                token: latest.token,
                support: latest.count,
                confidence,
            })
        }
    }
}

#[derive(Clone, Copy)]
struct DraftCandidate {
    token: u32,
    support: u32,
    confidence: f32,
}

fn prediction_passes(support: u32, confidence: f32, min_support: u32, conf_threshold: f32) -> bool {
    support >= min_support && confidence >= conf_threshold
}

fn push_prediction_context_token(buf: &mut [u32; 4], len: &mut usize, token: u32) {
    if *len < 4 {
        buf[*len] = token;
        *len += 1;
    } else {
        buf[0] = buf[1];
        buf[1] = buf[2];
        buf[2] = buf[3];
        buf[3] = token;
    }
}

#[derive(Default)]
struct NgramPredictionMapStats {
    continuations: usize,
    accepted_feedback: u64,
    rejected_feedback: u64,
}

fn ngram_prediction_map_stats<K>(map: &HashMap<K, NgramPrediction>) -> NgramPredictionMapStats {
    let mut stats = NgramPredictionMapStats::default();
    for prediction in map.values() {
        stats.continuations = stats
            .continuations
            .saturating_add(prediction.continuations.len());
        for continuation in &prediction.continuations {
            stats.accepted_feedback = stats
                .accepted_feedback
                .saturating_add(continuation.accepted as u64);
            stats.rejected_feedback = stats
                .rejected_feedback
                .saturating_add(continuation.rejected as u64);
        }
    }
    stats
}

fn update_prediction_and_prune<K>(
    map: &mut HashMap<K, NgramPrediction>,
    key: K,
    token: u32,
    last_seen: u64,
    max_entries: usize,
    prune_batch: usize,
    from_prompt: bool,
) where
    K: Clone + Eq + std::hash::Hash,
{
    match map.get_mut(&key) {
        Some(p) => {
            p.record(token, last_seen, from_prompt);
        }
        None => {
            map.insert(key, NgramPrediction::new(token, last_seen, from_prompt));
            prune_oldest_predictions(map, max_entries, prune_batch);
        }
    }
}

fn prune_oldest_predictions<K>(
    map: &mut HashMap<K, NgramPrediction>,
    max_entries: usize,
    prune_batch: usize,
) where
    K: Clone + Eq + std::hash::Hash,
{
    if map.len() <= max_entries {
        return;
    }

    let overflow = map.len().saturating_sub(max_entries);
    let remove_count = prune_batch.max(overflow).min(map.len());
    let mut oldest: Vec<(K, u64)> = Vec::with_capacity(remove_count);
    for (key, prediction) in map.iter() {
        let last_seen = prediction.key_last_seen;
        if oldest.len() < remove_count {
            oldest.push((key.clone(), last_seen));
            continue;
        }

        let Some((newest_old_index, newest_old_last_seen)) = oldest
            .iter()
            .enumerate()
            .max_by_key(|(_, (_, last_seen))| *last_seen)
            .map(|(index, (_, last_seen))| (index, *last_seen))
        else {
            continue;
        };
        if last_seen < newest_old_last_seen {
            oldest[newest_old_index] = (key.clone(), last_seen);
        }
    }

    for (key, _) in oldest {
        map.remove(&key);
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
    draft_policy: NgramDraftPolicy,
    sampling: MlxSamplingParams,
    repetition_tokens: &[u32],
    rng: &mut Xorshift64,
) -> Vec<u32> {
    let mut sampling_probs_buf = Vec::new();
    let mut sampling_logits_buf = Vec::new();
    let mut sampling_candidates_buf = Vec::new();
    ngram_accel_decode_step_with_sampling_buffers(
        cfg,
        weights,
        cache,
        ngram,
        last_token,
        draft,
        draft_policy,
        sampling,
        repetition_tokens,
        rng,
        &mut sampling_probs_buf,
        &mut sampling_logits_buf,
        &mut sampling_candidates_buf,
    )
}

/// `draft_policy` MUST be the exact policy that produced `draft` (see
/// `NgramTable::record_draft_feedback`); it is replayed verbatim to attribute
/// verifier feedback to the same context/candidate that was actually drafted.
#[allow(clippy::too_many_arguments)]
pub fn ngram_accel_decode_step_with_sampling_buffers(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    ngram: &mut NgramTable,
    last_token: u32,
    draft: &[u32],
    draft_policy: NgramDraftPolicy,
    sampling: MlxSamplingParams,
    repetition_tokens: &[u32],
    rng: &mut Xorshift64,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
) -> Vec<u32> {
    if draft.is_empty() || sampling.uses_repetition_penalty() {
        return single_decode_with_turboquant_context(
            cfg,
            weights,
            cache,
            ngram,
            last_token,
            sampling,
            repetition_tokens,
            rng,
            None,
            sampling_probs_buf,
            sampling_logits_buf,
            sampling_candidates_buf,
        );
    }

    if cfg.linear_attention.is_some() {
        return ngram_accel_decode_step_linear_safe(
            cfg,
            weights,
            cache,
            ngram,
            last_token,
            draft,
            draft_policy,
            sampling,
            rng,
            sampling_probs_buf,
            sampling_logits_buf,
            sampling_candidates_buf,
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
        sampling,
        effective_speculative_accept_threshold(),
        rng,
        sampling_probs_buf,
        sampling_logits_buf,
        sampling_candidates_buf,
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
    ngram.record_draft_feedback(draft, verification.accept_count, draft_policy);
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
    draft_policy: NgramDraftPolicy,
    sampling: MlxSamplingParams,
    rng: &mut Xorshift64,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
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
        sampling,
        effective_speculative_accept_threshold(),
        rng,
        sampling_probs_buf,
        sampling_logits_buf,
        sampling_candidates_buf,
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

    ngram.record_draft_feedback(draft, verification.accept_count, draft_policy);
    ngram.feed(&draft[..verification.accept_count]);
    ngram.feed(&verification.result[verification.accept_count..]); // correction or bonus

    verification.result
}

pub(crate) fn ngram_feedback_policy(cfg: &ModelConfig) -> (u32, f32) {
    if cfg.linear_attention.is_some() {
        (LINEAR_MIN_NGRAM_SUPPORT, DRAFT_CONFIDENCE_THRESHOLD)
    } else {
        (1, DRAFT_CONFIDENCE_THRESHOLD)
    }
}

/// For each draft position i, gather `softmax(logits_all[i] / T)[draft[i]]` —
/// the model's probability for that specific draft token.
///
/// Returns a lazy `[n]` f32 array; caller must include it in the combined `eval()`
/// before calling `.data_f32()`.  Only called when `temperature > 0`.
///
/// Index mapping: `verify_input = [last_token, D0, D1, …, D_{n-1}]`.
/// `logits_all[i]` is the prediction *after* `verify_input[i]`, so
/// `logits_all[i]` predicts `D_i = draft[i]` — no off-by-one.
fn gather_draft_token_probs_lazy(
    logits_all: &MlxArray,
    draft: &[u32],
    vocab: i32,
    temperature: f32,
) -> MlxArray {
    let n = draft.len();
    let inv_temp_val = 1.0_f32 / temperature;
    let inv_temp = MlxArray::from_raw_data(
        &inv_temp_val as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[],
        MlxDtype::Float32,
    );
    let scaled = multiply(logits_all, &inv_temp, None);
    let probs = softmax(&scaled, -1, None); // [verify_len, vocab] lazy
    let flat_indices: Vec<i32> = (0..n).map(|i| i as i32 * vocab + draft[i] as i32).collect();
    let flat_idx_arr = MlxArray::from_raw_data(
        flat_indices.as_ptr() as *const u8,
        flat_indices.len() * 4,
        &[n as i32],
        MlxDtype::Int32,
    );
    let probs_flat = reshape(&probs, &[-1_i32], None);
    take(&probs_flat, &flat_idx_arr, 0, None) // [n] lazy
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
    sampling: MlxSamplingParams,
    accept_threshold: f32,
    rng: &mut Xorshift64,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
) -> DraftVerification {
    // Verification sequence: [last_token, D0, D1, ... D_{n-1}].
    let mut verify_input = Vec::with_capacity(1 + draft.len());
    verify_input.push(last_token);
    verify_input.extend_from_slice(draft);
    let verify_len = verify_input.len();

    // One causal forward pass -> [verify_len, vocab_size] f32 logits.
    let logits_all = forward_all_positions(cfg, weights, &verify_input, cache, token_offset);
    cache.seq_len += verify_len;

    // Build argmax and, when speculative threshold is active, the draft-token
    // probability vector — both lazily, evaluated in a single blocking call.
    let predicted_arr = argmax(&logits_all, None);
    let vocab = cfg.vocab_size as i32;

    // Compute per-draft-position softmax probabilities when the threshold is
    // active and temperature > 0.  Bundled into the same eval() as argmax and
    // KV refs to avoid a second GPU sync.
    let use_speculative = accept_threshold > 0.0 && sampling.temperature > 0.0 && !draft.is_empty();
    let draft_probs_arr_opt: Option<MlxArray> = if use_speculative {
        Some(gather_draft_token_probs_lazy(
            &logits_all,
            draft,
            vocab,
            sampling.temperature,
        ))
    } else {
        None
    };

    let kv_refs = cache.collect_eval_refs();
    let mut targets: Vec<&MlxArray> = Vec::with_capacity(2 + kv_refs.len());
    targets.push(&predicted_arr);
    if let Some(ref arr) = draft_probs_arr_opt {
        targets.push(arr);
    }
    targets.extend(kv_refs);
    eval(&targets);

    // Both logits_all (transitive dep of predicted_arr) and all targets are now
    // materialised.  KV backing buffers are flat — no separate materialize_cache
    // call needed in the caller.
    let predicted: Vec<u32> = predicted_arr.data_u32().to_vec();
    let draft_probs: Vec<f32> = draft_probs_arr_opt
        .as_ref()
        .map(|arr| arr.data_f32().to_vec())
        .unwrap_or_default();

    // Accept/reject.
    // predicted[i] = argmax prediction for token AFTER verify_input[i] = draft[i].
    // Accept when:
    //   (a) argmax agrees with the draft, OR
    //   (b) speculative threshold: p_target(draft[i]) >= accept_threshold.
    // Case (b) accepts slightly-off-argmax drafts when the draft token still holds
    // high probability mass, mirroring the --mtp-optimistic strategy in lightning.
    let mut result: Vec<u32> = Vec::new();
    let mut accept_count = 0usize;

    for i in 0..draft.len() {
        let argmax_match = predicted[i] == draft[i];
        let prob_accept = !draft_probs.is_empty() && draft_probs[i] >= accept_threshold;

        if argmax_match || prob_accept {
            result.push(draft[i]);
            accept_count += 1;
        } else {
            // Correction token: sample at position i.  Slice one logit row to
            // avoid copying the full (1+draft_len)×vocab tensor to CPU.
            let tok = sample_logit_row(
                &logits_all,
                predicted[i],
                i,
                vocab,
                sampling,
                rng,
                sampling_probs_buf,
                sampling_logits_buf,
                sampling_candidates_buf,
            );
            result.push(tok);
            break;
        }
    }

    // Bonus: if ALL draft tokens were accepted, sample the next token for free.
    if accept_count == draft.len() {
        let pos = draft.len();
        let tok = sample_logit_row(
            &logits_all,
            predicted[pos],
            pos,
            vocab,
            sampling,
            rng,
            sampling_probs_buf,
            sampling_logits_buf,
            sampling_candidates_buf,
        );
        result.push(tok);
    }

    DraftVerification {
        accept_count,
        committed_len: token_offset + 1 + accept_count,
        result,
    }
}

pub(crate) fn recompute_committed_prefix(
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

    forward_all_positions_update_cache(cfg, weights, &commit_input, cache, token_offset);
    cache.seq_len += commit_input.len();
    let kv_refs = cache.collect_eval_refs();
    eval(&kv_refs);
}

/// Sample token at `pos` in the flattened `[verify_len, vocab]` logit buffer.
/// Falls back to `argmax_tok` when temperature is 0 or the buffer is empty.
/// Sample one token from `logits_all` at sequence position `pos`.
///
/// `logits_all` has shape `[verify_len, vocab]` and is already materialised.
/// Slices one `[1, vocab]` row — avoids copying the full multi-position
/// tensor to CPU when temperature > 0.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_logit_row(
    logits_all: &MlxArray,
    argmax_tok: u32,
    pos: usize,
    vocab: i32,
    sampling: MlxSamplingParams,
    rng: &mut Xorshift64,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
) -> u32 {
    if sampling.temperature <= 0.0 {
        return argmax_tok;
    }
    if logits_all.shape().len() == 1 {
        if let Some(tok) = sample_categorical_with_topk_gpu(logits_all, sampling, &[], rng)
            .or_else(|| sample_categorical_with_topp_gpu(logits_all, sampling, &[], rng))
        {
            return tok;
        }
        eval(&[logits_all]);
        return sample_categorical_into(
            logits_all.data_f32(),
            sampling,
            &[],
            rng,
            sampling_probs_buf,
            sampling_logits_buf,
            sampling_candidates_buf,
        );
    }
    let p = pos as i32;
    let row = slice(logits_all, &[p, 0], &[p + 1, vocab], &[1, 1], None);
    if let Some(tok) = sample_categorical_with_topk_gpu(&row, sampling, &[], rng)
        .or_else(|| sample_categorical_with_topp_gpu(&row, sampling, &[], rng))
    {
        return tok;
    }
    eval(&[&row]);
    sample_categorical_into(
        row.data_f32(),
        sampling,
        &[],
        rng,
        sampling_probs_buf,
        sampling_logits_buf,
        sampling_candidates_buf,
    )
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
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
) -> Vec<u32> {
    let mut probs_buf = Vec::new();
    let mut logits_buf = Vec::new();
    let mut candidates_buf = Vec::new();
    single_decode_with_turboquant_context(
        cfg,
        weights,
        cache,
        ngram,
        last_token,
        sampling_request.params,
        sampling_request.repetition_tokens,
        rng,
        None,
        &mut probs_buf,
        &mut logits_buf,
        &mut candidates_buf,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn single_decode_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    ngram: &mut NgramTable,
    last_token: u32,
    sampling: MlxSamplingParams,
    repetition_tokens: &[u32],
    rng: &mut Xorshift64,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
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

    let tok = if sampling.temperature > 0.0
        && !sampling.uses_repetition_penalty()
        && sampling.top_k == 0
        && sampling.top_p >= 1.0
    {
        // GPU-side sampling: no logits transfer to CPU.
        // The forward pass already updated the KV cache (it's in logits' graph);
        // sample_categorical_gpu evals the token internally.
        sample_categorical_gpu(&logits, sampling.temperature)
    } else if let Some(tok) =
        sample_categorical_with_topk_gpu(&logits, sampling, repetition_tokens, rng)
            .or_else(|| sample_categorical_with_topp_gpu(&logits, sampling, repetition_tokens, rng))
    {
        // GPU-selected candidate set: transfers a few hundred (index, prob)
        // pairs instead of the full vocab (1 MB of f32 at 262k) per token.
        tok
    } else if sampling.temperature > 0.0 || sampling.uses_repetition_penalty() {
        let kv_refs = cache.collect_eval_refs();
        let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
        targets.push(&logits);
        targets.extend(kv_refs);
        eval(&targets);
        sample_categorical_into(
            logits.data_f32(),
            sampling,
            repetition_tokens,
            rng,
            sampling_probs_buf,
            sampling_logits_buf,
            sampling_candidates_buf,
        )
    } else {
        let token_arr = argmax(&logits, None);
        let kv_refs = cache.collect_eval_refs();
        let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
        targets.push(&token_arr);
        targets.extend(kv_refs);
        eval(&targets);
        token_arr.first_u32_unchecked()
    };

    ngram.observe(tok, false);
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

    fn continuations_contain(prediction: &NgramPrediction, token: u32) -> bool {
        prediction
            .continuations
            .iter()
            .any(|stats| stats.token == token)
    }

    #[test]
    fn parse_confidence_threshold_default_when_unset() {
        assert_eq!(parse_confidence_threshold(None), DRAFT_CONFIDENCE_THRESHOLD);
    }

    #[test]
    fn parse_confidence_threshold_accepts_valid_values() {
        assert_eq!(parse_confidence_threshold(Some("0.0")), 0.0);
        assert_eq!(parse_confidence_threshold(Some("0.5")), 0.5);
        assert_eq!(parse_confidence_threshold(Some("1.0")), 1.0);
    }

    #[test]
    #[should_panic(expected = "AX_NGRAM_CONFIDENCE_THRESHOLD")]
    fn parse_confidence_threshold_rejects_above_one() {
        let _ = parse_confidence_threshold(Some("1.5"));
    }

    #[test]
    #[should_panic(expected = "AX_NGRAM_CONFIDENCE_THRESHOLD")]
    fn parse_confidence_threshold_rejects_negative() {
        let _ = parse_confidence_threshold(Some("-0.1"));
    }

    #[test]
    #[should_panic(expected = "AX_NGRAM_CONFIDENCE_THRESHOLD")]
    fn parse_confidence_threshold_rejects_nonsense() {
        let _ = parse_confidence_threshold(Some("not-a-number"));
    }

    #[test]
    #[should_panic(expected = "AX_NGRAM_CONFIDENCE_THRESHOLD")]
    fn parse_confidence_threshold_rejects_nan() {
        let _ = parse_confidence_threshold(Some("NaN"));
    }

    #[test]
    fn classify_prompt_class_short_prompt_is_non_repeating() {
        assert_eq!(classify_prompt_class(&[]), PROMPT_CLASS_NON_REPEATING);
        assert_eq!(
            classify_prompt_class(&[1, 2, 3]),
            PROMPT_CLASS_NON_REPEATING
        );
    }

    #[test]
    fn classify_prompt_class_linear_sequence_is_non_repeating() {
        let tokens: Vec<u32> = (0..32).collect();
        assert_eq!(
            classify_prompt_class(&tokens),
            PROMPT_CLASS_NON_REPEATING,
            "all-unique 4-grams must classify as non-repeating"
        );
    }

    #[test]
    fn classify_prompt_class_short_cycle_is_repeating() {
        // Cycle of period 3 over 12 tokens: 4-grams overlap heavily.
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
        assert_eq!(
            classify_prompt_class(&tokens),
            PROMPT_CLASS_REPEATING,
            "short-cycle repetition must classify as repeating"
        );
    }

    #[test]
    fn classify_prompt_class_single_repeat_block_is_non_repeating() {
        // Two copies of a 4-token block — only one repeated 4-gram, ratio > 0.5.
        let tokens = vec![1, 2, 3, 4, 1, 2, 3, 4];
        assert_eq!(
            classify_prompt_class(&tokens),
            PROMPT_CLASS_NON_REPEATING,
            "a single 4-gram repeat is below the repeating threshold"
        );
    }

    #[test]
    fn classify_prompt_class_boundary_at_half_ratio_is_repeating() {
        // Cycle [1,2,3,4] over 8 tokens: 4-grams are (1234),(2341),(3412),(4123),(1234)
        // → 4 unique out of 5 total → 4/5 = 0.8 → non-repeating.
        let tokens = vec![1, 2, 3, 4, 1, 2, 3, 4];
        assert_eq!(classify_prompt_class(&tokens), PROMPT_CLASS_NON_REPEATING);

        // Extend to 16 tokens of the same cycle: 13 total 4-grams, 4 unique → 4/13 ≈ 0.31 → repeating.
        let tokens: Vec<u32> = std::iter::repeat_n([1u32, 2, 3, 4], 4).flatten().collect();
        assert_eq!(classify_prompt_class(&tokens), PROMPT_CLASS_REPEATING);
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
        // tail=[3,1,2,3]; longer contexts and trigrams reconstruct the cycle.
        let draft = t.predict(4);
        assert_eq!(draft, vec![1, 2, 3, 1]);
    }

    #[test]
    fn predict_prefers_fourgram_over_trigram_and_bigram() {
        let mut t = NgramTable::new();
        // Make trigram(1,2,3) prefer 10, then add one more specific
        // fourgram(0,1,2,3) continuation to 99.
        t.feed(&[1, 2, 3, 10, 8, 1, 2, 3, 10]);
        t.feed(&[0, 1, 2, 3, 99]);
        t.feed(&[0, 1, 2, 3]);

        assert_eq!(
            t.predict(1),
            vec![99],
            "the most specific matching context should be used first"
        );
        assert_eq!(
            t.predict_with_min_support(1, 2),
            vec![10],
            "if the fourgram is too sparse, prediction should fall back to the supported trigram"
        );
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
    fn prediction_keeps_majority_continuation_after_late_outlier() {
        let mut t = NgramTable::new();
        t.feed(&[1, 2, 3, 1, 2, 3, 1, 2, 9]);
        t.feed(&[5, 1, 2]);

        assert_eq!(
            t.predict(1),
            vec![3],
            "late one-off continuation must not replace the majority continuation"
        );
        assert_eq!(
            t.predict_with_confidence(1, 2, 0.4),
            vec![3],
            "majority support should satisfy the linear-attention draft filters"
        );
    }

    #[test]
    fn prediction_breaks_equal_support_ties_by_recency() {
        let mut t = NgramTable::new();
        t.feed(&[1, 2, 10, 1, 2, 20]);
        t.feed(&[5, 1, 2]);

        assert_eq!(
            t.predict(1),
            vec![20],
            "equal-support continuations should prefer the newest local pattern"
        );
        assert!(
            t.predict_with_confidence(1, 1, 0.75).is_empty(),
            "recency tie-break must not bypass confidence filtering"
        );
    }

    #[test]
    fn llama_map_latest_policy_is_available_behind_flag_shape() {
        let mut t = NgramTable::new();
        t.feed(&[1, 2, 3, 1, 2, 3, 1, 2, 9]);
        t.feed(&[5, 1, 2]);

        let default = t.predict_with_policy(NgramDraftPolicy {
            variant: NgramPolicyVariant::MajorityRecency,
            max_len: 1,
            min_support: 1,
            confidence_threshold: 0.0,
            adaptive_match_len: false,
            bypass_prompt_min_support: false,
            min_context_len: 2,
        });
        let latest = t.predict_with_policy(NgramDraftPolicy {
            variant: NgramPolicyVariant::LlamaMapLatest,
            max_len: 1,
            min_support: 1,
            confidence_threshold: 0.0,
            adaptive_match_len: false,
            bypass_prompt_min_support: false,
            min_context_len: 2,
        });

        assert_eq!(default.draft, vec![3]);
        assert_eq!(
            latest.draft,
            vec![9],
            "llama-map/latest policy should model overwrite-style continuation lookup"
        );
    }

    #[test]
    fn adaptive_match_len_caps_sparse_drafts_per_step() {
        let policy = NgramDraftPolicy {
            variant: NgramPolicyVariant::MajorityRecency,
            max_len: 6,
            min_support: 1,
            confidence_threshold: 0.0,
            adaptive_match_len: true,
            bypass_prompt_min_support: false,
            min_context_len: 2,
        };

        let mut one_off = table_from_sequence(&[1, 2, 3, 4]);
        one_off.tail = [1, 2].into_iter().collect();
        assert_eq!(
            one_off.predict_with_policy(policy).draft,
            vec![3, 4],
            "support=1 at step 0 caps total to 2; support=1 at step 1 keeps ceiling at 2"
        );

        let mut repeated = table_from_sequence(&[1, 2, 3, 4, 9, 1, 2, 3, 4]);
        repeated.tail = [1, 2].into_iter().collect();
        assert_eq!(
            repeated.predict_with_policy(policy).draft,
            vec![3, 4, 9],
            "support=2 at step 0 allows 3; sparse tail step does not extend ceiling further"
        );
    }

    #[test]
    fn adaptive_match_len_sparse_mid_chain_step_truncates_early() {
        // Chain: (1,2)→3 (support=2, high-confidence), (2,3)→X (support=1, sparse mid).
        // Without per-step cap the chain continues past the sparse step indefinitely.
        // With per-step cap the ceiling is tightened at the sparse step to draft.len+2,
        // so the draft stops at length 3.
        let policy = NgramDraftPolicy {
            variant: NgramPolicyVariant::MajorityRecency,
            max_len: 6,
            min_support: 1,
            confidence_threshold: 0.0,
            adaptive_match_len: true,
            bypass_prompt_min_support: false,
            min_context_len: 2,
        };
        let mut t = NgramTable::new();
        t.feed(&[1, 2, 3, 4, 9, 1, 2, 3, 7, 8]);
        t.tail = [1, 2].into_iter().collect();
        let draft = t.predict_with_policy(policy);
        // Step 0: (1,2)→3, support=2, step_budget=3, allowed_len=3.
        // Step 1: (2,3)→X, support=1, step_budget=3, allowed_len stays 3.
        // Step 2: draft.len=2 < 3, one more token allowed. draft.len hits 3, loop exits.
        assert!(
            draft.draft.len() <= 3,
            "sparse mid-chain step must cap draft at 3, got {:?}",
            draft.draft
        );
    }

    #[test]
    fn draft_outcome_labels_no_candidate_and_confidence_filtering() {
        let empty = NgramTable::new();
        let no_candidate = empty.predict_with_policy(NgramDraftPolicy::majority(1, 1, 0.4));
        assert_eq!(no_candidate.draft, Vec::<u32>::new());
        assert_eq!(
            no_candidate.rejection,
            Some(NgramDraftRejection::NoCandidate)
        );

        let mut contested = NgramTable::new();
        contested.feed(&[3, 1, 2, 3, 1, 4, 3, 1, 5]);
        contested.feed(&[9, 3, 1]);
        let filtered = contested.predict_with_policy(NgramDraftPolicy::majority(1, 1, 0.4));
        assert_eq!(filtered.draft, Vec::<u32>::new());
        assert_eq!(
            filtered.rejection,
            Some(NgramDraftRejection::ConfidenceFiltered)
        );
    }

    #[test]
    fn prompt_sourced_bigrams_bypass_linear_min_support_threshold() {
        // Simulates what happens when seed_generation_ngram_from_prompt seeds
        // the table and then the first few output tokens position the tail so
        // that a prompt bigram is the relevant lookup key.
        //
        // Prompt [1,2,3] seeds bigram (1,2)→3 with prompt_count=1.
        // Then output [9,1,2] runs through the table (simulating first decode
        // steps), placing the tail at [3,9,1,2].
        // Prediction now looks for bigram (1,2)→3 — found with prompt_count=1.
        let mut t = NgramTable::new();
        // Seed from prompt: creates bigram (1,2)→3 with prompt_count=1.
        t.feed_from_prompt(&[1, 2, 3]);
        // Simulate a few output tokens to reposition the tail.
        t.feed(&[9, 1, 2]);
        // With min_support=2 and bypass_prompt_min_support=true (the policy
        // used during actual linear-attention decode), bigram (1,2)→3 has
        // prompt_count=1 → min_support is lowered to 1 → draft is produced.
        let bypass_policy = NgramDraftPolicy {
            variant: NgramPolicyVariant::MajorityRecency,
            max_len: 1,
            min_support: 2,
            confidence_threshold: 0.0,
            adaptive_match_len: false,
            bypass_prompt_min_support: true,
            min_context_len: 2,
        };
        let draft = t.predict_with_policy(bypass_policy);
        assert_eq!(
            draft.draft,
            vec![3],
            "prompt-sourced bigram should draft despite min_support=2 when bypass enabled"
        );

        // Without bypass, the same policy blocks the draft (count=1 < min_support=2).
        let strict_policy = NgramDraftPolicy {
            bypass_prompt_min_support: false,
            ..bypass_policy
        };
        let blocked_bypass = t.predict_with_policy(strict_policy);
        assert_eq!(
            blocked_bypass.draft,
            Vec::<u32>::new(),
            "prompt-sourced bigram must be blocked when bypass is disabled"
        );

        // Same sequence without prompt source: bigram (1,2)→3 has count=1
        // but prompt_count=0 → min_support=2 is NOT bypassed even with bypass=true.
        let mut t2 = NgramTable::new();
        t2.feed(&[1, 2, 3]);
        t2.feed(&[9, 1, 2]);
        let blocked = t2.predict_with_policy(bypass_policy);
        assert_eq!(
            blocked.draft,
            Vec::<u32>::new(),
            "output-only bigram with count=1 should not draft at min_support=2"
        );
    }

    #[test]
    fn min_context_len_forbids_bigram_only_match() {
        // Same setup as the bypass test: prompt [1,2,3] seeds bigram (1,2)→3 and
        // output [9,1,2] positions the tail so only a 2-token (bigram) context
        // matches. min_context_len=2 (default) drafts it; min_context_len=3
        // forbids the ambiguous bigram fallback, so no draft is produced — this
        // is the lever the MTP-stacked path uses to lift n-gram accept.
        let mut t = NgramTable::new();
        t.feed_from_prompt(&[1, 2, 3]);
        t.feed(&[9, 1, 2]);
        let allow_bigram = NgramDraftPolicy {
            variant: NgramPolicyVariant::MajorityRecency,
            max_len: 1,
            min_support: 2,
            confidence_threshold: 0.0,
            adaptive_match_len: false,
            bypass_prompt_min_support: true,
            min_context_len: 2,
        };
        assert_eq!(
            t.predict_with_policy(allow_bigram).draft,
            vec![3],
            "bigram match drafts when min_context_len allows a 2-token context"
        );
        let forbid_bigram = NgramDraftPolicy {
            min_context_len: 3,
            ..allow_bigram
        };
        assert_eq!(
            t.predict_with_policy(forbid_bigram).draft,
            Vec::<u32>::new(),
            "min_context_len=3 must forbid the ambiguous bigram-only match"
        );
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
    fn table_evicts_oldest_contexts_when_cap_is_exceeded() {
        let token_count = MAX_CONTEXTS_PER_ORDER + 20;
        let tokens: Vec<u32> = (0..token_count as u32).collect();
        let t = table_from_sequence(&tokens);

        assert!(t.bigrams.len() <= MAX_CONTEXTS_PER_ORDER);
        assert!(t.trigrams.len() <= MAX_CONTEXTS_PER_ORDER);
        assert!(t.fourgrams.len() <= MAX_CONTEXTS_PER_ORDER);
        assert!(!t.bigrams.contains_key(&(0, 1)));
        assert!(!t.trigrams.contains_key(&(0, 1, 2)));
        assert!(!t.fourgrams.contains_key(&(0, 1, 2, 3)));

        let last_recorded = token_count as u32 - 1;
        assert!(
            t.bigrams
                .contains_key(&(last_recorded - 2, last_recorded - 1))
        );
        assert!(t.trigrams.contains_key(&(
            last_recorded - 3,
            last_recorded - 2,
            last_recorded - 1
        )));
        assert!(t.fourgrams.contains_key(&(
            last_recorded - 4,
            last_recorded - 3,
            last_recorded - 2,
            last_recorded - 1
        )));
    }

    #[test]
    fn context_evicts_oldest_continuations_when_cap_is_exceeded() {
        let mut t = NgramTable::new();
        t.feed(&[1, 2, 10, 1, 2, 20, 1, 2, 30, 1, 2, 40, 1, 2, 50]);
        t.feed(&[9, 1, 2]);

        let prediction = t
            .bigrams
            .get(&(1, 2))
            .expect("bigram context should be retained");
        assert!(prediction.continuations.len() <= MAX_CONTINUATIONS_PER_CONTEXT);
        assert!(!continuations_contain(prediction, 10));
        for token in [20, 30, 40, 50] {
            assert!(continuations_contain(prediction, token));
        }
        assert_eq!(prediction.total, MAX_CONTINUATIONS_PER_CONTEXT as u32);
        assert_eq!(t.predict(1), vec![50]);
    }

    #[test]
    fn stats_report_context_and_continuation_pressure() {
        let t = table_from_sequence(&[1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2]);
        let stats = t.stats();

        assert_eq!(
            stats.total_contexts,
            stats.bigram_contexts + stats.trigram_contexts + stats.fourgram_contexts
        );
        assert!(stats.bigram_contexts > 0);
        assert!(stats.trigram_contexts > 0);
        assert!(stats.fourgram_contexts > 0);
        assert!(stats.total_continuations >= stats.total_contexts);
        assert_eq!(stats.accepted_feedback, 0);
        assert_eq!(stats.rejected_feedback, 0);
        assert_eq!(stats.max_contexts_per_order, MAX_CONTEXTS_PER_ORDER);
        assert_eq!(
            stats.max_continuations_per_context,
            MAX_CONTINUATIONS_PER_CONTEXT
        );
        assert_eq!(stats.context_prune_batch, CONTEXT_PRUNE_BATCH);
    }

    #[test]
    fn stats_report_verifier_feedback_without_tokens() {
        let mut t = NgramTable::new();
        t.feed(&[1, 2, 3, 9, 1, 2]);

        t.record_draft_feedback(&[3], 0, NgramDraftPolicy::majority(4, 1, 0.4));
        let rejected = t.stats();
        assert_eq!(rejected.accepted_feedback, 0);
        assert_eq!(rejected.rejected_feedback, 1);

        t.record_draft_feedback(&[3], 1, NgramDraftPolicy::majority(4, 1, 0.0));
        let accepted = t.stats();
        assert_eq!(accepted.accepted_feedback, 1);
        assert_eq!(accepted.rejected_feedback, 1);
    }

    #[test]
    fn context_pruning_evicts_oldest_entries_in_batches() {
        let max_entries = 128;
        let prune_batch = 16;
        let mut map = HashMap::new();
        for token in 0..=max_entries as u32 {
            map.insert(token, NgramPrediction::new(token, token as u64, false));
        }

        prune_oldest_predictions(&mut map, max_entries, prune_batch);

        assert_eq!(map.len(), max_entries + 1 - prune_batch);
        for token in 0..prune_batch as u32 {
            assert!(!map.contains_key(&token));
        }
        assert!(
            map.contains_key(&(max_entries as u32)),
            "newest contexts should survive batched pruning"
        );
    }

    #[test]
    fn context_pruning_removes_full_overflow_when_larger_than_batch() {
        let max_entries = 128;
        let prune_batch = 16;
        let overflow = prune_batch + 9;
        let mut map = HashMap::new();
        for token in 0..(max_entries + overflow) as u32 {
            map.insert(token, NgramPrediction::new(token, token as u64, false));
        }

        prune_oldest_predictions(&mut map, max_entries, prune_batch);

        assert_eq!(map.len(), max_entries);
        for token in 0..overflow as u32 {
            assert!(!map.contains_key(&token));
        }
        assert!(map.contains_key(&((max_entries + overflow - 1) as u32)));
    }

    #[test]
    fn updating_existing_context_does_not_trigger_prune() {
        let max_entries = 2;
        let prune_batch = 1;
        let mut map = HashMap::new();
        update_prediction_and_prune(&mut map, 1, 10, 1, max_entries, prune_batch, false);
        update_prediction_and_prune(&mut map, 2, 20, 2, max_entries, prune_batch, false);

        update_prediction_and_prune(&mut map, 1, 11, 3, max_entries, prune_batch, false);

        assert_eq!(
            map.len(),
            max_entries,
            "existing-key updates should not enter the prune path"
        );
        assert!(map.contains_key(&1));
        assert!(map.contains_key(&2));

        update_prediction_and_prune(&mut map, 3, 30, 4, max_entries, prune_batch, false);
        assert_eq!(map.len(), max_entries);
        assert!(
            !map.contains_key(&2),
            "oldest untouched context should prune first"
        );
    }

    #[test]
    fn rejected_draft_feedback_suppresses_low_quality_context() {
        let mut t = NgramTable::new();
        t.feed(&[1, 2, 3, 9, 1, 2]);

        assert_eq!(t.predict_with_confidence(1, 1, 0.4), vec![3]);
        t.record_draft_feedback(&[3], 0, NgramDraftPolicy::majority(1, 1, 0.4));

        assert!(
            t.predict_with_confidence(1, 1, 0.4).is_empty(),
            "one verifier rejection should suppress a one-off candidate under confidence gating"
        );
        assert_eq!(
            t.predict(1),
            vec![3],
            "raw predict remains available for callers that intentionally disable confidence gating"
        );
    }

    #[test]
    fn accepted_draft_feedback_keeps_context_above_threshold() {
        let mut t = NgramTable::new();
        t.feed(&[1, 2, 3, 9, 1, 2]);

        t.record_draft_feedback(&[3], 1, NgramDraftPolicy::majority(1, 1, 0.4));

        assert_eq!(
            t.predict_with_confidence(1, 1, 0.4),
            vec![3],
            "accepted verifier feedback should keep a candidate eligible"
        );
    }

    #[test]
    fn draft_feedback_only_marks_verified_prefix_and_first_reject() {
        let mut t = NgramTable::new();
        t.feed(&[1, 2, 3, 4, 1, 2]);

        let draft = t.predict_with_confidence(2, 1, 0.4);
        assert_eq!(draft, vec![3, 4]);

        t.record_draft_feedback(&draft, 1, NgramDraftPolicy::majority(2, 1, 0.4));

        let first = t
            .bigrams
            .get(&(1, 2))
            .and_then(NgramPrediction::selected_continuation)
            .expect("first draft context should be tracked");
        assert_eq!(first.accepted, 1);
        assert_eq!(first.rejected, 0);

        let second = t
            .trigrams
            .get(&(1, 2, 3))
            .and_then(NgramPrediction::selected_continuation)
            .expect("first rejected context should be tracked");
        assert_eq!(second.accepted, 0);
        assert_eq!(second.rejected, 1);
    }

    #[test]
    fn draft_feedback_updates_fallback_context_that_created_draft() {
        let mut t = NgramTable::new();
        t.feed(&[
            0, 1, 2, 3, 98, // fourgram and trigram observe 98
            0, 1, 2, 3, 97, // fourgram and trigram observe 97
            5, 1, 2, 3, 99, // trigram observes a supported 99 without this fourgram
            0, 1, 2, 3, 99, // fourgram and trigram observe 99
            0, 1, 2, 3, // put the tail back on the contested four-token context
        ]);

        assert_eq!(
            t.predict_with_confidence(1, 1, 0.4),
            vec![99],
            "fourgram confidence is too low, so the predictor should fall back to the trigram"
        );

        t.record_draft_feedback(&[99], 0, NgramDraftPolicy::majority(1, 1, 0.4));

        let fourgram = t
            .fourgrams
            .get(&(0, 1, 2, 3))
            .and_then(NgramPrediction::selected_continuation)
            .expect("contested fourgram continuation should be tracked");
        assert_eq!(fourgram.token, 99);
        assert_eq!(
            fourgram.rejected, 0,
            "feedback must not be charged to a longer context that failed the predictor gate"
        );

        let trigram = t
            .trigrams
            .get(&(1, 2, 3))
            .and_then(NgramPrediction::selected_continuation)
            .expect("fallback trigram continuation should be tracked");
        assert_eq!(trigram.token, 99);
        assert_eq!(trigram.rejected, 1);
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

    // ---------------------------------------------------------------------------
    // Speculative accept threshold constant / env-var parsing
    // ---------------------------------------------------------------------------

    #[test]
    fn speculative_accept_threshold_default() {
        assert_eq!(
            parse_speculative_accept_threshold(None),
            NGRAM_SPECULATIVE_ACCEPT_THRESHOLD,
        );
    }

    #[test]
    fn speculative_accept_threshold_parse_valid() {
        assert!((parse_speculative_accept_threshold(Some("0.0")) - 0.0).abs() < 1e-6);
        assert!((parse_speculative_accept_threshold(Some("0.3")) - 0.3).abs() < 1e-6);
        assert!((parse_speculative_accept_threshold(Some("1.0")) - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic]
    fn speculative_accept_threshold_parse_out_of_range() {
        parse_speculative_accept_threshold(Some("1.5"));
    }

    #[test]
    #[should_panic]
    fn speculative_accept_threshold_parse_invalid_string() {
        parse_speculative_accept_threshold(Some("not_a_float"));
    }
}
