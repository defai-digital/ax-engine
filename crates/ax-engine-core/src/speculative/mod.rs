//! Speculative decoding — CPU draft + GPU target verification.
//!
//! # Algorithm (Leviathan et al., 2023)
//!
//! Each generation step produces up to K+1 tokens instead of 1:
//!
//! 1. **Draft**: run the small draft model autoregressively for K steps on CPU,
//!    collecting draft tokens and their per-token probabilities.
//! 2. **Verify**: run the large target model on all K+1 tokens
//!    (`[last_accepted] + draft_tokens`) to obtain K+1 target logit distributions.
//! 3. **Accept/Reject**: for each draft token `i`:
//!    - sample `r ~ Uniform(0, 1)`
//!    - if `r < target_prob[i] / draft_prob[i]`: **accept**
//!    - else: **reject** — sample a correction token from `max(0, p_target - p_draft)`,
//!      rewind the target KV cache to `accepted_pos`, emit the correction token.
//! 4. If all K tokens accepted: sample a **bonus token** from `target_logits[K]`.
//!
//! # Performance characteristics
//!
//! - Draft K steps on CPU (lightweight model) vs. 1 step on GPU (large model).
//! - Target verification: `forward_batch_all_logits` over K+1 positions.
//!   Supported architectures route this through the GPU batch path and retain a
//!   serial fallback when batch logits are unavailable.
//! - Average accepted tokens per step: E[accepted] = K * acceptance_rate + 1 bonus.
//!
//! # Usage
//!
//! ```ignore
//! let mut decoder = SpeculativeDecoder::load("./draft.gguf", 4)?;
//! let result = decoder.generate_step(last_token, position, &target_model,
//!                                    &mut target_kv, &target_weights, &mut sampler)?;
//! for tok in result.tokens { /* emit */ }
//! ```

use std::path::Path;
use std::time::Duration;

use crate::gguf::MappedModel;
use crate::kv::{ModelKv, ModelKvSnapshot};
use crate::metrics::counters::OpTimer;
use crate::model::{LlamaModel, WeightStore};
use crate::sampling::Sampler;

fn qwen35_speculative_branch_verify_enabled() -> bool {
    match std::env::var("AX_QWEN35_SPEC_VERIFY_BRANCH") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

pub fn target_verify_mode_label(model: &LlamaModel, kv: &ModelKv) -> &'static str {
    if model.arch_name() == "qwen35" && kv.as_qwen35().is_some() {
        if qwen35_speculative_branch_verify_enabled() {
            "qwen35_branch"
        } else {
            "snapshot_replay"
        }
    } else if kv.supports_snapshot() {
        "snapshot_replay"
    } else {
        "truncate_only"
    }
}

/// Timing and shape details for one speculative decode step.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecStepMetrics {
    /// CPU draft generation time for the K proposed tokens.
    pub draft_duration: Duration,
    /// Target verification time for `[last_token] + draft_tokens`.
    pub verify_duration: Duration,
    /// Time spent preparing the target verify path before the forward call.
    pub verify_prepare_duration: Duration,
    /// Time spent inside the target verify forward itself.
    pub verify_forward_duration: Duration,
    /// Time spent cleaning up the target verify path after forward.
    pub verify_cleanup_duration: Duration,
    /// Acceptance/rejection bookkeeping time.
    pub accept_duration: Duration,
    /// Number of draft tokens proposed this step.
    pub drafted_tokens: usize,
    /// Number of verification positions evaluated this step.
    pub verified_positions: usize,
}

impl SpecStepMetrics {
    /// Total measured wall time for the step.
    pub fn total_duration(&self) -> Duration {
        self.draft_duration + self.verify_duration + self.accept_duration
    }
}

/// Result of one speculative decode step.
pub struct SpecStep {
    /// Accepted (and bonus) tokens produced this step. Always ≥ 1.
    pub tokens: Vec<u32>,
    /// Number of draft tokens that were accepted (0..=k).
    pub n_accepted: usize,
    /// Timing breakdown for the speculative step.
    pub metrics: SpecStepMetrics,
}

/// Speculative decoder: wraps a small CPU draft model for use with a large target model.
pub struct SpeculativeDecoder {
    /// Small draft model (runs on CpuBackend).
    draft_model: LlamaModel,
    /// GGUF mapping kept alive for the draft model's weights.
    _draft_mapped: MappedModel,
    /// Persistent draft KV synchronized to the accepted history.
    draft_kv: ModelKv,
    /// Number of tokens to speculate per step.
    k: usize,
    /// Reusable Qwen3.5 target-side verify branch slot, when enabled.
    target_qwen35_branch_slot: Option<usize>,
}

enum TargetVerifyState {
    Standard {
        snapshot: Option<ModelKvSnapshot>,
    },
    Qwen35Branch {
        source_seq_len: usize,
        source_slot: usize,
        branch_slot: usize,
    },
}

impl SpeculativeDecoder {
    /// Load a draft model from a GGUF file and create a `SpeculativeDecoder`.
    ///
    /// The draft model always uses `CpuBackend` (forced via `LlamaModel::new`).
    /// `k` is the number of lookahead tokens per step; 4 is a good default.
    pub fn load(draft_path: &str, k: usize) -> anyhow::Result<Self> {
        let draft_mapped = MappedModel::open(Path::new(draft_path))?;
        let draft_config = crate::model::ModelConfig::from_gguf(&draft_mapped.header)?;
        validate_speculative_config(k, draft_config.vocab_size, draft_config.vocab_size)?;
        let draft_model = LlamaModel::new(draft_config)?; // CPU backend
        validate_speculative_model_rollback("draft", &draft_model)?;
        let draft_kv = draft_model.create_model_kv();
        Ok(Self {
            draft_model,
            _draft_mapped: draft_mapped,
            draft_kv,
            k,
            target_qwen35_branch_slot: None,
        })
    }

    /// The lookahead length K.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Prewarm the target-side speculative verify path without advancing the
    /// shared attention timeline. For Qwen3.5 this allocates/reuses the branch
    /// slot and restores the original active slot immediately so subsequent
    /// measured steps avoid first-use branch setup noise.
    pub fn prewarm_target_verify_path(
        &mut self,
        target_model: &LlamaModel,
        target_kv: &mut ModelKv,
    ) -> anyhow::Result<()> {
        let verify_state = {
            let mut target_qwen35_branch_slot = self.target_qwen35_branch_slot;
            let verify_state = prepare_target_verify_state_with_branch_slot_hint(
                &mut target_qwen35_branch_slot,
                target_model,
                target_kv,
            )?;
            self.target_qwen35_branch_slot = target_qwen35_branch_slot;
            verify_state
        };

        if let TargetVerifyState::Qwen35Branch {
            source_seq_len,
            source_slot,
            ..
        } = &verify_state
        {
            target_kv.truncate_qwen35_attention_timeline(*source_seq_len)?;
            target_kv.set_qwen35_active_slot(*source_slot)?;
        }

        Ok(())
    }

    /// Run one speculative decode step starting from `last_token` at `position`.
    ///
    /// Returns a [`SpecStep`] containing 1..=K+1 tokens. The target KV cache is
    /// rewound if tokens are rejected so that its `seq_len` always equals
    /// `position + result.tokens.len()` after the call.
    ///
    /// # Arguments
    /// * `last_token` — the last accepted token (seeds both draft and target).
    /// * `position`   — KV position of `last_token` (= `target_kv.seq_len()`).
    /// * `target_model` — large target model.
    /// * `target_kv`  — KV cache for the target model (GPU or CPU).
    /// * `target_weights` — weight store for the target model.
    /// * `sampler`    — shared sampler (temperature/top-k/top-p).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_step(
        &mut self,
        history_tokens: &[u32],
        last_token: u32,
        position: usize,
        target_model: &LlamaModel,
        target_kv: &mut ModelKv,
        target_weights: &WeightStore,
        sampler: &mut Sampler,
    ) -> anyhow::Result<SpecStep> {
        self.validate_against_target(target_model)?;

        // ── Step 1: Draft K tokens on CPU ──────────────────────────────────
        let draft_weights = WeightStore::new(&self._draft_mapped);
        anyhow::ensure!(
            history_tokens.len() == position,
            "speculative draft history length ({}) must match position ({position})",
            history_tokens.len()
        );
        anyhow::ensure!(
            history_tokens.len() <= self.draft_model.config.context_length as usize,
            "draft history length ({}) exceeds draft model context ({})",
            history_tokens.len(),
            self.draft_model.config.context_length
        );

        if self.draft_kv.seq_len() > history_tokens.len() {
            self.draft_kv.truncate_to(history_tokens.len());
        }
        if self.draft_kv.seq_len() < history_tokens.len() {
            let mut sync_logits = vec![0.0f32; self.draft_model.config.vocab_size as usize];
            self.draft_model.forward_batch(
                &history_tokens[self.draft_kv.seq_len()..],
                &mut self.draft_kv,
                &draft_weights,
                &mut sync_logits,
            )?;
        }
        self.draft_model.sync_model_kv(&mut self.draft_kv);
        let draft_step_snapshot = self.draft_kv.snapshot();

        // Run draft model for K steps from position, seeded by last_token.
        // We collect the drafted tokens and their probabilities.
        let draft_timer = OpTimer::start();
        let vocab = self.draft_model.config.vocab_size as usize;
        let mut draft_tokens: Vec<u32> = Vec::with_capacity(self.k);
        let mut draft_probs_all: Vec<Vec<f32>> = Vec::with_capacity(self.k);
        let mut draft_logits = vec![0.0f32; vocab];
        let recent_limit = if sampler.config().repeat_last_n < 0 {
            usize::MAX
        } else {
            sampler.config().repeat_last_n as usize
        };
        let recent_start = history_tokens.len().saturating_sub(recent_limit);
        let mut draft_recent_tokens = history_tokens[recent_start..].to_vec();
        draft_recent_tokens.push(last_token);

        let mut draft_pos = self.draft_kv.seq_len();
        let mut prev_tok = last_token;

        for _ in 0..self.k {
            draft_logits.fill(0.0);
            self.draft_model.forward_single(
                prev_tok,
                draft_pos,
                &mut self.draft_kv,
                &draft_weights,
                &mut draft_logits,
            )?;

            // Draft token probability distribution
            let draft_probs = softmax(&draft_logits[..vocab]);
            let draft_tok = sampler.sample(&mut draft_logits, &draft_recent_tokens);
            draft_probs_all.push(draft_probs);
            draft_tokens.push(draft_tok);
            draft_recent_tokens.push(draft_tok);
            if recent_limit > 0 && draft_recent_tokens.len() > recent_limit {
                let drain = draft_recent_tokens.len() - recent_limit;
                draft_recent_tokens.drain(..drain);
            }

            prev_tok = draft_tok;
            draft_pos += 1;
        }
        let draft_duration = draft_timer.elapsed();

        // ── Step 2: Verify with target model ───────────────────────────────
        // Run target on [last_token] + draft_tokens (K+1 tokens total).
        let verify_tokens: Vec<u32> = std::iter::once(last_token)
            .chain(draft_tokens.iter().copied())
            .collect();

        let mut target_qwen35_branch_slot = self.target_qwen35_branch_slot;
        let verify_prepare_timer = OpTimer::start();
        let target_verify_state = prepare_target_verify_state_with_branch_slot_hint(
            &mut target_qwen35_branch_slot,
            target_model,
            target_kv,
        )?;
        self.target_qwen35_branch_slot = target_qwen35_branch_slot;
        let verify_prepare_duration = verify_prepare_timer.elapsed();

        let verify_forward_timer = OpTimer::start();
        let mut target_logits_all: Vec<f32> = Vec::new();
        if let Err(err) = target_model.forward_batch_all_logits(
            &verify_tokens,
            target_kv,
            target_weights,
            &mut target_logits_all,
        ) {
            let verify_forward_duration = verify_forward_timer.elapsed();
            let verify_cleanup_timer = OpTimer::start();
            abort_target_verify_state(
                &target_verify_state,
                target_model,
                target_kv,
                target_weights,
                position,
            )?;
            let verify_cleanup_duration = verify_cleanup_timer.elapsed();
            let _ = (
                verify_prepare_duration,
                verify_forward_duration,
                verify_cleanup_duration,
            );
            return Err(err);
        }
        let verify_forward_duration = verify_forward_timer.elapsed();
        let verify_duration = verify_prepare_duration + verify_forward_duration;
        // target_logits_all[i*vocab..(i+1)*vocab] are next-token logits after
        // consuming verify_tokens[i]. That means:
        // - slot 0 verifies draft_tokens[0]
        // - slot i verifies draft_tokens[i]
        // - slot k is the bonus position after all K drafts

        // ── Step 3: Accept / reject ─────────────────────────────────────────
        let accept_timer = OpTimer::start();
        let mut accepted: Vec<u32> = Vec::with_capacity(self.k + 1);
        let mut n_accepted = 0usize;

        // Position in target_kv where draft_tokens begin
        // (target_kv.seq_len() was incremented by forward_batch_all_logits for all K+1)
        let base_pos_after_verify = position + verify_tokens.len();

        for i in 0..self.k {
            let target_slot = logits_slot(&target_logits_all, i, vocab);
            let target_probs = softmax(target_slot);
            let t_prob = target_probs[draft_tokens[i] as usize].max(1e-9);
            let d_prob = draft_probs_all[i][draft_tokens[i] as usize].max(1e-9);

            let r: f32 = sampler.sample_uniform();
            if r < (t_prob / d_prob).min(1.0) {
                // Accept
                accepted.push(draft_tokens[i]);
                n_accepted += 1;
            } else {
                // Reject: sample correction from max(0, p_target - p_draft)
                let mut correction_probs = vec![0.0f32; vocab];
                let mut sum = 0.0f32;
                for j in 0..vocab {
                    let p = (target_probs[j] - draft_probs_all[i][j]).max(0.0);
                    correction_probs[j] = p;
                    sum += p;
                }
                if sum > 0.0 {
                    let inv_sum = 1.0 / sum;
                    for p in &mut correction_probs {
                        *p *= inv_sum;
                    }
                } else {
                    correction_probs.copy_from_slice(&target_probs);
                }
                let correction = sampler.sample_from_probs(&correction_probs);
                accepted.push(correction);

                // Restore target state to the start of the speculative window,
                // then replay last_token + accepted drafts when precise
                // truncate is unavailable.
                let rewind_pos = position + n_accepted + 1;
                let verify_cleanup_timer = OpTimer::start();
                restore_target_verify_state(
                    &target_verify_state,
                    target_model,
                    target_kv,
                    target_weights,
                    &verify_tokens[..n_accepted + 1],
                    rewind_pos,
                )?;
                restore_or_truncate_kv(
                    &self.draft_model,
                    &mut self.draft_kv,
                    &draft_weights,
                    draft_step_snapshot.as_ref(),
                    &verify_tokens[..n_accepted + 1],
                    position + n_accepted + 1,
                )?;
                let verify_cleanup_duration = verify_cleanup_timer.elapsed();

                return Ok(SpecStep {
                    tokens: accepted,
                    n_accepted,
                    metrics: SpecStepMetrics {
                        draft_duration,
                        verify_duration,
                        verify_prepare_duration,
                        verify_forward_duration,
                        verify_cleanup_duration,
                        accept_duration: accept_timer.elapsed(),
                        drafted_tokens: draft_tokens.len(),
                        verified_positions: verify_tokens.len(),
                    },
                });
            }
        }

        // All K tokens accepted: sample bonus token from target_logits[K]
        let bonus_slot = &mut logits_slot(&target_logits_all, self.k, vocab).to_vec();
        // Include history + last_token + accepted in context for repetition penalty
        let bonus_recent_limit = if sampler.config().repeat_last_n < 0 {
            usize::MAX
        } else {
            sampler.config().repeat_last_n as usize
        };
        let bonus_start = history_tokens.len().saturating_sub(bonus_recent_limit);
        let mut bonus_context =
            Vec::with_capacity(history_tokens[bonus_start..].len() + 1 + accepted.len());
        bonus_context.extend_from_slice(&history_tokens[bonus_start..]);
        bonus_context.push(last_token);
        bonus_context.extend_from_slice(&accepted);
        let bonus_tok = sampler.sample(bonus_slot, &bonus_context);
        accepted.push(bonus_tok);

        // target_kv is already at position + K + 1 (all K+1 forward_singles ran)
        // Verify it matches our expectation
        debug_assert_eq!(
            target_kv.seq_len(),
            base_pos_after_verify,
            "target KV seq_len mismatch after full acceptance"
        );
        // Draft KV is already at position + K from the K forward_single calls.
        // Do NOT call truncate_to here — for recurrent models (Qwen3.5) it would
        // destroy state, and for transformer models it is a no-op.
        debug_assert_eq!(
            self.draft_kv.seq_len(),
            position + self.k,
            "draft KV seq_len mismatch after full acceptance"
        );
        let verify_cleanup_timer = OpTimer::start();
        commit_target_verify_state(
            &target_verify_state,
            &mut self.target_qwen35_branch_slot,
            target_model,
            target_kv,
        )?;
        let verify_cleanup_duration = verify_cleanup_timer.elapsed();

        Ok(SpecStep {
            tokens: accepted,
            n_accepted: self.k,
            metrics: SpecStepMetrics {
                draft_duration,
                verify_duration,
                verify_prepare_duration,
                verify_forward_duration,
                verify_cleanup_duration,
                accept_duration: accept_timer.elapsed(),
                drafted_tokens: draft_tokens.len(),
                verified_positions: verify_tokens.len(),
            },
        })
    }

    fn validate_against_target(&self, target_model: &LlamaModel) -> anyhow::Result<()> {
        validate_speculative_config(
            self.k,
            self.draft_model.config.vocab_size,
            target_model.config.vocab_size,
        )?;
        validate_speculative_model_rollback("draft", &self.draft_model)?;
        validate_speculative_model_rollback("target", target_model)
    }
}

fn prepare_target_verify_state_with_branch_slot_hint(
    target_qwen35_branch_slot: &mut Option<usize>,
    target_model: &LlamaModel,
    target_kv: &mut ModelKv,
) -> anyhow::Result<TargetVerifyState> {
    if target_model.arch_name() == "qwen35"
        && target_kv.as_qwen35().is_some()
        && qwen35_speculative_branch_verify_enabled()
    {
        let source_seq_len = target_kv.seq_len();
        let source_slot = target_kv.qwen35_active_slot()?;
        let branch_slot =
            acquire_qwen35_verify_branch_slot(target_qwen35_branch_slot, target_kv, source_slot)?;
        if !target_model.try_clone_qwen35_recurrent_slot_via_backend(
            target_kv,
            source_slot,
            branch_slot,
        )? {
            target_model.sync_model_kv(target_kv);
            target_kv.clone_qwen35_recurrent_slot(source_slot, branch_slot)?;
        }
        target_kv.set_qwen35_active_slot(branch_slot)?;
        return Ok(TargetVerifyState::Qwen35Branch {
            source_seq_len,
            source_slot,
            branch_slot,
        });
    }

    target_model.sync_model_kv(target_kv);
    Ok(TargetVerifyState::Standard {
        snapshot: target_kv.snapshot(),
    })
}

fn acquire_qwen35_verify_branch_slot(
    target_qwen35_branch_slot: &mut Option<usize>,
    target_kv: &mut ModelKv,
    source_slot: usize,
) -> anyhow::Result<usize> {
    if let Some(branch_slot) = *target_qwen35_branch_slot
        && branch_slot != source_slot
        && target_kv
            .as_qwen35()
            .is_some_and(|qwen_kv| qwen_kv.has_recurrent_slot(branch_slot))
    {
        return Ok(branch_slot);
    }

    let branch_slot = target_kv.allocate_qwen35_recurrent_slot()?;
    *target_qwen35_branch_slot = Some(branch_slot);
    Ok(branch_slot)
}

fn validate_speculative_config(
    k: usize,
    draft_vocab_size: u32,
    target_vocab_size: u32,
) -> anyhow::Result<()> {
    anyhow::ensure!(k > 0, "speculative decoding requires k > 0");
    anyhow::ensure!(
        draft_vocab_size == target_vocab_size,
        "speculative decoding requires matching draft/target vocab sizes (draft={draft_vocab_size}, target={target_vocab_size})"
    );
    Ok(())
}

fn validate_speculative_model_rollback(label: &str, model: &LlamaModel) -> anyhow::Result<()> {
    if model
        .kv_plan_with_requirements(crate::backend::KvPlannerRequirements {
            require_precise_rollback: true,
            ..Default::default()
        })
        .is_ok()
    {
        return Ok(());
    }

    if model.create_model_kv().snapshot().is_some() {
        return Ok(());
    }

    anyhow::bail!(
        "speculative decoding does not support {label} model architecture '{}' because neither precise KV rollback nor snapshot replay is available",
        model.config.architecture
    );
}

fn restore_target_verify_state(
    verify_state: &TargetVerifyState,
    target_model: &LlamaModel,
    target_kv: &mut ModelKv,
    target_weights: &WeightStore,
    replay_tokens: &[u32],
    truncate_pos: usize,
) -> anyhow::Result<()> {
    match verify_state {
        TargetVerifyState::Standard { snapshot } => restore_or_truncate_kv(
            target_model,
            target_kv,
            target_weights,
            snapshot.as_ref(),
            replay_tokens,
            truncate_pos,
        ),
        TargetVerifyState::Qwen35Branch {
            source_seq_len,
            source_slot,
            ..
        } => {
            target_kv.truncate_qwen35_attention_timeline(*source_seq_len)?;
            target_kv.set_qwen35_active_slot(*source_slot)?;
            if !replay_tokens.is_empty() {
                let mut replay_logits = vec![0.0f32; target_model.config.vocab_size as usize];
                target_model.forward_batch(
                    replay_tokens,
                    target_kv,
                    target_weights,
                    &mut replay_logits,
                )?;
            }
            Ok(())
        }
    }
}

fn abort_target_verify_state(
    verify_state: &TargetVerifyState,
    target_model: &LlamaModel,
    target_kv: &mut ModelKv,
    target_weights: &WeightStore,
    position: usize,
) -> anyhow::Result<()> {
    restore_target_verify_state(
        verify_state,
        target_model,
        target_kv,
        target_weights,
        &[],
        position,
    )
}

fn commit_target_verify_state(
    verify_state: &TargetVerifyState,
    target_qwen35_branch_slot: &mut Option<usize>,
    _target_model: &LlamaModel,
    target_kv: &mut ModelKv,
) -> anyhow::Result<()> {
    if let TargetVerifyState::Qwen35Branch {
        source_slot,
        branch_slot,
        ..
    } = verify_state
    {
        target_kv.set_qwen35_active_slot(*branch_slot)?;
        *target_qwen35_branch_slot = Some(*source_slot);
    }
    Ok(())
}

fn restore_or_truncate_kv(
    model: &LlamaModel,
    kv: &mut ModelKv,
    weights: &WeightStore,
    snapshot: Option<&ModelKvSnapshot>,
    replay_tokens: &[u32],
    truncate_pos: usize,
) -> anyhow::Result<()> {
    if let Some(snapshot) = snapshot {
        kv.restore_snapshot(snapshot)?;
        if !replay_tokens.is_empty() {
            let mut replay_logits = vec![0.0f32; model.config.vocab_size as usize];
            model.forward_batch(replay_tokens, kv, weights, &mut replay_logits)?;
        }
        return Ok(());
    }

    kv.truncate_to(truncate_pos);
    Ok(())
}

/// Compute softmax of a logit slice, returning a probability vector.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let n = logits.len();
        return vec![1.0 / n as f32; n];
    }
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    probs
}

fn logits_slot(logits_all: &[f32], slot: usize, vocab: usize) -> &[f32] {
    &logits_all[slot * vocab..(slot + 1) * vocab]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    use crate::model::{LlamaModel, ModelConfig};

    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("speculative env test lock")
    }

    struct EnvVarRestore {
        key: String,
        previous: Option<OsString>,
    }

    impl Drop for EnvVarRestore {
        fn drop(&mut self) {
            match &self.previous {
                Some(prev) => unsafe {
                    std::env::set_var(&self.key, prev);
                },
                None => unsafe {
                    std::env::remove_var(&self.key);
                },
            }
        }
    }

    fn with_env_var<T>(key: &str, value: &str, f: impl FnOnce() -> T) -> T {
        let _guard = env_lock();
        let _restore = EnvVarRestore {
            key: key.to_string(),
            previous: std::env::var_os(key),
        };
        unsafe {
            std::env::set_var(key, value);
        }
        f()
    }

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: "llama".into(),
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            embedding_dim: 8,
            head_dim: 4,
            intermediate_dim: 16,
            vocab_size: 16,
            context_length: 64,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
            expert_intermediate_dim: None,
        }
    }

    #[test]
    fn test_truncate_to_resets_seq_len_gpu_variant() {
        // Test ModelKv::truncate_to via the Cpu variant (no GPU needed in tests)
        let model = LlamaModel::new(tiny_config()).unwrap();
        let mut kv = model.create_model_kv();
        assert_eq!(kv.seq_len(), 0);

        // truncate_to(0) on empty KV is a no-op
        kv.truncate_to(0);
        assert_eq!(kv.seq_len(), 0);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn test_softmax_max_is_largest() {
        let logits = vec![0.0f32, 1.0, 5.0, 2.0];
        let probs = softmax(&logits);
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 2, "max logit should have max prob");
    }

    #[test]
    fn test_softmax_uniform_logits() {
        let logits = vec![1.0f32; 8];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!(
                (p - 0.125).abs() < 1e-5,
                "uniform logits → uniform probs, got {p}"
            );
        }
    }

    #[test]
    fn test_validate_speculative_config_rejects_zero_k() {
        let err = validate_speculative_config(0, 16, 16).unwrap_err();
        assert!(err.to_string().contains("k > 0"));
    }

    #[test]
    fn test_validate_speculative_config_rejects_vocab_mismatch() {
        let err = validate_speculative_config(4, 16, 32).unwrap_err();
        assert!(
            err.to_string()
                .contains("matching draft/target vocab sizes")
        );
    }

    #[test]
    fn test_validate_speculative_config_accepts_matching_models() {
        validate_speculative_config(4, 16, 16).unwrap();
    }

    fn speculative_test_config(architecture: &str) -> crate::model::ModelConfig {
        crate::model::ModelConfig {
            architecture: architecture.into(),
            n_layers: 4,
            n_heads: 4,
            n_kv_heads: 4,
            embedding_dim: 512,
            head_dim: 128,
            intermediate_dim: 2048,
            context_length: 1024,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            expert_intermediate_dim: None,
            qwen35_full_attention_interval: if architecture == "qwen35" {
                Some(4)
            } else {
                None
            },
            qwen35_ssm_conv_kernel: if architecture == "qwen35" {
                Some(4)
            } else {
                None
            },
            qwen35_ssm_inner_size: if architecture == "qwen35" {
                Some(1024)
            } else {
                None
            },
            qwen35_ssm_state_size: if architecture == "qwen35" {
                Some(128)
            } else {
                None
            },
            qwen35_ssm_time_step_rank: if architecture == "qwen35" {
                Some(8)
            } else {
                None
            },
            qwen35_ssm_group_count: if architecture == "qwen35" {
                Some(2)
            } else {
                None
            },
        }
    }

    #[test]
    fn test_validate_speculative_model_rollback_accepts_qwen35_via_snapshot() {
        let model = LlamaModel::new(speculative_test_config("qwen35")).unwrap();
        validate_speculative_model_rollback("draft", &model).unwrap();
    }

    #[test]
    fn test_validate_speculative_model_rollback_accepts_llama() {
        let model = LlamaModel::new(speculative_test_config("llama")).unwrap();
        validate_speculative_model_rollback("target", &model).unwrap();
    }

    #[test]
    fn test_target_verify_mode_label_defaults_to_qwen35_branch() {
        let _guard = env_lock();
        let key = "AX_QWEN35_SPEC_VERIFY_BRANCH";
        let _restore = EnvVarRestore {
            key: key.to_string(),
            previous: std::env::var_os(key),
        };
        unsafe {
            std::env::remove_var(key);
        }

        let model = LlamaModel::new(speculative_test_config("qwen35")).unwrap();
        let kv = model.create_model_kv();
        assert_eq!(target_verify_mode_label(&model, &kv), "qwen35_branch");
    }

    #[test]
    fn test_target_verify_mode_label_honors_branch_disable_env() {
        with_env_var("AX_QWEN35_SPEC_VERIFY_BRANCH", "off", || {
            let model = LlamaModel::new(speculative_test_config("qwen35")).unwrap();
            let kv = model.create_model_kv();
            assert_eq!(target_verify_mode_label(&model, &kv), "snapshot_replay");
        });
    }

    #[test]
    fn test_prepare_target_verify_state_qwen35_forks_branch_and_commit_restores_source() {
        let model = LlamaModel::new(speculative_test_config("qwen35")).unwrap();
        let mut target_qwen35_branch_slot = None;
        let mut kv = model.create_model_kv();
        let kv_width = (model.config.n_kv_heads * model.config.head_dim) as usize;
        {
            let qwen_kv = kv.as_qwen35_mut().expect("expected qwen35 kv");
            let k0 = vec![1.0; kv_width];
            let v0 = vec![2.0; kv_width];
            qwen_kv.attention_append(3, &k0, &v0);
            qwen_kv.finalize_token();
            qwen_kv.conv_state_for_slot_mut(0, 0).fill(1.0);
            qwen_kv.recurrent_state_for_slot_mut(0, 0).fill(2.0);
        }

        let verify_state = prepare_target_verify_state_with_branch_slot_hint(
            &mut target_qwen35_branch_slot,
            &model,
            &mut kv,
        )
        .expect("qwen35 verify branch preparation should succeed");
        let (source_slot, branch_slot) = match &verify_state {
            TargetVerifyState::Qwen35Branch {
                source_slot,
                branch_slot,
                ..
            } => (*source_slot, *branch_slot),
            TargetVerifyState::Standard { .. } => panic!("expected qwen35 branch state"),
        };

        {
            let qwen_kv = kv.as_qwen35_mut().expect("expected qwen35 kv");
            assert_eq!(source_slot, 0);
            assert_eq!(qwen_kv.active_slot(), branch_slot);
            assert!(qwen_kv.has_recurrent_slot(branch_slot));
            qwen_kv.conv_state_for_slot_mut(branch_slot, 0).fill(9.0);
            qwen_kv
                .recurrent_state_for_slot_mut(branch_slot, 0)
                .fill(10.0);
            let k1 = vec![5.0; kv_width];
            let v1 = vec![6.0; kv_width];
            qwen_kv.attention_append(3, &k1, &v1);
            qwen_kv.finalize_token();
        }

        commit_target_verify_state(
            &verify_state,
            &mut target_qwen35_branch_slot,
            &model,
            &mut kv,
        )
        .expect("qwen35 verify branch commit should succeed");

        let qwen_kv = kv.as_qwen35().expect("expected qwen35 kv");
        assert_eq!(qwen_kv.active_slot(), branch_slot);
        assert_eq!(qwen_kv.seq_len(), 2);
        assert!(qwen_kv.has_recurrent_slot(branch_slot));
        assert!(
            qwen_kv
                .conv_state_for_slot(branch_slot, 0)
                .iter()
                .all(|&v| v == 9.0)
        );
        assert!(
            qwen_kv
                .recurrent_state_for_slot(branch_slot, 0)
                .iter()
                .all(|&v| v == 10.0)
        );
        assert_eq!(target_qwen35_branch_slot, Some(source_slot));
    }

    #[test]
    fn test_prepare_target_verify_state_qwen35_reuses_branch_slot_across_steps() {
        let model = LlamaModel::new(speculative_test_config("qwen35")).unwrap();
        let mut target_qwen35_branch_slot = None;
        let mut kv = model.create_model_kv();

        let first = prepare_target_verify_state_with_branch_slot_hint(
            &mut target_qwen35_branch_slot,
            &model,
            &mut kv,
        )
        .expect("first prepare should succeed");
        let first_branch_slot = match &first {
            TargetVerifyState::Qwen35Branch { branch_slot, .. } => *branch_slot,
            TargetVerifyState::Standard { .. } => panic!("expected qwen35 branch state"),
        };
        match &first {
            TargetVerifyState::Qwen35Branch {
                source_seq_len,
                source_slot,
                ..
            } => {
                kv.truncate_qwen35_attention_timeline(*source_seq_len)
                    .expect("truncate attention should succeed");
                kv.set_qwen35_active_slot(*source_slot)
                    .expect("restore source slot should succeed");
            }
            TargetVerifyState::Standard { .. } => panic!("expected qwen35 branch state"),
        }

        let second = prepare_target_verify_state_with_branch_slot_hint(
            &mut target_qwen35_branch_slot,
            &model,
            &mut kv,
        )
        .expect("second prepare should succeed");
        let second_branch_slot = match &second {
            TargetVerifyState::Qwen35Branch { branch_slot, .. } => *branch_slot,
            TargetVerifyState::Standard { .. } => panic!("expected qwen35 branch state"),
        };

        assert_eq!(first_branch_slot, second_branch_slot);
    }

    #[test]
    fn test_commit_target_verify_state_qwen35_handoffs_active_slot_and_reuses_old_source() {
        let model = LlamaModel::new(speculative_test_config("qwen35")).unwrap();
        let mut target_qwen35_branch_slot = None;
        let mut kv = model.create_model_kv();

        let first = prepare_target_verify_state_with_branch_slot_hint(
            &mut target_qwen35_branch_slot,
            &model,
            &mut kv,
        )
        .expect("first prepare should succeed");
        let (first_source_slot, first_branch_slot) = match &first {
            TargetVerifyState::Qwen35Branch {
                source_slot,
                branch_slot,
                ..
            } => (*source_slot, *branch_slot),
            TargetVerifyState::Standard { .. } => panic!("expected qwen35 branch state"),
        };

        commit_target_verify_state(&first, &mut target_qwen35_branch_slot, &model, &mut kv)
            .expect("handoff commit should succeed");

        {
            let qwen_kv = kv.as_qwen35().expect("expected qwen35 kv");
            assert_eq!(qwen_kv.active_slot(), first_branch_slot);
            assert_eq!(target_qwen35_branch_slot, Some(first_source_slot));
        }

        let second = prepare_target_verify_state_with_branch_slot_hint(
            &mut target_qwen35_branch_slot,
            &model,
            &mut kv,
        )
        .expect("second prepare should succeed");
        let (second_source_slot, second_branch_slot) = match &second {
            TargetVerifyState::Qwen35Branch {
                source_slot,
                branch_slot,
                ..
            } => (*source_slot, *branch_slot),
            TargetVerifyState::Standard { .. } => panic!("expected qwen35 branch state"),
        };

        assert_eq!(second_source_slot, first_branch_slot);
        assert_eq!(second_branch_slot, first_source_slot);
    }

    #[test]
    fn test_prepare_target_verify_state_qwen35_uses_attention_truncate_from_gpu_dirty_state() {
        let model = LlamaModel::new(speculative_test_config("qwen35")).unwrap();
        let mut target_qwen35_branch_slot = None;
        let mut kv = model.create_model_kv();
        let device = ax_engine_metal::MetalDevice::new().expect("metal device");
        let kv_width = (model.config.n_kv_heads * model.config.head_dim) as usize;
        let k0 = vec![3.0f32; kv_width];
        let v0 = vec![4.0f32; kv_width];
        {
            let qwen_kv = kv.as_qwen35_mut().expect("expected qwen35 kv");
            qwen_kv
                .enable_gpu_attention(&device, crate::kv::GpuKvDtype::F32)
                .expect("enable gpu attention");
            let gpu_attention = qwen_kv
                .gpu_attention_mut()
                .expect("gpu attention should exist");
            gpu_attention.append_layer(3, &k0, &v0);
            gpu_attention.finalize_token();
            qwen_kv.mark_attention_cpu_dirty();
            qwen_kv.finalize_token();
            assert!(
                qwen_kv
                    .attention_k_slice_including_current(3, 1)
                    .iter()
                    .all(|&v| v == 0.0)
            );
        }

        let verify_state = prepare_target_verify_state_with_branch_slot_hint(
            &mut target_qwen35_branch_slot,
            &model,
            &mut kv,
        )
        .expect("prepare should succeed");

        let source_seq_len = match &verify_state {
            TargetVerifyState::Qwen35Branch { source_seq_len, .. } => *source_seq_len,
            TargetVerifyState::Standard { .. } => panic!("expected qwen35 branch state"),
        };

        {
            let qwen_kv = kv.as_qwen35_mut().expect("expected qwen35 kv");
            qwen_kv.attention_append(3, &vec![9.0; kv_width], &vec![10.0; kv_width]);
            qwen_kv.finalize_token();
        }
        kv.truncate_qwen35_attention_timeline(source_seq_len)
            .expect("truncate attention should succeed");
        kv.sync_qwen35_attention_timeline_if_needed()
            .expect("sync truncated attention should succeed");

        let qwen_kv = kv.as_qwen35().expect("expected qwen35 kv");
        assert_eq!(qwen_kv.seq_len(), 1);
        assert_eq!(
            qwen_kv.attention_k_slice_including_current(3, 1),
            k0.as_slice()
        );
        assert_eq!(
            qwen_kv.attention_v_slice_including_current(3, 1),
            v0.as_slice()
        );
    }

    #[test]
    fn test_logits_slot_maps_first_draft_to_slot_zero() {
        let logits_all: Vec<f32> = (0..12).map(|i| i as f32).collect();
        assert_eq!(logits_slot(&logits_all, 0, 4), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(logits_slot(&logits_all, 1, 4), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(logits_slot(&logits_all, 2, 4), &[8.0, 9.0, 10.0, 11.0]);
    }
}
