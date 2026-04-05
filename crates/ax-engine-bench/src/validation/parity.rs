//! Backend parity probe for comparing CPU reference logits against another backend.

use std::path::Path;

use ax_engine_core::backend::BackendConfig;
use ax_engine_core::gguf::MappedModel;
use ax_engine_core::model::{InferenceModel, ModelConfig, WeightStore};
use ax_engine_core::sampling::argmax;
use ax_engine_core::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};

/// Parity probe mode.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ParityMode {
    /// Compare standard full-prompt prefill followed by greedy serial decode.
    Decode,
    /// Compare the speculative verification batch:
    /// `forward_batch_all_logits([last_token] + draft_tokens)`.
    SpeculativeVerify { draft_tokens: usize },
}

impl ParityMode {
    fn label(self) -> String {
        match self {
            Self::Decode => "decode".to_string(),
            Self::SpeculativeVerify { draft_tokens } => {
                format!("speculative-verify(k={draft_tokens})")
            }
        }
    }
}

/// Parity probe configuration.
pub struct ParityConfig {
    /// Model path (GGUF file).
    pub model_path: String,
    /// Prompt text used to seed the shared context.
    pub prompt: String,
    /// Number of greedy decode steps to compare after prefill.
    pub decode_tokens: usize,
    /// Backend compared against the CPU reference.
    pub compare_backend: BackendConfig,
    /// Number of top tokens to retain in each step summary.
    pub top_tokens: usize,
    /// Maximum acceptable absolute logit delta before flagging divergence.
    pub max_abs_tolerance: f32,
    /// Probe mode.
    pub mode: ParityMode,
}

impl Default for ParityConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: "Hello".to_string(),
            decode_tokens: 8,
            compare_backend: BackendConfig::Hybrid,
            top_tokens: 5,
            max_abs_tolerance: 1e-3,
            mode: ParityMode::Decode,
        }
    }
}

/// Rendered summary of one token in the top-k set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityToken {
    pub token_id: u32,
    pub text: String,
    pub logit: f32,
}

/// One parity comparison point: either prefill output or one decode step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityStep {
    /// Human-readable phase label.
    pub phase: String,
    /// Next-token position represented by these logits.
    pub next_position: usize,
    /// Token consumed to reach this state, when applicable.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consumed_token: Option<ParityToken>,
    /// CPU reference argmax token.
    pub reference_top: ParityToken,
    /// Compared backend argmax token.
    pub compare_top: ParityToken,
    /// Whether the argmax token matches.
    pub argmax_match: bool,
    /// Maximum absolute logit delta across the vocabulary.
    pub max_abs_diff: f32,
    /// Mean absolute logit delta across the vocabulary.
    pub mean_abs_diff: f32,
    /// Root-mean-square logit delta across the vocabulary.
    pub rms_diff: f32,
    /// CPU top-k tokens.
    pub reference_top_tokens: Vec<ParityToken>,
    /// Compared backend top-k tokens.
    pub compare_top_tokens: Vec<ParityToken>,
}

/// First detected divergence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityDivergence {
    pub phase: String,
    pub next_position: usize,
    pub reason: String,
}

/// Result of a parity probe run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityResult {
    pub model: String,
    pub prompt: String,
    pub prompt_tokens: usize,
    pub probe_mode: String,
    pub decode_tokens: usize,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verify_draft_tokens: Option<usize>,
    pub compare_backend: String,
    pub reference_prefill_plan: String,
    pub compare_prefill_plan: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub support_note: Option<String>,
    pub max_abs_tolerance: f32,
    pub prefill: ParityStep,
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub verify_steps: Vec<ParityStep>,
    pub decode_steps: Vec<ParityStep>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_divergence: Option<ParityDivergence>,
}

impl ParityResult {
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn print_summary(&self) {
        eprintln!("=== Backend Parity Probe ===");
        eprintln!("Model:              {}", self.model);
        eprintln!("Prompt:             {:?}", self.prompt);
        eprintln!("Prompt tokens:      {}", self.prompt_tokens);
        eprintln!("Probe mode:         {}", self.probe_mode);
        if let Some(verify_draft_tokens) = self.verify_draft_tokens {
            eprintln!("Verify draft toks:  {verify_draft_tokens}");
        } else {
            eprintln!("Decode steps:       {}", self.decode_tokens);
        }
        eprintln!("Compare backend:    {}", self.compare_backend);
        eprintln!("Prefill CPU plan:   {}", self.reference_prefill_plan);
        eprintln!("Prefill target plan: {}", self.compare_prefill_plan);
        if let Some(note) = &self.support_note {
            eprintln!("Support:            {note}");
        }
        eprintln!("Max abs tolerance:  {:.3e}", self.max_abs_tolerance);
        eprintln!();
        print_step("Prefill", &self.prefill);
        for step in &self.verify_steps {
            eprintln!();
            print_step(&step.phase, step);
        }
        for step in &self.decode_steps {
            eprintln!();
            print_step(&step.phase, step);
        }
        eprintln!();
        if let Some(divergence) = &self.first_divergence {
            eprintln!(
                "First divergence:   {} @ next_pos={} ({})",
                divergence.phase, divergence.next_position, divergence.reason
            );
        } else {
            eprintln!("First divergence:   none within configured tolerance");
        }
    }
}

pub fn run_parity(config: &ParityConfig) -> anyhow::Result<ParityResult> {
    anyhow::ensure!(
        !config.model_path.is_empty(),
        "parity probe requires a non-empty model path"
    );
    anyhow::ensure!(
        !config.prompt.is_empty(),
        "parity probe requires a non-empty prompt"
    );
    anyhow::ensure!(config.top_tokens > 0, "top_tokens must be > 0");
    if let ParityMode::SpeculativeVerify { draft_tokens } = config.mode {
        anyhow::ensure!(
            draft_tokens > 0,
            "speculative verify parity requires draft_tokens > 0"
        );
    }

    let mapped = MappedModel::open(Path::new(&config.model_path))?;
    let tokenizer = Tokenizer::from_gguf(&mapped.header)?;
    let model_config = ModelConfig::from_gguf(&mapped.header)?;
    let weights = WeightStore::new(&mapped);
    let support_note = crate::support_note(&mapped);

    let prompt_tokens = tokenizer.encode(&config.prompt, true);
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "prompt encoded to zero tokens; cannot run parity probe"
    );

    let reference_backend = ax_engine_core::backend::create_backend(BackendConfig::Cpu)?;
    crate::configure_backend_for_model(
        &*reference_backend,
        &config.model_path,
        &mapped,
        &model_config,
    )?;
    let reference_model = InferenceModel::with_backend(model_config.clone(), reference_backend)?;

    let compare_backend = ax_engine_core::backend::create_backend(config.compare_backend)?;
    crate::configure_backend_for_model(
        &*compare_backend,
        &config.model_path,
        &mapped,
        &model_config,
    )?;
    let compare_model = InferenceModel::with_backend(model_config.clone(), compare_backend)?;

    match config.mode {
        ParityMode::Decode => run_decode_parity(
            config,
            &tokenizer,
            &weights,
            &prompt_tokens,
            support_note,
            &reference_model,
            &compare_model,
            model_config.vocab_size as usize,
        ),
        ParityMode::SpeculativeVerify { draft_tokens } => run_speculative_verify_parity(
            config,
            &tokenizer,
            &weights,
            &prompt_tokens,
            support_note,
            &reference_model,
            &compare_model,
            model_config.vocab_size as usize,
            draft_tokens,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_decode_parity(
    config: &ParityConfig,
    tokenizer: &Tokenizer,
    weights: &WeightStore,
    prompt_tokens: &[u32],
    support_note: Option<String>,
    reference_model: &InferenceModel,
    compare_model: &InferenceModel,
    vocab_size: usize,
) -> anyhow::Result<ParityResult> {
    let mut reference_kv = reference_model.create_model_kv_for_weights(weights);
    let mut compare_kv = compare_model.create_model_kv_for_weights(weights);
    let reference_prefill_plan =
        reference_model.prefill_plan_summary(weights, &reference_kv, prompt_tokens.len())?;
    let compare_prefill_plan =
        compare_model.prefill_plan_summary(weights, &compare_kv, prompt_tokens.len())?;

    let mut reference_logits = vec![0.0f32; vocab_size];
    let mut compare_logits = vec![0.0f32; vocab_size];
    reference_model.forward_batch(
        prompt_tokens,
        &mut reference_kv,
        weights,
        &mut reference_logits,
    )?;
    compare_model.forward_batch(prompt_tokens, &mut compare_kv, weights, &mut compare_logits)?;

    let prefill = summarize_step(
        "prefill".to_string(),
        prompt_tokens.len(),
        None,
        &reference_logits,
        &compare_logits,
        tokenizer,
        config.top_tokens,
    );
    let mut first_divergence = divergence_for_step(&prefill, config.max_abs_tolerance);

    let mut decode_steps = Vec::with_capacity(config.decode_tokens);
    let mut position = prompt_tokens.len();
    let mut next_reference_logits = vec![0.0f32; vocab_size];
    let mut next_compare_logits = vec![0.0f32; vocab_size];

    for step_idx in 0..config.decode_tokens {
        let drive_token = argmax(&reference_logits);
        let drive_summary = parity_token(
            tokenizer,
            drive_token,
            reference_logits[drive_token as usize],
        );

        next_reference_logits.fill(0.0);
        next_compare_logits.fill(0.0);
        reference_model.forward_single(
            drive_token,
            position,
            &mut reference_kv,
            weights,
            &mut next_reference_logits,
        )?;
        compare_model.forward_single(
            drive_token,
            position,
            &mut compare_kv,
            weights,
            &mut next_compare_logits,
        )?;

        let step = summarize_step(
            format!("decode[{step_idx}]"),
            position + 1,
            Some(drive_summary),
            &next_reference_logits,
            &next_compare_logits,
            tokenizer,
            config.top_tokens,
        );
        if first_divergence.is_none() {
            first_divergence = divergence_for_step(&step, config.max_abs_tolerance);
        }
        decode_steps.push(step);

        std::mem::swap(&mut reference_logits, &mut next_reference_logits);
        std::mem::swap(&mut compare_logits, &mut next_compare_logits);
        position += 1;
    }

    Ok(ParityResult {
        model: config.model_path.clone(),
        prompt: config.prompt.clone(),
        prompt_tokens: prompt_tokens.len(),
        probe_mode: config.mode.label(),
        decode_tokens: config.decode_tokens,
        verify_draft_tokens: None,
        compare_backend: backend_label(config.compare_backend).to_string(),
        reference_prefill_plan,
        compare_prefill_plan,
        support_note,
        max_abs_tolerance: config.max_abs_tolerance,
        prefill,
        verify_steps: Vec::new(),
        decode_steps,
        first_divergence,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_speculative_verify_parity(
    config: &ParityConfig,
    tokenizer: &Tokenizer,
    weights: &WeightStore,
    prompt_tokens: &[u32],
    support_note: Option<String>,
    reference_model: &InferenceModel,
    compare_model: &InferenceModel,
    vocab_size: usize,
    draft_tokens: usize,
) -> anyhow::Result<ParityResult> {
    let mut reference_prefill_kv = reference_model.create_model_kv_for_weights(weights);
    let mut compare_prefill_kv = compare_model.create_model_kv_for_weights(weights);
    let reference_prefill_plan = reference_model.prefill_plan_summary(
        weights,
        &reference_prefill_kv,
        prompt_tokens.len(),
    )?;
    let compare_prefill_plan =
        compare_model.prefill_plan_summary(weights, &compare_prefill_kv, prompt_tokens.len())?;

    let mut reference_prefill_logits = vec![0.0f32; vocab_size];
    let mut compare_prefill_logits = vec![0.0f32; vocab_size];
    reference_model.forward_batch(
        prompt_tokens,
        &mut reference_prefill_kv,
        weights,
        &mut reference_prefill_logits,
    )?;
    compare_model.forward_batch(
        prompt_tokens,
        &mut compare_prefill_kv,
        weights,
        &mut compare_prefill_logits,
    )?;

    let prefill = summarize_step(
        "prefill".to_string(),
        prompt_tokens.len(),
        None,
        &reference_prefill_logits,
        &compare_prefill_logits,
        tokenizer,
        config.top_tokens,
    );
    let mut first_divergence = divergence_for_step(&prefill, config.max_abs_tolerance);

    let prefix_tokens = &prompt_tokens[..prompt_tokens.len().saturating_sub(1)];
    let last_token = *prompt_tokens
        .last()
        .expect("prompt token validation should prevent empty prompts");

    let mut draft_kv = reference_model.create_model_kv_for_weights(weights);
    let mut scratch_logits = vec![0.0f32; vocab_size];
    if !prefix_tokens.is_empty() {
        reference_model.forward_batch(
            prefix_tokens,
            &mut draft_kv,
            weights,
            &mut scratch_logits,
        )?;
    }
    reference_model.sync_model_kv(&mut draft_kv);

    let mut greedy_draft_tokens = Vec::with_capacity(draft_tokens);
    let mut drive_token = last_token;
    let mut drive_position = prefix_tokens.len();
    for _ in 0..draft_tokens {
        scratch_logits.fill(0.0);
        reference_model.forward_single(
            drive_token,
            drive_position,
            &mut draft_kv,
            weights,
            &mut scratch_logits,
        )?;
        let drafted_token = argmax(&scratch_logits);
        greedy_draft_tokens.push(drafted_token);
        drive_token = drafted_token;
        drive_position += 1;
    }

    let verify_tokens: Vec<u32> = std::iter::once(last_token)
        .chain(greedy_draft_tokens.iter().copied())
        .collect();

    let mut reference_verify_kv = reference_model.create_model_kv_for_weights(weights);
    let mut compare_verify_kv = compare_model.create_model_kv_for_weights(weights);
    if !prefix_tokens.is_empty() {
        reference_model.forward_batch(
            prefix_tokens,
            &mut reference_verify_kv,
            weights,
            &mut scratch_logits,
        )?;
        compare_model.forward_batch(
            prefix_tokens,
            &mut compare_verify_kv,
            weights,
            &mut scratch_logits,
        )?;
    }
    reference_model.sync_model_kv(&mut reference_verify_kv);
    compare_model.sync_model_kv(&mut compare_verify_kv);

    let mut reference_logits_all = Vec::new();
    let mut compare_logits_all = Vec::new();
    reference_model.forward_batch_all_logits(
        &verify_tokens,
        &mut reference_verify_kv,
        weights,
        &mut reference_logits_all,
    )?;
    compare_model.forward_batch_all_logits(
        &verify_tokens,
        &mut compare_verify_kv,
        weights,
        &mut compare_logits_all,
    )?;

    let mut verify_steps = Vec::with_capacity(verify_tokens.len());
    for (step_idx, &consumed_token) in verify_tokens.iter().enumerate() {
        let step = summarize_step(
            format!("verify[{step_idx}]"),
            prefix_tokens.len() + step_idx + 1,
            Some(parity_token(tokenizer, consumed_token, 0.0)),
            logits_slot(&reference_logits_all, step_idx, vocab_size),
            logits_slot(&compare_logits_all, step_idx, vocab_size),
            tokenizer,
            config.top_tokens,
        );
        if first_divergence.is_none() {
            first_divergence = divergence_for_step(&step, config.max_abs_tolerance);
        }
        verify_steps.push(step);
    }

    Ok(ParityResult {
        model: config.model_path.clone(),
        prompt: config.prompt.clone(),
        prompt_tokens: prompt_tokens.len(),
        probe_mode: config.mode.label(),
        decode_tokens: 0,
        verify_draft_tokens: Some(draft_tokens),
        compare_backend: backend_label(config.compare_backend).to_string(),
        reference_prefill_plan,
        compare_prefill_plan,
        support_note,
        max_abs_tolerance: config.max_abs_tolerance,
        prefill,
        verify_steps,
        decode_steps: Vec::new(),
        first_divergence,
    })
}

fn print_step(label: &str, step: &ParityStep) {
    eprintln!(
        "{}: next_pos={} argmax_match={} max_abs={:.6} mean_abs={:.6} rms={:.6}",
        label,
        step.next_position,
        step.argmax_match,
        step.max_abs_diff,
        step.mean_abs_diff,
        step.rms_diff,
    );
    if let Some(token) = &step.consumed_token {
        eprintln!(
            "  consumed:   {} {:?} logit={:.6}",
            token.token_id, token.text, token.logit
        );
    }
    eprintln!(
        "  reference:  {} {:?} logit={:.6}",
        step.reference_top.token_id, step.reference_top.text, step.reference_top.logit
    );
    eprintln!(
        "  compare:    {} {:?} logit={:.6}",
        step.compare_top.token_id, step.compare_top.text, step.compare_top.logit
    );
    eprintln!(
        "  ref top-k:  {}",
        format_top_tokens(&step.reference_top_tokens)
    );
    eprintln!(
        "  cmp top-k:  {}",
        format_top_tokens(&step.compare_top_tokens)
    );
}

fn logits_slot(logits_all: &[f32], slot_idx: usize, vocab_size: usize) -> &[f32] {
    let start = slot_idx * vocab_size;
    let end = start + vocab_size;
    &logits_all[start..end]
}

fn format_top_tokens(tokens: &[ParityToken]) -> String {
    tokens
        .iter()
        .map(|token| format!("{}:{:?}@{:.4}", token.token_id, token.text, token.logit))
        .collect::<Vec<_>>()
        .join(", ")
}

fn summarize_step(
    phase: String,
    next_position: usize,
    consumed_token: Option<ParityToken>,
    reference_logits: &[f32],
    compare_logits: &[f32],
    tokenizer: &Tokenizer,
    top_tokens: usize,
) -> ParityStep {
    assert_eq!(reference_logits.len(), compare_logits.len());

    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    for (&reference, &compare) in reference_logits.iter().zip(compare_logits.iter()) {
        let diff = (reference - compare).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff as f64;
        sum_sq_diff += (diff * diff) as f64;
    }
    let len = reference_logits.len().max(1) as f64;
    let reference_top = argmax(reference_logits);
    let compare_top = argmax(compare_logits);

    ParityStep {
        phase,
        next_position,
        consumed_token,
        reference_top: parity_token(
            tokenizer,
            reference_top,
            reference_logits[reference_top as usize],
        ),
        compare_top: parity_token(tokenizer, compare_top, compare_logits[compare_top as usize]),
        argmax_match: reference_top == compare_top,
        max_abs_diff,
        mean_abs_diff: (sum_abs_diff / len) as f32,
        rms_diff: (sum_sq_diff / len).sqrt() as f32,
        reference_top_tokens: top_k_tokens(reference_logits, tokenizer, top_tokens),
        compare_top_tokens: top_k_tokens(compare_logits, tokenizer, top_tokens),
    }
}

fn divergence_for_step(step: &ParityStep, max_abs_tolerance: f32) -> Option<ParityDivergence> {
    if !step.argmax_match {
        return Some(ParityDivergence {
            phase: step.phase.clone(),
            next_position: step.next_position,
            reason: format!(
                "argmax mismatch: reference={} compare={}",
                step.reference_top.token_id, step.compare_top.token_id
            ),
        });
    }

    if step.max_abs_diff > max_abs_tolerance {
        return Some(ParityDivergence {
            phase: step.phase.clone(),
            next_position: step.next_position,
            reason: format!(
                "max_abs_diff {:.6} exceeded tolerance {:.6}",
                step.max_abs_diff, max_abs_tolerance
            ),
        });
    }

    None
}

fn top_k_tokens(logits: &[f32], tokenizer: &Tokenizer, k: usize) -> Vec<ParityToken> {
    let mut indexed: Vec<usize> = (0..logits.len()).collect();
    indexed.sort_unstable_by(|&lhs, &rhs| logits[rhs].total_cmp(&logits[lhs]));
    indexed
        .into_iter()
        .take(k)
        .map(|idx| parity_token(tokenizer, idx as u32, logits[idx]))
        .collect()
}

fn parity_token(tokenizer: &Tokenizer, token_id: u32, logit: f32) -> ParityToken {
    ParityToken {
        token_id,
        text: tokenizer
            .render_token(token_id)
            .or_else(|| tokenizer.decode_token(token_id).map(str::to_owned))
            .unwrap_or_else(|| "<special>".to_string()),
        logit,
    }
}

fn backend_label(config: BackendConfig) -> &'static str {
    match config {
        BackendConfig::Cpu => "cpu",
        BackendConfig::Metal => "metal",
        BackendConfig::Hybrid => "hybrid",
        BackendConfig::HybridCpuDecode => "hybrid-cpu-decode",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use ax_engine_core::gguf::MetadataValue;
    use ax_engine_core::gguf::header::GgufHeader;

    fn tiny_tokenizer() -> Tokenizer {
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            MetadataValue::Array(vec![
                MetadataValue::String("a".into()),
                MetadataValue::String("b".into()),
                MetadataValue::String("c".into()),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.scores".into(),
            MetadataValue::Array(vec![
                MetadataValue::Float32(0.0),
                MetadataValue::Float32(0.0),
                MetadataValue::Float32(0.0),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".into(),
            MetadataValue::Array(vec![
                MetadataValue::Int32(1),
                MetadataValue::Int32(1),
                MetadataValue::Int32(1),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.model".into(),
            MetadataValue::String("llama".into()),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".into(),
            MetadataValue::Uint32(0),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".into(),
            MetadataValue::Uint32(2),
        );
        metadata.insert(
            "tokenizer.ggml.add_bos_token".into(),
            MetadataValue::Bool(false),
        );
        metadata.insert(
            "tokenizer.ggml.add_eos_token".into(),
            MetadataValue::Bool(false),
        );
        metadata.insert(
            "tokenizer.ggml.add_space_prefix".into(),
            MetadataValue::Bool(false),
        );

        let header = GgufHeader {
            version: 3,
            tensor_count: 0,
            metadata,
        };
        Tokenizer::from_gguf(&header).expect("tiny tokenizer")
    }

    #[test]
    fn test_divergence_for_step_prefers_argmax_mismatch() {
        let tokenizer = tiny_tokenizer();
        let step = summarize_step(
            "decode[0]".into(),
            3,
            None,
            &[1.0, 5.0, 0.0],
            &[4.0, 3.0, 0.0],
            &tokenizer,
            2,
        );
        let divergence = divergence_for_step(&step, 10.0).expect("divergence");
        assert!(divergence.reason.contains("argmax mismatch"));
    }

    #[test]
    fn test_divergence_for_step_uses_tolerance_when_argmax_matches() {
        let tokenizer = tiny_tokenizer();
        let step = summarize_step(
            "prefill".into(),
            2,
            None,
            &[5.0, 1.0, 0.0],
            &[4.0, 1.0, 0.0],
            &tokenizer,
            2,
        );
        let divergence = divergence_for_step(&step, 0.5).expect("divergence");
        assert!(divergence.reason.contains("exceeded tolerance"));
    }

    #[test]
    fn test_parity_mode_label_formats_speculative_verify() {
        assert_eq!(ParityMode::Decode.label(), "decode");
        assert_eq!(
            ParityMode::SpeculativeVerify { draft_tokens: 3 }.label(),
            "speculative-verify(k=3)"
        );
    }

    #[test]
    fn test_logits_slot_returns_requested_vocab_window() {
        let logits_all = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(logits_slot(&logits_all, 0, 3), &[1.0, 2.0, 3.0]);
        assert_eq!(logits_slot(&logits_all, 1, 3), &[4.0, 5.0, 6.0]);
    }
}
