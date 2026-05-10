use std::collections::BTreeSet;
use std::fmt;

use crate::ids::RequestId;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    pub repetition_context_size: Option<u32>,
    pub seed: u64,
    pub deterministic: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context_size: None,
            seed: 0,
            deterministic: true,
        }
    }
}

pub(crate) fn sampling_params_allow_deterministic_argmax_fast_path(
    params: &SamplingParams,
) -> bool {
    params.deterministic
        && params.temperature <= 0.0
        && params.top_k == 0
        && params.top_p >= 1.0
        && params.repetition_penalty == 1.0
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EosToken,
    MaxOutputTokens,
    Cancelled,
    Error,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledToken {
    pub request_id: RequestId,
    pub token_id: u32,
    pub stop_reason: Option<StopReason>,
    pub logprob: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplerRequest {
    pub request_id: RequestId,
    pub previous_token: u32,
    pub logits: Option<Vec<f32>>,
    pub recent_tokens: Vec<u32>,
    pub generated_len: u32,
    pub max_output_tokens: u32,
    pub sampling_params: SamplingParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplerInput {
    pub requests: Vec<SamplerRequest>,
}

pub trait TokenSampler: fmt::Debug + Send + Sync {
    fn sample(&self, input: SamplerInput) -> Vec<SampledToken>;
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DeterministicSampler;

impl TokenSampler for DeterministicSampler {
    fn sample(&self, input: SamplerInput) -> Vec<SampledToken> {
        input
            .requests
            .into_iter()
            .map(|request| {
                let sampled_from_logits = request.logits.as_ref().and_then(|logits| {
                    sample_argmax_with_logprob_and_repetition_penalty(
                        logits,
                        request.sampling_params.repetition_penalty,
                        &request.recent_tokens,
                    )
                });
                let invalid_logits = request.logits.is_some() && sampled_from_logits.is_none();
                let token_id = if invalid_logits {
                    0
                } else {
                    sampled_from_logits
                        .map(|(token_id, _)| token_id)
                        .unwrap_or_else(|| request.previous_token.saturating_add(1))
                };
                let logprob = sampled_from_logits
                    .map(|(_, logprob)| logprob)
                    .or_else(|| (!invalid_logits).then_some(0.0));
                let stop_reason = if invalid_logits {
                    Some(StopReason::Error)
                } else if request.generated_len.saturating_add(1) >= request.max_output_tokens {
                    Some(StopReason::MaxOutputTokens)
                } else {
                    None
                };

                SampledToken {
                    request_id: request.request_id,
                    token_id,
                    stop_reason,
                    logprob,
                }
            })
            .collect()
    }
}

pub(crate) fn sample_argmax_with_logprob(logits: &[f32]) -> Option<(u32, f32)> {
    sample_argmax_with_logprob_and_repetition_penalty(logits, 1.0, &[])
}

fn sample_argmax_with_logprob_and_repetition_penalty(
    logits: &[f32],
    repetition_penalty: f32,
    recent_tokens: &[u32],
) -> Option<(u32, f32)> {
    let adjusted_logits;
    let logits = if repetition_penalty_applies(repetition_penalty, recent_tokens) {
        adjusted_logits = logits_with_repetition_penalty(logits, repetition_penalty, recent_tokens);
        adjusted_logits.as_slice()
    } else {
        logits
    };

    let (best_index, &best_logit) = logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))?;
    if !best_logit.is_finite() {
        return None;
    }

    let normalizer = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| (value - best_logit).exp())
        .sum::<f32>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return None;
    }

    Some((best_index as u32, -normalizer.ln()))
}

pub(crate) fn recent_repetition_tokens(
    prompt_tokens: &[u32],
    generated_tokens: &[u32],
    context_size: Option<u32>,
) -> Vec<u32> {
    let total_len = prompt_tokens.len().saturating_add(generated_tokens.len());
    let keep_len = context_size
        .map(|size| size as usize)
        .unwrap_or(total_len)
        .min(total_len);
    if keep_len == 0 {
        return Vec::new();
    }

    let start = total_len - keep_len;
    let mut tokens = Vec::with_capacity(keep_len);
    if start < prompt_tokens.len() {
        tokens.extend_from_slice(&prompt_tokens[start..]);
        tokens.extend_from_slice(generated_tokens);
    } else {
        tokens.extend_from_slice(&generated_tokens[start - prompt_tokens.len()..]);
    }
    tokens
}

fn repetition_penalty_applies(repetition_penalty: f32, recent_tokens: &[u32]) -> bool {
    repetition_penalty.is_finite()
        && repetition_penalty > 0.0
        && repetition_penalty != 1.0
        && !recent_tokens.is_empty()
}

fn logits_with_repetition_penalty(
    logits: &[f32],
    repetition_penalty: f32,
    recent_tokens: &[u32],
) -> Vec<f32> {
    let mut adjusted = logits.to_vec();
    let mut seen_tokens = BTreeSet::new();
    for token in recent_tokens {
        if !seen_tokens.insert(*token) {
            continue;
        }
        let Some(logit) = adjusted.get_mut(*token as usize) else {
            continue;
        };
        if *logit < 0.0 {
            *logit *= repetition_penalty;
        } else {
            *logit /= repetition_penalty;
        }
    }
    adjusted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sampler_marks_max_output_boundary() {
        let sampler = DeterministicSampler;
        let sampled = sampler.sample(SamplerInput {
            requests: vec![
                SamplerRequest {
                    request_id: RequestId(1),
                    previous_token: 10,
                    logits: None,
                    recent_tokens: Vec::new(),
                    generated_len: 0,
                    max_output_tokens: 1,
                    sampling_params: SamplingParams::default(),
                },
                SamplerRequest {
                    request_id: RequestId(2),
                    previous_token: 20,
                    logits: None,
                    recent_tokens: Vec::new(),
                    generated_len: 0,
                    max_output_tokens: 4,
                    sampling_params: SamplingParams::default(),
                },
            ],
        });

        assert_eq!(sampled[0].token_id, 11);
        assert_eq!(sampled[0].stop_reason, Some(StopReason::MaxOutputTokens));
        assert_eq!(sampled[1].token_id, 21);
        assert_eq!(sampled[1].stop_reason, None);
    }

    #[test]
    fn deterministic_sampler_prefers_argmax_logits_when_present() {
        let sampler = DeterministicSampler;
        let sampled = sampler.sample(SamplerInput {
            requests: vec![SamplerRequest {
                request_id: RequestId(1),
                previous_token: 99,
                logits: Some(vec![0.25, 1.5, -0.5, 0.75]),
                recent_tokens: Vec::new(),
                generated_len: 0,
                max_output_tokens: 4,
                sampling_params: SamplingParams::default(),
            }],
        });

        assert_eq!(sampled[0].token_id, 1);
        assert_eq!(sampled[0].stop_reason, None);
        assert!(
            sampled[0]
                .logprob
                .is_some_and(|logprob| logprob.is_finite() && logprob < 0.0)
        );
    }

    #[test]
    fn deterministic_sampler_marks_non_finite_logits_as_error() {
        let sampler = DeterministicSampler;
        let sampled = sampler.sample(SamplerInput {
            requests: vec![SamplerRequest {
                request_id: RequestId(1),
                previous_token: 99,
                logits: Some(vec![f32::NAN, f32::INFINITY]),
                recent_tokens: Vec::new(),
                generated_len: 0,
                max_output_tokens: 4,
                sampling_params: SamplingParams::default(),
            }],
        });

        assert_eq!(sampled[0].token_id, 0);
        assert_eq!(sampled[0].stop_reason, Some(StopReason::Error));
        assert_eq!(sampled[0].logprob, None);
    }

    #[test]
    fn deterministic_argmax_fast_path_only_allows_default_greedy_sampling() {
        assert!(sampling_params_allow_deterministic_argmax_fast_path(
            &SamplingParams::default()
        ));

        let params = SamplingParams {
            temperature: 0.7,
            ..SamplingParams::default()
        };
        assert!(!sampling_params_allow_deterministic_argmax_fast_path(
            &params
        ));
    }

    #[test]
    fn deterministic_sampler_applies_repetition_penalty_to_recent_tokens() {
        let sampler = DeterministicSampler;
        let sampling_params = SamplingParams {
            repetition_penalty: 2.0,
            ..SamplingParams::default()
        };
        let sampled = sampler.sample(SamplerInput {
            requests: vec![SamplerRequest {
                request_id: RequestId(1),
                previous_token: 99,
                logits: Some(vec![1.0, 1.8, 0.9]),
                recent_tokens: vec![1],
                generated_len: 0,
                max_output_tokens: 4,
                sampling_params,
            }],
        });

        assert_eq!(sampled[0].token_id, 0);
    }

    #[test]
    fn recent_repetition_tokens_respects_context_size_across_prompt_and_generated() {
        assert_eq!(
            recent_repetition_tokens(&[1, 2, 3], &[4, 5], Some(3)),
            vec![3, 4, 5]
        );
        assert_eq!(
            recent_repetition_tokens(&[1, 2, 3], &[4, 5], Some(1)),
            vec![5]
        );
    }
}
