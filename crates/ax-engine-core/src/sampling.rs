use std::fmt;

use crate::ids::RequestId;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
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
                let sampled_from_logits = request
                    .logits
                    .as_ref()
                    .and_then(|logits| sample_argmax_with_logprob(logits));
                let token_id = sampled_from_logits
                    .map(|(token_id, _)| token_id)
                    .unwrap_or_else(|| request.previous_token.saturating_add(1));
                let logprob = sampled_from_logits
                    .map(|(_, logprob)| logprob)
                    .or(Some(0.0));
                let stop_reason =
                    if request.generated_len.saturating_add(1) >= request.max_output_tokens {
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
    let (best_index, &best_logit) = logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))?;
    if !best_logit.is_finite() {
        return None;
    }

    let max_logit = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp)?;
    let normalizer = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| (value - max_logit).exp())
        .sum::<f32>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return None;
    }

    Some((best_index as u32, best_logit - max_logit - normalizer.ln()))
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
                    generated_len: 0,
                    max_output_tokens: 1,
                    sampling_params: SamplingParams::default(),
                },
                SamplerRequest {
                    request_id: RequestId(2),
                    previous_token: 20,
                    logits: None,
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
                generated_len: 0,
                max_output_tokens: 4,
                sampling_params: SamplingParams::default(),
            }],
        });

        assert_eq!(sampled[0].token_id, 1);
        assert_eq!(sampled[0].stop_reason, None);
        assert!(sampled[0]
            .logprob
            .is_some_and(|logprob| logprob.is_finite() && logprob < 0.0));
    }

    #[test]
    fn deterministic_argmax_fast_path_only_allows_default_greedy_sampling() {
        assert!(sampling_params_allow_deterministic_argmax_fast_path(
            &SamplingParams::default()
        ));

        let mut params = SamplingParams::default();
        params.temperature = 0.7;
        assert!(!sampling_params_allow_deterministic_argmax_fast_path(
            &params
        ));
    }
}
