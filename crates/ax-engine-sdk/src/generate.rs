use std::collections::BTreeMap;

use ax_engine_core::{RequestSnapshot, RequestState, RouteMetadata, SamplingParams, StopReason};
use serde::{Deserialize, Serialize};

use crate::backend::RuntimeReport;
use crate::request::{EngineStepReport, SessionRequestReport, SessionRequestState};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateRequest {
    pub model_id: String,
    pub input_tokens: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_text: Option<String>,
    pub max_output_tokens: u32,
    #[serde(default)]
    pub sampling: GenerateSampling,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateResponse {
    pub request_id: u64,
    pub model_id: String,
    pub prompt_tokens: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_text: Option<String>,
    pub output_tokens: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_token_logprobs: Vec<Option<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_text: Option<String>,
    /// Token count reported by a compatibility backend when token arrays are not available.
    /// When set, takes precedence over `prompt_tokens.len()` for usage reporting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_token_count: Option<u32>,
    /// Token count reported by a compatibility backend when token arrays are not available.
    /// When set, takes precedence over `output_tokens.len()` for usage reporting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_count: Option<u32>,
    pub status: GenerateStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<GenerateFinishReason>,
    pub step_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_step: Option<u64>,
    pub route: GenerateRouteReport,
    pub runtime: RuntimeReport,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateStreamRequestEvent {
    pub request: SessionRequestReport,
    pub runtime: RuntimeReport,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateStreamStepEvent {
    pub request: SessionRequestReport,
    pub step: EngineStepReport,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub delta_tokens: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub delta_token_logprobs: Vec<Option<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_text: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateStreamResponseEvent {
    pub response: GenerateResponse,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", tag = "event", content = "data")]
#[allow(clippy::large_enum_variant)]
pub enum GenerateStreamEvent {
    Request(GenerateStreamRequestEvent),
    Step(GenerateStreamStepEvent),
    Response(GenerateStreamResponseEvent),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GenerateRouteReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_plan: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_route: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub barrier_mode: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub crossover_decisions: BTreeMap<String, u32>,
}

impl GenerateRouteReport {
    pub fn from_route(route: &RouteMetadata) -> Self {
        let crossover_decisions = route
            .crossover_decisions
            .iter()
            .cloned()
            .collect::<BTreeMap<_, _>>();

        Self {
            execution_plan: route.execution_plan.clone(),
            attention_route: route.attention_route.clone(),
            kv_mode: route.kv_mode.clone(),
            prefix_cache_path: route.prefix_cache_path.clone(),
            barrier_mode: route.barrier_mode.clone(),
            crossover_decisions,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerateStatus {
    Finished,
    Cancelled,
    Failed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerateFinishReason {
    Stop,
    MaxOutputTokens,
    Cancelled,
    Error,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateSampling {
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: u32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub deterministic: Option<bool>,
}

impl Default for GenerateSampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: 0,
            deterministic: None,
        }
    }
}

impl GenerateSampling {
    pub fn into_core(self, default_deterministic: bool) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
            deterministic: self.deterministic.unwrap_or(default_deterministic),
        }
    }
}

impl GenerateResponse {
    /// Returns a prompt token count only when AX has an authoritative source for it:
    /// either explicit backend-reported usage or a concrete prompt token array.
    pub fn known_prompt_token_count(&self) -> Option<u32> {
        self.prompt_token_count.or_else(|| {
            if !self.prompt_tokens.is_empty() || self.prompt_text.is_none() {
                Some(self.prompt_tokens.len() as u32)
            } else {
                None
            }
        })
    }

    /// Returns a completion token count only when AX has an authoritative source for it:
    /// either explicit backend-reported usage or a concrete output token array.
    pub fn known_output_token_count(&self) -> Option<u32> {
        self.output_token_count.or_else(|| {
            if !self.output_tokens.is_empty() || self.output_text.is_none() {
                Some(self.output_tokens.len() as u32)
            } else {
                None
            }
        })
    }

    pub fn known_usage(&self) -> Option<(u32, u32)> {
        Some((
            self.known_prompt_token_count()?,
            self.known_output_token_count()?,
        ))
    }

    pub fn from_snapshot(
        snapshot: RequestSnapshot,
        step_count: u64,
        ttft_step: Option<u64>,
        runtime: RuntimeReport,
    ) -> Self {
        let status = GenerateStatus::from_request_state(snapshot.state);
        let finish_reason =
            GenerateFinishReason::from_request_state(snapshot.state, snapshot.terminal_stop_reason);

        Self {
            request_id: snapshot.request_id.0,
            model_id: snapshot.model_id.0,
            prompt_tokens: snapshot.prompt_tokens,
            prompt_text: None,
            output_tokens: snapshot.generated_tokens,
            output_token_logprobs: snapshot.generated_token_logprobs,
            output_text: None,
            prompt_token_count: None,
            output_token_count: None,
            status,
            finish_reason,
            step_count,
            ttft_step,
            route: GenerateRouteReport::from_route(&snapshot.route_metadata_hint),
            runtime,
        }
    }

    pub fn from_report(
        report: SessionRequestReport,
        step_count: u64,
        ttft_step: Option<u64>,
        runtime: RuntimeReport,
    ) -> Self {
        let status = GenerateStatus::from_session_state(report.state);
        let finish_reason = report.finish_reason.or_else(|| {
            GenerateFinishReason::from_session_state(report.state, report.terminal_stop_reason)
        });

        Self {
            request_id: report.request_id,
            model_id: report.model_id,
            prompt_tokens: report.prompt_tokens,
            prompt_text: None,
            output_tokens: report.output_tokens,
            output_token_logprobs: report.output_token_logprobs,
            output_text: None,
            prompt_token_count: None,
            output_token_count: None,
            status,
            finish_reason,
            step_count,
            ttft_step,
            route: report.route,
            runtime,
        }
    }
}

impl GenerateStreamEvent {
    pub fn event_name(&self) -> &'static str {
        match self {
            Self::Request(_) => "request",
            Self::Step(_) => "step",
            Self::Response(_) => "response",
        }
    }
}

impl GenerateStatus {
    pub fn from_request_state(state: RequestState) -> Self {
        match state {
            RequestState::Finished => Self::Finished,
            RequestState::Cancelled => Self::Cancelled,
            RequestState::Failed => Self::Failed,
            RequestState::Waiting
            | RequestState::Runnable
            | RequestState::BlockedOnMemory
            | RequestState::Running => Self::Failed,
        }
    }

    pub fn from_session_state(state: SessionRequestState) -> Self {
        match state {
            SessionRequestState::Finished => Self::Finished,
            SessionRequestState::Cancelled => Self::Cancelled,
            SessionRequestState::Failed => Self::Failed,
            SessionRequestState::Waiting
            | SessionRequestState::Runnable
            | SessionRequestState::BlockedOnMemory
            | SessionRequestState::Running => Self::Failed,
        }
    }
}

impl GenerateFinishReason {
    pub fn from_request_state(
        state: RequestState,
        terminal_stop_reason: Option<StopReason>,
    ) -> Option<Self> {
        match state {
            RequestState::Finished => match terminal_stop_reason {
                Some(StopReason::EosToken) => Some(Self::Stop),
                Some(StopReason::MaxOutputTokens) | None => Some(Self::MaxOutputTokens),
                Some(StopReason::Cancelled) => Some(Self::Cancelled),
                Some(StopReason::Error) => Some(Self::Error),
            },
            RequestState::Cancelled => Some(Self::Cancelled),
            RequestState::Failed => Some(Self::Error),
            RequestState::Waiting
            | RequestState::Runnable
            | RequestState::BlockedOnMemory
            | RequestState::Running => None,
        }
    }

    pub fn from_session_state(
        state: SessionRequestState,
        terminal_stop_reason: Option<StopReason>,
    ) -> Option<Self> {
        match state {
            SessionRequestState::Finished => match terminal_stop_reason {
                Some(StopReason::EosToken) => Some(Self::Stop),
                Some(StopReason::MaxOutputTokens) | None => Some(Self::MaxOutputTokens),
                Some(StopReason::Cancelled) => Some(Self::Cancelled),
                Some(StopReason::Error) => Some(Self::Error),
            },
            SessionRequestState::Cancelled => Some(Self::Cancelled),
            SessionRequestState::Failed => Some(Self::Error),
            SessionRequestState::Waiting
            | SessionRequestState::Runnable
            | SessionRequestState::BlockedOnMemory
            | SessionRequestState::Running => None,
        }
    }
}

const fn default_top_p() -> f32 {
    1.0
}

const fn default_repetition_penalty() -> f32 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{CapabilityReport, ResolutionPolicy, SelectedBackend, SupportTier};
    use crate::request::{SessionRequestReport, SessionRequestState};
    use ax_engine_core::StopReason;

    #[test]
    fn route_report_preserves_crossover_decisions() {
        let route = RouteMetadata {
            execution_plan: Some("phase1.qwen3_dense".to_string()),
            attention_route: Some("paged_decode".to_string()),
            kv_mode: Some("paged".to_string()),
            prefix_cache_path: Some("live_request_share".to_string()),
            barrier_mode: None,
            crossover_decisions: vec![("prefix_reused_tokens".to_string(), 64)],
        };

        let report = GenerateRouteReport::from_route(&route);

        assert_eq!(
            report.crossover_decisions.get("prefix_reused_tokens"),
            Some(&64)
        );
        assert_eq!(
            report.prefix_cache_path.as_deref(),
            Some("live_request_share")
        );
    }

    #[test]
    fn response_from_report_preserves_terminal_metadata() {
        let response = GenerateResponse::from_report(
            SessionRequestReport {
                request_id: 9,
                model_id: "qwen3_dense".to_string(),
                state: SessionRequestState::Finished,
                prompt_tokens: vec![1, 2, 3],
                processed_prompt_tokens: 3,
                output_tokens: vec![4, 5],
                output_token_logprobs: vec![Some(-0.25), Some(-0.5)],
                prompt_len: 3,
                output_len: 2,
                max_output_tokens: 2,
                cancel_requested: false,
                execution_plan_ref: Some("phase1.qwen3_dense.paged_decode".to_string()),
                route: GenerateRouteReport {
                    execution_plan: Some("phase1.qwen3_dense.paged_decode".to_string()),
                    attention_route: Some("qwen3_dense_paged_decode".to_string()),
                    kv_mode: Some("paged_metadata".to_string()),
                    prefix_cache_path: None,
                    barrier_mode: Some("serial".to_string()),
                    crossover_decisions: BTreeMap::new(),
                },
                finish_reason: Some(GenerateFinishReason::MaxOutputTokens),
                terminal_stop_reason: Some(StopReason::MaxOutputTokens),
                last_error: None,
            },
            3,
            Some(2),
            RuntimeReport {
                selected_backend: SelectedBackend::AxNative,
                support_tier: SupportTier::NativePreview,
                resolution_policy: ResolutionPolicy::StrictNative,
                capabilities: CapabilityReport::native_preview(),
                fallback_reason: None,
                host: Default::default(),
                metal_toolchain: Default::default(),
                native_runtime: None,
                native_model: None,
            },
        );

        assert_eq!(response.request_id, 9);
        assert_eq!(response.status, GenerateStatus::Finished);
        assert_eq!(
            response.finish_reason,
            Some(GenerateFinishReason::MaxOutputTokens)
        );
        assert_eq!(response.step_count, 3);
        assert_eq!(response.ttft_step, Some(2));
        assert_eq!(response.output_tokens, vec![4, 5]);
        assert_eq!(
            response.output_token_logprobs,
            vec![Some(-0.25), Some(-0.5)]
        );
        assert_eq!(response.prompt_text, None);
        assert_eq!(response.output_text, None);
    }

    #[test]
    fn response_from_report_preserves_stop_finish_reason() {
        let response = GenerateResponse::from_report(
            SessionRequestReport {
                request_id: 10,
                model_id: "qwen3_dense".to_string(),
                state: SessionRequestState::Finished,
                prompt_tokens: vec![1, 2, 3],
                processed_prompt_tokens: 3,
                output_tokens: vec![4],
                output_token_logprobs: vec![Some(-0.125)],
                prompt_len: 3,
                output_len: 1,
                max_output_tokens: 4,
                cancel_requested: false,
                execution_plan_ref: None,
                route: GenerateRouteReport {
                    execution_plan: None,
                    attention_route: None,
                    kv_mode: None,
                    prefix_cache_path: None,
                    barrier_mode: None,
                    crossover_decisions: BTreeMap::new(),
                },
                finish_reason: Some(GenerateFinishReason::Stop),
                terminal_stop_reason: Some(StopReason::EosToken),
                last_error: None,
            },
            2,
            Some(2),
            RuntimeReport {
                selected_backend: SelectedBackend::AxNative,
                support_tier: SupportTier::NativePreview,
                resolution_policy: ResolutionPolicy::StrictNative,
                capabilities: CapabilityReport::native_preview(),
                fallback_reason: None,
                host: Default::default(),
                metal_toolchain: Default::default(),
                native_runtime: None,
                native_model: None,
            },
        );

        assert_eq!(response.finish_reason, Some(GenerateFinishReason::Stop));
    }

    #[test]
    fn known_usage_uses_token_arrays_for_token_native_paths() {
        let response = GenerateResponse {
            request_id: 1,
            model_id: "qwen3_dense".to_string(),
            prompt_tokens: vec![1, 2, 3],
            prompt_text: None,
            output_tokens: vec![4, 5],
            output_token_logprobs: vec![None, None],
            output_text: None,
            prompt_token_count: None,
            output_token_count: None,
            status: GenerateStatus::Finished,
            finish_reason: Some(GenerateFinishReason::Stop),
            step_count: 2,
            ttft_step: Some(1),
            route: GenerateRouteReport {
                execution_plan: None,
                attention_route: None,
                kv_mode: None,
                prefix_cache_path: None,
                barrier_mode: None,
                crossover_decisions: BTreeMap::new(),
            },
            runtime: RuntimeReport {
                selected_backend: SelectedBackend::AxNative,
                support_tier: SupportTier::NativePreview,
                resolution_policy: ResolutionPolicy::StrictNative,
                capabilities: CapabilityReport::native_preview(),
                fallback_reason: None,
                host: Default::default(),
                metal_toolchain: Default::default(),
                native_runtime: None,
                native_model: None,
            },
        };

        assert_eq!(response.known_prompt_token_count(), Some(3));
        assert_eq!(response.known_output_token_count(), Some(2));
        assert_eq!(response.known_usage(), Some((3, 2)));
    }

    #[test]
    fn known_usage_prefers_backend_reported_counts_for_text_backends() {
        let response = GenerateResponse {
            request_id: 2,
            model_id: "qwen3_dense".to_string(),
            prompt_tokens: Vec::new(),
            prompt_text: Some("hello".to_string()),
            output_tokens: Vec::new(),
            output_token_logprobs: Vec::new(),
            output_text: Some("world".to_string()),
            prompt_token_count: Some(7),
            output_token_count: Some(3),
            status: GenerateStatus::Finished,
            finish_reason: Some(GenerateFinishReason::Stop),
            step_count: 0,
            ttft_step: None,
            route: GenerateRouteReport {
                execution_plan: None,
                attention_route: None,
                kv_mode: None,
                prefix_cache_path: None,
                barrier_mode: None,
                crossover_decisions: BTreeMap::new(),
            },
            runtime: RuntimeReport {
                selected_backend: SelectedBackend::Mlx,
                support_tier: SupportTier::Compatibility,
                resolution_policy: ResolutionPolicy::AllowCompat,
                capabilities: CapabilityReport::compatibility_baseline(),
                fallback_reason: Some("native preview not ready".to_string()),
                host: Default::default(),
                metal_toolchain: Default::default(),
                native_runtime: None,
                native_model: None,
            },
        };

        assert_eq!(response.known_prompt_token_count(), Some(7));
        assert_eq!(response.known_output_token_count(), Some(3));
        assert_eq!(response.known_usage(), Some((7, 3)));
    }

    #[test]
    fn known_usage_omits_unknown_text_only_counts() {
        let response = GenerateResponse {
            request_id: 3,
            model_id: "qwen3_dense".to_string(),
            prompt_tokens: Vec::new(),
            prompt_text: Some("hello".to_string()),
            output_tokens: Vec::new(),
            output_token_logprobs: Vec::new(),
            output_text: Some("world".to_string()),
            prompt_token_count: None,
            output_token_count: None,
            status: GenerateStatus::Finished,
            finish_reason: Some(GenerateFinishReason::MaxOutputTokens),
            step_count: 0,
            ttft_step: None,
            route: GenerateRouteReport {
                execution_plan: Some("compatibility.mlx.blocking_cli".to_string()),
                attention_route: None,
                kv_mode: None,
                prefix_cache_path: None,
                barrier_mode: None,
                crossover_decisions: BTreeMap::new(),
            },
            runtime: RuntimeReport {
                selected_backend: SelectedBackend::Mlx,
                support_tier: SupportTier::Compatibility,
                resolution_policy: ResolutionPolicy::AllowCompat,
                capabilities: CapabilityReport::compatibility_cli_baseline(),
                fallback_reason: Some("native preview not ready".to_string()),
                host: Default::default(),
                metal_toolchain: Default::default(),
                native_runtime: None,
                native_model: None,
            },
        };

        assert_eq!(response.known_prompt_token_count(), None);
        assert_eq!(response.known_output_token_count(), None);
        assert_eq!(response.known_usage(), None);
    }
}
