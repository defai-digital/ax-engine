use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_sdk::{GenerateFinishReason, GenerateRequest, GenerateSampling, GenerateStreamEvent};

use super::proto;
use crate::app_state::LiveState;

pub(super) fn proto_sampling_to_sdk(s: proto::GenerateSampling) -> GenerateSampling {
    GenerateSampling {
        temperature: s.temperature,
        top_p: if s.top_p == 0.0 { 1.0 } else { s.top_p },
        top_k: s.top_k,
        min_p: None,
        repetition_penalty: if s.repetition_penalty == 0.0 {
            1.0
        } else {
            s.repetition_penalty
        },
        repetition_context_size: None,
        seed: s.seed,
        deterministic: None,
        ignore_eos: false,
    }
}

pub(super) fn proto_to_generate_request(
    live: &LiveState,
    req: proto::GenerateRequest,
) -> GenerateRequest {
    let sampling = req.sampling.map(proto_sampling_to_sdk).unwrap_or_default();
    GenerateRequest {
        model_id: live.model_id.to_string(),
        input_tokens: req.input_tokens,
        input_text: if req.input_text.is_empty() {
            None
        } else {
            Some(req.input_text)
        },
        multimodal_inputs: Default::default(),
        max_output_tokens: if req.max_output_tokens == 0 {
            256
        } else {
            req.max_output_tokens
        },
        sampling,
        stop_sequences: Vec::new(),
        metadata: if req.metadata.is_empty() {
            None
        } else {
            Some(req.metadata)
        },
    }
}

pub(super) fn sdk_response_to_proto(r: ax_engine_sdk::GenerateResponse) -> proto::GenerateResponse {
    proto::GenerateResponse {
        request_id: r.request_id,
        model_id: r.model_id,
        prompt_tokens: r.prompt_tokens,
        output_tokens: r.output_tokens,
        output_text: r.output_text.unwrap_or_default(),
        status: format!("{:?}", r.status),
        finish_reason: r.finish_reason.map(finish_reason_str).unwrap_or_default(),
        step_count: r.step_count,
    }
}

fn sdk_request_report_to_proto(r: &ax_engine_sdk::SessionRequestReport) -> proto::RequestReport {
    proto::RequestReport {
        request_id: r.request_id,
        model_id: r.model_id.clone(),
        state: format!("{:?}", r.state),
        prompt_tokens: r.prompt_tokens.clone(),
        processed_prompt_tokens: r.processed_prompt_tokens,
        output_tokens: r.output_tokens.clone(),
        prompt_len: r.prompt_len,
        output_len: r.output_len,
        max_output_tokens: r.max_output_tokens,
        cancel_requested: r.cancel_requested,
    }
}

pub(super) fn sdk_stream_event_to_proto(event: GenerateStreamEvent) -> proto::GenerateStreamEvent {
    match event {
        GenerateStreamEvent::Request(payload) => proto::GenerateStreamEvent {
            event: "request".to_string(),
            request: Some(sdk_request_report_to_proto(&payload.request)),
            step: None,
            response: None,
        },
        GenerateStreamEvent::Step(payload) => proto::GenerateStreamEvent {
            event: "step".to_string(),
            request: None,
            step: Some(proto::GenerateStepEvent {
                request: Some(sdk_request_report_to_proto(&payload.request)),
                step: Some(proto::StepReport {
                    scheduled_requests: payload.step.scheduled_requests,
                    scheduled_tokens: payload.step.scheduled_tokens,
                    ttft_events: payload.step.ttft_events,
                    prefix_hits: payload.step.prefix_hits,
                    kv_usage_blocks: payload.step.kv_usage_blocks,
                    evictions: payload.step.evictions,
                    cpu_time_us: payload.step.cpu_time_us,
                    runner_time_us: payload.step.runner_time_us,
                }),
                delta_tokens: payload.delta_tokens,
                delta_text: payload.delta_text.unwrap_or_default(),
            }),
            response: None,
        },
        GenerateStreamEvent::Response(payload) => proto::GenerateStreamEvent {
            event: "response".to_string(),
            request: None,
            step: None,
            response: Some(sdk_response_to_proto(payload.response)),
        },
    }
}

pub(super) fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub(super) fn finish_reason_str(fr: GenerateFinishReason) -> String {
    match fr {
        GenerateFinishReason::Stop => "stop".to_string(),
        GenerateFinishReason::MaxOutputTokens => "length".to_string(),
        GenerateFinishReason::ContentFilter => "content_filter".to_string(),
        GenerateFinishReason::Cancelled => "cancelled".to_string(),
        GenerateFinishReason::Error => "error".to_string(),
    }
}
