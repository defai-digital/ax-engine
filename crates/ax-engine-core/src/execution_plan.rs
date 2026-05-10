use std::fmt;

use crate::request::RequestSnapshot;
use crate::scheduler::{ROUTE_BARRIER_MODE_SERIAL, ROUTE_KV_MODE_PAGED_METADATA, RouteMetadata};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionPlanBinding {
    pub execution_plan_ref: String,
    pub route_metadata: RouteMetadata,
}

pub trait ExecutionPlanResolver: fmt::Debug + Send + Sync {
    fn resolve(&self, snapshot: &RequestSnapshot) -> Option<ExecutionPlanBinding>;
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DeterministicExecutionPlanResolver;

impl ExecutionPlanResolver for DeterministicExecutionPlanResolver {
    fn resolve(&self, snapshot: &RequestSnapshot) -> Option<ExecutionPlanBinding> {
        let phase = if snapshot.processed_prompt_tokens < snapshot.prompt_len {
            "dense_prefill"
        } else if snapshot.generated_len < snapshot.max_output_tokens {
            "paged_decode"
        } else {
            return None;
        };

        let model_slug = model_slug(&snapshot.model_id.0);
        let execution_plan_ref = format!("phase1.{}.{}", model_slug, phase);
        Some(binding_for_phase(&model_slug, &execution_plan_ref, phase))
    }
}

fn binding_for_phase(
    model_slug: &str,
    execution_plan_ref: &str,
    phase: &str,
) -> ExecutionPlanBinding {
    ExecutionPlanBinding {
        execution_plan_ref: execution_plan_ref.to_string(),
        route_metadata: RouteMetadata {
            execution_plan: Some(execution_plan_ref.to_string()),
            attention_route: Some(attention_route_label(model_slug, phase)),
            kv_mode: Some(ROUTE_KV_MODE_PAGED_METADATA.into()),
            prefix_cache_path: None,
            barrier_mode: Some(ROUTE_BARRIER_MODE_SERIAL.into()),
            crossover_decisions: Vec::new(),
        },
    }
}

fn model_slug(model_id: &str) -> String {
    let mut slug = String::with_capacity(model_id.len());
    let mut previous_was_separator = false;

    for ch in model_id.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            previous_was_separator = false;
        } else if !previous_was_separator && !slug.is_empty() {
            slug.push('_');
            previous_was_separator = true;
        }
    }

    slug.trim_matches('_').to_string()
}

fn attention_route_label(model_slug: &str, phase: &str) -> String {
    match phase {
        "dense_prefill" => format!("{model_slug}_prefill"),
        "paged_decode" => format!("{model_slug}_paged_decode"),
        other => format!("{model_slug}_{other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{ModelId, RequestId, SequenceNo};
    use crate::request::{RequestSnapshot, RequestState};

    fn make_snapshot(
        model_id: &str,
        prompt_len: u32,
        processed_prompt_tokens: u32,
        generated_len: u32,
        max_output_tokens: u32,
    ) -> RequestSnapshot {
        RequestSnapshot {
            request_id: RequestId(1),
            model_id: ModelId(model_id.into()),
            arrival_sequence: SequenceNo(1),
            state: RequestState::Runnable,
            prompt_tokens: vec![1; prompt_len as usize],
            processed_prompt_tokens,
            generated_tokens: vec![2; generated_len as usize],
            generated_token_logprobs: vec![Some(0.0); generated_len as usize],
            prompt_len,
            generated_len,
            max_output_tokens,
            cancel_requested: false,
            execution_plan_ref: None,
            route_metadata_hint: RouteMetadata::empty(),
            terminal_stop_reason: None,
            last_error: None,
        }
    }

    #[test]
    fn resolves_prefill_plan_for_unprocessed_prompt_work() {
        let resolver = DeterministicExecutionPlanResolver;
        let binding = resolver
            .resolve(&make_snapshot("Qwen3.5-27B", 8, 4, 0, 16))
            .expect("prefill work should resolve a plan");

        assert_eq!(
            binding.execution_plan_ref,
            "phase1.qwen3_5_27b.dense_prefill"
        );
        assert_eq!(
            binding.route_metadata.attention_route.as_deref(),
            Some("qwen3_5_27b_prefill")
        );
        assert_eq!(
            binding.route_metadata.kv_mode.as_deref(),
            Some("paged_metadata")
        );
        assert_eq!(
            binding.route_metadata.barrier_mode.as_deref(),
            Some("serial")
        );
    }

    #[test]
    fn resolves_decode_plan_after_prompt_is_fully_processed() {
        let resolver = DeterministicExecutionPlanResolver;
        let binding = resolver
            .resolve(&make_snapshot("gemma-4-27b-it", 8, 8, 1, 16))
            .expect("decode work should resolve a plan");

        assert_eq!(
            binding.execution_plan_ref,
            "phase1.gemma_4_27b_it.paged_decode"
        );
        assert_eq!(
            binding.route_metadata.attention_route.as_deref(),
            Some("gemma_4_27b_it_paged_decode")
        );
    }

    #[test]
    fn returns_none_when_request_has_no_remaining_work() {
        let resolver = DeterministicExecutionPlanResolver;

        assert_eq!(resolver.resolve(&make_snapshot("qwen3", 4, 4, 4, 4)), None);
    }
}
