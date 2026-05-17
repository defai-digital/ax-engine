use crate::generate::GenerateRouteReport;
use crate::request::{EngineStepReport, SessionRequestReport};

use super::LLAMA_CPP_STREAM_EXECUTION_PLAN;

// Crossover-decision keys that monotonically grow within a single Session.generate()
// call and would be silently zeroed by a plain last-step-wins merge. Prefix-reuse
// counters land on the prefill step only; without max-merge they are overwritten by
// the subsequent decode steps, which is the bug the W1 evidence report identified.
const MONOTONIC_CROSSOVER_DECISION_KEYS: &[&str] = &[
    "prefix_reused_requests",
    "live_share_hits",
    "retained_cache_hits",
    "prefix_reused_blocks",
    "prefix_reused_tokens",
    "blocked_prefix_reuse_requests",
    "blocked_prefix_reuse_blocks",
    "blocked_prefix_reuse_tokens",
    "max_prefix_blocks_reused_per_request",
    "branch_prefill_requests",
    "branch_decode_requests",
    "branch_prefill_tail_tokens",
    "branch_decode_tokens",
];

// Apply per-step route metadata onto the route accumulated so far for the same
// request. String fields are last-wins, except `prefix_cache_path` keeps a more
// informative stored value rather than being clobbered by the decode-step default
// of "metadata_lookup". Crossover decisions keyed by `MONOTONIC_CROSSOVER_DECISION_KEYS`
// are merged via max(); all others fall back to last-wins to preserve current
// behaviour for non-prefix-reuse telemetry.
pub(super) fn merge_native_route_into(stored: &mut GenerateRouteReport, new: GenerateRouteReport) {
    if new.execution_plan.is_some() {
        stored.execution_plan = new.execution_plan;
    }
    if new.attention_route.is_some() {
        stored.attention_route = new.attention_route;
    }
    if new.kv_mode.is_some() {
        stored.kv_mode = new.kv_mode;
    }
    if new.barrier_mode.is_some() {
        stored.barrier_mode = new.barrier_mode;
    }
    match (
        stored.prefix_cache_path.as_deref(),
        new.prefix_cache_path.as_deref(),
    ) {
        (Some(existing), Some("metadata_lookup")) if existing != "metadata_lookup" => {
            // Keep the more informative stored value.
        }
        (_, Some(_)) => stored.prefix_cache_path = new.prefix_cache_path,
        _ => {}
    }
    for (key, new_val) in new.crossover_decisions {
        if MONOTONIC_CROSSOVER_DECISION_KEYS.contains(&key.as_str()) {
            let slot = stored.crossover_decisions.entry(key).or_insert(0);
            *slot = (*slot).max(new_val);
        } else {
            stored.crossover_decisions.insert(key, new_val);
        }
    }
}

pub(super) fn llama_cpp_stream_route() -> GenerateRouteReport {
    GenerateRouteReport::with_execution_plan(LLAMA_CPP_STREAM_EXECUTION_PLAN)
}

pub(super) fn apply_native_step_route_to_report(
    report: &mut SessionRequestReport,
    step: &EngineStepReport,
) {
    if let Some(route) = step.route.as_ref() {
        merge_native_route_into(&mut report.route, route.clone());
    }
}
