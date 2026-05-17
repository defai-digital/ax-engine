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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_native_route_preserves_prefix_reuse_counters_across_steps() {
        // Simulate a Session.generate() call: prefill step writes prefix-reuse
        // telemetry; subsequent decode steps land with zero prefix-reuse values
        // and a default "metadata_lookup" prefix_cache_path. Without the merge
        // fix, the prefill values are silently overwritten.
        let mut stored = GenerateRouteReport {
            execution_plan: Some("native.mlx.qwen3".to_string()),
            attention_route: Some("mlx_native".to_string()),
            kv_mode: Some("paged".to_string()),
            prefix_cache_path: Some("retained_prompt_prefix_cache".to_string()),
            barrier_mode: Some("eager".to_string()),
            crossover_decisions: [
                ("retained_cache_hits".to_string(), 1u32),
                ("prefix_reused_blocks".to_string(), 32),
                ("prefix_reused_tokens".to_string(), 512),
                ("ax_mlx_kv_logical_tokens".to_string(), 519),
            ]
            .into_iter()
            .collect(),
        };
        let decode_step = GenerateRouteReport {
            execution_plan: Some("native.mlx.qwen3".to_string()),
            attention_route: Some("mlx_native".to_string()),
            kv_mode: Some("paged".to_string()),
            prefix_cache_path: Some("metadata_lookup".to_string()),
            barrier_mode: Some("eager".to_string()),
            crossover_decisions: [
                ("retained_cache_hits".to_string(), 0u32),
                ("prefix_reused_blocks".to_string(), 0),
                ("prefix_reused_tokens".to_string(), 0),
                ("ax_mlx_kv_logical_tokens".to_string(), 520),
            ]
            .into_iter()
            .collect(),
        };

        merge_native_route_into(&mut stored, decode_step);

        assert_eq!(
            stored.prefix_cache_path.as_deref(),
            Some("retained_prompt_prefix_cache"),
            "decode-step default should not clobber a real prefix_cache_path"
        );
        assert_eq!(
            stored.crossover_decisions.get("retained_cache_hits"),
            Some(&1)
        );
        assert_eq!(
            stored.crossover_decisions.get("prefix_reused_blocks"),
            Some(&32)
        );
        assert_eq!(
            stored.crossover_decisions.get("prefix_reused_tokens"),
            Some(&512)
        );
        assert_eq!(
            stored.crossover_decisions.get("ax_mlx_kv_logical_tokens"),
            Some(&520),
            "non-monotonic counters take the latest value"
        );
    }

    #[test]
    fn merge_native_route_takes_max_for_monotonic_counters() {
        let mut stored = GenerateRouteReport {
            crossover_decisions: [("prefix_reused_blocks".to_string(), 4u32)]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let new = GenerateRouteReport {
            crossover_decisions: [("prefix_reused_blocks".to_string(), 32u32)]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        merge_native_route_into(&mut stored, new);
        assert_eq!(
            stored.crossover_decisions.get("prefix_reused_blocks"),
            Some(&32)
        );

        let smaller = GenerateRouteReport {
            crossover_decisions: [("prefix_reused_blocks".to_string(), 1u32)]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        merge_native_route_into(&mut stored, smaller);
        assert_eq!(
            stored.crossover_decisions.get("prefix_reused_blocks"),
            Some(&32),
            "smaller monotonic value must not overwrite the max"
        );
    }

    #[test]
    fn merge_native_route_upgrades_metadata_lookup_to_real_path() {
        let mut stored = GenerateRouteReport {
            prefix_cache_path: Some("metadata_lookup".to_string()),
            ..Default::default()
        };
        let new = GenerateRouteReport {
            prefix_cache_path: Some("retained_prompt_prefix_cache".to_string()),
            ..Default::default()
        };
        merge_native_route_into(&mut stored, new);
        assert_eq!(
            stored.prefix_cache_path.as_deref(),
            Some("retained_prompt_prefix_cache")
        );
    }
}
