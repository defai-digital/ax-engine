use std::path::Path;

use ax_engine_core::gguf::MappedModel;
use ax_engine_core::memory::MemoryBudget;
use ax_engine_core::model::LlamaModel;

pub mod baseline;
pub mod microbench;
pub mod parity;
pub mod perf;
pub mod prefill_gap;
pub mod prefill_profile;
pub mod profile;
pub mod report;
pub mod soak;

pub(crate) fn configure_backend_for_model(
    backend: &dyn ax_engine_core::backend::Backend,
    model_path: &str,
    mapped: &MappedModel,
) -> anyhow::Result<()> {
    let profile_model_name = Path::new(model_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::to_owned)
        .or_else(|| mapped.header.get_str("general.name").map(str::to_owned))
        .or_else(|| mapped.header.architecture().map(str::to_owned))
        .unwrap_or_else(|| "default".to_string());
    let profile_quant = mapped
        .predominant_quant()
        .map(|dtype| dtype.to_string())
        .unwrap_or_else(|| "default".to_string());
    let profile_architecture = mapped.header.architecture().unwrap_or("default");
    backend.configure_for_model(&profile_model_name, &profile_quant, profile_architecture)
}

pub(crate) fn report_planned_kv_budget(
    mapped: &MappedModel,
    model: &LlamaModel,
) -> anyhow::Result<()> {
    let kv_plan = model.kv_plan();
    let kv_memory = kv_plan.memory_estimate();
    let kv_capacity = kv_plan.capacity_policy();
    let model_bytes = mapped.total_tensor_bytes();

    MemoryBudget::check_combined(model_bytes, kv_memory.initial_bytes as u64)?;
    if let Ok(max_summary) = MemoryBudget::summary(model_bytes, kv_memory.max_bytes as u64)
        && max_summary.required_bytes > max_summary.allowed_bytes
    {
        eprintln!(
            "Warning: model + max planned KV footprint ({:.1}GB) exceeds budget ({:.1}GB). Initial allocation still fits, but long-context growth may pressure memory.",
            max_summary.required_bytes as f64 / 1e9,
            max_summary.allowed_bytes as f64 / 1e9,
        );
    }

    eprintln!(
        "KV plan: {} | Rollback: {} | Capacity {}→+{} up to {} tok | Initial {:.1}MB | Max {:.1}MB",
        kv_plan.summary_label(),
        kv_plan.rollback_policy().label(),
        kv_capacity.initial_tokens,
        kv_capacity.growth_tokens,
        kv_capacity.max_tokens,
        kv_memory.initial_bytes as f64 / 1024.0 / 1024.0,
        kv_memory.max_bytes as f64 / 1024.0 / 1024.0,
    );

    Ok(())
}

pub(crate) fn support_note(mapped: &MappedModel) -> Option<String> {
    mapped.support_note().map(str::to_owned)
}

pub(crate) fn q5k_prefill_mode(prefill_plan: &str) -> Option<String> {
    prefill_plan
        .split_whitespace()
        .find_map(|part| part.strip_prefix("q5k_prefill=").map(str::to_owned))
}

pub(crate) fn prefill_plan_field(prefill_plan: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    prefill_plan
        .split_whitespace()
        .find_map(|part| part.strip_prefix(&prefix).map(str::to_owned))
}

pub(crate) fn prefill_mode(prefill_plan: &str) -> String {
    prefill_plan_field(prefill_plan, "mode").unwrap_or_default()
}

pub(crate) fn prefill_bool_field(prefill_plan: &str, key: &str) -> Option<bool> {
    match prefill_plan_field(prefill_plan, key)?.as_str() {
        "on" | "true" => Some(true),
        "off" | "false" => Some(false),
        _ => None,
    }
}

pub(crate) fn prefill_route_family(prefill_plan: &str) -> String {
    if matches!(
        prefill_plan_field(prefill_plan, "kv").as_deref(),
        Some("qwen35_hybrid")
    ) {
        return "qwen35_hybrid".to_string();
    }
    match prefill_mode(prefill_plan).as_str() {
        "gpu_batch" => "dense_gpu_batch".to_string(),
        "gpu_chunked" => "dense_gpu_chunked".to_string(),
        "serial" => "serial_prefill".to_string(),
        _ => String::new(),
    }
}

pub(crate) fn prefill_route_detail(prefill_plan: &str) -> String {
    match prefill_route_family(prefill_plan).as_str() {
        "qwen35_hybrid" => prefill_plan_field(prefill_plan, "recurrent")
            .unwrap_or_else(|| "backend_owned".to_string()),
        "dense_gpu_batch" => prefill_plan_field(prefill_plan, "attn_route")
            .unwrap_or_else(|| "generic_gpu_batch".to_string()),
        "dense_gpu_chunked" => prefill_plan_field(prefill_plan, "chunk")
            .map(|chunk| format!("chunk_{chunk}"))
            .unwrap_or_else(|| "generic_gpu_chunked".to_string()),
        "serial_prefill" => prefill_plan_field(prefill_plan, "reason")
            .unwrap_or_else(|| "cpu_or_fallback".to_string()),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q5k_prefill_mode_extracts_known_prefill_label() {
        assert_eq!(
            q5k_prefill_mode("mode=gpu_batch kv=f16 attn=local q5k_prefill=small_n"),
            Some("small_n".into())
        );
        assert_eq!(q5k_prefill_mode("mode=gpu_batch kv=f16 attn=local"), None);
    }

    #[test]
    fn test_prefill_route_helpers_extract_dense_metadata() {
        let plan =
            "mode=gpu_batch kv=f16 qkv=fused split_rope=on attn=local attn_route=cache/stable";
        assert_eq!(prefill_mode(plan), "gpu_batch");
        assert_eq!(prefill_plan_field(plan, "qkv"), Some("fused".into()));
        assert_eq!(prefill_bool_field(plan, "split_rope"), Some(true));
        assert_eq!(prefill_route_family(plan), "dense_gpu_batch");
        assert_eq!(prefill_route_detail(plan), "cache/stable");
    }

    #[test]
    fn test_prefill_route_helpers_classify_qwen35_hybrid() {
        let plan = "mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned";
        assert_eq!(prefill_route_family(plan), "qwen35_hybrid");
        assert_eq!(prefill_route_detail(plan), "backend_owned");
    }
}
