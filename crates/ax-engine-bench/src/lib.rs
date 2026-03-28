use std::path::Path;

use ax_engine_core::gguf::MappedModel;
use ax_engine_core::memory::MemoryBudget;
use ax_engine_core::model::LlamaModel;

pub mod baseline;
pub mod microbench;
pub mod parity;
pub mod perf;
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
}
