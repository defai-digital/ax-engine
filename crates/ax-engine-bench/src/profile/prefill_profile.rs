//! Batched-prefill profiler.
//!
//! Measures the real `forward_batch` path used by throughput-mode inference and
//! records coarse timing around host setup, GPU execution, and readback. This
//! is intentionally separate from the decode hot-path profiler because the
//! performance questions are different: prefill is dominated by one batched GPU
//! pass, not repeated single-token decode steps.

use std::path::Path;
use std::time::Duration;

use ax_engine_core::backend::RuntimePolicy;
use ax_engine_core::gguf::MappedModel;
use ax_engine_core::kv::ModelKv;
use ax_engine_core::metrics::OpBreakdown;
use ax_engine_core::metrics::counters::OpTimer;
use ax_engine_core::model::{InferenceModel, ModelConfig, WeightStore};
use ax_engine_core::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};

pub use crate::arch_config::{
    Qwen35AlphaBetaStorageMode, Qwen35PrefillDTypeAudit, Qwen35RecurrentStateMode,
};

/// Prefill profiler configuration.
#[derive(Debug, Clone)]
pub struct PrefillProfileConfig {
    /// Model path (GGUF file).
    pub model_path: String,
    /// Number of prompt tokens to process.
    pub prompt_tokens: usize,
    /// Number of unprofiled warmup prefills before measurement.
    pub warmup_iters: usize,
    /// Fan the same prompt out across this many Qwen3.5 recurrent slots on a
    /// shared attention timeline. Only valid for qwen35 and values >= 1.
    pub qwen35_shared_timeline_slots: usize,
    /// Optional source recurrent slot for Qwen3.5 shared-timeline prefill.
    pub qwen35_shared_timeline_source_slot: Option<usize>,
    /// Experimental Qwen3.5 recurrent state mode for GPU prefill handoff.
    pub qwen35_recurrent_state_mode: Qwen35RecurrentStateMode,
    /// Experimental Qwen3.5 alpha/beta scratch storage mode for recurrent handoff.
    pub qwen35_alpha_beta_storage_mode: Qwen35AlphaBetaStorageMode,
    /// Prime Qwen3.5 recurrent Metal slot buffers before timed prefill.
    pub qwen35_prime_slot_buffers: bool,
    /// Run one unmeasured prefill on the same KV before timing the measured prefill.
    pub qwen35_prewarm_prefill_same_kv: bool,
    /// Force Qwen3.5 recurrent prefill to bypass model-side GPU QKV handoff and use backend state batch.
    pub qwen35_force_backend_state_batch: bool,
    /// Force a specific local HD128 attention prefill route for benchmarking.
    pub local_hd128_route: LocalPrefillHd128Route,
}

impl Default for PrefillProfileConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt_tokens: 512,
            warmup_iters: 1,
            qwen35_shared_timeline_slots: 1,
            qwen35_shared_timeline_source_slot: None,
            qwen35_recurrent_state_mode: Qwen35RecurrentStateMode::Auto,
            qwen35_alpha_beta_storage_mode: Qwen35AlphaBetaStorageMode::Auto,
            qwen35_prime_slot_buffers: false,
            qwen35_prewarm_prefill_same_kv: false,
            qwen35_force_backend_state_batch: false,
            local_hd128_route: LocalPrefillHd128Route::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LocalPrefillHd128Route {
    #[default]
    Auto,
    AxBc64,
    Fa2SimdHd128,
    Fa2HalfHd128,
}

impl LocalPrefillHd128Route {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::AxBc64 => "ax_bc64",
            Self::Fa2SimdHd128 => "fa2_simd_hd128",
            Self::Fa2HalfHd128 => "fa2_half_hd128",
        }
    }

    pub fn expected_attention_route_prefix(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            Self::AxBc64 => Some("ax_bc64/"),
            Self::Fa2SimdHd128 => Some("fa2_simd_hd128/"),
            Self::Fa2HalfHd128 => Some("fa2_half_hd128/"),
        }
    }
}

fn default_qwen35_shared_timeline_slots() -> usize {
    1
}

fn effective_qwen35_shared_timeline_tokens(prompt_tokens: usize, slot_count: usize) -> usize {
    prompt_tokens.saturating_mul(slot_count.max(1))
}

fn qwen35_prefill_recurrent_state_mode_auto_for_tokens(
    prompt_tokens: usize,
) -> Qwen35RecurrentStateMode {
    if prompt_tokens <= 32 || prompt_tokens >= 96 {
        Qwen35RecurrentStateMode::BackendOwned
    } else {
        Qwen35RecurrentStateMode::CpuAlias
    }
}

fn prepare_qwen35_shared_timeline_source_slot(
    model: &InferenceModel,
    kv: &mut ModelKv,
    source_slot: Option<usize>,
) -> anyhow::Result<()> {
    let Some(source_slot) = source_slot else {
        return Ok(());
    };
    anyhow::ensure!(
        model.arch_name() == "qwen35",
        "qwen35 shared timeline source slot requires a qwen35 model"
    );
    if source_slot == 0 {
        return Ok(());
    }
    kv.clone_qwen35_recurrent_slot(0, source_slot)?;
    Ok(())
}

fn qwen35_effective_source_slot(
    kv: &ModelKv,
    qwen35_shared_timeline_source_slot: Option<usize>,
) -> anyhow::Result<usize> {
    if let Some(source_slot) = qwen35_shared_timeline_source_slot {
        Ok(source_slot)
    } else {
        kv.qwen35_active_slot()
    }
}

fn merged_support_note(
    support_note: Option<String>,
    profile_note: Option<&'static str>,
) -> Option<String> {
    match (support_note, profile_note) {
        (Some(note), Some(profile_note)) => Some(format!("{note} | {profile_note}")),
        (Some(note), None) => Some(note),
        (None, Some(profile_note)) => Some(profile_note.to_string()),
        (None, None) => None,
    }
}

fn prefill_profile_support_note(model: &InferenceModel) -> Option<&'static str> {
    if model.arch_name() == "qwen35" && model.use_gpu_decode() && model.metal_device().is_some() {
        Some(
            "profile: Qwen3.5 prefill timing follows the native unified batch path; wall time, GPU aggregate, submit counters, and recurrent batch phase buckets are authoritative",
        )
    } else {
        None
    }
}

fn classify_prefill_profile_route(
    prefill_plan: &str,
    recurrent_qkv_handoff_layers: u64,
    recurrent_qkv_fast_path_eligible_layers: u64,
) -> (String, String) {
    let family = crate::prefill_route_family(prefill_plan);
    let detail = if family == "qwen35_hybrid" {
        if recurrent_qkv_handoff_layers > 0 {
            "recurrent_handoff_fast_path".to_string()
        } else if recurrent_qkv_fast_path_eligible_layers > 0 {
            "recurrent_fast_path_without_handoff".to_string()
        } else {
            "recurrent_cpu_shaped_or_serial".to_string()
        }
    } else {
        crate::prefill_route_detail(prefill_plan)
    };
    (family, detail)
}

#[allow(clippy::too_many_arguments)]
fn build_qwen35_dtype_audit(
    model: &InferenceModel,
    runtime_policy: Option<&RuntimePolicy>,
    recurrent_counters: &ax_engine_core::backend::metal::Qwen35RecurrentBatchPerfCounters,
    prompt_tokens: usize,
    requested_mode: Qwen35RecurrentStateMode,
    requested_alpha_beta_mode: Qwen35AlphaBetaStorageMode,
    requested_prime_slot_buffers: bool,
    requested_same_kv_prewarm: bool,
    requested_force_backend_state_batch: bool,
) -> Option<Qwen35PrefillDTypeAudit> {
    if model.arch_name() != "qwen35" {
        return None;
    }
    let prefers_f16_io = runtime_policy
        .map(RuntimePolicy::batch_prefill_prefers_f16_io)
        .unwrap_or(false);
    let observed_state_path = if recurrent_counters.qkv_handoff_cpu_alias_layers > 0
        && recurrent_counters.qkv_handoff_slot_buffer_layers == 0
    {
        "cpu_alias_only"
    } else if recurrent_counters.qkv_handoff_slot_buffer_layers > 0
        && recurrent_counters.qkv_handoff_cpu_alias_layers == 0
    {
        "slot_buffer_only"
    } else if recurrent_counters.qkv_handoff_cpu_alias_layers > 0
        && recurrent_counters.qkv_handoff_slot_buffer_layers > 0
    {
        "mixed"
    } else {
        "inactive"
    };
    let observed_state_owner = if recurrent_counters.qkv_handoff_slot_buffer_layers > 0
        && recurrent_counters.qkv_handoff_cpu_materialization_layers == 0
        && recurrent_counters.qkv_handoff_backend_carryover_layers > 0
    {
        "backend_owned"
    } else if recurrent_counters.qkv_handoff_slot_buffer_layers > 0
        && recurrent_counters.qkv_handoff_cpu_materialization_layers == 0
        && recurrent_counters.qkv_handoff_backend_carryover_layers == 0
        && recurrent_counters.qkv_handoff_backend_zero_init_layers > 0
    {
        "backend_zero_initialized"
    } else if recurrent_counters.qkv_handoff_slot_buffer_layers > 0
        && recurrent_counters.qkv_handoff_cpu_materialization_layers > 0
        && recurrent_counters.qkv_handoff_backend_carryover_layers == 0
    {
        "cpu_materialized"
    } else if recurrent_counters.qkv_handoff_slot_buffer_layers > 0
        && recurrent_counters.qkv_handoff_cpu_materialization_layers > 0
        && recurrent_counters.qkv_handoff_backend_carryover_layers > 0
    {
        "mixed"
    } else if recurrent_counters.qkv_handoff_cpu_alias_layers > 0 {
        "cpu_materialized"
    } else if recurrent_counters.qkv_handoff_slot_buffer_layers > 0 {
        "already_synced"
    } else {
        "inactive"
    };
    let recurrent_state_batch_kind = if recurrent_counters.state_batch_backend_native_layers > 0
        && recurrent_counters.state_batch_cpu_direct_layers == 0
        && recurrent_counters.state_batch_cpu_direct_materialized_from_backend_layers == 0
        && recurrent_counters.state_batch_cpu_gathered_layers == 0
        && recurrent_counters.state_batch_cpu_gathered_materialized_from_backend_layers == 0
    {
        "backend_native"
    } else if recurrent_counters.state_batch_cpu_direct_layers > 0
        && recurrent_counters.state_batch_cpu_direct_materialized_from_backend_layers == 0
        && recurrent_counters.state_batch_cpu_gathered_layers == 0
        && recurrent_counters.state_batch_cpu_gathered_materialized_from_backend_layers == 0
    {
        "cpu_direct"
    } else if recurrent_counters.state_batch_cpu_direct_materialized_from_backend_layers > 0
        && recurrent_counters.state_batch_cpu_direct_layers == 0
        && recurrent_counters.state_batch_cpu_gathered_layers == 0
        && recurrent_counters.state_batch_cpu_gathered_materialized_from_backend_layers == 0
    {
        "cpu_direct_materialized_from_backend"
    } else if recurrent_counters.state_batch_cpu_gathered_layers > 0
        && recurrent_counters.state_batch_cpu_direct_layers == 0
        && recurrent_counters.state_batch_cpu_direct_materialized_from_backend_layers == 0
        && recurrent_counters.state_batch_cpu_gathered_materialized_from_backend_layers == 0
    {
        "cpu_gathered"
    } else if recurrent_counters.state_batch_cpu_gathered_materialized_from_backend_layers > 0
        && recurrent_counters.state_batch_backend_native_layers == 0
        && recurrent_counters.state_batch_cpu_direct_layers == 0
        && recurrent_counters.state_batch_cpu_direct_materialized_from_backend_layers == 0
        && recurrent_counters.state_batch_cpu_gathered_layers == 0
    {
        "cpu_gathered_materialized_from_backend"
    } else if recurrent_counters.state_batch_backend_native_layers > 0
        || recurrent_counters.state_batch_cpu_direct_layers > 0
        || recurrent_counters.state_batch_cpu_direct_materialized_from_backend_layers > 0
        || recurrent_counters.state_batch_cpu_gathered_layers > 0
        || recurrent_counters.state_batch_cpu_gathered_materialized_from_backend_layers > 0
    {
        "mixed"
    } else if recurrent_counters.qkv_handoff_layers > 0 {
        "model_side_handoff"
    } else {
        "inactive"
    };
    let effective_recurrent_state_mode = match observed_state_path {
        "cpu_alias_only" => "cpu_alias",
        "slot_buffer_only" => {
            if observed_state_owner == "backend_owned"
                && matches!(
                    requested_mode,
                    Qwen35RecurrentStateMode::BackendOwned | Qwen35RecurrentStateMode::Auto
                )
            {
                "backend_owned"
            } else {
                "slot_buffer"
            }
        }
        "mixed" => "mixed",
        _ => "inactive",
    };
    let implicit_prime = match requested_mode {
        Qwen35RecurrentStateMode::BackendOwned => true,
        Qwen35RecurrentStateMode::Auto => matches!(
            qwen35_prefill_recurrent_state_mode_auto_for_tokens(prompt_tokens),
            Qwen35RecurrentStateMode::BackendOwned
        ),
        _ => false,
    };
    let effective_slot_buffer_priming = (requested_prime_slot_buffers || implicit_prime)
        && matches!(
            effective_recurrent_state_mode,
            "slot_buffer" | "backend_owned"
        );
    let effective_alpha_beta_storage_dtype = match requested_alpha_beta_mode {
        Qwen35AlphaBetaStorageMode::F16 => "f16_storage_f32_compute",
        Qwen35AlphaBetaStorageMode::F32 | Qwen35AlphaBetaStorageMode::Auto => "f32",
    };
    Some(Qwen35PrefillDTypeAudit {
        requested_recurrent_state_mode: requested_mode,
        effective_recurrent_state_mode: effective_recurrent_state_mode.to_string(),
        requested_alpha_beta_storage_mode: requested_alpha_beta_mode,
        effective_alpha_beta_storage_dtype: effective_alpha_beta_storage_dtype.to_string(),
        requested_slot_buffer_priming: requested_prime_slot_buffers,
        effective_slot_buffer_priming,
        requested_same_kv_prewarm,
        effective_same_kv_prewarm: requested_same_kv_prewarm,
        requested_force_backend_state_batch,
        effective_force_backend_state_batch: requested_force_backend_state_batch
            && recurrent_state_batch_kind != "model_side_handoff",
        runtime_batch_prefill_prefers_f16_io: prefers_f16_io,
        dense_batch_projection_wrong_type_suspected: !prefers_f16_io,
        recurrent_state_logical_dtype: "f32".to_string(),
        recurrent_state_storage: "cpu_visible_vec_f32_or_shared_alias".to_string(),
        recurrent_snapshot_dtype: "f32".to_string(),
        recurrent_slot_mut_api_dtype: "&mut [f32]".to_string(),
        recurrent_batch_scratch_dtype: "f32".to_string(),
        recurrent_handoff_alpha_beta_dtype: "f32".to_string(),
        recurrent_handoff_observed_state_path: observed_state_path.to_string(),
        recurrent_handoff_observed_state_owner: observed_state_owner.to_string(),
        recurrent_handoff_cpu_alias_layers: recurrent_counters.qkv_handoff_cpu_alias_layers,
        recurrent_handoff_slot_buffer_layers: recurrent_counters.qkv_handoff_slot_buffer_layers,
        recurrent_handoff_backend_carryover_layers: recurrent_counters
            .qkv_handoff_backend_carryover_layers,
        recurrent_handoff_backend_zero_init_layers: recurrent_counters
            .qkv_handoff_backend_zero_init_layers,
        recurrent_handoff_cpu_materialization_layers: recurrent_counters
            .qkv_handoff_cpu_materialization_layers,
        recurrent_handoff_fused_tail_layers: recurrent_counters.qkv_handoff_fused_tail_layers,
        recurrent_state_batch_kind: recurrent_state_batch_kind.to_string(),
        recurrent_state_batch_backend_native_layers: recurrent_counters
            .state_batch_backend_native_layers,
        recurrent_state_batch_cpu_direct_layers: recurrent_counters.state_batch_cpu_direct_layers,
        recurrent_state_batch_cpu_direct_materialized_from_backend_layers: recurrent_counters
            .state_batch_cpu_direct_materialized_from_backend_layers,
        recurrent_state_batch_cpu_gathered_layers: recurrent_counters
            .state_batch_cpu_gathered_layers,
        recurrent_state_batch_cpu_gathered_materialized_from_backend_layers: recurrent_counters
            .state_batch_cpu_gathered_materialized_from_backend_layers,
        recurrent_f32_contract_ceiling_suspected: true,
    })
}

fn prime_qwen35_slot_buffers_if_requested(
    model: &InferenceModel,
    kv: &mut ModelKv,
    qwen35_shared_timeline_source_slot: Option<usize>,
    prompt_tokens: usize,
    qwen35_recurrent_state_mode: Qwen35RecurrentStateMode,
    qwen35_prime_slot_buffers: bool,
) -> anyhow::Result<()> {
    let implicit_prime = match qwen35_recurrent_state_mode {
        Qwen35RecurrentStateMode::BackendOwned => true,
        Qwen35RecurrentStateMode::Auto => matches!(
            qwen35_prefill_recurrent_state_mode_auto_for_tokens(prompt_tokens),
            Qwen35RecurrentStateMode::BackendOwned
        ),
        _ => false,
    };
    if (!qwen35_prime_slot_buffers && !implicit_prime) || model.arch_name() != "qwen35" {
        return Ok(());
    }
    prepare_qwen35_shared_timeline_source_slot(model, kv, qwen35_shared_timeline_source_slot)?;
    let slot_idx = qwen35_effective_source_slot(kv, qwen35_shared_timeline_source_slot)?;
    model.prime_qwen35_recurrent_slot_buffers(kv, &[slot_idx])
}

/// Result of a prefill profiling run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillProfileResult {
    pub model: String,
    pub prompt_tokens: usize,
    #[serde(default)]
    pub effective_prompt_tokens: usize,
    pub total_ms: f64,
    pub tok_per_sec: f64,
    #[serde(default)]
    pub effective_tok_per_sec: f64,
    #[serde(default)]
    pub prefill_plan: String,
    #[serde(default)]
    pub prefill_mode: String,
    #[serde(default)]
    pub prefill_route_family: String,
    #[serde(default)]
    pub prefill_route_detail: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_attention_route: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_qkv_plan: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_split_rope_append: Option<bool>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q5k_prefill_mode: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub support_note: Option<String>,
    #[serde(default = "default_qwen35_shared_timeline_slots")]
    pub qwen35_shared_timeline_slots: usize,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qwen35_shared_timeline_source_slot: Option<usize>,
    #[serde(default)]
    pub qwen35_recurrent_state_mode: Qwen35RecurrentStateMode,
    #[serde(default)]
    pub qwen35_alpha_beta_storage_mode: Qwen35AlphaBetaStorageMode,
    #[serde(default)]
    pub qwen35_prime_slot_buffers: bool,
    #[serde(default)]
    pub qwen35_prewarm_prefill_same_kv: bool,
    #[serde(default)]
    pub qwen35_force_backend_state_batch: bool,
    #[serde(default)]
    pub local_hd128_route: LocalPrefillHd128Route,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qwen35_dtype_audit: Option<Qwen35PrefillDTypeAudit>,
    #[serde(default)]
    pub prefill_command_buffers: f64,
    #[serde(default)]
    pub prefill_buffer_barriers: f64,
    #[serde(default)]
    pub prefill_command_buffers_per_tok: f64,
    #[serde(default)]
    pub prefill_buffer_barriers_per_tok: f64,
    #[serde(default)]
    pub gpu_pct: f64,
    #[serde(default)]
    pub gpu_encode_pct: f64,
    #[serde(default)]
    pub gpu_execute_pct: f64,
    #[serde(default)]
    pub gpu_execute_layers_pct: f64,
    #[serde(default)]
    pub gpu_execute_output_pct: f64,
    #[serde(default)]
    pub gpu_readback_pct: f64,
    #[serde(default)]
    pub gpu_encode_layer_norm_pct: f64,
    #[serde(default)]
    pub gpu_encode_layer_qkv_pct: f64,
    #[serde(default)]
    pub gpu_encode_layer_rope_pct: f64,
    #[serde(default)]
    pub gpu_encode_layer_kv_append_pct: f64,
    #[serde(default)]
    pub gpu_encode_layer_attention_pct: f64,
    #[serde(default)]
    pub gpu_encode_layer_out_proj_pct: f64,
    #[serde(default)]
    pub gpu_encode_layer_ffn_pct: f64,
    #[serde(default)]
    pub gpu_encode_layer_residual_pct: f64,
    #[serde(default)]
    pub matmul_pct: f64,
    #[serde(default)]
    pub attention_pct: f64,
    #[serde(default)]
    pub recurrent_pct: f64,
    #[serde(default)]
    pub recurrent_batch_conv_pct: f64,
    #[serde(default)]
    pub recurrent_batch_pack_pct: f64,
    #[serde(default)]
    pub recurrent_batch_gated_delta_pct: f64,
    #[serde(default)]
    pub recurrent_batch_unpack_pct: f64,
    #[serde(default)]
    pub dequant_pct: f64,
    #[serde(default)]
    pub rope_pct: f64,
    #[serde(default)]
    pub norm_pct: f64,
    #[serde(default)]
    pub other_pct: f64,
    #[serde(default)]
    pub gpu_ms: f64,
    #[serde(default)]
    pub gpu_encode_ms: f64,
    #[serde(default)]
    pub gpu_execute_ms: f64,
    #[serde(default)]
    pub gpu_execute_layers_ms: f64,
    #[serde(default)]
    pub gpu_execute_output_ms: f64,
    #[serde(default)]
    pub gpu_readback_ms: f64,
    #[serde(default)]
    pub gpu_encode_layer_norm_ms: f64,
    #[serde(default)]
    pub gpu_encode_layer_qkv_ms: f64,
    #[serde(default)]
    pub gpu_encode_layer_rope_ms: f64,
    #[serde(default)]
    pub gpu_encode_layer_kv_append_ms: f64,
    #[serde(default)]
    pub gpu_encode_layer_attention_ms: f64,
    #[serde(default)]
    pub gpu_encode_layer_out_proj_ms: f64,
    #[serde(default)]
    pub gpu_encode_layer_ffn_ms: f64,
    #[serde(default)]
    pub gpu_encode_layer_residual_ms: f64,
    #[serde(default)]
    pub matmul_ms: f64,
    #[serde(default)]
    pub attention_ms: f64,
    #[serde(default)]
    pub recurrent_ms: f64,
    #[serde(default)]
    pub recurrent_batch_conv_ms: f64,
    #[serde(default)]
    pub recurrent_batch_pack_ms: f64,
    #[serde(default)]
    pub recurrent_batch_gated_delta_ms: f64,
    #[serde(default)]
    pub recurrent_batch_unpack_ms: f64,
    #[serde(default)]
    pub recurrent_batch_qkv_handoff_ms: f64,
    #[serde(default)]
    pub recurrent_qkv_handoff_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_handoff_fused_tail_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_gpu_projection_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_fast_path_eligible_layers: u64,
    #[serde(default)]
    pub recurrent_gpu_ssm_projection_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_fast_reject_state_size_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_fast_reject_group_divisibility_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_fast_reject_missing_batch_scratches_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_fast_reject_q_capacity_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_fast_reject_k_capacity_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_fast_reject_v_capacity_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_fast_reject_gate_capacity_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_fast_reject_up_capacity_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_handoff_cpu_alias_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_handoff_slot_buffer_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_handoff_backend_carryover_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_handoff_backend_zero_init_layers: u64,
    #[serde(default)]
    pub recurrent_qkv_handoff_cpu_materialization_layers: u64,
    #[serde(default)]
    pub recurrent_state_batch_backend_native_layers: u64,
    #[serde(default)]
    pub recurrent_state_batch_cpu_direct_layers: u64,
    #[serde(default)]
    pub recurrent_state_batch_cpu_direct_materialized_from_backend_layers: u64,
    #[serde(default)]
    pub recurrent_state_batch_cpu_gathered_layers: u64,
    #[serde(default)]
    pub recurrent_state_batch_cpu_gathered_materialized_from_backend_layers: u64,
    #[serde(default)]
    pub dequant_ms: f64,
    #[serde(default)]
    pub rope_ms: f64,
    #[serde(default)]
    pub norm_ms: f64,
}

fn run_prefill_once(
    model: &InferenceModel,
    kv: &mut ModelKv,
    prompt_tokens: &[u32],
    qwen35_shared_timeline_slots: usize,
    qwen35_shared_timeline_source_slot: Option<usize>,
    weights: &WeightStore,
    logits: &mut [f32],
) -> anyhow::Result<()> {
    prepare_qwen35_shared_timeline_source_slot(model, kv, qwen35_shared_timeline_source_slot)?;
    if let Some(source_slot) = qwen35_shared_timeline_source_slot {
        model.forward_batch_qwen35_shared_timeline_forked_from_slot(
            prompt_tokens,
            kv,
            source_slot,
            qwen35_shared_timeline_slots,
            weights,
            logits,
        )
    } else {
        model.forward_batch_qwen35_shared_timeline_forked(
            prompt_tokens,
            kv,
            qwen35_shared_timeline_slots,
            weights,
            logits,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_profiled_once(
    model: &InferenceModel,
    kv: &mut ModelKv,
    prompt_tokens: &[u32],
    qwen35_shared_timeline_slots: usize,
    qwen35_shared_timeline_source_slot: Option<usize>,
    weights: &WeightStore,
    logits: &mut [f32],
    ops: &mut OpBreakdown,
) -> anyhow::Result<()> {
    prepare_qwen35_shared_timeline_source_slot(model, kv, qwen35_shared_timeline_source_slot)?;
    if let Some(source_slot) = qwen35_shared_timeline_source_slot {
        model.forward_batch_profiled_qwen35_shared_timeline_forked_from_slot(
            prompt_tokens,
            kv,
            source_slot,
            qwen35_shared_timeline_slots,
            weights,
            logits,
            ops,
        )
    } else {
        model.forward_batch_profiled_qwen35_shared_timeline_forked(
            prompt_tokens,
            kv,
            qwen35_shared_timeline_slots,
            weights,
            logits,
            ops,
        )
    }
}

/// Run the batched-prefill profiler.
pub fn run_prefill_profile(config: &PrefillProfileConfig) -> anyhow::Result<PrefillProfileResult> {
    run_prefill_profile_with_backend(
        config,
        ax_engine_core::backend::create_backend(ax_engine_core::backend::BackendConfig::default())?,
    )
}

pub fn run_prefill_profile_with_backend(
    config: &PrefillProfileConfig,
    backend: Box<dyn ax_engine_core::backend::Backend>,
) -> anyhow::Result<PrefillProfileResult> {
    let mapped = MappedModel::open(Path::new(&config.model_path))?;
    let model_config = ModelConfig::from_gguf(&mapped.header)?;
    crate::configure_backend_for_model(&*backend, &config.model_path, &mapped, &model_config)?;
    let runtime_policy = backend.runtime_policy();
    let tokenizer = Tokenizer::from_gguf(&mapped.header)?;
    let model = InferenceModel::with_backend(model_config.clone(), backend)?;
    crate::report_planned_kv_budget(&mapped, &model)?;
    let support_note = merged_support_note(
        crate::support_note(&mapped),
        prefill_profile_support_note(&model),
    );
    let weights = WeightStore::new(&mapped);
    model.prepare_runtime_for_weights(&weights)?;

    let vocab_size = model_config.vocab_size as usize;
    let prompt_tokens = build_fixed_prompt(&tokenizer, config.prompt_tokens);

    eprintln!(
        "Prefill profile: {} layers, {:.0}MB, {} warmup + {} prompt tokens",
        model_config.n_layers,
        mapped.total_tensor_bytes() as f64 / 1024.0 / 1024.0,
        config.warmup_iters,
        config.prompt_tokens,
    );

    let prefill_plan = {
        let kv = model.create_model_kv_for_weights(&weights);
        model.prefill_plan_summary(&weights, &kv, prompt_tokens.len())?
    };

    for _ in 0..config.warmup_iters {
        let mut warmup_kv = model.create_model_kv_for_weights(&weights);
        prime_qwen35_slot_buffers_if_requested(
            &model,
            &mut warmup_kv,
            config.qwen35_shared_timeline_source_slot,
            prompt_tokens.len(),
            config.qwen35_recurrent_state_mode,
            config.qwen35_prime_slot_buffers,
        )?;
        let mut warmup_logits = vec![0.0f32; vocab_size];
        warmup_logits.fill(0.0);
        run_prefill_once(
            &model,
            &mut warmup_kv,
            &prompt_tokens,
            config.qwen35_shared_timeline_slots,
            config.qwen35_shared_timeline_source_slot,
            &weights,
            &mut warmup_logits,
        )?;
    }

    let mut kv = model.create_model_kv_for_weights(&weights);
    prime_qwen35_slot_buffers_if_requested(
        &model,
        &mut kv,
        config.qwen35_shared_timeline_source_slot,
        prompt_tokens.len(),
        config.qwen35_recurrent_state_mode,
        config.qwen35_prime_slot_buffers,
    )?;
    if config.qwen35_prewarm_prefill_same_kv && model.arch_name() == "qwen35" {
        let mut prewarm_logits = vec![0.0f32; vocab_size];
        prewarm_logits.fill(0.0);
        run_prefill_once(
            &model,
            &mut kv,
            &prompt_tokens,
            config.qwen35_shared_timeline_slots,
            config.qwen35_shared_timeline_source_slot,
            &weights,
            &mut prewarm_logits,
        )?;
        let slot_idx =
            qwen35_effective_source_slot(&kv, config.qwen35_shared_timeline_source_slot)?;
        model.prime_qwen35_recurrent_slot_buffers(&mut kv, &[slot_idx])?;
    }
    model.prepare_prefill_for_weights(&weights, &mut kv, prompt_tokens.len())?;
    let mut logits = vec![0.0f32; vocab_size];

    let mut ops = OpBreakdown::new();
    model.reset_metal_perf_counters();
    let wall_timer = OpTimer::start();
    logits.fill(0.0);
    run_prefill_profiled_once(
        &model,
        &mut kv,
        &prompt_tokens,
        config.qwen35_shared_timeline_slots,
        config.qwen35_shared_timeline_source_slot,
        &weights,
        &mut logits,
        &mut ops,
    )?;
    let wall_time = wall_timer.elapsed();
    let counters = model.read_metal_perf_counters();
    let recurrent_counters = model.read_qwen35_recurrent_batch_perf_counters();
    let qwen35_dtype_audit = build_qwen35_dtype_audit(
        &model,
        runtime_policy.as_ref(),
        &recurrent_counters,
        prompt_tokens.len(),
        config.qwen35_recurrent_state_mode,
        config.qwen35_alpha_beta_storage_mode,
        config.qwen35_prime_slot_buffers,
        config.qwen35_prewarm_prefill_same_kv,
        config.qwen35_force_backend_state_batch,
    );

    let wall_ms = wall_time.as_secs_f64() * 1000.0;
    let tracked_ms = ops.total().as_secs_f64() * 1000.0;
    let other_ms = (wall_ms - tracked_ms).max(0.0);
    let prefill_mode = crate::prefill_mode(&prefill_plan);
    let prefill_attention_route = crate::prefill_plan_field(&prefill_plan, "attn_route");
    let prefill_qkv_plan = crate::prefill_plan_field(&prefill_plan, "qkv");
    let prefill_split_rope_append = crate::prefill_bool_field(&prefill_plan, "split_rope");
    let q5k_prefill_mode = crate::q5k_prefill_mode(&prefill_plan);
    let (prefill_route_family, prefill_route_detail) = classify_prefill_profile_route(
        &prefill_plan,
        recurrent_counters.qkv_handoff_layers,
        recurrent_counters.qkv_fast_path_eligible_layers,
    );
    let effective_prompt_tokens = effective_qwen35_shared_timeline_tokens(
        prompt_tokens.len(),
        config.qwen35_shared_timeline_slots,
    );
    let pct = |d: Duration| -> f64 {
        if wall_ms > 0.0 {
            d.as_secs_f64() * 1000.0 / wall_ms * 100.0
        } else {
            0.0
        }
    };

    Ok(PrefillProfileResult {
        model: config.model_path.clone(),
        prompt_tokens: prompt_tokens.len(),
        effective_prompt_tokens,
        total_ms: wall_ms,
        tok_per_sec: if wall_time.as_secs_f64() > 0.0 {
            prompt_tokens.len() as f64 / wall_time.as_secs_f64()
        } else {
            0.0
        },
        effective_tok_per_sec: if wall_time.as_secs_f64() > 0.0 {
            effective_prompt_tokens as f64 / wall_time.as_secs_f64()
        } else {
            0.0
        },
        prefill_mode,
        prefill_route_family,
        prefill_route_detail,
        prefill_attention_route,
        prefill_qkv_plan,
        prefill_split_rope_append,
        q5k_prefill_mode,
        prefill_plan,
        support_note,
        qwen35_shared_timeline_slots: config.qwen35_shared_timeline_slots,
        qwen35_shared_timeline_source_slot: config.qwen35_shared_timeline_source_slot,
        qwen35_recurrent_state_mode: config.qwen35_recurrent_state_mode,
        qwen35_alpha_beta_storage_mode: config.qwen35_alpha_beta_storage_mode,
        qwen35_prime_slot_buffers: config.qwen35_prime_slot_buffers,
        qwen35_prewarm_prefill_same_kv: config.qwen35_prewarm_prefill_same_kv,
        qwen35_force_backend_state_batch: config.qwen35_force_backend_state_batch,
        local_hd128_route: config.local_hd128_route,
        qwen35_dtype_audit,
        prefill_command_buffers: counters.command_buffers as f64,
        prefill_buffer_barriers: counters.buffer_barriers as f64,
        prefill_command_buffers_per_tok: if prompt_tokens.is_empty() {
            0.0
        } else {
            counters.command_buffers as f64 / prompt_tokens.len() as f64
        },
        prefill_buffer_barriers_per_tok: if prompt_tokens.is_empty() {
            0.0
        } else {
            counters.buffer_barriers as f64 / prompt_tokens.len() as f64
        },
        gpu_pct: pct(ops.gpu),
        gpu_encode_pct: pct(ops.gpu_encode),
        gpu_execute_pct: pct(ops.gpu_execute),
        gpu_execute_layers_pct: pct(ops.gpu_execute_layers),
        gpu_execute_output_pct: pct(ops.gpu_execute_output),
        gpu_readback_pct: pct(ops.gpu_readback),
        gpu_encode_layer_norm_pct: pct(ops.gpu_encode_layer_norm),
        gpu_encode_layer_qkv_pct: pct(ops.gpu_encode_layer_qkv),
        gpu_encode_layer_rope_pct: pct(ops.gpu_encode_layer_rope),
        gpu_encode_layer_kv_append_pct: pct(ops.gpu_encode_layer_kv_append),
        gpu_encode_layer_attention_pct: pct(ops.gpu_encode_layer_attention),
        gpu_encode_layer_out_proj_pct: pct(ops.gpu_encode_layer_out_proj),
        gpu_encode_layer_ffn_pct: pct(ops.gpu_encode_layer_ffn),
        gpu_encode_layer_residual_pct: pct(ops.gpu_encode_layer_residual),
        matmul_pct: pct(ops.matmul),
        attention_pct: pct(ops.attention),
        recurrent_pct: pct(ops.recurrent),
        recurrent_batch_conv_pct: if wall_ms > 0.0 {
            recurrent_counters.conv_ns as f64 / 1_000_000.0 / wall_ms * 100.0
        } else {
            0.0
        },
        recurrent_batch_pack_pct: if wall_ms > 0.0 {
            recurrent_counters.pack_ns as f64 / 1_000_000.0 / wall_ms * 100.0
        } else {
            0.0
        },
        recurrent_batch_gated_delta_pct: if wall_ms > 0.0 {
            recurrent_counters.gated_delta_ns as f64 / 1_000_000.0 / wall_ms * 100.0
        } else {
            0.0
        },
        recurrent_batch_unpack_pct: if wall_ms > 0.0 {
            recurrent_counters.unpack_ns as f64 / 1_000_000.0 / wall_ms * 100.0
        } else {
            0.0
        },
        dequant_pct: pct(ops.dequant),
        rope_pct: pct(ops.rope),
        norm_pct: pct(ops.norm),
        other_pct: if wall_ms > 0.0 {
            other_ms / wall_ms * 100.0
        } else {
            0.0
        },
        gpu_ms: ops.gpu.as_secs_f64() * 1000.0,
        gpu_encode_ms: ops.gpu_encode.as_secs_f64() * 1000.0,
        gpu_execute_ms: ops.gpu_execute.as_secs_f64() * 1000.0,
        gpu_execute_layers_ms: ops.gpu_execute_layers.as_secs_f64() * 1000.0,
        gpu_execute_output_ms: ops.gpu_execute_output.as_secs_f64() * 1000.0,
        gpu_readback_ms: ops.gpu_readback.as_secs_f64() * 1000.0,
        gpu_encode_layer_norm_ms: ops.gpu_encode_layer_norm.as_secs_f64() * 1000.0,
        gpu_encode_layer_qkv_ms: ops.gpu_encode_layer_qkv.as_secs_f64() * 1000.0,
        gpu_encode_layer_rope_ms: ops.gpu_encode_layer_rope.as_secs_f64() * 1000.0,
        gpu_encode_layer_kv_append_ms: ops.gpu_encode_layer_kv_append.as_secs_f64() * 1000.0,
        gpu_encode_layer_attention_ms: ops.gpu_encode_layer_attention.as_secs_f64() * 1000.0,
        gpu_encode_layer_out_proj_ms: ops.gpu_encode_layer_out_proj.as_secs_f64() * 1000.0,
        gpu_encode_layer_ffn_ms: ops.gpu_encode_layer_ffn.as_secs_f64() * 1000.0,
        gpu_encode_layer_residual_ms: ops.gpu_encode_layer_residual.as_secs_f64() * 1000.0,
        matmul_ms: ops.matmul.as_secs_f64() * 1000.0,
        attention_ms: ops.attention.as_secs_f64() * 1000.0,
        recurrent_ms: ops.recurrent.as_secs_f64() * 1000.0,
        recurrent_batch_conv_ms: recurrent_counters.conv_ns as f64 / 1_000_000.0,
        recurrent_batch_pack_ms: recurrent_counters.pack_ns as f64 / 1_000_000.0,
        recurrent_batch_gated_delta_ms: recurrent_counters.gated_delta_ns as f64 / 1_000_000.0,
        recurrent_batch_unpack_ms: recurrent_counters.unpack_ns as f64 / 1_000_000.0,
        recurrent_batch_qkv_handoff_ms: recurrent_counters.qkv_handoff_ns as f64 / 1_000_000.0,
        recurrent_qkv_handoff_layers: recurrent_counters.qkv_handoff_layers,
        recurrent_qkv_handoff_fused_tail_layers: recurrent_counters.qkv_handoff_fused_tail_layers,
        recurrent_qkv_gpu_projection_layers: recurrent_counters.qkv_gpu_projection_layers,
        recurrent_qkv_fast_path_eligible_layers: recurrent_counters.qkv_fast_path_eligible_layers,
        recurrent_gpu_ssm_projection_layers: recurrent_counters.gpu_ssm_projection_layers,
        recurrent_qkv_fast_reject_state_size_layers: recurrent_counters
            .qkv_fast_reject_state_size_layers,
        recurrent_qkv_fast_reject_group_divisibility_layers: recurrent_counters
            .qkv_fast_reject_group_divisibility_layers,
        recurrent_qkv_fast_reject_missing_batch_scratches_layers: recurrent_counters
            .qkv_fast_reject_missing_batch_scratches_layers,
        recurrent_qkv_fast_reject_q_capacity_layers: recurrent_counters
            .qkv_fast_reject_q_capacity_layers,
        recurrent_qkv_fast_reject_k_capacity_layers: recurrent_counters
            .qkv_fast_reject_k_capacity_layers,
        recurrent_qkv_fast_reject_v_capacity_layers: recurrent_counters
            .qkv_fast_reject_v_capacity_layers,
        recurrent_qkv_fast_reject_gate_capacity_layers: recurrent_counters
            .qkv_fast_reject_gate_capacity_layers,
        recurrent_qkv_fast_reject_up_capacity_layers: recurrent_counters
            .qkv_fast_reject_up_capacity_layers,
        recurrent_qkv_handoff_cpu_alias_layers: recurrent_counters.qkv_handoff_cpu_alias_layers,
        recurrent_qkv_handoff_slot_buffer_layers: recurrent_counters.qkv_handoff_slot_buffer_layers,
        recurrent_qkv_handoff_backend_carryover_layers: recurrent_counters
            .qkv_handoff_backend_carryover_layers,
        recurrent_qkv_handoff_backend_zero_init_layers: recurrent_counters
            .qkv_handoff_backend_zero_init_layers,
        recurrent_qkv_handoff_cpu_materialization_layers: recurrent_counters
            .qkv_handoff_cpu_materialization_layers,
        recurrent_state_batch_backend_native_layers: recurrent_counters
            .state_batch_backend_native_layers,
        recurrent_state_batch_cpu_direct_layers: recurrent_counters.state_batch_cpu_direct_layers,
        recurrent_state_batch_cpu_direct_materialized_from_backend_layers: recurrent_counters
            .state_batch_cpu_direct_materialized_from_backend_layers,
        recurrent_state_batch_cpu_gathered_layers: recurrent_counters
            .state_batch_cpu_gathered_layers,
        recurrent_state_batch_cpu_gathered_materialized_from_backend_layers: recurrent_counters
            .state_batch_cpu_gathered_materialized_from_backend_layers,
        dequant_ms: ops.dequant.as_secs_f64() * 1000.0,
        rope_ms: ops.rope.as_secs_f64() * 1000.0,
        norm_ms: ops.norm.as_secs_f64() * 1000.0,
    })
}

fn build_fixed_prompt(tokenizer: &Tokenizer, prompt_tokens: usize) -> Vec<u32> {
    let prompt = tokenizer.encode("The quick brown fox jumps over the lazy dog. ", true);
    let mut tokens = Vec::new();
    while tokens.len() < prompt_tokens {
        tokens.extend_from_slice(&prompt);
    }
    tokens.truncate(prompt_tokens);
    tokens
}

impl PrefillProfileResult {
    fn effective_prompt_tokens_resolved(&self) -> usize {
        if self.effective_prompt_tokens > 0 {
            self.effective_prompt_tokens
        } else {
            effective_qwen35_shared_timeline_tokens(
                self.prompt_tokens,
                self.qwen35_shared_timeline_slots,
            )
        }
    }

    fn effective_tok_per_sec_resolved(&self) -> f64 {
        if self.effective_tok_per_sec > 0.0 {
            self.effective_tok_per_sec
        } else {
            self.tok_per_sec * self.qwen35_shared_timeline_slots.max(1) as f64
        }
    }

    pub fn print_summary(&self) {
        eprintln!();
        eprintln!("=== Prefill Profile ===");
        eprintln!("Model:       {}", self.model);
        eprintln!("Prompt:      {} tokens", self.prompt_tokens);
        if self.qwen35_shared_timeline_slots > 1 {
            eprintln!(
                "Qwen35Fanout:{:>4} shared-timeline slots",
                self.qwen35_shared_timeline_slots
            );
            if let Some(source_slot) = self.qwen35_shared_timeline_source_slot {
                eprintln!("Qwen35Src:   slot {source_slot}");
            }
            eprintln!(
                "Effective:   {} slot-tok ({:.1} tok/s)",
                self.effective_prompt_tokens_resolved(),
                self.effective_tok_per_sec_resolved(),
            );
        }
        eprintln!("PrefillPlan: {}", self.prefill_plan);
        if !self.prefill_route_family.is_empty() {
            eprintln!(
                "PrefillRoute:{} / {}",
                self.prefill_route_family, self.prefill_route_detail
            );
        }
        if let Some(mode) = &self.q5k_prefill_mode {
            eprintln!("Q5KPrefill:  {mode}");
        }
        if let Some(note) = &self.support_note {
            eprintln!("Support:     {note}");
        }
        if self.local_hd128_route != LocalPrefillHd128Route::Auto {
            eprintln!("RouteForce:  {}", self.local_hd128_route.label());
        }
        if let Some(audit) = &self.qwen35_dtype_audit {
            eprintln!(
                "Qwen35Mode:  requested={} | effective={}",
                audit.requested_recurrent_state_mode.label(),
                audit.effective_recurrent_state_mode,
            );
            eprintln!(
                "Qwen35DType: dense_f16_io={} | recurrent_state={} | alpha_beta={}/{} | prime={} | same_kv_prewarm={} | force_backend_batch={} | observed_path={} | owner={} | batch_kind={}",
                audit.runtime_batch_prefill_prefers_f16_io,
                audit.recurrent_state_logical_dtype,
                audit.requested_alpha_beta_storage_mode.label(),
                audit.effective_alpha_beta_storage_dtype,
                audit.effective_slot_buffer_priming,
                audit.effective_same_kv_prewarm,
                audit.effective_force_backend_state_batch,
                audit.recurrent_handoff_observed_state_path,
                audit.recurrent_handoff_observed_state_owner,
                audit.recurrent_state_batch_kind,
            );
        }
        eprintln!(
            "Wall time:   {:.1}ms ({:.1} tok/s)",
            self.total_ms, self.tok_per_sec,
        );
        if self.prefill_command_buffers > 0.0 || self.prefill_buffer_barriers > 0.0 {
            eprintln!(
                "GPU Submit:  {:.1} cmd, {:.1} barriers  ({:.3} cmd/tok, {:.3} barriers/tok)",
                self.prefill_command_buffers,
                self.prefill_buffer_barriers,
                self.prefill_command_buffers_per_tok,
                self.prefill_buffer_barriers_per_tok,
            );
        }
        eprintln!();
        eprintln!(
            "  GPU:       {:6.1}ms  ({:5.1}%)",
            self.gpu_ms, self.gpu_pct
        );
        eprintln!(
            "  GPU Enc:   {:6.1}ms  ({:5.1}%)",
            self.gpu_encode_ms, self.gpu_encode_pct
        );
        eprintln!(
            "  GPU Exec:  {:6.1}ms  ({:5.1}%)",
            self.gpu_execute_ms, self.gpu_execute_pct
        );
        if self.gpu_execute_layers_ms > 0.0 || self.gpu_execute_output_ms > 0.0 {
            eprintln!(
                "    Layers:  {:6.1}ms  ({:5.1}%)",
                self.gpu_execute_layers_ms, self.gpu_execute_layers_pct
            );
            eprintln!(
                "    Output:  {:6.1}ms  ({:5.1}%)",
                self.gpu_execute_output_ms, self.gpu_execute_output_pct
            );
        }
        eprintln!(
            "  GPU RBack: {:6.1}ms  ({:5.1}%)",
            self.gpu_readback_ms, self.gpu_readback_pct
        );
        if self.gpu_encode_layer_norm_ms > 0.0
            || self.gpu_encode_layer_qkv_ms > 0.0
            || self.gpu_encode_layer_rope_ms > 0.0
            || self.gpu_encode_layer_kv_append_ms > 0.0
            || self.gpu_encode_layer_attention_ms > 0.0
            || self.gpu_encode_layer_out_proj_ms > 0.0
            || self.gpu_encode_layer_ffn_ms > 0.0
            || self.gpu_encode_layer_residual_ms > 0.0
        {
            eprintln!(
                "    Enc Norm:{:6.1}ms  ({:5.1}%)",
                self.gpu_encode_layer_norm_ms, self.gpu_encode_layer_norm_pct
            );
            eprintln!(
                "    Enc QKV: {:6.1}ms  ({:5.1}%)",
                self.gpu_encode_layer_qkv_ms, self.gpu_encode_layer_qkv_pct
            );
            eprintln!(
                "    Enc RoPE:{:6.1}ms  ({:5.1}%)",
                self.gpu_encode_layer_rope_ms, self.gpu_encode_layer_rope_pct
            );
            eprintln!(
                "    Enc KV:  {:6.1}ms  ({:5.1}%)",
                self.gpu_encode_layer_kv_append_ms, self.gpu_encode_layer_kv_append_pct
            );
            eprintln!(
                "    Enc Attn:{:6.1}ms  ({:5.1}%)",
                self.gpu_encode_layer_attention_ms, self.gpu_encode_layer_attention_pct
            );
            eprintln!(
                "    Enc Out: {:6.1}ms  ({:5.1}%)",
                self.gpu_encode_layer_out_proj_ms, self.gpu_encode_layer_out_proj_pct
            );
            eprintln!(
                "    Enc FFN: {:6.1}ms  ({:5.1}%)",
                self.gpu_encode_layer_ffn_ms, self.gpu_encode_layer_ffn_pct
            );
            eprintln!(
                "    Enc Res: {:6.1}ms  ({:5.1}%)",
                self.gpu_encode_layer_residual_ms, self.gpu_encode_layer_residual_pct
            );
        }
        eprintln!(
            "  Matmul:    {:6.1}ms  ({:5.1}%)",
            self.matmul_ms, self.matmul_pct
        );
        eprintln!(
            "  Attention: {:6.1}ms  ({:5.1}%)",
            self.attention_ms, self.attention_pct
        );
        if self.recurrent_ms > 0.0 || self.recurrent_pct > 0.0 {
            eprintln!(
                "  Recurrent: {:6.1}ms  ({:5.1}%)",
                self.recurrent_ms, self.recurrent_pct
            );
            if self.recurrent_batch_conv_ms > 0.0
                || self.recurrent_batch_pack_ms > 0.0
                || self.recurrent_batch_gated_delta_ms > 0.0
                || self.recurrent_batch_unpack_ms > 0.0
                || self.recurrent_batch_qkv_handoff_ms > 0.0
            {
                eprintln!(
                    "    BatchConv:{:5.1}ms  ({:5.1}%)",
                    self.recurrent_batch_conv_ms, self.recurrent_batch_conv_pct
                );
                eprintln!(
                    "    BatchPack:{:5.1}ms  ({:5.1}%)",
                    self.recurrent_batch_pack_ms, self.recurrent_batch_pack_pct
                );
                eprintln!(
                    "    BatchGDN: {:5.1}ms  ({:5.1}%)",
                    self.recurrent_batch_gated_delta_ms, self.recurrent_batch_gated_delta_pct
                );
                eprintln!(
                    "    BatchUnpk:{:5.1}ms  ({:5.1}%)",
                    self.recurrent_batch_unpack_ms, self.recurrent_batch_unpack_pct
                );
                if self.recurrent_batch_qkv_handoff_ms > 0.0 {
                    eprintln!(
                        "    BatchFast:{:5.1}ms  ({:5.1}%)",
                        self.recurrent_batch_qkv_handoff_ms,
                        if self.total_ms > 0.0 {
                            self.recurrent_batch_qkv_handoff_ms * 100.0 / self.total_ms
                        } else {
                            0.0
                        }
                    );
                }
                if self.recurrent_qkv_handoff_layers > 0
                    || self.recurrent_gpu_ssm_projection_layers > 0
                {
                    eprintln!(
                        "    Route:     qkv-proj {} | qkv-fast {} | qkv-handoff {} | fused-tail {} | gpu-ssm {}",
                        self.recurrent_qkv_gpu_projection_layers,
                        self.recurrent_qkv_fast_path_eligible_layers,
                        self.recurrent_qkv_handoff_layers,
                        self.recurrent_qkv_handoff_fused_tail_layers,
                        self.recurrent_gpu_ssm_projection_layers
                    );
                    if self.recurrent_qkv_handoff_cpu_alias_layers > 0
                        || self.recurrent_qkv_handoff_slot_buffer_layers > 0
                    {
                        eprintln!(
                            "    State:     cpu-alias {} | slot-buffer {}",
                            self.recurrent_qkv_handoff_cpu_alias_layers,
                            self.recurrent_qkv_handoff_slot_buffer_layers
                        );
                        if self.recurrent_qkv_handoff_backend_carryover_layers > 0
                            || self.recurrent_qkv_handoff_backend_zero_init_layers > 0
                            || self.recurrent_qkv_handoff_cpu_materialization_layers > 0
                        {
                            eprintln!(
                                "    Owner:     carryover {} | zero-init {} | materialized {}",
                                self.recurrent_qkv_handoff_backend_carryover_layers,
                                self.recurrent_qkv_handoff_backend_zero_init_layers,
                                self.recurrent_qkv_handoff_cpu_materialization_layers
                            );
                        }
                        if self.recurrent_state_batch_backend_native_layers > 0
                            || self.recurrent_state_batch_cpu_direct_layers > 0
                            || self
                                .recurrent_state_batch_cpu_direct_materialized_from_backend_layers
                                > 0
                            || self.recurrent_state_batch_cpu_gathered_layers > 0
                            || self
                                .recurrent_state_batch_cpu_gathered_materialized_from_backend_layers
                                > 0
                        {
                            eprintln!(
                                "    BatchKind: native {} | cpu-direct {} | cpu-direct-from-backend {} | gathered {} | gathered-from-backend {}",
                                self.recurrent_state_batch_backend_native_layers,
                                self.recurrent_state_batch_cpu_direct_layers,
                                self.recurrent_state_batch_cpu_direct_materialized_from_backend_layers,
                                self.recurrent_state_batch_cpu_gathered_layers,
                                self.recurrent_state_batch_cpu_gathered_materialized_from_backend_layers
                            );
                        }
                    }
                    if self.recurrent_qkv_fast_path_eligible_layers == 0
                        && (self.recurrent_qkv_fast_reject_state_size_layers > 0
                            || self.recurrent_qkv_fast_reject_group_divisibility_layers > 0
                            || self.recurrent_qkv_fast_reject_missing_batch_scratches_layers > 0
                            || self.recurrent_qkv_fast_reject_q_capacity_layers > 0
                            || self.recurrent_qkv_fast_reject_k_capacity_layers > 0
                            || self.recurrent_qkv_fast_reject_v_capacity_layers > 0
                            || self.recurrent_qkv_fast_reject_gate_capacity_layers > 0
                            || self.recurrent_qkv_fast_reject_up_capacity_layers > 0)
                    {
                        eprintln!(
                            "    Reject:    state {} | group {} | scratch {} | q {} | k {} | v {} | gate {} | up {}",
                            self.recurrent_qkv_fast_reject_state_size_layers,
                            self.recurrent_qkv_fast_reject_group_divisibility_layers,
                            self.recurrent_qkv_fast_reject_missing_batch_scratches_layers,
                            self.recurrent_qkv_fast_reject_q_capacity_layers,
                            self.recurrent_qkv_fast_reject_k_capacity_layers,
                            self.recurrent_qkv_fast_reject_v_capacity_layers,
                            self.recurrent_qkv_fast_reject_gate_capacity_layers,
                            self.recurrent_qkv_fast_reject_up_capacity_layers
                        );
                    }
                }
            }
        }
        eprintln!(
            "  Dequant:   {:6.1}ms  ({:5.1}%)",
            self.dequant_ms, self.dequant_pct
        );
        eprintln!(
            "  RoPE:      {:6.1}ms  ({:5.1}%)",
            self.rope_ms, self.rope_pct
        );
        eprintln!(
            "  Norm:      {:6.1}ms  ({:5.1}%)",
            self.norm_ms, self.norm_pct
        );
        eprintln!(
            "  Other:     {:6.1}ms  ({:5.1}%)",
            (self.total_ms
                - self.gpu_ms
                - self.matmul_ms
                - self.attention_ms
                - self.recurrent_ms
                - self.dequant_ms
                - self.rope_ms
                - self.norm_ms)
                .max(0.0),
            self.other_pct,
        );
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_core::backend::BackendConfig;
    use ax_engine_core::model::config::{GateActivation, RopeScaling};

    fn tiny_qwen35_config() -> ModelConfig {
        ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 4,
            n_heads: 2,
            n_kv_heads: 2,
            embedding_dim: 8,
            head_dim: 4,
            intermediate_dim: 16,
            context_length: 32,
            vocab_size: 8,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            expert_intermediate_dim: None,
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(8),
            qwen35_ssm_state_size: Some(2),
            qwen35_ssm_time_step_rank: Some(4),
            qwen35_ssm_group_count: Some(2),
            gemma4_head_dim_swa: None,
            gemma4_head_dim_global: None,
            gemma4_n_kv_heads_swa: None,
            gemma4_n_kv_heads_global: None,
            gemma4_rope_dim_swa: None,
            gemma4_rope_dim_global: None,
            final_logit_softcapping: None,
        }
    }

    #[test]
    fn test_prefill_profile_config_defaults() {
        let c = PrefillProfileConfig::default();
        assert_eq!(c.prompt_tokens, 512);
        assert_eq!(c.warmup_iters, 1);
        assert_eq!(c.qwen35_shared_timeline_slots, 1);
        assert_eq!(
            c.qwen35_recurrent_state_mode,
            Qwen35RecurrentStateMode::Auto
        );
        assert_eq!(
            c.qwen35_alpha_beta_storage_mode,
            Qwen35AlphaBetaStorageMode::Auto
        );
        assert!(!c.qwen35_prime_slot_buffers);
        assert!(!c.qwen35_prewarm_prefill_same_kv);
        assert!(!c.qwen35_force_backend_state_batch);
        assert_eq!(c.local_hd128_route, LocalPrefillHd128Route::Auto);
    }

    #[test]
    fn test_merged_support_note_appends_prefill_profile_note() {
        let merged =
            merged_support_note(Some("support".into()), Some("profile: native unified path"))
                .unwrap();
        assert_eq!(merged, "support | profile: native unified path");
    }

    #[test]
    fn test_merged_support_note_uses_profile_note_when_support_note_missing() {
        let merged = merged_support_note(None, Some("profile: native unified path")).unwrap();
        assert_eq!(merged, "profile: native unified path");
    }

    #[test]
    fn test_prefill_profile_result_json() {
        let result = PrefillProfileResult {
            model: "test.gguf".into(),
            prompt_tokens: 512,
            effective_prompt_tokens: 512,
            total_ms: 100.0,
            tok_per_sec: 5120.0,
            effective_tok_per_sec: 5120.0,
            prefill_plan: "mode=gpu_batch".into(),
            prefill_mode: "gpu_batch".into(),
            prefill_route_family: "dense_gpu_batch".into(),
            prefill_route_detail: "generic_gpu_batch".into(),
            prefill_attention_route: None,
            prefill_qkv_plan: None,
            prefill_split_rope_append: None,
            q5k_prefill_mode: None,
            support_note: None,
            qwen35_shared_timeline_slots: 1,
            qwen35_shared_timeline_source_slot: None,
            qwen35_recurrent_state_mode: Qwen35RecurrentStateMode::Auto,
            qwen35_alpha_beta_storage_mode: Qwen35AlphaBetaStorageMode::Auto,
            qwen35_prime_slot_buffers: false,
            qwen35_prewarm_prefill_same_kv: false,
            qwen35_force_backend_state_batch: false,
            local_hd128_route: LocalPrefillHd128Route::Auto,
            qwen35_dtype_audit: None,
            prefill_command_buffers: 1.0,
            prefill_buffer_barriers: 10.0,
            prefill_command_buffers_per_tok: 1.0 / 512.0,
            prefill_buffer_barriers_per_tok: 10.0 / 512.0,
            gpu_pct: 95.0,
            gpu_encode_pct: 0.5,
            gpu_execute_pct: 94.0,
            gpu_execute_layers_pct: 0.0,
            gpu_execute_output_pct: 0.0,
            gpu_readback_pct: 0.1,
            gpu_encode_layer_norm_pct: 0.2,
            gpu_encode_layer_qkv_pct: 0.1,
            gpu_encode_layer_rope_pct: 0.05,
            gpu_encode_layer_kv_append_pct: 0.05,
            gpu_encode_layer_attention_pct: 0.05,
            gpu_encode_layer_out_proj_pct: 0.02,
            gpu_encode_layer_ffn_pct: 0.02,
            gpu_encode_layer_residual_pct: 0.01,
            matmul_pct: 0.0,
            attention_pct: 0.0,
            recurrent_pct: 0.0,
            recurrent_batch_conv_pct: 0.0,
            recurrent_batch_pack_pct: 0.0,
            recurrent_batch_gated_delta_pct: 0.0,
            recurrent_batch_unpack_pct: 0.0,
            dequant_pct: 0.0,
            rope_pct: 0.0,
            norm_pct: 0.0,
            other_pct: 5.0,
            gpu_ms: 95.0,
            gpu_encode_ms: 0.5,
            gpu_execute_ms: 94.0,
            gpu_execute_layers_ms: 0.0,
            gpu_execute_output_ms: 0.0,
            gpu_readback_ms: 0.1,
            gpu_encode_layer_norm_ms: 0.2,
            gpu_encode_layer_qkv_ms: 0.1,
            gpu_encode_layer_rope_ms: 0.05,
            gpu_encode_layer_kv_append_ms: 0.05,
            gpu_encode_layer_attention_ms: 0.05,
            gpu_encode_layer_out_proj_ms: 0.02,
            gpu_encode_layer_ffn_ms: 0.02,
            gpu_encode_layer_residual_ms: 0.01,
            matmul_ms: 0.0,
            attention_ms: 0.0,
            recurrent_ms: 0.0,
            recurrent_batch_conv_ms: 0.0,
            recurrent_batch_pack_ms: 0.0,
            recurrent_batch_gated_delta_ms: 0.0,
            recurrent_batch_unpack_ms: 0.0,
            recurrent_batch_qkv_handoff_ms: 0.0,
            recurrent_qkv_handoff_layers: 0,
            recurrent_qkv_handoff_fused_tail_layers: 0,
            recurrent_qkv_gpu_projection_layers: 0,
            recurrent_qkv_fast_path_eligible_layers: 0,
            recurrent_gpu_ssm_projection_layers: 0,
            recurrent_qkv_fast_reject_state_size_layers: 0,
            recurrent_qkv_fast_reject_group_divisibility_layers: 0,
            recurrent_qkv_fast_reject_missing_batch_scratches_layers: 0,
            recurrent_qkv_fast_reject_q_capacity_layers: 0,
            recurrent_qkv_fast_reject_k_capacity_layers: 0,
            recurrent_qkv_fast_reject_v_capacity_layers: 0,
            recurrent_qkv_fast_reject_gate_capacity_layers: 0,
            recurrent_qkv_fast_reject_up_capacity_layers: 0,
            recurrent_qkv_handoff_cpu_alias_layers: 0,
            recurrent_qkv_handoff_slot_buffer_layers: 0,
            recurrent_qkv_handoff_backend_carryover_layers: 0,
            recurrent_qkv_handoff_backend_zero_init_layers: 0,
            recurrent_qkv_handoff_cpu_materialization_layers: 0,
            recurrent_state_batch_backend_native_layers: 0,
            recurrent_state_batch_cpu_direct_layers: 0,
            recurrent_state_batch_cpu_direct_materialized_from_backend_layers: 0,
            recurrent_state_batch_cpu_gathered_layers: 0,
            recurrent_state_batch_cpu_gathered_materialized_from_backend_layers: 0,
            dequant_ms: 0.0,
            rope_ms: 0.0,
            norm_ms: 0.0,
        };
        let json = result.to_json().unwrap();
        assert!(json.contains("\"prompt_tokens\": 512"));
        assert!(json.contains("\"prefill_plan\": \"mode=gpu_batch\""));
        assert!(json.contains("\"prefill_mode\": \"gpu_batch\""));
        assert!(json.contains("\"prefill_route_family\": \"dense_gpu_batch\""));
    }

    #[test]
    fn test_qwen35_recurrent_state_mode_env_values() {
        assert_eq!(Qwen35RecurrentStateMode::Auto.as_env_value(), None);
        assert_eq!(
            Qwen35RecurrentStateMode::CpuAlias.as_env_value(),
            Some("cpu_alias")
        );
        assert_eq!(
            Qwen35RecurrentStateMode::SlotBuffer.as_env_value(),
            Some("slot_buffer")
        );
        assert_eq!(
            Qwen35RecurrentStateMode::BackendOwned.as_env_value(),
            Some("backend_owned")
        );
    }

    #[test]
    fn test_qwen35_alpha_beta_storage_mode_env_values() {
        assert_eq!(Qwen35AlphaBetaStorageMode::Auto.as_env_value(), None);
        assert_eq!(Qwen35AlphaBetaStorageMode::F32.as_env_value(), Some("f32"));
        assert_eq!(Qwen35AlphaBetaStorageMode::F16.as_env_value(), Some("f16"));
    }

    #[test]
    fn test_qwen35_recurrent_state_mode_auto_helper_is_prompt_aware() {
        assert_eq!(
            qwen35_prefill_recurrent_state_mode_auto_for_tokens(32),
            Qwen35RecurrentStateMode::BackendOwned
        );
        assert_eq!(
            qwen35_prefill_recurrent_state_mode_auto_for_tokens(64),
            Qwen35RecurrentStateMode::CpuAlias
        );
        assert_eq!(
            qwen35_prefill_recurrent_state_mode_auto_for_tokens(96),
            Qwen35RecurrentStateMode::BackendOwned
        );
    }

    #[test]
    fn test_qwen35_dtype_audit_serializes_requested_mode() {
        let audit = Qwen35PrefillDTypeAudit {
            requested_recurrent_state_mode: Qwen35RecurrentStateMode::SlotBuffer,
            effective_recurrent_state_mode: "slot_buffer".into(),
            requested_alpha_beta_storage_mode: Qwen35AlphaBetaStorageMode::F16,
            effective_alpha_beta_storage_dtype: "f16_storage_f32_compute".into(),
            requested_slot_buffer_priming: true,
            effective_slot_buffer_priming: true,
            requested_same_kv_prewarm: false,
            effective_same_kv_prewarm: false,
            requested_force_backend_state_batch: false,
            effective_force_backend_state_batch: false,
            runtime_batch_prefill_prefers_f16_io: true,
            dense_batch_projection_wrong_type_suspected: false,
            recurrent_state_logical_dtype: "f32".into(),
            recurrent_state_storage: "cpu_visible_vec_f32_or_shared_alias".into(),
            recurrent_snapshot_dtype: "f32".into(),
            recurrent_slot_mut_api_dtype: "&mut [f32]".into(),
            recurrent_batch_scratch_dtype: "f32".into(),
            recurrent_handoff_alpha_beta_dtype: "f32".into(),
            recurrent_handoff_observed_state_path: "slot_buffer_only".into(),
            recurrent_handoff_observed_state_owner: "backend_owned".into(),
            recurrent_handoff_cpu_alias_layers: 0,
            recurrent_handoff_slot_buffer_layers: 8,
            recurrent_handoff_backend_carryover_layers: 8,
            recurrent_handoff_backend_zero_init_layers: 0,
            recurrent_handoff_cpu_materialization_layers: 0,
            recurrent_handoff_fused_tail_layers: 0,
            recurrent_state_batch_kind: "backend_native".into(),
            recurrent_state_batch_backend_native_layers: 8,
            recurrent_state_batch_cpu_direct_layers: 0,
            recurrent_state_batch_cpu_direct_materialized_from_backend_layers: 0,
            recurrent_state_batch_cpu_gathered_layers: 0,
            recurrent_state_batch_cpu_gathered_materialized_from_backend_layers: 0,
            recurrent_f32_contract_ceiling_suspected: true,
        };
        let json = serde_json::to_string(&audit).unwrap();
        assert!(json.contains("\"requested_recurrent_state_mode\":\"slot_buffer\""));
        assert!(json.contains("\"effective_recurrent_state_mode\":\"slot_buffer\""));
        assert!(json.contains("\"requested_alpha_beta_storage_mode\":\"f16\""));
        assert!(json.contains("\"effective_slot_buffer_priming\":true"));
        assert!(json.contains("\"recurrent_handoff_slot_buffer_layers\":8"));
        assert!(json.contains("\"recurrent_handoff_observed_state_owner\":\"backend_owned\""));
        assert!(json.contains("\"recurrent_state_batch_kind\":\"backend_native\""));
    }

    #[test]
    fn test_build_qwen35_dtype_audit_reports_backend_owned_effective_mode() {
        let model_config = tiny_qwen35_config();
        let model = InferenceModel::with_backend(
            model_config,
            ax_engine_core::backend::create_backend(BackendConfig::default()).expect("backend"),
        )
        .expect("model");
        let recurrent_counters = ax_engine_core::backend::metal::Qwen35RecurrentBatchPerfCounters {
            qkv_handoff_slot_buffer_layers: 8,
            qkv_handoff_backend_carryover_layers: 8,
            qkv_handoff_layers: 8,
            ..Default::default()
        };
        let audit = build_qwen35_dtype_audit(
            &model,
            None,
            &recurrent_counters,
            128,
            Qwen35RecurrentStateMode::BackendOwned,
            Qwen35AlphaBetaStorageMode::Auto,
            false,
            false,
            false,
        )
        .expect("audit");
        assert_eq!(audit.effective_recurrent_state_mode, "backend_owned");
        assert_eq!(
            audit.recurrent_handoff_observed_state_owner,
            "backend_owned"
        );
        assert!(audit.effective_slot_buffer_priming);
        assert!(!audit.effective_same_kv_prewarm);
        assert!(!audit.effective_force_backend_state_batch);
    }

    #[test]
    fn test_build_qwen35_dtype_audit_auto_backend_owned_marks_effective_prime() {
        let model_config = tiny_qwen35_config();
        let model = InferenceModel::with_backend(
            model_config,
            ax_engine_core::backend::create_backend(BackendConfig::default()).expect("backend"),
        )
        .expect("model");
        let recurrent_counters = ax_engine_core::backend::metal::Qwen35RecurrentBatchPerfCounters {
            qkv_handoff_slot_buffer_layers: 8,
            qkv_handoff_backend_carryover_layers: 8,
            qkv_handoff_layers: 8,
            ..Default::default()
        };
        let audit = build_qwen35_dtype_audit(
            &model,
            None,
            &recurrent_counters,
            128,
            Qwen35RecurrentStateMode::Auto,
            Qwen35AlphaBetaStorageMode::Auto,
            false,
            false,
            false,
        )
        .expect("audit");
        assert_eq!(audit.effective_recurrent_state_mode, "backend_owned");
        assert!(audit.effective_slot_buffer_priming);
    }
}
