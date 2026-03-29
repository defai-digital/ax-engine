//! Prefill gap attribution and baseline comparison.
//!
//! This module wraps `prefill_profile` and adds:
//! - baseline ingestion from AX/llama-style JSON artifacts
//! - route classification
//! - command-buffer pressure classification
//! - actionable recommendation generation

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::prefill_profile::{self, PrefillProfileConfig, PrefillProfileResult};

#[derive(Debug, Clone)]
pub struct PrefillGapConfig {
    pub profile: PrefillProfileConfig,
    pub baseline_json: Option<String>,
    pub baseline_prefill_tok_per_sec: Option<f64>,
    pub baseline_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillBaseline {
    pub label: String,
    pub prefill_tok_per_sec: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillGapReport {
    pub result: PrefillProfileResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline: Option<PrefillBaseline>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_gap_pct: Option<f64>,
    pub route_family: String,
    pub route_detail: String,
    pub command_buffer_pressure: String,
    pub qwen35_fast_path_alive: bool,
    pub recommendations: Vec<String>,
}

impl PrefillGapReport {
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn print_summary(&self) {
        self.result.print_summary();
        eprintln!();
        eprintln!("=== Prefill Gap ===");
        eprintln!("Route:       {} / {}", self.route_family, self.route_detail);
        eprintln!("CmdPressure: {}", self.command_buffer_pressure);
        eprintln!(
            "Qwen35Fast:  {}",
            if self.qwen35_fast_path_alive {
                "alive"
            } else {
                "inactive"
            }
        );
        if let Some(baseline) = &self.baseline {
            eprintln!(
                "Baseline:    {} {:.1} tok/s",
                baseline.label, baseline.prefill_tok_per_sec
            );
        }
        if let Some(ratio) = self.prefill_ratio {
            eprintln!(
                "Ratio:       {:.1}% of baseline ({:+.1}% gap)",
                ratio * 100.0,
                self.prefill_gap_pct.unwrap_or(0.0)
            );
        }
        if !self.recommendations.is_empty() {
            eprintln!("Focus:");
            for rec in &self.recommendations {
                eprintln!("  - {rec}");
            }
        }
    }
}

pub fn run_prefill_gap(config: &PrefillGapConfig) -> anyhow::Result<PrefillGapReport> {
    run_prefill_gap_with_backend(
        config,
        ax_engine_core::backend::create_backend(ax_engine_core::backend::BackendConfig::default())?,
    )
}

pub fn run_prefill_gap_with_backend(
    config: &PrefillGapConfig,
    backend: Box<dyn ax_engine_core::backend::Backend>,
) -> anyhow::Result<PrefillGapReport> {
    let result = prefill_profile::run_prefill_profile_with_backend(&config.profile, backend)?;
    build_prefill_gap_report(
        result,
        config.baseline_json.as_deref(),
        config.baseline_prefill_tok_per_sec,
        config.baseline_label.as_deref(),
    )
}

pub fn build_prefill_gap_report(
    result: PrefillProfileResult,
    baseline_json: Option<&str>,
    baseline_prefill_tok_per_sec: Option<f64>,
    baseline_label: Option<&str>,
) -> anyhow::Result<PrefillGapReport> {
    let baseline = if let Some(path) = baseline_json {
        Some(load_prefill_baseline(path, baseline_label)?)
    } else {
        baseline_prefill_tok_per_sec.map(|prefill_tok_per_sec| PrefillBaseline {
            label: baseline_label.unwrap_or("baseline").to_string(),
            prefill_tok_per_sec,
            source: Some("inline".to_string()),
        })
    };

    let prefill_ratio = baseline.as_ref().and_then(|b| {
        if b.prefill_tok_per_sec > 0.0 {
            Some(result.tok_per_sec / b.prefill_tok_per_sec)
        } else {
            None
        }
    });
    let prefill_gap_pct = prefill_ratio.map(|ratio| (ratio - 1.0) * 100.0);
    let (route_family, route_detail) = classify_prefill_route(&result);
    let command_buffer_pressure = classify_command_buffer_pressure(&result);
    let qwen35_fast_path_alive = result.recurrent_qkv_fast_path_eligible_layers > 0
        && result.recurrent_qkv_handoff_layers > 0;
    let recommendations = build_recommendations(
        &result,
        &route_family,
        &command_buffer_pressure,
        prefill_ratio,
    );

    Ok(PrefillGapReport {
        result,
        baseline,
        prefill_ratio,
        prefill_gap_pct,
        route_family,
        route_detail,
        command_buffer_pressure,
        qwen35_fast_path_alive,
        recommendations,
    })
}

fn load_prefill_baseline(
    path: &str,
    label_override: Option<&str>,
) -> anyhow::Result<PrefillBaseline> {
    let content = fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    let prefill_tok_per_sec = json
        .get("prefill_tok_per_sec_median")
        .and_then(|v| v.as_f64())
        .or_else(|| json.get("prefill_tok_per_sec").and_then(|v| v.as_f64()))
        .or_else(|| json.get("tok_per_sec").and_then(|v| v.as_f64()))
        .ok_or_else(|| anyhow::anyhow!("baseline JSON missing prefill tok/s field"))?;
    let label = label_override
        .map(str::to_owned)
        .or_else(|| {
            json.get("label")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
        })
        .or_else(|| {
            Path::new(path)
                .file_name()
                .and_then(|s| s.to_str())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "baseline".to_string());
    let source = json
        .get("source")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .or_else(|| Some(path.to_string()));
    Ok(PrefillBaseline {
        label,
        prefill_tok_per_sec,
        source,
    })
}

fn classify_prefill_route(result: &PrefillProfileResult) -> (String, String) {
    if !result.prefill_route_family.is_empty() {
        let detail = if result.prefill_route_detail.is_empty() {
            "unknown".to_string()
        } else {
            result.prefill_route_detail.clone()
        };
        return (result.prefill_route_family.clone(), detail);
    }
    if result.prefill_plan.contains("qwen35_hybrid") {
        let detail = if result.recurrent_qkv_handoff_layers > 0 {
            "recurrent_handoff_fast_path"
        } else if result.recurrent_qkv_fast_path_eligible_layers > 0 {
            "recurrent_fast_path_without_handoff"
        } else {
            "recurrent_cpu_shaped_or_serial"
        };
        return ("qwen35_hybrid".to_string(), detail.to_string());
    }
    if result.prefill_plan.contains("mode=gpu_batch") {
        if let Some(attn_route) = result
            .prefill_plan
            .split_whitespace()
            .find_map(|part| part.strip_prefix("attn_route="))
        {
            return ("dense_gpu_batch".to_string(), attn_route.to_string());
        }
        return (
            "dense_gpu_batch".to_string(),
            "generic_gpu_batch".to_string(),
        );
    }
    if result.prefill_plan.contains("mode=serial") {
        return ("serial_prefill".to_string(), "cpu_or_fallback".to_string());
    }
    ("unknown".to_string(), "unknown".to_string())
}

fn classify_command_buffer_pressure(result: &PrefillProfileResult) -> String {
    let cmd_per_tok = result.prefill_command_buffers_per_tok;
    if cmd_per_tok <= 0.05 {
        "single_cb_like".to_string()
    } else if cmd_per_tok <= 0.5 {
        "low".to_string()
    } else if cmd_per_tok <= 1.5 {
        "moderate".to_string()
    } else {
        "high".to_string()
    }
}

fn build_recommendations(
    result: &PrefillProfileResult,
    route_family: &str,
    command_buffer_pressure: &str,
    prefill_ratio: Option<f64>,
) -> Vec<String> {
    let mut recs = Vec::new();
    if let Some(ratio) = prefill_ratio
        && ratio < 0.75
    {
        recs.push("Prefill gap is still large; prioritize execution-shape and runtime attribution over more profile tuning.".to_string());
    }
    if route_family == "qwen35_hybrid" {
        if let Some(audit) = &result.qwen35_dtype_audit
            && !audit.dense_batch_projection_wrong_type_suspected
            && audit.recurrent_f32_contract_ceiling_suspected
        {
            recs.push("Dense prefill dtype routing looks healthy; prioritize recurrent state/storage representation over more dense kernel tuning.".to_string());
        }
        if result.recurrent_qkv_handoff_slot_buffer_layers > 0
            && result.recurrent_qkv_handoff_cpu_materialization_layers > 0
        {
            recs.push("Slot-buffer handoff is still materializing CPU-visible recurrent state; prioritize device-primary recurrent ownership before more kernel tuning.".to_string());
        }
        if result.recurrent_qkv_handoff_slot_buffer_layers > 0
            && result.recurrent_qkv_handoff_backend_carryover_layers > 0
            && result.recurrent_qkv_handoff_cpu_materialization_layers == 0
        {
            recs.push("Slot-buffer handoff is already reusing backend-resident recurrent state; remaining gap is more likely execution-shape and persistent residency than wrong-type routing.".to_string());
        }
        if result.recurrent_qkv_handoff_slot_buffer_layers > 0
            && result.recurrent_qkv_handoff_backend_zero_init_layers > 0
            && result.recurrent_qkv_handoff_cpu_materialization_layers == 0
            && result.recurrent_qkv_handoff_backend_carryover_layers == 0
        {
            recs.push("Slot-buffer handoff is avoiding CPU copies only for pristine zero state; prioritize persistent backend-owned recurrent state for mutated slots next.".to_string());
        }
        if result.recurrent_state_batch_cpu_gathered_layers > 0
            || result.recurrent_state_batch_cpu_gathered_materialized_from_backend_layers > 0
        {
            recs.push("Recurrent batch interface is still falling back to gathered CPU state; prioritize slot-native batch execution before more micro-tuning.".to_string());
        }
        if result.recurrent_state_batch_backend_native_layers > 0
            && result.recurrent_state_batch_cpu_gathered_layers == 0
            && result.recurrent_state_batch_cpu_gathered_materialized_from_backend_layers == 0
        {
            recs.push("Recurrent batch interface is already hitting backend-native slot-buffer execution; focus next on persistent ownership and submission collapse.".to_string());
        }
        if result.recurrent_qkv_handoff_layers == 0 {
            recs.push("Qwen3.5 recurrent qkv-handoff fast path is inactive; restore handoff before any further tuning.".to_string());
        } else if command_buffer_pressure == "high" {
            recs.push("Collapse Qwen3.5 recurrent prefill submissions; current command-buffer density still indicates fragmented recurrent execution.".to_string());
        } else {
            recs.push("Continue pushing device-primary recurrent residency and state_indices-grade batching; profile policy is no longer the main blocker.".to_string());
        }
    } else if route_family == "dense_gpu_batch" {
        if command_buffer_pressure != "single_cb_like" && command_buffer_pressure != "low" {
            recs.push("Audit dense prefill execution graph and command-buffer shape; remaining gap is more likely graph/runtime than kernel constants.".to_string());
        }
        if result.dequant_pct > 5.0 {
            recs.push("Prompt front-end dequant/gather is materially visible; evaluate batch embedding/dequant cleanup.".to_string());
        }
        if result.matmul_pct > 60.0 {
            recs.push("Dense prefill remains matmul-dominant; verify route coverage before introducing new kernel variants.".to_string());
        }
    }
    if recs.is_empty() {
        recs.push("No dominant single bottleneck detected; use this report as the stable regression baseline for the next prefill experiment.".to_string());
    }
    recs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefill_profile::PrefillProfileResult;

    fn sample_prefill_result() -> PrefillProfileResult {
        PrefillProfileResult {
            model: "test.gguf".into(),
            prompt_tokens: 64,
            effective_prompt_tokens: 64,
            total_ms: 100.0,
            tok_per_sec: 200.0,
            effective_tok_per_sec: 200.0,
            prefill_plan: "mode=gpu_batch kv=qwen35_hybrid recurrent=backend_owned".into(),
            prefill_mode: "gpu_batch".into(),
            prefill_route_family: "qwen35_hybrid".into(),
            prefill_route_detail: "recurrent_handoff_fast_path".into(),
            prefill_attention_route: None,
            prefill_qkv_plan: None,
            prefill_split_rope_append: None,
            q5k_prefill_mode: None,
            support_note: None,
            kernel_profile_path: None,
            qwen35_shared_timeline_slots: 1,
            qwen35_shared_timeline_source_slot: None,
            qwen35_recurrent_state_mode: crate::prefill_profile::Qwen35RecurrentStateMode::Auto,
            qwen35_alpha_beta_storage_mode:
                crate::prefill_profile::Qwen35AlphaBetaStorageMode::Auto,
            qwen35_prime_slot_buffers: false,
            qwen35_prewarm_prefill_same_kv: false,
            qwen35_force_backend_state_batch: false,
            qwen35_dtype_audit: Some(crate::prefill_profile::Qwen35PrefillDTypeAudit {
                requested_recurrent_state_mode:
                    crate::prefill_profile::Qwen35RecurrentStateMode::Auto,
                effective_recurrent_state_mode: "mixed".into(),
                requested_alpha_beta_storage_mode:
                    crate::prefill_profile::Qwen35AlphaBetaStorageMode::Auto,
                effective_alpha_beta_storage_dtype: "f32".into(),
                requested_slot_buffer_priming: false,
                effective_slot_buffer_priming: false,
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
                recurrent_handoff_observed_state_path: "mixed".into(),
                recurrent_handoff_observed_state_owner: "mixed".into(),
                recurrent_handoff_cpu_alias_layers: 12,
                recurrent_handoff_slot_buffer_layers: 12,
                recurrent_handoff_backend_carryover_layers: 8,
                recurrent_handoff_backend_zero_init_layers: 0,
                recurrent_handoff_cpu_materialization_layers: 4,
                recurrent_handoff_fused_tail_layers: 0,
                recurrent_state_batch_kind: "mixed".into(),
                recurrent_state_batch_backend_native_layers: 0,
                recurrent_state_batch_cpu_direct_layers: 12,
                recurrent_state_batch_cpu_direct_materialized_from_backend_layers: 0,
                recurrent_state_batch_cpu_gathered_layers: 0,
                recurrent_state_batch_cpu_gathered_materialized_from_backend_layers: 12,
                recurrent_f32_contract_ceiling_suspected: true,
            }),
            prefill_command_buffers: 129.0,
            prefill_buffer_barriers: 0.0,
            prefill_command_buffers_per_tok: 2.0,
            prefill_buffer_barriers_per_tok: 0.0,
            gpu_pct: 90.0,
            gpu_encode_pct: 0.0,
            gpu_execute_pct: 90.0,
            gpu_execute_layers_pct: 0.0,
            gpu_execute_output_pct: 0.0,
            gpu_readback_pct: 0.0,
            gpu_encode_layer_norm_pct: 0.0,
            gpu_encode_layer_qkv_pct: 0.0,
            gpu_encode_layer_rope_pct: 0.0,
            gpu_encode_layer_kv_append_pct: 0.0,
            gpu_encode_layer_attention_pct: 0.0,
            gpu_encode_layer_out_proj_pct: 0.0,
            gpu_encode_layer_ffn_pct: 0.0,
            gpu_encode_layer_residual_pct: 0.0,
            matmul_pct: 70.0,
            attention_pct: 0.0,
            recurrent_pct: 20.0,
            recurrent_batch_conv_pct: 0.0,
            recurrent_batch_pack_pct: 0.0,
            recurrent_batch_gated_delta_pct: 0.0,
            recurrent_batch_unpack_pct: 0.0,
            dequant_pct: 0.0,
            rope_pct: 0.0,
            norm_pct: 0.0,
            other_pct: 10.0,
            gpu_ms: 90.0,
            gpu_encode_ms: 0.0,
            gpu_execute_ms: 90.0,
            gpu_execute_layers_ms: 0.0,
            gpu_execute_output_ms: 0.0,
            gpu_readback_ms: 0.0,
            gpu_encode_layer_norm_ms: 0.0,
            gpu_encode_layer_qkv_ms: 0.0,
            gpu_encode_layer_rope_ms: 0.0,
            gpu_encode_layer_kv_append_ms: 0.0,
            gpu_encode_layer_attention_ms: 0.0,
            gpu_encode_layer_out_proj_ms: 0.0,
            gpu_encode_layer_ffn_ms: 0.0,
            gpu_encode_layer_residual_ms: 0.0,
            matmul_ms: 70.0,
            attention_ms: 0.0,
            recurrent_ms: 20.0,
            recurrent_batch_conv_ms: 0.0,
            recurrent_batch_pack_ms: 0.0,
            recurrent_batch_gated_delta_ms: 0.0,
            recurrent_batch_unpack_ms: 0.0,
            recurrent_batch_qkv_handoff_ms: 10.0,
            recurrent_qkv_handoff_layers: 24,
            recurrent_qkv_handoff_fused_tail_layers: 0,
            recurrent_qkv_gpu_projection_layers: 24,
            recurrent_qkv_fast_path_eligible_layers: 24,
            recurrent_gpu_ssm_projection_layers: 24,
            recurrent_qkv_fast_reject_state_size_layers: 0,
            recurrent_qkv_fast_reject_group_divisibility_layers: 0,
            recurrent_qkv_fast_reject_missing_batch_scratches_layers: 0,
            recurrent_qkv_fast_reject_q_capacity_layers: 0,
            recurrent_qkv_fast_reject_k_capacity_layers: 0,
            recurrent_qkv_fast_reject_v_capacity_layers: 0,
            recurrent_qkv_fast_reject_gate_capacity_layers: 0,
            recurrent_qkv_fast_reject_up_capacity_layers: 0,
            recurrent_qkv_handoff_cpu_alias_layers: 12,
            recurrent_qkv_handoff_slot_buffer_layers: 12,
            recurrent_qkv_handoff_backend_carryover_layers: 8,
            recurrent_qkv_handoff_backend_zero_init_layers: 0,
            recurrent_qkv_handoff_cpu_materialization_layers: 4,
            recurrent_state_batch_backend_native_layers: 0,
            recurrent_state_batch_cpu_direct_layers: 12,
            recurrent_state_batch_cpu_direct_materialized_from_backend_layers: 0,
            recurrent_state_batch_cpu_gathered_layers: 0,
            recurrent_state_batch_cpu_gathered_materialized_from_backend_layers: 12,
            dequant_ms: 0.0,
            rope_ms: 0.0,
            norm_ms: 0.0,
        }
    }

    #[test]
    fn test_classify_prefill_route_qwen35_handoff() {
        let result = sample_prefill_result();
        let (family, detail) = classify_prefill_route(&result);
        assert_eq!(family, "qwen35_hybrid");
        assert_eq!(detail, "recurrent_handoff_fast_path");
    }

    #[test]
    fn test_build_prefill_gap_report_from_inline_baseline() {
        let report =
            build_prefill_gap_report(sample_prefill_result(), None, Some(400.0), Some("llama"))
                .unwrap();
        assert_eq!(report.baseline.unwrap().label, "llama");
        assert_eq!(report.command_buffer_pressure, "high");
        assert!(report.prefill_ratio.unwrap() < 1.0);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_load_prefill_baseline_reads_bench_json() {
        let dir = std::env::temp_dir();
        let path = dir.join("ax-engine-prefill-baseline-test.json");
        std::fs::write(
            &path,
            r#"{
              "label":"llama.cpp",
              "prefill_tok_per_sec_median": 720.5,
              "source":"unit"
            }"#,
        )
        .unwrap();
        let baseline = load_prefill_baseline(path.to_str().unwrap(), None).unwrap();
        assert_eq!(baseline.label, "llama.cpp");
        assert_eq!(baseline.prefill_tok_per_sec, 720.5);
        let _ = std::fs::remove_file(path);
    }
}
