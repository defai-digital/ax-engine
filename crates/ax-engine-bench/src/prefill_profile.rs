//! Batched-prefill profiler.
//!
//! Measures the real `forward_batch` path used by throughput-mode inference and
//! records coarse timing around host setup, GPU execution, and readback. This
//! is intentionally separate from the decode hot-path profiler because the
//! performance questions are different: prefill is dominated by one batched GPU
//! pass, not repeated single-token decode steps.

use std::path::Path;
use std::time::Duration;

use ax_engine_core::gguf::MappedModel;
use ax_engine_core::metrics::OpBreakdown;
use ax_engine_core::metrics::counters::OpTimer;
use ax_engine_core::model::{LlamaModel, ModelConfig, WeightStore};
use ax_engine_core::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};

/// Prefill profiler configuration.
pub struct PrefillProfileConfig {
    /// Model path (GGUF file).
    pub model_path: String,
    /// Number of prompt tokens to process.
    pub prompt_tokens: usize,
    /// Number of unprofiled warmup prefills before measurement.
    pub warmup_iters: usize,
    /// Optional kernel profile override path used for this run.
    pub kernel_profile_path: Option<String>,
}

impl Default for PrefillProfileConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt_tokens: 512,
            warmup_iters: 1,
            kernel_profile_path: None,
        }
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

fn prefill_profile_support_note(model: &LlamaModel) -> Option<&'static str> {
    if model.arch_name() == "qwen35" && model.use_gpu_decode() && model.metal_device().is_some() {
        Some(
            "profile: Qwen3.5 prefill timing follows the native unified batch path; per-op buckets are still incomplete, so wall time, GPU aggregate, and submit counters are authoritative",
        )
    } else {
        None
    }
}

/// Result of a prefill profiling run.
#[derive(Debug, Serialize, Deserialize)]
pub struct PrefillProfileResult {
    pub model: String,
    pub prompt_tokens: usize,
    pub total_ms: f64,
    pub tok_per_sec: f64,
    #[serde(default)]
    pub prefill_plan: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q5k_prefill_mode: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub support_note: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_profile_path: Option<String>,
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
    pub dequant_ms: f64,
    #[serde(default)]
    pub rope_ms: f64,
    #[serde(default)]
    pub norm_ms: f64,
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
    crate::configure_backend_for_model(&*backend, &config.model_path, &mapped)?;
    let model_config = ModelConfig::from_gguf(&mapped.header)?;
    let tokenizer = Tokenizer::from_gguf(&mapped.header)?;
    let model = LlamaModel::with_backend(model_config.clone(), backend)?;
    crate::report_planned_kv_budget(&mapped, &model)?;
    let support_note = merged_support_note(
        crate::support_note(&mapped),
        prefill_profile_support_note(&model),
    );
    let weights = WeightStore::new(&mapped);

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
        let mut warmup_logits = vec![0.0f32; vocab_size];
        warmup_logits.fill(0.0);
        model.forward_batch(&prompt_tokens, &mut warmup_kv, &weights, &mut warmup_logits)?;
    }

    let mut kv = model.create_model_kv_for_weights(&weights);
    let mut logits = vec![0.0f32; vocab_size];

    let mut ops = OpBreakdown::new();
    model.reset_metal_perf_counters();
    let wall_timer = OpTimer::start();
    logits.fill(0.0);
    model.forward_batch_profiled(&prompt_tokens, &mut kv, &weights, &mut logits, &mut ops)?;
    let wall_time = wall_timer.elapsed();
    let counters = model.read_metal_perf_counters();

    let wall_ms = wall_time.as_secs_f64() * 1000.0;
    let tracked_ms = ops.total().as_secs_f64() * 1000.0;
    let other_ms = (wall_ms - tracked_ms).max(0.0);
    let q5k_prefill_mode = crate::q5k_prefill_mode(&prefill_plan);
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
        total_ms: wall_ms,
        tok_per_sec: if wall_time.as_secs_f64() > 0.0 {
            prompt_tokens.len() as f64 / wall_time.as_secs_f64()
        } else {
            0.0
        },
        prefill_plan,
        q5k_prefill_mode,
        support_note,
        kernel_profile_path: config.kernel_profile_path.clone(),
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
    pub fn print_summary(&self) {
        eprintln!();
        eprintln!("=== Prefill Profile ===");
        eprintln!("Model:       {}", self.model);
        eprintln!("Prompt:      {} tokens", self.prompt_tokens);
        eprintln!("PrefillPlan: {}", self.prefill_plan);
        if let Some(mode) = &self.q5k_prefill_mode {
            eprintln!("Q5KPrefill:  {mode}");
        }
        if let Some(note) = &self.support_note {
            eprintln!("Support:     {note}");
        }
        if let Some(path) = &self.kernel_profile_path {
            eprintln!("KernelProf:  {path}");
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

    #[test]
    fn test_prefill_profile_config_defaults() {
        let c = PrefillProfileConfig::default();
        assert_eq!(c.prompt_tokens, 512);
        assert_eq!(c.warmup_iters, 1);
        assert_eq!(c.kernel_profile_path, None);
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
            total_ms: 100.0,
            tok_per_sec: 5120.0,
            prefill_plan: "mode=gpu_batch".into(),
            q5k_prefill_mode: None,
            support_note: None,
            kernel_profile_path: None,
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
            dequant_ms: 0.0,
            rope_ms: 0.0,
            norm_ms: 0.0,
        };
        let json = result.to_json().unwrap();
        assert!(json.contains("\"prompt_tokens\": 512"));
        assert!(json.contains("\"prefill_plan\": \"mode=gpu_batch\""));
    }
}
