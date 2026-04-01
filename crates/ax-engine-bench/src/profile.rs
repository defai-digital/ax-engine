//! Decode hot-path profiler.
//!
//! Runs inference with observational timing on the latency-mode decode path.
//! The profiler must not change command submission shape; it records coarse GPU
//! buckets and CPU-side timings around the same single-token decode path used
//! by normal latency-mode execution.

use std::path::Path;
use std::time::Duration;

use ax_engine_core::gguf::MappedModel;
use ax_engine_core::metrics::OpBreakdown;
use ax_engine_core::metrics::counters::OpTimer;
use ax_engine_core::model::{
    DecodeIntent, LlamaModel, ModelConfig, WeightStore, select_decode_mode,
};
use ax_engine_core::sampling::{Sampler, SamplingConfig};
use ax_engine_core::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};

/// Profile configuration.
pub struct ProfileConfig {
    /// Model path (GGUF file).
    pub model_path: String,
    /// Number of warmup tokens to generate before profiling.
    pub warmup_tokens: usize,
    /// Number of tokens to profile.
    pub profile_tokens: usize,
    /// Optional kernel profile override path used for this run.
    pub kernel_profile_path: Option<String>,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            warmup_tokens: 16,
            profile_tokens: 64,
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

fn profile_support_note(model: &LlamaModel) -> Option<&'static str> {
    if model.arch_name() == "qwen35" && model.use_gpu_decode() && model.metal_device().is_some() {
        Some(
            "profile: Qwen3.5 decode timing breakdown follows the native hybrid single-CB path; encode buckets are coarse CPU-side issue timings and execute remains aggregate",
        )
    } else {
        None
    }
}

/// Result of a profiling run.
#[derive(Debug, Serialize, Deserialize)]
pub struct ProfileResult {
    /// Model path.
    pub model: String,
    /// Number of profiled tokens.
    pub tokens: usize,
    /// Total wall-clock time for profiled tokens.
    pub total_ms: f64,
    /// Average per-token latency.
    pub avg_tok_ms: f64,
    /// Benchmark intent for this profile run.
    #[serde(default)]
    pub decode_intent: String,
    /// Decode mode used during profiling.
    #[serde(default)]
    pub decode_mode: String,
    /// Compact prefill execution-plan summary.
    #[serde(default)]
    pub prefill_plan: String,
    /// Parsed `mode=...` from the prefill plan, when present.
    #[serde(default)]
    pub prefill_mode: String,
    /// Normalized prefill runtime family derived from the plan.
    #[serde(default)]
    pub prefill_route_family: String,
    /// Normalized prefill runtime detail derived from the plan.
    #[serde(default)]
    pub prefill_route_detail: String,
    /// Parsed `attn_route=...` from the prefill plan, when present.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_attention_route: Option<String>,
    /// Parsed `qkv=...` from the prefill plan, when present.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_qkv_plan: Option<String>,
    /// Parsed `split_rope=...` from the prefill plan, when present.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_split_rope_append: Option<bool>,
    /// Parsed `q5k_prefill=...` mode from the prefill plan, when present.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q5k_prefill_mode: Option<String>,
    /// Compact decode execution-plan summary.
    #[serde(default)]
    pub decode_plan: String,
    /// Optional support-boundary note for the loaded model quant family.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub support_note: Option<String>,
    /// Optional explanation when a faster mode was not used.
    #[serde(default)]
    pub decode_fallback_reason: Option<String>,
    /// Optional kernel profile override path used for this run.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_profile_path: Option<String>,
    /// Total Metal command buffers used during the profiled decode loop.
    #[serde(default)]
    pub decode_command_buffers: f64,
    /// Total explicit Metal buffer barriers used during the profiled decode loop.
    #[serde(default)]
    pub decode_buffer_barriers: f64,
    /// Average Metal command buffers per profiled token.
    #[serde(default)]
    pub decode_command_buffers_per_tok: f64,
    /// Average explicit Metal buffer barriers per profiled token.
    #[serde(default)]
    pub decode_buffer_barriers_per_tok: f64,
    /// Maximum Metal command buffers observed for any profiled token.
    #[serde(default)]
    pub decode_command_buffers_per_tok_max: f64,
    /// Maximum explicit Metal buffer barriers observed for any profiled token.
    #[serde(default)]
    pub decode_buffer_barriers_per_tok_max: f64,
    /// GPU decode wall time percentage (Metal decode path end-to-end).
    #[serde(default)]
    pub gpu_pct: f64,
    /// Host-side encode/setup share inside profiled GPU decode step.
    #[serde(default)]
    pub gpu_encode_pct: f64,
    /// GPU execute/sync share inside profiled GPU decode step.
    #[serde(default)]
    pub gpu_execute_pct: f64,
    /// GPU execute share for transformer layers.
    #[serde(default)]
    pub gpu_execute_layers_pct: f64,
    /// GPU execute share for final norm + lm_head output stage.
    #[serde(default)]
    pub gpu_execute_output_pct: f64,
    /// GPU readback share.
    #[serde(default)]
    pub gpu_readback_pct: f64,
    /// GPU encode layer RMSNorm share.
    #[serde(default)]
    pub gpu_encode_layer_norm_pct: f64,
    /// GPU encode layer QKV projection share.
    #[serde(default)]
    pub gpu_encode_layer_qkv_pct: f64,
    /// GPU encode layer RoPE share.
    #[serde(default)]
    pub gpu_encode_layer_rope_pct: f64,
    /// GPU encode layer KV append share.
    #[serde(default)]
    pub gpu_encode_layer_kv_append_pct: f64,
    /// GPU encode layer attention kernel share.
    #[serde(default)]
    pub gpu_encode_layer_attention_pct: f64,
    /// GPU encode layer output projection share.
    #[serde(default)]
    pub gpu_encode_layer_out_proj_pct: f64,
    /// GPU encode layer FFN block share.
    #[serde(default)]
    pub gpu_encode_layer_ffn_pct: f64,
    /// GPU encode layer residual adds share.
    #[serde(default)]
    pub gpu_encode_layer_residual_pct: f64,
    /// Matmul percentage of total tracked time.
    pub matmul_pct: f64,
    /// Input-side projection matmul percentage.
    #[serde(default)]
    pub matmul_input_proj_pct: f64,
    /// Output-side projection matmul percentage.
    #[serde(default)]
    pub matmul_output_proj_pct: f64,
    /// LM-head matmul percentage.
    #[serde(default)]
    pub matmul_lm_head_pct: f64,
    /// Attention percentage.
    pub attention_pct: f64,
    /// Qwen3.5 recurrent-block percentage.
    #[serde(default)]
    pub recurrent_pct: f64,
    /// Dequant percentage.
    pub dequant_pct: f64,
    /// RoPE percentage.
    pub rope_pct: f64,
    /// Norm percentage.
    pub norm_pct: f64,
    /// Sampling percentage.
    pub sampling_pct: f64,
    /// Unaccounted time percentage (allocation, cache ops, etc.).
    pub other_pct: f64,
    /// Raw GPU decode wall time in milliseconds.
    #[serde(default)]
    pub gpu_ms: f64,
    /// Raw GPU encode/setup time in milliseconds.
    #[serde(default)]
    pub gpu_encode_ms: f64,
    /// Raw GPU execute/sync time in milliseconds.
    #[serde(default)]
    pub gpu_execute_ms: f64,
    /// Raw GPU execute time for transformer layers in milliseconds.
    #[serde(default)]
    pub gpu_execute_layers_ms: f64,
    /// Raw GPU execute time for output stage in milliseconds.
    #[serde(default)]
    pub gpu_execute_output_ms: f64,
    /// Raw GPU readback time in milliseconds.
    #[serde(default)]
    pub gpu_readback_ms: f64,
    /// Raw GPU encode layer RMSNorm time in milliseconds.
    #[serde(default)]
    pub gpu_encode_layer_norm_ms: f64,
    /// Raw GPU encode layer QKV projection time in milliseconds.
    #[serde(default)]
    pub gpu_encode_layer_qkv_ms: f64,
    /// Raw GPU encode layer RoPE time in milliseconds.
    #[serde(default)]
    pub gpu_encode_layer_rope_ms: f64,
    /// Raw GPU encode layer KV append time in milliseconds.
    #[serde(default)]
    pub gpu_encode_layer_kv_append_ms: f64,
    /// Raw GPU encode layer attention kernel time in milliseconds.
    #[serde(default)]
    pub gpu_encode_layer_attention_ms: f64,
    /// Raw GPU encode layer output projection time in milliseconds.
    #[serde(default)]
    pub gpu_encode_layer_out_proj_ms: f64,
    /// Raw GPU encode layer FFN time in milliseconds.
    #[serde(default)]
    pub gpu_encode_layer_ffn_ms: f64,
    /// Raw GPU encode layer residual-add time in milliseconds.
    #[serde(default)]
    pub gpu_encode_layer_residual_ms: f64,
    /// Raw matmul time in milliseconds.
    pub matmul_ms: f64,
    /// Raw input-side projection matmul time in milliseconds.
    #[serde(default)]
    pub matmul_input_proj_ms: f64,
    /// Raw output-side projection matmul time in milliseconds.
    #[serde(default)]
    pub matmul_output_proj_ms: f64,
    /// Raw LM-head matmul time in milliseconds.
    #[serde(default)]
    pub matmul_lm_head_ms: f64,
    /// Raw attention time in milliseconds.
    pub attention_ms: f64,
    /// Raw Qwen3.5 recurrent-block time in milliseconds.
    #[serde(default)]
    pub recurrent_ms: f64,
    /// Raw dequant time in milliseconds.
    pub dequant_ms: f64,
    /// Raw rope time in milliseconds.
    pub rope_ms: f64,
    /// Raw norm time in milliseconds.
    pub norm_ms: f64,
    /// Raw sampling time in milliseconds.
    pub sampling_ms: f64,
}

#[derive(Debug, Clone)]
pub struct KernelRegression {
    pub op: &'static str,
    pub baseline_ms: f64,
    pub current_ms: f64,
    pub delta_ms: f64,
    pub delta_pct_vs_baseline: f64,
    pub share_delta_pp: f64,
}

#[derive(Debug, Clone)]
pub struct ProfileComparison {
    pub baseline_total_ms: f64,
    pub current_total_ms: f64,
    pub baseline_avg_tok_ms: f64,
    pub current_avg_tok_ms: f64,
    pub regressions: Vec<KernelRegression>,
}

/// Run the decode hot-path profiler.
pub fn run_profile(config: &ProfileConfig) -> anyhow::Result<ProfileResult> {
    run_profile_with_backend(
        config,
        ax_engine_core::backend::create_backend(ax_engine_core::backend::BackendConfig::default())?,
    )
}

pub fn run_profile_with_backend(
    config: &ProfileConfig,
    backend: Box<dyn ax_engine_core::backend::Backend>,
) -> anyhow::Result<ProfileResult> {
    let mapped = MappedModel::open(Path::new(&config.model_path))?;
    let model_config = ModelConfig::from_gguf(&mapped.header)?;
    crate::configure_backend_for_model(&*backend, &config.model_path, &mapped, &model_config)?;
    let tokenizer = Tokenizer::from_gguf(&mapped.header)?;
    let model = LlamaModel::with_backend(model_config.clone(), backend)?;
    crate::report_planned_kv_budget(&mapped, &model)?;
    let support_note =
        merged_support_note(crate::support_note(&mapped), profile_support_note(&model));
    let weights = WeightStore::new(&mapped);

    let vocab_size = model_config.vocab_size as usize;
    let sampling_cfg = SamplingConfig::default();
    let mut sampler = Sampler::new(sampling_cfg);

    eprintln!(
        "Profiling: {} layers, {:.0}MB, {} warmup + {} profile tokens",
        model_config.n_layers,
        mapped.total_tensor_bytes() as f64 / 1024.0 / 1024.0,
        config.warmup_tokens,
        config.profile_tokens,
    );

    let mut kv = model.create_model_kv_for_weights(&weights);
    let mut logits = vec![0.0f32; vocab_size];

    // Seed the KV with the same batched prefill path used by the normal
    // runtime so profile baselines reflect the real warm-start state.
    let prompt_tokens = tokenizer.encode("The meaning of life is", true);
    let prefill_plan = model.prefill_plan_summary(&weights, &kv, prompt_tokens.len())?;
    logits.fill(0.0);
    model.forward_batch(&prompt_tokens, &mut kv, &weights, &mut logits)?;

    let mut position = prompt_tokens.len();
    let mut next_token = sampler.sample(&mut logits, &prompt_tokens);

    // Warmup decode (un-profiled, settles caches and branch predictors)
    for _ in 0..config.warmup_tokens {
        if tokenizer.is_eos(next_token) {
            break;
        }
        logits.fill(0.0);
        model.forward_single(next_token, position, &mut kv, &weights, &mut logits)?;
        position += 1;
        next_token = sampler.sample(&mut logits, &[]);
    }

    eprintln!("Warmup done at position {position}, starting profile...");

    let selection = select_decode_mode(&model, &kv, DecodeIntent::Latency, false);
    let plan_summary = model.decode_plan_summary(&kv, DecodeIntent::Latency, false);

    // Profiled decode
    let mut ops = OpBreakdown::new();
    let mut tokens_generated = 0usize;
    let wall_timer = OpTimer::start();
    let mut decode_command_buffers = 0.0f64;
    let mut decode_buffer_barriers = 0.0f64;
    let mut decode_command_buffers_per_tok_max = 0.0f64;
    let mut decode_buffer_barriers_per_tok_max = 0.0f64;

    for _ in 0..config.profile_tokens {
        if tokenizer.is_eos(next_token) {
            break;
        }

        logits.fill(0.0);
        model.reset_metal_perf_counters();
        model.forward_single_profiled(
            next_token,
            position,
            &mut kv,
            &weights,
            &mut logits,
            &mut ops,
        )?;

        // Track sampling time
        let t = OpTimer::start();
        next_token = sampler.sample(&mut logits, &[]);
        ops.sampling += t.elapsed();

        let decode_counters = model.read_metal_perf_counters();
        decode_command_buffers += decode_counters.command_buffers as f64;
        decode_buffer_barriers += decode_counters.buffer_barriers as f64;
        decode_command_buffers_per_tok_max =
            decode_command_buffers_per_tok_max.max(decode_counters.command_buffers as f64);
        decode_buffer_barriers_per_tok_max =
            decode_buffer_barriers_per_tok_max.max(decode_counters.buffer_barriers as f64);

        position += 1;
        tokens_generated += 1;
    }

    let wall_time = wall_timer.elapsed();

    // Compute breakdown
    let tracked = ops.total();
    let wall_ms = wall_time.as_secs_f64() * 1000.0;
    let tracked_ms = tracked.as_secs_f64() * 1000.0;
    let other_ms = (wall_ms - tracked_ms).max(0.0);

    let pct = |d: Duration| -> f64 {
        if wall_ms > 0.0 {
            d.as_secs_f64() * 1000.0 / wall_ms * 100.0
        } else {
            0.0
        }
    };

    let prefill_mode = crate::prefill_mode(&prefill_plan);
    let prefill_route_family = crate::prefill_route_family(&prefill_plan);
    let prefill_route_detail = crate::prefill_route_detail(&prefill_plan);
    let prefill_attention_route = crate::prefill_plan_field(&prefill_plan, "attn_route");
    let prefill_qkv_plan = crate::prefill_plan_field(&prefill_plan, "qkv");
    let prefill_split_rope_append = crate::prefill_bool_field(&prefill_plan, "split_rope");
    Ok(ProfileResult {
        model: config.model_path.clone(),
        tokens: tokens_generated,
        total_ms: wall_ms,
        avg_tok_ms: if tokens_generated > 0 {
            wall_ms / tokens_generated as f64
        } else {
            0.0
        },
        decode_intent: selection.intent.to_string(),
        decode_mode: selection.mode.to_string(),
        prefill_mode,
        prefill_route_family,
        prefill_route_detail,
        prefill_attention_route,
        prefill_qkv_plan,
        prefill_split_rope_append,
        q5k_prefill_mode: crate::q5k_prefill_mode(&prefill_plan),
        prefill_plan,
        decode_plan: plan_summary,
        support_note,
        decode_fallback_reason: selection.fallback_reason,
        kernel_profile_path: config.kernel_profile_path.clone(),
        decode_command_buffers,
        decode_buffer_barriers,
        decode_command_buffers_per_tok: if tokens_generated > 0 {
            decode_command_buffers / tokens_generated as f64
        } else {
            0.0
        },
        decode_buffer_barriers_per_tok: if tokens_generated > 0 {
            decode_buffer_barriers / tokens_generated as f64
        } else {
            0.0
        },
        decode_command_buffers_per_tok_max,
        decode_buffer_barriers_per_tok_max,
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
        matmul_input_proj_pct: pct(ops.matmul_input_proj),
        matmul_output_proj_pct: pct(ops.matmul_output_proj),
        matmul_lm_head_pct: pct(ops.matmul_lm_head),
        attention_pct: pct(ops.attention),
        recurrent_pct: pct(ops.recurrent),
        dequant_pct: pct(ops.dequant),
        rope_pct: pct(ops.rope),
        norm_pct: pct(ops.norm),
        sampling_pct: pct(ops.sampling),
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
        matmul_input_proj_ms: ops.matmul_input_proj.as_secs_f64() * 1000.0,
        matmul_output_proj_ms: ops.matmul_output_proj.as_secs_f64() * 1000.0,
        matmul_lm_head_ms: ops.matmul_lm_head.as_secs_f64() * 1000.0,
        attention_ms: ops.attention.as_secs_f64() * 1000.0,
        recurrent_ms: ops.recurrent.as_secs_f64() * 1000.0,
        dequant_ms: ops.dequant.as_secs_f64() * 1000.0,
        rope_ms: ops.rope.as_secs_f64() * 1000.0,
        norm_ms: ops.norm.as_secs_f64() * 1000.0,
        sampling_ms: ops.sampling.as_secs_f64() * 1000.0,
    })
}

impl ProfileResult {
    fn op_rows(&self) -> [(&'static str, f64, f64); 24] {
        [
            ("gpu", self.gpu_ms, self.gpu_pct),
            ("gpu_encode", self.gpu_encode_ms, self.gpu_encode_pct),
            ("gpu_execute", self.gpu_execute_ms, self.gpu_execute_pct),
            (
                "gpu_execute_layers",
                self.gpu_execute_layers_ms,
                self.gpu_execute_layers_pct,
            ),
            (
                "gpu_execute_output",
                self.gpu_execute_output_ms,
                self.gpu_execute_output_pct,
            ),
            ("gpu_readback", self.gpu_readback_ms, self.gpu_readback_pct),
            (
                "gpu_encode_layer_norm",
                self.gpu_encode_layer_norm_ms,
                self.gpu_encode_layer_norm_pct,
            ),
            (
                "gpu_encode_layer_qkv",
                self.gpu_encode_layer_qkv_ms,
                self.gpu_encode_layer_qkv_pct,
            ),
            (
                "gpu_encode_layer_rope",
                self.gpu_encode_layer_rope_ms,
                self.gpu_encode_layer_rope_pct,
            ),
            (
                "gpu_encode_layer_kv_append",
                self.gpu_encode_layer_kv_append_ms,
                self.gpu_encode_layer_kv_append_pct,
            ),
            (
                "gpu_encode_layer_attention",
                self.gpu_encode_layer_attention_ms,
                self.gpu_encode_layer_attention_pct,
            ),
            (
                "gpu_encode_layer_out_proj",
                self.gpu_encode_layer_out_proj_ms,
                self.gpu_encode_layer_out_proj_pct,
            ),
            (
                "gpu_encode_layer_ffn",
                self.gpu_encode_layer_ffn_ms,
                self.gpu_encode_layer_ffn_pct,
            ),
            (
                "gpu_encode_layer_residual",
                self.gpu_encode_layer_residual_ms,
                self.gpu_encode_layer_residual_pct,
            ),
            ("matmul", self.matmul_ms, self.matmul_pct),
            (
                "matmul_input_proj",
                self.matmul_input_proj_ms,
                self.matmul_input_proj_pct,
            ),
            (
                "matmul_output_proj",
                self.matmul_output_proj_ms,
                self.matmul_output_proj_pct,
            ),
            (
                "matmul_lm_head",
                self.matmul_lm_head_ms,
                self.matmul_lm_head_pct,
            ),
            ("attention", self.attention_ms, self.attention_pct),
            ("recurrent", self.recurrent_ms, self.recurrent_pct),
            ("dequant", self.dequant_ms, self.dequant_pct),
            ("rope", self.rope_ms, self.rope_pct),
            ("norm", self.norm_ms, self.norm_pct),
            ("sampling", self.sampling_ms, self.sampling_pct),
        ]
    }

    pub fn compare_against(&self, baseline: &ProfileResult) -> ProfileComparison {
        let mut regressions = Vec::new();
        let current_rows = self.op_rows();
        let baseline_rows = baseline.op_rows();
        for (idx, (name, cur_ms, cur_pct)) in current_rows.iter().enumerate() {
            let (_, base_ms, base_pct) = baseline_rows[idx];
            let delta_ms = *cur_ms - base_ms;
            let delta_pct_vs_baseline = if base_ms > 0.0 {
                delta_ms / base_ms * 100.0
            } else {
                0.0
            };
            regressions.push(KernelRegression {
                op: name,
                baseline_ms: base_ms,
                current_ms: *cur_ms,
                delta_ms,
                delta_pct_vs_baseline,
                share_delta_pp: *cur_pct - base_pct,
            });
        }
        regressions.sort_by(|a, b| b.delta_ms.total_cmp(&a.delta_ms));
        ProfileComparison {
            baseline_total_ms: baseline.total_ms,
            current_total_ms: self.total_ms,
            baseline_avg_tok_ms: baseline.avg_tok_ms,
            current_avg_tok_ms: self.avg_tok_ms,
            regressions,
        }
    }

    pub fn print_regression_summary(&self, baseline: &ProfileResult, top_n: usize) {
        let cmp = self.compare_against(baseline);
        eprintln!();
        eprintln!("=== Kernel Regression Report ===");
        eprintln!(
            "Wall: {:.2}ms -> {:.2}ms  (delta {:+.2}ms, {:+.2}%)",
            cmp.baseline_total_ms,
            cmp.current_total_ms,
            cmp.current_total_ms - cmp.baseline_total_ms,
            if cmp.baseline_total_ms > 0.0 {
                (cmp.current_total_ms - cmp.baseline_total_ms) / cmp.baseline_total_ms * 100.0
            } else {
                0.0
            }
        );
        eprintln!(
            "Per-token: {:.3}ms -> {:.3}ms  (delta {:+.3}ms)",
            cmp.baseline_avg_tok_ms,
            cmp.current_avg_tok_ms,
            cmp.current_avg_tok_ms - cmp.baseline_avg_tok_ms
        );
        eprintln!();
        eprintln!("Top {} regression kernels:", top_n);
        for r in cmp.regressions.iter().take(top_n) {
            eprintln!(
                "  {:9}  {:+7.2}ms  ({:+6.2}%, {:+5.2} pp share)  [{:.2} -> {:.2} ms]",
                r.op,
                r.delta_ms,
                r.delta_pct_vs_baseline,
                r.share_delta_pp,
                r.baseline_ms,
                r.current_ms
            );
        }
    }

    /// Print a human-readable breakdown.
    pub fn print_summary(&self) {
        eprintln!();
        eprintln!("=== Decode Hot-Path Profile ===");
        eprintln!("Model:       {}", self.model);
        eprintln!("Tokens:      {}", self.tokens);
        eprintln!("Intent:      {}", self.decode_intent);
        eprintln!("Mode:        {}", self.decode_mode);
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
        eprintln!("Plan:        {}", self.decode_plan);
        if let Some(note) = &self.support_note {
            eprintln!("Support:     {note}");
        }
        if let Some(reason) = &self.decode_fallback_reason {
            eprintln!("Fallback:    {reason}");
        }
        if let Some(path) = &self.kernel_profile_path {
            eprintln!("KernelProf:  {path}");
        }
        eprintln!(
            "Wall time:   {:.1}ms ({:.2}ms/tok)",
            self.total_ms, self.avg_tok_ms,
        );
        if self.decode_command_buffers > 0.0 || self.decode_buffer_barriers > 0.0 {
            eprintln!(
                "GPU Submit:  {:.1} cmd/tok, {:.1} barriers/tok  ({:.0} cmd, {:.0} barriers)",
                self.decode_command_buffers_per_tok,
                self.decode_buffer_barriers_per_tok,
                self.decode_command_buffers,
                self.decode_buffer_barriers,
            );
            eprintln!(
                "DecodeShape: CB max {:.0} | barriers max {:.0}",
                self.decode_command_buffers_per_tok_max, self.decode_buffer_barriers_per_tok_max,
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
                "  GPU Lyrs:  {:6.1}ms  ({:5.1}%)",
                self.gpu_execute_layers_ms, self.gpu_execute_layers_pct
            );
            eprintln!(
                "  GPU Out:   {:6.1}ms  ({:5.1}%)",
                self.gpu_execute_output_ms, self.gpu_execute_output_pct
            );
        }
        eprintln!(
            "  GPU RBack: {:6.1}ms  ({:5.1}%)",
            self.gpu_readback_ms, self.gpu_readback_pct
        );
        eprintln!(
            "  GPU E-Nrm: {:6.1}ms  ({:5.1}%)",
            self.gpu_encode_layer_norm_ms, self.gpu_encode_layer_norm_pct
        );
        eprintln!(
            "  GPU E-QKV: {:6.1}ms  ({:5.1}%)",
            self.gpu_encode_layer_qkv_ms, self.gpu_encode_layer_qkv_pct
        );
        eprintln!(
            "  GPU E-Rope:{:6.1}ms  ({:5.1}%)",
            self.gpu_encode_layer_rope_ms, self.gpu_encode_layer_rope_pct
        );
        eprintln!(
            "  GPU E-KV:  {:6.1}ms  ({:5.1}%)",
            self.gpu_encode_layer_kv_append_ms, self.gpu_encode_layer_kv_append_pct
        );
        eprintln!(
            "  GPU E-Attn:{:6.1}ms  ({:5.1}%)",
            self.gpu_encode_layer_attention_ms, self.gpu_encode_layer_attention_pct
        );
        eprintln!(
            "  GPU E-Proj:{:6.1}ms  ({:5.1}%)",
            self.gpu_encode_layer_out_proj_ms, self.gpu_encode_layer_out_proj_pct
        );
        eprintln!(
            "  GPU E-FFN: {:6.1}ms  ({:5.1}%)",
            self.gpu_encode_layer_ffn_ms, self.gpu_encode_layer_ffn_pct
        );
        eprintln!(
            "  GPU E-Res: {:6.1}ms  ({:5.1}%)",
            self.gpu_encode_layer_residual_ms, self.gpu_encode_layer_residual_pct
        );
        eprintln!(
            "  Matmul:    {:6.1}ms  ({:5.1}%)",
            self.matmul_ms, self.matmul_pct
        );
        if self.matmul_input_proj_ms > 0.0 || self.matmul_input_proj_pct > 0.0 {
            eprintln!(
                "  Matmul In: {:6.1}ms  ({:5.1}%)",
                self.matmul_input_proj_ms, self.matmul_input_proj_pct
            );
        }
        if self.matmul_output_proj_ms > 0.0 || self.matmul_output_proj_pct > 0.0 {
            eprintln!(
                "  Matmul Out:{:6.1}ms  ({:5.1}%)",
                self.matmul_output_proj_ms, self.matmul_output_proj_pct
            );
        }
        if self.matmul_lm_head_ms > 0.0 || self.matmul_lm_head_pct > 0.0 {
            eprintln!(
                "  Matmul LM: {:6.1}ms  ({:5.1}%)",
                self.matmul_lm_head_ms, self.matmul_lm_head_pct
            );
        }
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
            "  Norm:      {:6.1}ms  ({:5.1}%)",
            self.norm_ms, self.norm_pct
        );
        eprintln!(
            "  RoPE:      {:6.1}ms  ({:5.1}%)",
            self.rope_ms, self.rope_pct
        );
        eprintln!(
            "  Sampling:  {:6.1}ms  ({:5.1}%)",
            self.sampling_ms, self.sampling_pct
        );
        eprintln!(
            "  Other:     {:6.1}ms  ({:5.1}%)",
            (self.total_ms
                - self.gpu_ms
                - self.matmul_ms
                - self.attention_ms
                - self.recurrent_ms
                - self.dequant_ms
                - self.norm_ms
                - self.rope_ms
                - self.sampling_ms)
                .max(0.0),
            self.other_pct,
        );
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Parse a JSON-encoded profile result.
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q5k_support_note() -> String {
        ax_engine_core::gguf::mmap::support_note_for_q5k_layer_presence(true)
            .unwrap()
            .to_string()
    }

    #[test]
    fn test_merged_support_note_appends_profile_note() {
        let merged =
            merged_support_note(Some("support".into()), Some("profile: host path")).unwrap();
        assert_eq!(merged, "support | profile: host path");
    }

    #[test]
    fn test_merged_support_note_uses_profile_note_when_support_note_missing() {
        let merged = merged_support_note(None, Some("profile: host path")).unwrap();
        assert_eq!(merged, "profile: host path");
    }

    #[test]
    fn test_profile_config_defaults() {
        let c = ProfileConfig::default();
        assert_eq!(c.warmup_tokens, 16);
        assert_eq!(c.profile_tokens, 64);
        assert_eq!(c.kernel_profile_path, None);
    }

    #[test]
    fn test_profile_result_summary_formatting() {
        let r = ProfileResult {
            model: "test.gguf".into(),
            tokens: 64,
            total_ms: 1000.0,
            avg_tok_ms: 15.625,
            decode_intent: "latency".into(),
            decode_mode: "single_cb".into(),
            prefill_plan: "mode=gpu_batch q5k_prefill=small_n".into(),
            prefill_mode: "gpu_batch".into(),
            prefill_route_family: "dense_gpu_batch".into(),
            prefill_route_detail: "generic_gpu_batch".into(),
            prefill_attention_route: None,
            prefill_qkv_plan: None,
            prefill_split_rope_append: None,
            q5k_prefill_mode: Some("small_n".into()),
            decode_plan: "sync=single_cb scratch=gpu_shared".into(),
            support_note: Some(q5k_support_note()),
            decode_fallback_reason: None,
            kernel_profile_path: None,
            decode_command_buffers: 64.0,
            decode_buffer_barriers: 128.0,
            decode_command_buffers_per_tok: 1.0,
            decode_buffer_barriers_per_tok: 2.0,
            decode_command_buffers_per_tok_max: 1.0,
            decode_buffer_barriers_per_tok_max: 4.0,
            gpu_pct: 70.0,
            gpu_encode_pct: 8.0,
            gpu_execute_pct: 62.0,
            gpu_execute_layers_pct: 57.0,
            gpu_execute_output_pct: 5.0,
            gpu_readback_pct: 0.2,
            gpu_encode_layer_norm_pct: 6.0,
            gpu_encode_layer_qkv_pct: 16.0,
            gpu_encode_layer_rope_pct: 2.0,
            gpu_encode_layer_kv_append_pct: 1.0,
            gpu_encode_layer_attention_pct: 14.0,
            gpu_encode_layer_out_proj_pct: 8.0,
            gpu_encode_layer_ffn_pct: 8.0,
            gpu_encode_layer_residual_pct: 2.0,
            matmul_pct: 60.0,
            matmul_input_proj_pct: 20.0,
            matmul_output_proj_pct: 25.0,
            matmul_lm_head_pct: 15.0,
            attention_pct: 15.0,
            recurrent_pct: 0.0,
            dequant_pct: 10.0,
            rope_pct: 3.0,
            norm_pct: 2.0,
            sampling_pct: 1.0,
            other_pct: 9.0,
            gpu_ms: 700.0,
            gpu_encode_ms: 80.0,
            gpu_execute_ms: 620.0,
            gpu_execute_layers_ms: 570.0,
            gpu_execute_output_ms: 50.0,
            gpu_readback_ms: 2.0,
            gpu_encode_layer_norm_ms: 60.0,
            gpu_encode_layer_qkv_ms: 160.0,
            gpu_encode_layer_rope_ms: 20.0,
            gpu_encode_layer_kv_append_ms: 10.0,
            gpu_encode_layer_attention_ms: 140.0,
            gpu_encode_layer_out_proj_ms: 80.0,
            gpu_encode_layer_ffn_ms: 80.0,
            gpu_encode_layer_residual_ms: 20.0,
            matmul_ms: 600.0,
            matmul_input_proj_ms: 200.0,
            matmul_output_proj_ms: 250.0,
            matmul_lm_head_ms: 150.0,
            attention_ms: 150.0,
            recurrent_ms: 0.0,
            dequant_ms: 100.0,
            rope_ms: 30.0,
            norm_ms: 20.0,
            sampling_ms: 10.0,
        };
        // Just verify it doesn't panic
        r.print_summary();
    }

    #[test]
    fn test_profile_result_to_json() {
        let r = ProfileResult {
            model: "test.gguf".into(),
            tokens: 32,
            total_ms: 500.0,
            avg_tok_ms: 15.625,
            decode_intent: "latency".into(),
            decode_mode: "single_cb".into(),
            prefill_plan: "mode=gpu_batch q5k_prefill=small_n".into(),
            prefill_mode: "gpu_batch".into(),
            prefill_route_family: "dense_gpu_batch".into(),
            prefill_route_detail: "generic_gpu_batch".into(),
            prefill_attention_route: None,
            prefill_qkv_plan: None,
            prefill_split_rope_append: None,
            q5k_prefill_mode: Some("small_n".into()),
            decode_plan: "sync=single_cb scratch=gpu_shared".into(),
            support_note: Some(q5k_support_note()),
            decode_fallback_reason: None,
            kernel_profile_path: None,
            decode_command_buffers: 32.0,
            decode_buffer_barriers: 64.0,
            decode_command_buffers_per_tok: 1.0,
            decode_buffer_barriers_per_tok: 2.0,
            decode_command_buffers_per_tok_max: 1.0,
            decode_buffer_barriers_per_tok_max: 3.0,
            gpu_pct: 60.0,
            gpu_encode_pct: 10.0,
            gpu_execute_pct: 50.0,
            gpu_execute_layers_pct: 45.0,
            gpu_execute_output_pct: 5.0,
            gpu_readback_pct: 0.2,
            gpu_encode_layer_norm_pct: 5.0,
            gpu_encode_layer_qkv_pct: 15.0,
            gpu_encode_layer_rope_pct: 2.0,
            gpu_encode_layer_kv_append_pct: 1.0,
            gpu_encode_layer_attention_pct: 12.0,
            gpu_encode_layer_out_proj_pct: 6.0,
            gpu_encode_layer_ffn_pct: 6.0,
            gpu_encode_layer_residual_pct: 1.0,
            matmul_pct: 65.0,
            matmul_input_proj_pct: 24.0,
            matmul_output_proj_pct: 26.0,
            matmul_lm_head_pct: 15.0,
            attention_pct: 12.0,
            recurrent_pct: 0.0,
            dequant_pct: 8.0,
            rope_pct: 4.0,
            norm_pct: 3.0,
            sampling_pct: 1.0,
            other_pct: 7.0,
            gpu_ms: 300.0,
            gpu_encode_ms: 50.0,
            gpu_execute_ms: 250.0,
            gpu_execute_layers_ms: 225.0,
            gpu_execute_output_ms: 25.0,
            gpu_readback_ms: 1.0,
            gpu_encode_layer_norm_ms: 25.0,
            gpu_encode_layer_qkv_ms: 75.0,
            gpu_encode_layer_rope_ms: 10.0,
            gpu_encode_layer_kv_append_ms: 5.0,
            gpu_encode_layer_attention_ms: 60.0,
            gpu_encode_layer_out_proj_ms: 30.0,
            gpu_encode_layer_ffn_ms: 30.0,
            gpu_encode_layer_residual_ms: 5.0,
            matmul_ms: 325.0,
            matmul_input_proj_ms: 120.0,
            matmul_output_proj_ms: 130.0,
            matmul_lm_head_ms: 75.0,
            attention_ms: 60.0,
            recurrent_ms: 0.0,
            dequant_ms: 40.0,
            rope_ms: 20.0,
            norm_ms: 15.0,
            sampling_ms: 5.0,
        };
        let json = r.to_json().unwrap();
        assert!(json.contains("\"matmul_pct\": 65.0"));
        assert!(json.contains("\"tokens\": 32"));
        assert!(json.contains("\"decode_command_buffers\": 32.0"));
        assert!(json.contains(&format!("\"support_note\": \"{}\"", q5k_support_note())));
        assert!(json.contains("\"q5k_prefill_mode\": \"small_n\""));
    }

    #[test]
    fn test_profile_result_zero_tokens() {
        let r = ProfileResult {
            model: "test.gguf".into(),
            tokens: 0,
            total_ms: 0.0,
            avg_tok_ms: 0.0,
            decode_intent: "latency".into(),
            decode_mode: "sequential".into(),
            prefill_plan: "mode=serial".into(),
            prefill_mode: "serial".into(),
            prefill_route_family: "serial_prefill".into(),
            prefill_route_detail: "cpu_or_fallback".into(),
            prefill_attention_route: None,
            prefill_qkv_plan: None,
            prefill_split_rope_append: None,
            q5k_prefill_mode: None,
            decode_plan: "sync=sequential scratch=cpu".into(),
            support_note: None,
            decode_fallback_reason: None,
            kernel_profile_path: None,
            decode_command_buffers: 0.0,
            decode_buffer_barriers: 0.0,
            decode_command_buffers_per_tok: 0.0,
            decode_buffer_barriers_per_tok: 0.0,
            decode_command_buffers_per_tok_max: 0.0,
            decode_buffer_barriers_per_tok_max: 0.0,
            gpu_pct: 0.0,
            gpu_encode_pct: 0.0,
            gpu_execute_pct: 0.0,
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
            matmul_pct: 0.0,
            matmul_input_proj_pct: 0.0,
            matmul_output_proj_pct: 0.0,
            matmul_lm_head_pct: 0.0,
            attention_pct: 0.0,
            recurrent_pct: 0.0,
            dequant_pct: 0.0,
            rope_pct: 0.0,
            norm_pct: 0.0,
            sampling_pct: 0.0,
            other_pct: 0.0,
            gpu_ms: 0.0,
            gpu_encode_ms: 0.0,
            gpu_execute_ms: 0.0,
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
            matmul_ms: 0.0,
            matmul_input_proj_ms: 0.0,
            matmul_output_proj_ms: 0.0,
            matmul_lm_head_ms: 0.0,
            attention_ms: 0.0,
            recurrent_ms: 0.0,
            dequant_ms: 0.0,
            rope_ms: 0.0,
            norm_ms: 0.0,
            sampling_ms: 0.0,
        };
        r.print_summary();
        assert_eq!(r.tokens, 0);
    }

    #[test]
    fn test_profile_compare_ranks_regressions() {
        let baseline = ProfileResult {
            model: "base.gguf".into(),
            tokens: 64,
            total_ms: 1000.0,
            avg_tok_ms: 15.625,
            decode_intent: "latency".into(),
            decode_mode: "single_cb".into(),
            prefill_plan: "mode=gpu_batch".into(),
            prefill_mode: "gpu_batch".into(),
            prefill_route_family: "dense_gpu_batch".into(),
            prefill_route_detail: "generic_gpu_batch".into(),
            prefill_attention_route: None,
            prefill_qkv_plan: None,
            prefill_split_rope_append: None,
            q5k_prefill_mode: None,
            decode_plan: "sync=single_cb scratch=gpu_shared".into(),
            support_note: None,
            decode_fallback_reason: None,
            kernel_profile_path: None,
            decode_command_buffers: 64.0,
            decode_buffer_barriers: 0.0,
            decode_command_buffers_per_tok: 1.0,
            decode_buffer_barriers_per_tok: 0.0,
            decode_command_buffers_per_tok_max: 1.0,
            decode_buffer_barriers_per_tok_max: 0.0,
            gpu_pct: 55.0,
            gpu_encode_pct: 5.0,
            gpu_execute_pct: 50.0,
            gpu_execute_layers_pct: 45.0,
            gpu_execute_output_pct: 5.0,
            gpu_readback_pct: 0.2,
            gpu_encode_layer_norm_pct: 5.0,
            gpu_encode_layer_qkv_pct: 15.0,
            gpu_encode_layer_rope_pct: 2.0,
            gpu_encode_layer_kv_append_pct: 1.0,
            gpu_encode_layer_attention_pct: 12.0,
            gpu_encode_layer_out_proj_pct: 6.0,
            gpu_encode_layer_ffn_pct: 6.0,
            gpu_encode_layer_residual_pct: 1.0,
            matmul_pct: 50.0,
            matmul_input_proj_pct: 18.0,
            matmul_output_proj_pct: 20.0,
            matmul_lm_head_pct: 12.0,
            attention_pct: 20.0,
            recurrent_pct: 0.0,
            dequant_pct: 10.0,
            rope_pct: 5.0,
            norm_pct: 5.0,
            sampling_pct: 2.0,
            other_pct: 8.0,
            gpu_ms: 550.0,
            gpu_encode_ms: 50.0,
            gpu_execute_ms: 500.0,
            gpu_execute_layers_ms: 450.0,
            gpu_execute_output_ms: 50.0,
            gpu_readback_ms: 2.0,
            gpu_encode_layer_norm_ms: 50.0,
            gpu_encode_layer_qkv_ms: 150.0,
            gpu_encode_layer_rope_ms: 20.0,
            gpu_encode_layer_kv_append_ms: 10.0,
            gpu_encode_layer_attention_ms: 120.0,
            gpu_encode_layer_out_proj_ms: 60.0,
            gpu_encode_layer_ffn_ms: 60.0,
            gpu_encode_layer_residual_ms: 10.0,
            matmul_ms: 500.0,
            matmul_input_proj_ms: 180.0,
            matmul_output_proj_ms: 200.0,
            matmul_lm_head_ms: 120.0,
            attention_ms: 200.0,
            recurrent_ms: 0.0,
            dequant_ms: 100.0,
            rope_ms: 50.0,
            norm_ms: 50.0,
            sampling_ms: 20.0,
        };
        let current = ProfileResult {
            model: "cur.gguf".into(),
            tokens: 64,
            total_ms: 1150.0,
            avg_tok_ms: 17.96875,
            decode_intent: "latency".into(),
            decode_mode: "single_cb".into(),
            prefill_plan: "mode=gpu_batch".into(),
            prefill_mode: "gpu_batch".into(),
            prefill_route_family: "dense_gpu_batch".into(),
            prefill_route_detail: "generic_gpu_batch".into(),
            prefill_attention_route: None,
            prefill_qkv_plan: None,
            prefill_split_rope_append: None,
            q5k_prefill_mode: None,
            decode_plan: "sync=single_cb scratch=gpu_shared".into(),
            support_note: None,
            decode_fallback_reason: None,
            kernel_profile_path: None,
            decode_command_buffers: 64.0,
            decode_buffer_barriers: 0.0,
            decode_command_buffers_per_tok: 1.0,
            decode_buffer_barriers_per_tok: 0.0,
            decode_command_buffers_per_tok_max: 1.0,
            decode_buffer_barriers_per_tok_max: 0.0,
            gpu_pct: 58.0,
            gpu_encode_pct: 7.0,
            gpu_execute_pct: 51.0,
            gpu_execute_layers_pct: 44.0,
            gpu_execute_output_pct: 7.0,
            gpu_readback_pct: 0.2,
            gpu_encode_layer_norm_pct: 5.5,
            gpu_encode_layer_qkv_pct: 15.5,
            gpu_encode_layer_rope_pct: 2.0,
            gpu_encode_layer_kv_append_pct: 1.0,
            gpu_encode_layer_attention_pct: 12.5,
            gpu_encode_layer_out_proj_pct: 6.0,
            gpu_encode_layer_ffn_pct: 6.0,
            gpu_encode_layer_residual_pct: 1.5,
            matmul_pct: 52.0,
            matmul_input_proj_pct: 19.0,
            matmul_output_proj_pct: 21.0,
            matmul_lm_head_pct: 12.0,
            attention_pct: 24.0,
            recurrent_pct: 0.0,
            dequant_pct: 8.0,
            rope_pct: 5.0,
            norm_pct: 4.0,
            sampling_pct: 2.0,
            other_pct: 5.0,
            gpu_ms: 667.0,
            gpu_encode_ms: 70.0,
            gpu_execute_ms: 597.0,
            gpu_execute_layers_ms: 515.0,
            gpu_execute_output_ms: 82.0,
            gpu_readback_ms: 2.0,
            gpu_encode_layer_norm_ms: 64.0,
            gpu_encode_layer_qkv_ms: 178.0,
            gpu_encode_layer_rope_ms: 23.0,
            gpu_encode_layer_kv_append_ms: 12.0,
            gpu_encode_layer_attention_ms: 143.0,
            gpu_encode_layer_out_proj_ms: 68.0,
            gpu_encode_layer_ffn_ms: 68.0,
            gpu_encode_layer_residual_ms: 17.0,
            matmul_ms: 598.0,
            matmul_input_proj_ms: 218.0,
            matmul_output_proj_ms: 241.0,
            matmul_lm_head_ms: 138.0,
            attention_ms: 276.0,
            recurrent_ms: 0.0,
            dequant_ms: 92.0,
            rope_ms: 57.5,
            norm_ms: 46.0,
            sampling_ms: 23.0,
        };
        let cmp = current.compare_against(&baseline);
        assert_eq!(cmp.regressions[0].op, "gpu");
        assert!(cmp.regressions[0].delta_ms > 0.0);
    }
}
