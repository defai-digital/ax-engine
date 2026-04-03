use std::fmt;
use std::time::Duration;

use ax_engine_metal::PerfCounters;

use crate::kv::ModelKv;
use crate::metrics::counters::OpTimer;
use crate::model::execution_plan::{DecodeExecutionPlan, DecodeScratchPlan, DecodeSyncPlan};
use crate::model::{LlamaModel, WeightStore};
use crate::sampling::{SampledTokenInfo, Sampler};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeIntent {
    Throughput,
    Latency,
}

impl fmt::Display for DecodeIntent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Throughput => f.write_str("throughput"),
            Self::Latency => f.write_str("latency"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMode {
    Sequential,
    SingleCb,
    Pipelined,
}

impl fmt::Display for DecodeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sequential => f.write_str("sequential"),
            Self::SingleCb => f.write_str("single_cb"),
            Self::Pipelined => f.write_str("pipelined"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeControl {
    Continue,
    Stop,
}

#[derive(Debug, Clone)]
pub struct DecodeSelection {
    pub intent: DecodeIntent,
    pub mode: DecodeMode,
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeRunConfig {
    pub intent: DecodeIntent,
    pub allow_pipelined: bool,
    pub top_logprobs: usize,
    pub collect_metal_perf: bool,
}

impl Default for DecodeRunConfig {
    fn default() -> Self {
        Self {
            intent: DecodeIntent::Throughput,
            allow_pipelined: true,
            top_logprobs: 0,
            collect_metal_perf: false,
        }
    }
}

#[derive(Debug)]
pub struct DecodeRunResult {
    pub selection: DecodeSelection,
    pub plan_summary: String,
    pub generated_tokens: u64,
    pub decode_duration: Duration,
    pub latencies: Vec<Duration>,
    pub metal_perf: Option<DecodeMetalPerfSummary>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeMetalPerfSummary {
    pub total_command_buffers: u64,
    pub total_buffer_barriers: u64,
    pub max_command_buffers_per_token: u64,
    pub max_buffer_barriers_per_token: u64,
    pub avg_command_buffers_per_token: f64,
    pub avg_buffer_barriers_per_token: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DecodePerfGateConfig {
    max_command_buffers_per_token: Option<u64>,
    max_buffer_barriers_per_token: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct DecodePerfObserver {
    config: DecodePerfGateConfig,
    totals: PerfCounters,
    max_per_token: PerfCounters,
    observed_tokens: u64,
}

impl DecodePerfObserver {
    fn from_config(model: &LlamaModel, collect_metal_perf: bool) -> Option<Self> {
        if model.metal_device().is_none() || !collect_metal_perf {
            return None;
        }
        Some(Self {
            config: DecodePerfGateConfig {
                max_command_buffers_per_token: env_u64(
                    "AX_METAL_DECODE_MAX_COMMAND_BUFFERS_PER_TOKEN",
                ),
                max_buffer_barriers_per_token: env_u64("AX_METAL_DECODE_MAX_BARRIERS_PER_TOKEN"),
            },
            totals: PerfCounters::default(),
            max_per_token: PerfCounters::default(),
            observed_tokens: 0,
        })
    }

    fn begin_token(&self, model: &LlamaModel) {
        model.reset_metal_perf_counters();
    }

    fn observe_token(&mut self, model: &LlamaModel, selection: &DecodeSelection, token_idx: usize) {
        let counters = model.read_metal_perf_counters();
        self.totals.command_buffers += counters.command_buffers;
        self.totals.buffer_barriers += counters.buffer_barriers;
        self.max_per_token.command_buffers = self
            .max_per_token
            .command_buffers
            .max(counters.command_buffers);
        self.max_per_token.buffer_barriers = self
            .max_per_token
            .buffer_barriers
            .max(counters.buffer_barriers);
        self.observed_tokens += 1;

        if let Some(limit) = self.config.max_command_buffers_per_token
            && counters.command_buffers > limit
        {
            tracing::warn!(
                arch = model.arch_name(),
                mode = %selection.mode,
                intent = %selection.intent,
                token_idx,
                command_buffers = counters.command_buffers,
                threshold = limit,
                "Decode perf gate exceeded command-buffer threshold"
            );
        }
        if let Some(limit) = self.config.max_buffer_barriers_per_token
            && counters.buffer_barriers > limit
        {
            tracing::warn!(
                arch = model.arch_name(),
                mode = %selection.mode,
                intent = %selection.intent,
                token_idx,
                buffer_barriers = counters.buffer_barriers,
                threshold = limit,
                "Decode perf gate exceeded barrier threshold"
            );
        }
    }

    fn finish(self, generated_tokens: u64) -> DecodeMetalPerfSummary {
        let denom = self.observed_tokens.max(generated_tokens).max(1) as f64;
        DecodeMetalPerfSummary {
            total_command_buffers: self.totals.command_buffers,
            total_buffer_barriers: self.totals.buffer_barriers,
            max_command_buffers_per_token: self.max_per_token.command_buffers,
            max_buffer_barriers_per_token: self.max_per_token.buffer_barriers,
            avg_command_buffers_per_token: self.totals.command_buffers as f64 / denom,
            avg_buffer_barriers_per_token: self.totals.buffer_barriers as f64 / denom,
        }
    }
}

fn env_u64(var: &str) -> Option<u64> {
    std::env::var(var)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|&value| value > 0)
}

fn log_decode_perf_summary(
    model: &LlamaModel,
    selection: &DecodeSelection,
    summary: &DecodeMetalPerfSummary,
) {
    tracing::info!(
        arch = model.arch_name(),
        mode = %selection.mode,
        intent = %selection.intent,
        total_command_buffers = summary.total_command_buffers,
        total_buffer_barriers = summary.total_buffer_barriers,
        max_command_buffers_per_token = summary.max_command_buffers_per_token,
        max_buffer_barriers_per_token = summary.max_buffer_barriers_per_token,
        avg_command_buffers_per_token = summary.avg_command_buffers_per_token,
        avg_buffer_barriers_per_token = summary.avg_buffer_barriers_per_token,
        "Decode Metal perf summary"
    );
}

pub fn select_decode_mode(
    model: &LlamaModel,
    kv: &ModelKv,
    intent: DecodeIntent,
    allow_pipelined: bool,
) -> DecodeSelection {
    DecodeExecutionPlan::for_model(model, kv, intent, allow_pipelined).selection
}

#[allow(clippy::too_many_arguments)]
pub fn run_decode<F>(
    model: &LlamaModel,
    weights: &WeightStore<'_>,
    tokenizer: &Tokenizer,
    kv: &mut ModelKv,
    sampler: &mut Sampler,
    history: &mut Vec<u32>,
    first_token: u32,
    first_token_info: Option<SampledTokenInfo>,
    position: usize,
    max_tokens: usize,
    config: DecodeRunConfig,
    on_token: F,
) -> anyhow::Result<DecodeRunResult>
where
    F: FnMut(u32, Option<&SampledTokenInfo>) -> anyhow::Result<DecodeControl>,
{
    let plan = DecodeExecutionPlan::for_model(model, kv, config.intent, config.allow_pipelined);
    let plan_summary = plan.summary_label();
    let selection = plan.selection.clone();
    let expects_gpu_subplan = matches!(
        (plan.sync, plan.scratch),
        (
            DecodeSyncPlan::Sequential,
            DecodeScratchPlan::HybridBackendOwned
        ) | (
            DecodeSyncPlan::SingleCommandBuffer | DecodeSyncPlan::Pipelined,
            DecodeScratchPlan::HybridBackendOwned
        ) | (
            DecodeSyncPlan::SingleCommandBuffer | DecodeSyncPlan::Pipelined,
            DecodeScratchPlan::SharedGpuScratch
        )
    );
    debug_assert!(
        expects_gpu_subplan == plan.gpu.is_some(),
        "decode execution plan GPU sub-plan must match the selected scratch/sync policy"
    );
    debug_assert!(
        matches!(
            (plan.sync, plan.scratch),
            (DecodeSyncPlan::Sequential, DecodeScratchPlan::CpuScratch)
                | (
                    DecodeSyncPlan::Sequential,
                    DecodeScratchPlan::HybridBackendOwned
                )
                | (
                    DecodeSyncPlan::SingleCommandBuffer | DecodeSyncPlan::Pipelined,
                    DecodeScratchPlan::HybridBackendOwned
                )
                | (
                    DecodeSyncPlan::SingleCommandBuffer | DecodeSyncPlan::Pipelined,
                    DecodeScratchPlan::SharedGpuScratch
                )
        ),
        "decode execution plan scratch policy must match sync mode"
    );
    match plan.sync {
        DecodeSyncPlan::Pipelined => run_pipelined_decode(
            model,
            weights,
            tokenizer,
            kv,
            sampler,
            history,
            first_token,
            first_token_info,
            position,
            max_tokens,
            config.top_logprobs,
            config.collect_metal_perf,
            selection,
            plan_summary,
            on_token,
        ),
        DecodeSyncPlan::Sequential | DecodeSyncPlan::SingleCommandBuffer => run_sequential_decode(
            model,
            weights,
            tokenizer,
            kv,
            sampler,
            history,
            first_token,
            first_token_info,
            position,
            max_tokens,
            config.top_logprobs,
            config.collect_metal_perf,
            selection,
            plan_summary,
            on_token,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_sequential_decode<F>(
    model: &LlamaModel,
    weights: &WeightStore<'_>,
    tokenizer: &Tokenizer,
    kv: &mut ModelKv,
    sampler: &mut Sampler,
    history: &mut Vec<u32>,
    first_token: u32,
    first_token_info: Option<SampledTokenInfo>,
    mut position: usize,
    max_tokens: usize,
    top_logprobs: usize,
    collect_metal_perf: bool,
    selection: DecodeSelection,
    plan_summary: String,
    mut on_token: F,
) -> anyhow::Result<DecodeRunResult>
where
    F: FnMut(u32, Option<&SampledTokenInfo>) -> anyhow::Result<DecodeControl>,
{
    let mut logits = vec![0.0f32; model.config.vocab_size as usize];
    let mut next_token = first_token;
    let mut next_token_info = first_token_info;
    let mut generated_tokens = 0u64;
    let mut latencies = Vec::new();
    let decode_timer = OpTimer::start();
    let mut perf_observer = DecodePerfObserver::from_config(model, collect_metal_perf);

    for step_idx in 0..max_tokens {
        if tokenizer.is_eos(next_token) {
            break;
        }

        if matches!(
            on_token(next_token, next_token_info.as_ref())?,
            DecodeControl::Stop
        ) {
            break;
        }
        history.push(next_token);
        generated_tokens += 1;

        let tok_timer = OpTimer::start();
        logits.fill(0.0);
        if let Some(observer) = perf_observer.as_ref() {
            observer.begin_token(model);
        }
        model.forward_single(next_token, position, kv, weights, &mut logits)?;
        if let Some(observer) = perf_observer.as_mut() {
            observer.observe_token(model, &selection, step_idx);
        }
        let tok_latency = tok_timer.elapsed();
        if selection.intent == DecodeIntent::Latency {
            latencies.push(tok_latency);
        }
        position += 1;

        if top_logprobs > 0 {
            let sampled = sampler.sample_with_logprobs(&mut logits, history, top_logprobs);
            next_token = sampled.token;
            next_token_info = Some(sampled);
        } else {
            next_token = sampler.sample(&mut logits, history);
            next_token_info = None;
        }
    }

    let metal_perf = perf_observer.map(|observer| {
        let summary = observer.finish(generated_tokens);
        log_decode_perf_summary(model, &selection, &summary);
        summary
    });

    Ok(DecodeRunResult {
        selection,
        plan_summary,
        generated_tokens,
        decode_duration: decode_timer.elapsed(),
        latencies,
        metal_perf,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_pipelined_decode<F>(
    model: &LlamaModel,
    weights: &WeightStore<'_>,
    tokenizer: &Tokenizer,
    kv: &mut ModelKv,
    sampler: &mut Sampler,
    history: &mut Vec<u32>,
    first_token: u32,
    first_token_info: Option<SampledTokenInfo>,
    start_position: usize,
    max_tokens: usize,
    top_logprobs: usize,
    collect_metal_perf: bool,
    selection: DecodeSelection,
    plan_summary: String,
    mut on_token: F,
) -> anyhow::Result<DecodeRunResult>
where
    F: FnMut(u32, Option<&SampledTokenInfo>) -> anyhow::Result<DecodeControl>,
{
    if tokenizer.is_eos(first_token) || max_tokens == 0 {
        return Ok(DecodeRunResult {
            selection,
            plan_summary,
            generated_tokens: 0,
            decode_duration: Duration::ZERO,
            latencies: Vec::new(),
            metal_perf: None,
        });
    }

    let metal_dev = model
        .metal_device()
        .ok_or_else(|| anyhow::anyhow!("pipelined decode requested without a Metal device"))?;
    let dim = model.config.embedding_dim as usize;
    let hidden_bytes = dim * std::mem::size_of::<f32>();
    let mut hidden_a = model.alloc_metal_buf(hidden_bytes)?;
    let mut hidden_b = model.alloc_metal_buf(hidden_bytes)?;

    model.prewarm_kv_capacity(kv, start_position + max_tokens + 2)?;
    if let Some(active_slot) = kv.as_qwen35().map(crate::kv::Qwen35Kv::active_slot) {
        model.prime_qwen35_recurrent_slot_buffers(kv, &[active_slot])?;
    }

    let mut logits = vec![0.0f32; model.config.vocab_size as usize];
    let mut position = start_position;
    let mut next_token = first_token;
    let mut next_token_info = first_token_info;
    let mut generated_tokens = 0u64;
    let decode_timer = OpTimer::start();
    let mut perf_observer = DecodePerfObserver::from_config(model, collect_metal_perf);

    let greedy = sampler.is_greedy() && top_logprobs == 0;
    let use_fused_argmax = greedy && model.supports_fused_argmax();

    model.embed_token_into(first_token, &hidden_a, weights)?;
    if let Some(observer) = perf_observer.as_ref() {
        observer.begin_token(model);
    }
    let pending = if use_fused_argmax {
        model.encode_pending_decode_step_with_argmax(&hidden_a, position, kv, weights)?
    } else {
        model.encode_pending_decode_step(&hidden_a, position, kv, weights)?
    }
    .ok_or_else(|| anyhow::anyhow!("pipelined decode became unavailable during setup"))?;
    let mut inflight = Some(metal_dev.commit_frame(pending));

    for step_idx in 0..max_tokens {
        if tokenizer.is_eos(next_token) {
            break;
        }

        let should_prepare_next = step_idx + 1 < max_tokens;
        let next_position = position + 1;
        let pending_next = if should_prepare_next {
            if use_fused_argmax {
                model.encode_pending_decode_step_with_argmax(
                    &hidden_b,
                    next_position,
                    kv,
                    weights,
                )?
            } else {
                model.encode_pending_decode_step(&hidden_b, next_position, kv, weights)?
            }
        } else {
            None
        };

        if let Some(frame) = inflight.take() {
            metal_dev.wait_frame(frame)?;
        }
        if let Some(observer) = perf_observer.as_mut() {
            observer.observe_token(model, &selection, step_idx);
        }
        model.advance_gpu_kv_token(kv);

        // Greedy decode fast paths:
        // - Fused: argmax dispatched inside the main CB → just read 4 bytes.
        // - CPU greedy: read logits back + CPU argmax (faster than a separate
        //   GPU CB round-trip on Apple UMA where readback is zero-copy).
        // - Full: read all logits to CPU for sampling with penalties/top-p.
        let (sampled, sampled_info) = if use_fused_argmax {
            let idx = model.read_gpu_argmax_result()?;
            (idx, None)
        } else if greedy {
            model.read_gpu_logits(&mut logits)?;
            let idx = crate::sampling::argmax(&logits);
            (idx, None)
        } else {
            model.read_gpu_logits(&mut logits)?;
            let info = if top_logprobs > 0 {
                Some(sampler.sample_with_logprobs(&mut logits, history, top_logprobs))
            } else {
                None
            };
            let token = info
                .as_ref()
                .map(|i| i.token)
                .unwrap_or_else(|| sampler.sample(&mut logits, history));
            (token, info)
        };

        if matches!(
            on_token(next_token, next_token_info.as_ref())?,
            DecodeControl::Stop
        ) {
            break;
        }
        history.push(next_token);
        generated_tokens += 1;

        let stop_after_emit = !should_prepare_next || tokenizer.is_eos(sampled);
        if stop_after_emit {
            break;
        }

        model.embed_token_into(sampled, &hidden_b, weights)?;
        let pending_next = pending_next.ok_or_else(|| {
            anyhow::anyhow!("pipelined decode became unavailable while preparing the next frame")
        })?;
        if let Some(observer) = perf_observer.as_ref() {
            observer.begin_token(model);
        }
        inflight = Some(metal_dev.commit_frame(pending_next));

        next_token = sampled;
        next_token_info = sampled_info;
        position = next_position;
        std::mem::swap(&mut hidden_a, &mut hidden_b);
    }

    let metal_perf = perf_observer.map(|observer| {
        let summary = observer.finish(generated_tokens);
        log_decode_perf_summary(model, &selection, &summary);
        summary
    });

    Ok(DecodeRunResult {
        selection,
        plan_summary,
        generated_tokens,
        decode_duration: decode_timer.elapsed(),
        latencies: Vec::new(),
        metal_perf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::{HybridCpuDecodeBackend, MetalBackend};
    use crate::model::config::{GateActivation, RopeScaling};

    fn tiny_config(arch: &str) -> crate::model::ModelConfig {
        crate::model::ModelConfig {
            architecture: arch.into(),
            n_layers: if arch == "qwen35" { 4 } else { 1 },
            n_heads: 2,
            n_kv_heads: 2,
            embedding_dim: 8,
            head_dim: 4,
            intermediate_dim: 16,
            context_length: 32,
            vocab_size: 8,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            has_qkv_bias: arch == "qwen3",
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: if arch == "gemma3" {
                GateActivation::GELU
            } else {
                GateActivation::SiLU
            },
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: RopeScaling::None,
            embed_scale: arch == "gemma3",
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            expert_intermediate_dim: None,
            qwen35_full_attention_interval: (arch == "qwen35").then_some(4),
            qwen35_ssm_conv_kernel: (arch == "qwen35").then_some(4),
            qwen35_ssm_inner_size: (arch == "qwen35").then_some(8),
            qwen35_ssm_state_size: (arch == "qwen35").then_some(2),
            qwen35_ssm_time_step_rank: (arch == "qwen35").then_some(4),
            qwen35_ssm_group_count: (arch == "qwen35").then_some(2),
        }
    }

    #[test]
    fn test_select_decode_mode_cpu_uses_sequential_decode() {
        let model = LlamaModel::new(tiny_config("llama")).unwrap();
        let kv = model.create_model_kv();
        let selection = select_decode_mode(&model, &kv, DecodeIntent::Throughput, true);

        assert_eq!(selection.mode, DecodeMode::Sequential);
        assert_eq!(
            selection.fallback_reason.as_deref(),
            Some("GPU decode unavailable; using sequential decode")
        );
    }

    #[test]
    fn test_select_decode_mode_hybrid_cpu_decode_uses_sequential_decode() {
        let Ok(backend) = HybridCpuDecodeBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("llama"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();

        assert!(!kv.is_gpu(), "HybridCpuDecode should allocate CPU KV");

        let selection = select_decode_mode(&model, &kv, DecodeIntent::Throughput, true);
        assert_eq!(selection.mode, DecodeMode::Sequential);
    }

    #[test]
    fn test_select_decode_mode_llama_metal_prefers_pipelined() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("llama"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        if !kv.is_gpu() {
            return;
        }

        let selection = select_decode_mode(&model, &kv, DecodeIntent::Throughput, true);
        assert_eq!(selection.mode, DecodeMode::Pipelined);
        assert!(selection.fallback_reason.is_none());
    }

    #[test]
    fn test_select_decode_mode_qwen3_metal_prefers_pipelined() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("qwen3"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        if !kv.is_gpu() {
            return;
        }

        let selection = select_decode_mode(&model, &kv, DecodeIntent::Throughput, true);
        assert_eq!(selection.mode, DecodeMode::Pipelined);
        assert!(selection.fallback_reason.is_none());
    }

    #[test]
    fn test_select_decode_mode_gemma3_metal_prefers_pipelined() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("gemma3"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        if !kv.is_gpu() {
            return;
        }

        let selection = select_decode_mode(&model, &kv, DecodeIntent::Throughput, true);
        assert_eq!(selection.mode, DecodeMode::Pipelined);
        assert!(selection.fallback_reason.is_none());
    }

    #[test]
    fn test_select_decode_mode_qwen35_hybrid_prefers_pipelined_when_available() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("qwen35"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        assert!(matches!(kv, ModelKv::Qwen35(_)));

        // AX_QWEN35_GPU_DECODE defaults to false, so pipelined decode is
        // unavailable and the selection falls back to SingleCb.
        let selection = select_decode_mode(&model, &kv, DecodeIntent::Throughput, true);
        assert_eq!(selection.mode, DecodeMode::SingleCb);
        assert!(selection.fallback_reason.is_some());
    }

    #[test]
    fn test_select_decode_mode_qwen35_hybrid_latency_uses_single_cb() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("qwen35"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        assert!(matches!(kv, ModelKv::Qwen35(_)));

        let selection = select_decode_mode(&model, &kv, DecodeIntent::Latency, false);
        assert_eq!(selection.mode, DecodeMode::SingleCb);
        assert!(selection.fallback_reason.is_none());
        let summary = model.decode_plan_summary(&kv, DecodeIntent::Latency, false);
        assert!(summary.starts_with("sync=single_cb scratch=hybrid_backend"));
        assert!(summary.contains("attn="));
    }

    #[test]
    fn test_select_decode_mode_latency_gpu_uses_single_cb() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("llama"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        if !kv.is_gpu() {
            return;
        }

        let selection = select_decode_mode(&model, &kv, DecodeIntent::Latency, false);
        assert_eq!(selection.mode, DecodeMode::SingleCb);
        assert!(selection.fallback_reason.is_none());
    }

    #[test]
    fn test_decode_perf_observer_finish_uses_observed_token_count() {
        let observer = DecodePerfObserver {
            config: DecodePerfGateConfig {
                max_command_buffers_per_token: None,
                max_buffer_barriers_per_token: None,
            },
            totals: PerfCounters {
                command_buffers: 6,
                buffer_barriers: 12,
            },
            max_per_token: PerfCounters {
                command_buffers: 3,
                buffer_barriers: 7,
            },
            observed_tokens: 2,
        };

        let summary = observer.finish(2);
        assert_eq!(summary.total_command_buffers, 6);
        assert_eq!(summary.total_buffer_barriers, 12);
        assert_eq!(summary.max_command_buffers_per_token, 3);
        assert_eq!(summary.max_buffer_barriers_per_token, 7);
        assert_eq!(summary.avg_command_buffers_per_token, 3.0);
        assert_eq!(summary.avg_buffer_barriers_per_token, 6.0);
    }

    #[test]
    fn test_env_u64_rejects_zero_and_invalid_values() {
        let key = "AX_TEST_DECODE_PERF_LIMIT";
        unsafe {
            std::env::remove_var(key);
        }
        assert_eq!(env_u64(key), None);

        unsafe {
            std::env::set_var(key, "0");
        }
        assert_eq!(env_u64(key), None);

        unsafe {
            std::env::set_var(key, "abc");
        }
        assert_eq!(env_u64(key), None);

        unsafe {
            std::env::set_var(key, "7");
        }
        assert_eq!(env_u64(key), Some(7));

        unsafe {
            std::env::remove_var(key);
        }
    }
}
