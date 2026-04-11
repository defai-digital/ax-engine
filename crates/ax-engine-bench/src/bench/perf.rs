//! Performance benchmark runner.
//!
//! Measures prefill tok/s, decode tok/s, and latency percentiles
//! for a given model and configuration.

use std::path::Path;
use std::thread;
use std::time::Duration;

use ax_engine_core::gguf::MappedModel;
use ax_engine_core::kv::ModelKv;
use ax_engine_core::metrics::LatencyHistogram;
use ax_engine_core::metrics::counters::OpTimer;
use ax_engine_core::model::{
    DecodeControl, DecodeIntent, DecodeMetalPerfSummary, DecodeRunConfig, DecodeSelection,
    InferenceModel, ModelConfig, WeightStore, run_decode,
};
use ax_engine_core::sampling::{Sampler, SamplingConfig};
use ax_engine_core::speculative::{SpeculativeDecoder, target_verify_mode_label};
use ax_engine_core::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};

/// Benchmark configuration.
pub struct BenchConfig {
    /// Model path (GGUF file).
    pub model_path: String,
    /// Number of prompt tokens to test prefill with.
    pub prompt_tokens: usize,
    /// Number of tokens to decode.
    pub decode_tokens: usize,
    /// Number of warmup iterations before measuring.
    pub warmup_iters: usize,
    /// Number of measurement iterations.
    pub measure_iters: usize,
    /// Enable deterministic mode (cooldown + repeated samples + median reporting).
    pub deterministic: bool,
    /// Number of repeated samples in deterministic mode.
    pub samples: usize,
    /// Cooldown between measured iterations in milliseconds.
    pub cooldown_ms: u64,
    /// Whether the benchmark is measuring throughput or latency semantics.
    pub intent: DecodeIntent,
    /// Fan the same prompt out across this many Qwen3.5 recurrent slots on a
    /// shared attention timeline during prefill. Only valid for qwen35 and
    /// values >= 1.
    pub qwen35_shared_timeline_slots: usize,
    /// Optional source recurrent slot for Qwen3.5 shared-timeline prefill.
    pub qwen35_shared_timeline_source_slot: Option<usize>,
}

/// Benchmark configuration for speculative decoding.
pub struct SpecBenchConfig {
    /// Path to target GGUF model file.
    pub model_path: String,
    /// Path to draft GGUF model file.
    pub draft_model_path: String,
    /// Number of prompt tokens to test prefill with.
    pub prompt_tokens: usize,
    /// Number of tokens to decode.
    pub decode_tokens: usize,
    /// Number of warmup iterations before measuring.
    pub warmup_iters: usize,
    /// Number of measurement iterations.
    pub measure_iters: usize,
    /// Enable deterministic mode (cooldown + repeated samples + median reporting).
    pub deterministic: bool,
    /// Number of repeated samples in deterministic mode.
    pub samples: usize,
    /// Cooldown between measured iterations in milliseconds.
    pub cooldown_ms: u64,
    /// Speculative lookahead length.
    pub speculative_k: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt_tokens: 512,
            decode_tokens: 128,
            warmup_iters: 2,
            measure_iters: 5,
            deterministic: false,
            samples: 1,
            cooldown_ms: 0,
            intent: DecodeIntent::Throughput,
            qwen35_shared_timeline_slots: 1,
            qwen35_shared_timeline_source_slot: None,
        }
    }
}

impl Default for SpecBenchConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            draft_model_path: String::new(),
            prompt_tokens: 512,
            decode_tokens: 128,
            warmup_iters: 2,
            measure_iters: 5,
            deterministic: false,
            samples: 1,
            cooldown_ms: 0,
            speculative_k: 4,
        }
    }
}

/// Results of a performance benchmark.
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchResult {
    /// Model name/path.
    pub model: String,
    /// Prompt token count.
    pub prompt_tokens: usize,
    /// Effective shared-timeline prompt advancement in slot-tokens.
    #[serde(default)]
    pub effective_prefill_tokens: usize,
    /// Decode token count.
    pub decode_tokens: usize,
    /// Average prefill throughput (tok/s).
    pub prefill_tok_per_sec: f64,
    /// Median prefill throughput (tok/s).
    pub prefill_tok_per_sec_median: f64,
    /// Average effective shared-timeline prefill throughput (slot-tok/s).
    #[serde(default)]
    pub effective_prefill_tok_per_sec: f64,
    /// Median effective shared-timeline prefill throughput (slot-tok/s).
    #[serde(default)]
    pub effective_prefill_tok_per_sec_median: f64,
    /// Average decode throughput (tok/s).
    pub decode_tok_per_sec: f64,
    /// Median decode throughput (tok/s).
    pub decode_tok_per_sec_median: f64,
    /// P50 decode latency per token.
    pub p50_latency: Duration,
    /// P95 decode latency per token.
    pub p95_latency: Duration,
    /// P99 decode latency per token.
    pub p99_latency: Duration,
    /// Average number of Metal command buffers used in prefill.
    pub prefill_command_buffers: f64,
    /// Average number of Metal buffer barriers used in prefill.
    pub prefill_buffer_barriers: f64,
    /// Average number of Metal command buffers used in decode.
    pub decode_command_buffers: f64,
    /// Average number of Metal buffer barriers used in decode.
    pub decode_buffer_barriers: f64,
    /// Average number of Metal command buffers per decoded token.
    #[serde(default)]
    pub decode_command_buffers_per_tok: f64,
    /// Average number of Metal buffer barriers per decoded token.
    #[serde(default)]
    pub decode_buffer_barriers_per_tok: f64,
    /// Maximum Metal command buffers observed for any decoded token.
    #[serde(default)]
    pub decode_command_buffers_per_tok_max: f64,
    /// Maximum Metal buffer barriers observed for any decoded token.
    #[serde(default)]
    pub decode_buffer_barriers_per_tok_max: f64,
    /// Benchmark intent (`throughput` or `latency`).
    pub decode_intent: String,
    /// Selected decode mode for measured runs.
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
    /// Fallback reason if a faster mode was requested but not used.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_fallback_reason: Option<String>,
    #[serde(default = "default_qwen35_shared_timeline_slots")]
    pub qwen35_shared_timeline_slots: usize,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qwen35_shared_timeline_source_slot: Option<usize>,
    /// Whether f16 KV cache was active.
    pub kv_f16: bool,
    /// True when deterministic mode was enabled.
    pub deterministic: bool,
    /// Number of repeated samples used.
    #[serde(default)]
    pub samples: usize,
    /// Cooldown between measured iterations (ms).
    #[serde(default)]
    pub cooldown_ms: u64,
}

/// Results of a speculative-decoding benchmark.
#[derive(Debug, Serialize, Deserialize)]
pub struct SpecBenchResult {
    /// Target model name/path.
    pub model: String,
    /// Draft model path.
    pub draft_model: String,
    /// Prompt token count.
    pub prompt_tokens: usize,
    /// Decode token count.
    pub decode_tokens: usize,
    /// Speculative lookahead length.
    pub speculative_k: usize,
    /// Average prefill throughput (tok/s).
    pub prefill_tok_per_sec: f64,
    /// Median prefill throughput (tok/s).
    pub prefill_tok_per_sec_median: f64,
    /// Average decode throughput (tok/s).
    pub decode_tok_per_sec: f64,
    /// Median decode throughput (tok/s).
    pub decode_tok_per_sec_median: f64,
    /// Compact prefill execution-plan summary.
    #[serde(default)]
    pub prefill_plan: String,
    /// Target verification implementation path used during speculative decode.
    #[serde(default)]
    pub target_verify_mode: String,
    /// Parsed `q5k_prefill=...` mode from the prefill plan, when present.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q5k_prefill_mode: Option<String>,
    /// Average accepted draft tokens per speculative step.
    pub avg_accepted_per_step: f64,
    /// Average draft time per speculative step.
    pub draft_ms_per_step: f64,
    /// Average verification time per speculative step.
    pub verify_ms_per_step: f64,
    /// Average verification prepare time per speculative step.
    #[serde(default)]
    pub verify_prepare_ms_per_step: f64,
    /// Average verification forward time per speculative step.
    #[serde(default)]
    pub verify_forward_ms_per_step: f64,
    /// Average verification cleanup time per speculative step.
    #[serde(default)]
    pub verify_cleanup_ms_per_step: f64,
    /// Average acceptance bookkeeping time per speculative step.
    pub accept_ms_per_step: f64,
    /// Average verification time per verified position.
    pub verify_ms_per_position: f64,
    /// Average draft time per drafted token.
    pub draft_ms_per_drafted_token: f64,
    /// Optional support-boundary note for the loaded model quant family.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub support_note: Option<String>,
    /// Whether f16 KV cache was active.
    pub kv_f16: bool,
    /// True when deterministic mode was enabled.
    pub deterministic: bool,
    /// Number of repeated samples used.
    #[serde(default)]
    pub samples: usize,
    /// Cooldown between measured iterations (ms).
    #[serde(default)]
    pub cooldown_ms: u64,
}

fn default_qwen35_shared_timeline_slots() -> usize {
    1
}

fn effective_qwen35_shared_timeline_tokens(prompt_tokens: usize, slot_count: usize) -> usize {
    prompt_tokens.saturating_mul(slot_count.max(1))
}

fn kv_dtype_label_from_plan(plan: &ax_engine_core::backend::KvPlan) -> (&'static str, bool) {
    match plan {
        ax_engine_core::backend::KvPlan::Cpu(_) => ("f32", false),
        ax_engine_core::backend::KvPlan::Gpu(plan) => match plan.dtype {
            ax_engine_core::kv::GpuKvDtype::F32 => ("f32", false),
            ax_engine_core::kv::GpuKvDtype::F16 => ("f16", true),
            ax_engine_core::kv::GpuKvDtype::Q8_0 => ("q8_0", false),
        },
        ax_engine_core::backend::KvPlan::Qwen35(_) => ("qwen35_hybrid", false),
    }
}

fn kv_dtype_label_from_plan_summary(plan: &str) -> Option<&str> {
    plan.split_whitespace()
        .find_map(|part| part.strip_prefix("kv="))
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

/// Run a performance benchmark.
pub fn run_benchmark(config: &BenchConfig) -> anyhow::Result<BenchResult> {
    run_benchmark_with_backend(
        config,
        ax_engine_core::backend::create_backend(ax_engine_core::backend::BackendConfig::default())?,
    )
}

/// Run a speculative-decoding benchmark.
pub fn run_speculative_benchmark(config: &SpecBenchConfig) -> anyhow::Result<SpecBenchResult> {
    run_speculative_benchmark_with_backend(
        config,
        ax_engine_core::backend::create_backend(ax_engine_core::backend::BackendConfig::default())?,
    )
}

pub fn run_benchmark_with_backend(
    config: &BenchConfig,
    backend: Box<dyn ax_engine_core::backend::Backend>,
) -> anyhow::Result<BenchResult> {
    let mapped = MappedModel::open(Path::new(&config.model_path))?;
    let model_config = ModelConfig::from_gguf(&mapped.header)?;
    crate::configure_backend_for_model(&*backend, &config.model_path, &mapped, &model_config)?;
    let tokenizer = Tokenizer::from_gguf(&mapped.header)?;
    let model = InferenceModel::with_backend(model_config.clone(), backend)?;
    crate::report_planned_kv_budget(&mapped, &model)?;
    let support_note = crate::support_note(&mapped);
    let weights = WeightStore::new(&mapped);
    model.prepare_runtime_for_weights(&weights)?;

    let vocab_size = model_config.vocab_size as usize;
    let sampling = SamplingConfig {
        temperature: 0.0, // greedy for determinism
        ..Default::default()
    };

    let kv_plan = model.kv_plan();
    let (kv_dtype, kv_f16) = kv_dtype_label_from_plan(&kv_plan);

    eprintln!(
        "Benchmarking: {} layers, {:.0}MB, KV dtype: {}",
        model_config.n_layers,
        mapped.total_tensor_bytes() as f64 / 1024.0 / 1024.0,
        kv_dtype,
    );
    eprintln!("Intent:      {}", config.intent);

    // Generate a fixed prompt of the desired length
    let prompt = tokenizer.encode("The quick brown fox jumps over the lazy dog. ", true);
    let mut prompt_tokens = Vec::new();
    while prompt_tokens.len() < config.prompt_tokens {
        prompt_tokens.extend_from_slice(&prompt);
    }
    prompt_tokens.truncate(config.prompt_tokens);

    // Warmup
    for _ in 0..config.warmup_iters {
        // v2: KV is created fresh inside run_single_bench, no reset needed.
        run_single_bench(
            &model,
            &weights,
            &tokenizer,
            &prompt_tokens,
            vocab_size,
            config.decode_tokens,
            &sampling,
            config.intent,
            config.qwen35_shared_timeline_slots,
            config.qwen35_shared_timeline_source_slot,
        )?;
    }

    // Measure
    let sample_count = if config.deterministic {
        config.samples.max(1)
    } else {
        1
    };

    let iters_per_sample = config.measure_iters.max(1);
    let cooldown = Duration::from_millis(config.cooldown_ms);

    let mut prefill_times = Vec::new();
    let mut decode_times = Vec::new();
    let mut sample_prefill_tok_per_sec = Vec::with_capacity(sample_count);
    let mut sample_decode_tok_per_sec = Vec::with_capacity(sample_count);
    let mut decode_token_counts = Vec::new();
    let mut latency = LatencyHistogram::new();
    let mut prefill_cmd_buf_counts = Vec::new();
    let mut prefill_barrier_counts = Vec::new();
    let mut decode_cmd_buf_counts = Vec::new();
    let mut decode_barrier_counts = Vec::new();
    let mut decode_cmd_buf_per_tok = Vec::new();
    let mut decode_barrier_per_tok = Vec::new();
    let mut decode_cmd_buf_per_tok_max = Vec::new();
    let mut decode_barrier_per_tok_max = Vec::new();
    let mut decode_selection: Option<DecodeSelection> = None;
    let mut prefill_plan: Option<String> = None;
    let mut decode_plan: Option<String> = None;

    for _sample in 0..sample_count {
        let mut sample_prefill_times = Vec::with_capacity(iters_per_sample);
        let mut sample_decode_times = Vec::with_capacity(iters_per_sample);
        let mut sample_decode_token_counts = Vec::with_capacity(iters_per_sample);

        for iter_idx in 0..iters_per_sample {
            // v2: KV is created fresh inside run_single_bench, no reset needed.
            let (
                prefill_dur,
                decode_dur,
                generated_tokens,
                iter_latency,
                prefill_cmd_bufs,
                prefill_barriers,
                decode_cmd_bufs,
                decode_barriers,
                decode_metal_perf,
                iter_selection,
                iter_prefill_plan,
                iter_plan,
            ) = run_single_bench(
                &model,
                &weights,
                &tokenizer,
                &prompt_tokens,
                vocab_size,
                config.decode_tokens,
                &sampling,
                config.intent,
                config.qwen35_shared_timeline_slots,
                config.qwen35_shared_timeline_source_slot,
            )?;
            match &decode_selection {
                None => decode_selection = Some(iter_selection.clone()),
                Some(existing)
                    if existing.mode != iter_selection.mode
                        || existing.fallback_reason != iter_selection.fallback_reason =>
                {
                    anyhow::bail!(
                        "decode mode changed across measured iterations: {} -> {}",
                        existing.mode,
                        iter_selection.mode
                    );
                }
                Some(_) => {}
            }
            match &prefill_plan {
                None => prefill_plan = Some(iter_prefill_plan.clone()),
                Some(existing) if existing != &iter_prefill_plan => {
                    anyhow::bail!(
                        "prefill execution plan changed across measured iterations: {} -> {}",
                        existing,
                        iter_prefill_plan
                    );
                }
                Some(_) => {}
            }
            match &decode_plan {
                None => decode_plan = Some(iter_plan.clone()),
                Some(existing) if existing != &iter_plan => {
                    anyhow::bail!(
                        "decode execution plan changed across measured iterations: {} -> {}",
                        existing,
                        iter_plan
                    );
                }
                Some(_) => {}
            }
            prefill_times.push(prefill_dur);
            decode_times.push(decode_dur);
            decode_token_counts.push(generated_tokens);
            sample_decode_token_counts.push(generated_tokens);
            sample_prefill_times.push(prefill_dur);
            sample_decode_times.push(decode_dur);
            prefill_cmd_buf_counts.push(prefill_cmd_bufs);
            prefill_barrier_counts.push(prefill_barriers);
            decode_cmd_buf_counts.push(decode_cmd_bufs);
            decode_barrier_counts.push(decode_barriers);
            if let Some(perf) = decode_metal_perf {
                decode_cmd_buf_per_tok.push(perf.avg_command_buffers_per_token);
                decode_barrier_per_tok.push(perf.avg_buffer_barriers_per_token);
                decode_cmd_buf_per_tok_max.push(perf.max_command_buffers_per_token as f64);
                decode_barrier_per_tok_max.push(perf.max_buffer_barriers_per_token as f64);
            } else {
                let denom = generated_tokens.max(1) as f64;
                decode_cmd_buf_per_tok.push(decode_cmd_bufs as f64 / denom);
                decode_barrier_per_tok.push(decode_barriers as f64 / denom);
                decode_cmd_buf_per_tok_max.push(0.0);
                decode_barrier_per_tok_max.push(0.0);
            }
            for sample in iter_latency {
                latency.record(sample);
            }

            if config.deterministic && config.cooldown_ms > 0 && iter_idx + 1 < iters_per_sample {
                thread::sleep(cooldown);
            }
        }

        let sample_avg_prefill = sample_prefill_times
            .iter()
            .map(|d| d.as_secs_f64())
            .sum::<f64>()
            / iters_per_sample as f64;
        let sample_avg_decode = sample_decode_times
            .iter()
            .map(|d| d.as_secs_f64())
            .sum::<f64>()
            / iters_per_sample as f64;

        sample_prefill_tok_per_sec.push(if sample_avg_prefill > 0.0 {
            config.prompt_tokens as f64 / sample_avg_prefill
        } else {
            0.0
        });
        let sample_avg_decode_tokens = sample_decode_token_counts.iter().copied().sum::<usize>()
            as f64
            / iters_per_sample as f64;
        sample_decode_tok_per_sec.push(if sample_avg_decode > 0.0 {
            sample_avg_decode_tokens / sample_avg_decode
        } else {
            0.0
        });

        if config.deterministic && config.cooldown_ms > 0 {
            thread::sleep(cooldown);
        }
    }

    // Compute averages
    let measured_iters = prefill_times.len().max(1) as f64;
    let avg_prefill: f64 =
        prefill_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / measured_iters;
    let avg_decode: f64 =
        decode_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / measured_iters;

    let prefill_tok_per_sec = if avg_prefill > 0.0 {
        config.prompt_tokens as f64 / avg_prefill
    } else {
        0.0
    };
    let avg_decode_tokens =
        decode_token_counts.iter().copied().sum::<usize>() as f64 / measured_iters;
    let decode_tok_per_sec = if avg_decode > 0.0 {
        avg_decode_tokens / avg_decode
    } else {
        0.0
    };

    let median = |vals: &mut [f64]| -> f64 {
        if vals.is_empty() {
            return 0.0;
        }
        vals.sort_by(f64::total_cmp);
        let mid = vals.len() / 2;
        if vals.len().is_multiple_of(2) {
            (vals[mid - 1] + vals[mid]) * 0.5
        } else {
            vals[mid]
        }
    };
    let prefill_tok_per_sec_median = median(&mut sample_prefill_tok_per_sec);
    let decode_tok_per_sec_median = median(&mut sample_decode_tok_per_sec);
    let effective_prefill_tokens = effective_qwen35_shared_timeline_tokens(
        config.prompt_tokens,
        config.qwen35_shared_timeline_slots,
    );

    let avg_u64 = |vals: &[u64]| -> f64 {
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().copied().sum::<u64>() as f64 / vals.len() as f64
        }
    };

    let prefill_plan = prefill_plan.unwrap_or_else(|| "mode=serial".to_string());
    Ok(BenchResult {
        model: config.model_path.clone(),
        prompt_tokens: config.prompt_tokens,
        effective_prefill_tokens,
        decode_tokens: config.decode_tokens,
        prefill_tok_per_sec,
        prefill_tok_per_sec_median,
        effective_prefill_tok_per_sec: prefill_tok_per_sec
            * config.qwen35_shared_timeline_slots as f64,
        effective_prefill_tok_per_sec_median: prefill_tok_per_sec_median
            * config.qwen35_shared_timeline_slots as f64,
        decode_tok_per_sec,
        decode_tok_per_sec_median,
        p50_latency: latency.p50().unwrap_or(Duration::ZERO),
        p95_latency: latency.p95().unwrap_or(Duration::ZERO),
        p99_latency: latency.p99().unwrap_or(Duration::ZERO),
        prefill_command_buffers: avg_u64(&prefill_cmd_buf_counts),
        prefill_buffer_barriers: avg_u64(&prefill_barrier_counts),
        decode_command_buffers: avg_u64(&decode_cmd_buf_counts),
        decode_buffer_barriers: avg_u64(&decode_barrier_counts),
        decode_command_buffers_per_tok: if decode_cmd_buf_per_tok.is_empty() {
            0.0
        } else {
            decode_cmd_buf_per_tok.iter().sum::<f64>() / decode_cmd_buf_per_tok.len() as f64
        },
        decode_buffer_barriers_per_tok: if decode_barrier_per_tok.is_empty() {
            0.0
        } else {
            decode_barrier_per_tok.iter().sum::<f64>() / decode_barrier_per_tok.len() as f64
        },
        decode_command_buffers_per_tok_max: decode_cmd_buf_per_tok_max
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(0.0),
        decode_buffer_barriers_per_tok_max: decode_barrier_per_tok_max
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(0.0),
        decode_intent: config.intent.to_string(),
        decode_mode: decode_selection
            .as_ref()
            .map(|s| s.mode.to_string())
            .unwrap_or_else(|| "sequential".to_string()),
        prefill_mode: crate::prefill_mode(&prefill_plan),
        prefill_route_family: crate::prefill_route_family(&prefill_plan),
        prefill_route_detail: crate::prefill_route_detail(&prefill_plan),
        prefill_attention_route: crate::prefill_plan_field(&prefill_plan, "attn_route"),
        prefill_qkv_plan: crate::prefill_plan_field(&prefill_plan, "qkv"),
        prefill_split_rope_append: crate::prefill_bool_field(&prefill_plan, "split_rope"),
        q5k_prefill_mode: crate::q5k_prefill_mode(&prefill_plan),
        prefill_plan,
        decode_plan: decode_plan.unwrap_or_else(|| "sync=sequential scratch=cpu".to_string()),
        support_note,
        decode_fallback_reason: decode_selection.and_then(|s| s.fallback_reason),
        qwen35_shared_timeline_slots: config.qwen35_shared_timeline_slots,
        qwen35_shared_timeline_source_slot: config.qwen35_shared_timeline_source_slot,
        kv_f16,
        deterministic: config.deterministic,
        samples: sample_count,
        cooldown_ms: config.cooldown_ms,
    })
}

pub fn run_speculative_benchmark_with_backend(
    config: &SpecBenchConfig,
    backend: Box<dyn ax_engine_core::backend::Backend>,
) -> anyhow::Result<SpecBenchResult> {
    let mapped = MappedModel::open(Path::new(&config.model_path))?;
    let model_config = ModelConfig::from_gguf(&mapped.header)?;
    crate::configure_backend_for_model(&*backend, &config.model_path, &mapped, &model_config)?;
    let tokenizer = Tokenizer::from_gguf(&mapped.header)?;
    let model = InferenceModel::with_backend(model_config.clone(), backend)?;
    crate::report_planned_kv_budget(&mapped, &model)?;
    let support_note = crate::support_note(&mapped);
    let weights = WeightStore::new(&mapped);
    model.prepare_runtime_for_weights(&weights)?;

    let sampling = SamplingConfig {
        temperature: 0.0,
        ..Default::default()
    };
    let kv_plan = model.kv_plan();
    let (kv_dtype, kv_f16) = kv_dtype_label_from_plan(&kv_plan);

    eprintln!(
        "Spec benchmark: target={} draft={} k={} layers={} {:.0}MB KV dtype={}",
        config.model_path,
        config.draft_model_path,
        config.speculative_k,
        model_config.n_layers,
        mapped.total_tensor_bytes() as f64 / 1024.0 / 1024.0,
        kv_dtype,
    );

    let prompt_tokens = build_fixed_prompt(&tokenizer, config.prompt_tokens);
    let prefill_plan = model.prefill_plan_summary(
        &weights,
        &model.create_model_kv_for_weights(&weights),
        prompt_tokens.len(),
    )?;
    let target_verify_mode =
        target_verify_mode_label(&model, &model.create_model_kv_for_weights(&weights)).to_string();

    let mut spec = SpeculativeDecoder::load(&config.draft_model_path, config.speculative_k)?;

    for _ in 0..config.warmup_iters {
        run_single_spec_bench(
            &model,
            &weights,
            &tokenizer,
            &prompt_tokens,
            config.decode_tokens,
            &sampling,
            &mut spec,
        )?;
    }

    let sample_count = if config.deterministic {
        config.samples.max(1)
    } else {
        1
    };
    let iters_per_sample = config.measure_iters.max(1);
    let cooldown = Duration::from_millis(config.cooldown_ms);

    let mut prefill_times = Vec::new();
    let mut decode_times = Vec::new();
    let mut sample_prefill_tok_per_sec = Vec::with_capacity(sample_count);
    let mut sample_decode_tok_per_sec = Vec::with_capacity(sample_count);
    let mut decode_token_counts = Vec::new();
    let mut accepted_per_step_values = Vec::new();
    let mut draft_ms_per_step_values = Vec::new();
    let mut verify_ms_per_step_values = Vec::new();
    let mut verify_prepare_ms_per_step_values = Vec::new();
    let mut verify_forward_ms_per_step_values = Vec::new();
    let mut verify_cleanup_ms_per_step_values = Vec::new();
    let mut accept_ms_per_step_values = Vec::new();
    let mut verify_ms_per_position_values = Vec::new();
    let mut draft_ms_per_drafted_token_values = Vec::new();

    for _sample in 0..sample_count {
        let mut sample_prefill_times = Vec::with_capacity(iters_per_sample);
        let mut sample_decode_times = Vec::with_capacity(iters_per_sample);
        let mut sample_decode_token_counts = Vec::with_capacity(iters_per_sample);

        for iter_idx in 0..iters_per_sample {
            let (
                prefill_dur,
                decode_dur,
                generated_tokens,
                avg_accepted_per_step,
                draft_ms_per_step,
                verify_ms_per_step,
                verify_prepare_ms_per_step,
                verify_forward_ms_per_step,
                verify_cleanup_ms_per_step,
                accept_ms_per_step,
                verify_ms_per_position,
                draft_ms_per_drafted_token,
            ) = run_single_spec_bench(
                &model,
                &weights,
                &tokenizer,
                &prompt_tokens,
                config.decode_tokens,
                &sampling,
                &mut spec,
            )?;

            prefill_times.push(prefill_dur);
            decode_times.push(decode_dur);
            decode_token_counts.push(generated_tokens);
            sample_prefill_times.push(prefill_dur);
            sample_decode_times.push(decode_dur);
            sample_decode_token_counts.push(generated_tokens);
            accepted_per_step_values.push(avg_accepted_per_step);
            draft_ms_per_step_values.push(draft_ms_per_step);
            verify_ms_per_step_values.push(verify_ms_per_step);
            verify_prepare_ms_per_step_values.push(verify_prepare_ms_per_step);
            verify_forward_ms_per_step_values.push(verify_forward_ms_per_step);
            verify_cleanup_ms_per_step_values.push(verify_cleanup_ms_per_step);
            accept_ms_per_step_values.push(accept_ms_per_step);
            verify_ms_per_position_values.push(verify_ms_per_position);
            draft_ms_per_drafted_token_values.push(draft_ms_per_drafted_token);

            if config.deterministic && config.cooldown_ms > 0 && iter_idx + 1 < iters_per_sample {
                thread::sleep(cooldown);
            }
        }

        let sample_avg_prefill = sample_prefill_times
            .iter()
            .map(|d| d.as_secs_f64())
            .sum::<f64>()
            / iters_per_sample as f64;
        let sample_avg_decode = sample_decode_times
            .iter()
            .map(|d| d.as_secs_f64())
            .sum::<f64>()
            / iters_per_sample as f64;

        sample_prefill_tok_per_sec.push(if sample_avg_prefill > 0.0 {
            config.prompt_tokens as f64 / sample_avg_prefill
        } else {
            0.0
        });

        let sample_avg_decode_tokens = sample_decode_token_counts.iter().copied().sum::<usize>()
            as f64
            / iters_per_sample as f64;
        sample_decode_tok_per_sec.push(if sample_avg_decode > 0.0 {
            sample_avg_decode_tokens / sample_avg_decode
        } else {
            0.0
        });

        if config.deterministic && config.cooldown_ms > 0 {
            thread::sleep(cooldown);
        }
    }

    let measured_iters = prefill_times.len().max(1) as f64;
    let avg_prefill = prefill_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / measured_iters;
    let avg_decode = decode_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / measured_iters;
    let prefill_tok_per_sec = if avg_prefill > 0.0 {
        config.prompt_tokens as f64 / avg_prefill
    } else {
        0.0
    };
    let avg_decode_tokens =
        decode_token_counts.iter().copied().sum::<usize>() as f64 / measured_iters;
    let decode_tok_per_sec = if avg_decode > 0.0 {
        avg_decode_tokens / avg_decode
    } else {
        0.0
    };

    let median = |vals: &mut [f64]| -> f64 {
        if vals.is_empty() {
            return 0.0;
        }
        vals.sort_by(f64::total_cmp);
        let mid = vals.len() / 2;
        if vals.len().is_multiple_of(2) {
            (vals[mid - 1] + vals[mid]) * 0.5
        } else {
            vals[mid]
        }
    };
    let avg_metric = |vals: &[f64]| -> f64 {
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().copied().sum::<f64>() / vals.len() as f64
        }
    };

    Ok(SpecBenchResult {
        model: config.model_path.clone(),
        draft_model: config.draft_model_path.clone(),
        prompt_tokens: config.prompt_tokens,
        decode_tokens: config.decode_tokens,
        speculative_k: config.speculative_k,
        prefill_tok_per_sec,
        prefill_tok_per_sec_median: median(&mut sample_prefill_tok_per_sec),
        decode_tok_per_sec,
        decode_tok_per_sec_median: median(&mut sample_decode_tok_per_sec),
        q5k_prefill_mode: crate::q5k_prefill_mode(&prefill_plan),
        prefill_plan,
        target_verify_mode,
        avg_accepted_per_step: avg_metric(&accepted_per_step_values),
        draft_ms_per_step: avg_metric(&draft_ms_per_step_values),
        verify_ms_per_step: avg_metric(&verify_ms_per_step_values),
        verify_prepare_ms_per_step: avg_metric(&verify_prepare_ms_per_step_values),
        verify_forward_ms_per_step: avg_metric(&verify_forward_ms_per_step_values),
        verify_cleanup_ms_per_step: avg_metric(&verify_cleanup_ms_per_step_values),
        accept_ms_per_step: avg_metric(&accept_ms_per_step_values),
        verify_ms_per_position: avg_metric(&verify_ms_per_position_values),
        draft_ms_per_drafted_token: avg_metric(&draft_ms_per_drafted_token_values),
        support_note,
        kv_f16,
        deterministic: config.deterministic,
        samples: sample_count,
        cooldown_ms: config.cooldown_ms,
    })
}

/// Run a single benchmark iteration.
/// Returns:
/// (prefill_duration, decode_duration, generated_tokens, per_token_latencies,
///  prefill_command_buffers, prefill_buffer_barriers,
///  decode_command_buffers, decode_buffer_barriers, decode_metal_perf, decode_selection,
///  prefill_plan_summary, decode_plan_summary).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn run_single_bench(
    model: &InferenceModel,
    weights: &WeightStore,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    vocab_size: usize,
    decode_count: usize,
    sampling_config: &SamplingConfig,
    intent: DecodeIntent,
    qwen35_shared_timeline_slots: usize,
    qwen35_shared_timeline_source_slot: Option<usize>,
) -> anyhow::Result<(
    Duration,
    Duration,
    usize,
    Vec<Duration>,
    u64,
    u64,
    u64,
    u64,
    Option<DecodeMetalPerfSummary>,
    DecodeSelection,
    String,
    String,
)> {
    let mut kv = model.create_model_kv_for_weights(weights);
    let mut logits = vec![0.0f32; vocab_size];
    let mut sampler = Sampler::new(bench_sampling_config(sampling_config, tokenizer));

    // Prefill
    model.prepare_prefill_for_weights(weights, &mut kv, prompt_tokens.len())?;
    let prefill_plan = model.prefill_plan_summary(weights, &kv, prompt_tokens.len())?;
    model.reset_metal_perf_counters();
    let prefill_timer = OpTimer::start();
    logits.fill(0.0);
    run_prefill_once(
        model,
        &mut kv,
        prompt_tokens,
        qwen35_shared_timeline_slots,
        qwen35_shared_timeline_source_slot,
        weights,
        &mut logits,
    )?;
    let prefill_dur = prefill_timer.elapsed();
    let prefill_counters = model.read_metal_perf_counters();

    // Decode
    model.reset_metal_perf_counters();
    let mut history = prompt_tokens.to_vec();
    let first_token = sampler.sample(&mut logits, &history);
    let decode_plan = model.decode_plan_summary_for_weights(weights, &kv, intent, true)?;
    let decode = run_decode(
        model,
        weights,
        tokenizer,
        &mut kv,
        &mut sampler,
        &mut history,
        first_token,
        None,
        prompt_tokens.len(),
        decode_count,
        DecodeRunConfig {
            intent,
            allow_pipelined: true,
            top_logprobs: 0,
            collect_metal_perf: true,
        },
        |_tok, _info| Ok(DecodeControl::Continue),
    )?;
    let decode_dur = decode.decode_duration;
    let decode_counters = model.read_metal_perf_counters();

    Ok((
        prefill_dur,
        decode_dur,
        decode.generated_tokens as usize,
        decode.latencies,
        prefill_counters.command_buffers,
        prefill_counters.buffer_barriers,
        decode_counters.command_buffers,
        decode_counters.buffer_barriers,
        decode.metal_perf,
        decode.selection,
        prefill_plan,
        decode_plan,
    ))
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

fn bench_sampling_config(base: &SamplingConfig, tokenizer: &Tokenizer) -> SamplingConfig {
    let mut config = base.clone();
    for stop_token in [Some(tokenizer.eos_id()), tokenizer.eot_id()]
        .into_iter()
        .flatten()
    {
        if !config.banned_token_ids.contains(&stop_token) {
            config.banned_token_ids.push(stop_token);
        }
    }
    config
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_single_spec_bench(
    model: &InferenceModel,
    weights: &WeightStore,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    decode_count: usize,
    sampling_config: &SamplingConfig,
    spec: &mut SpeculativeDecoder,
) -> anyhow::Result<(
    Duration,
    Duration,
    usize,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
)> {
    let vocab_size = model.config.vocab_size as usize;
    let mut kv = model.create_model_kv_for_weights(weights);
    let mut logits = vec![0.0f32; vocab_size];
    let mut sampler = Sampler::new(bench_sampling_config(sampling_config, tokenizer));

    let prefill_timer = OpTimer::start();
    logits.fill(0.0);
    model.forward_batch(prompt_tokens, &mut kv, weights, &mut logits)?;
    let prefill_dur = prefill_timer.elapsed();

    let mut history = prompt_tokens.to_vec();
    let mut last_token = sampler.sample(&mut logits, &history);
    let mut position = prompt_tokens.len();
    let mut generated_tokens = 0usize;
    let mut total_steps = 0usize;
    let mut total_accepted = 0usize;
    let mut total_draft_duration = Duration::ZERO;
    let mut total_verify_duration = Duration::ZERO;
    let mut total_verify_prepare_duration = Duration::ZERO;
    let mut total_verify_forward_duration = Duration::ZERO;
    let mut total_verify_cleanup_duration = Duration::ZERO;
    let mut total_accept_duration = Duration::ZERO;
    let mut total_verified_positions = 0usize;
    let mut total_drafted_tokens = 0usize;

    spec.prewarm_target_verify_path(model, &mut kv)?;

    let decode_timer = OpTimer::start();

    loop {
        if generated_tokens >= decode_count || tokenizer.is_eos(last_token) {
            break;
        }

        let remaining = decode_count - generated_tokens;
        let step_k = spec.k().min(remaining).max(1);

        let step = if step_k < spec.k() {
            logits.fill(0.0);
            model.forward_single(last_token, position, &mut kv, weights, &mut logits)?;
            history.push(last_token);
            let tok = sampler.sample(&mut logits, &history);
            ax_engine_core::speculative::SpecStep {
                tokens: vec![tok],
                n_accepted: 0,
                metrics: Default::default(),
            }
        } else {
            spec.generate_step(
                &history,
                last_token,
                position,
                model,
                &mut kv,
                weights,
                &mut sampler,
            )?
        };

        total_steps += 1;
        total_accepted += step.n_accepted;
        total_draft_duration += step.metrics.draft_duration;
        total_verify_duration += step.metrics.verify_duration;
        total_verify_prepare_duration += step.metrics.verify_prepare_duration;
        total_verify_forward_duration += step.metrics.verify_forward_duration;
        total_verify_cleanup_duration += step.metrics.verify_cleanup_duration;
        total_accept_duration += step.metrics.accept_duration;
        total_verified_positions += step.metrics.verified_positions;
        total_drafted_tokens += step.metrics.drafted_tokens;

        let n_emitted = step.tokens.len().saturating_sub(1);
        history.push(last_token);
        history.extend_from_slice(&step.tokens[..n_emitted]);
        for &tok in &step.tokens[..n_emitted] {
            if tokenizer.is_eos(tok) || generated_tokens >= decode_count {
                break;
            }
            generated_tokens += 1;
        }

        if let Some(&tok) = step.tokens.last() {
            last_token = tok;
            position += step.tokens.len();
        } else {
            break;
        }
    }

    let decode_dur = decode_timer.elapsed();
    let steps = total_steps.max(1) as f64;

    Ok((
        prefill_dur,
        decode_dur,
        generated_tokens,
        total_accepted as f64 / steps,
        total_draft_duration.as_secs_f64() * 1000.0 / steps,
        total_verify_duration.as_secs_f64() * 1000.0 / steps,
        total_verify_prepare_duration.as_secs_f64() * 1000.0 / steps,
        total_verify_forward_duration.as_secs_f64() * 1000.0 / steps,
        total_verify_cleanup_duration.as_secs_f64() * 1000.0 / steps,
        total_accept_duration.as_secs_f64() * 1000.0 / steps,
        if total_verified_positions > 0 {
            total_verify_duration.as_secs_f64() * 1000.0 / total_verified_positions as f64
        } else {
            0.0
        },
        if total_drafted_tokens > 0 {
            total_draft_duration.as_secs_f64() * 1000.0 / total_drafted_tokens as f64
        } else {
            0.0
        },
    ))
}

impl BenchResult {
    fn effective_prefill_tokens_resolved(&self) -> usize {
        if self.effective_prefill_tokens > 0 {
            self.effective_prefill_tokens
        } else {
            effective_qwen35_shared_timeline_tokens(
                self.prompt_tokens,
                self.qwen35_shared_timeline_slots,
            )
        }
    }

    fn effective_prefill_tok_per_sec_resolved(&self) -> f64 {
        if self.effective_prefill_tok_per_sec > 0.0 {
            self.effective_prefill_tok_per_sec
        } else {
            self.prefill_tok_per_sec * self.qwen35_shared_timeline_slots.max(1) as f64
        }
    }

    fn effective_prefill_tok_per_sec_median_resolved(&self) -> f64 {
        if self.effective_prefill_tok_per_sec_median > 0.0 {
            self.effective_prefill_tok_per_sec_median
        } else {
            self.prefill_tok_per_sec_median * self.qwen35_shared_timeline_slots.max(1) as f64
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Parse a JSON-encoded benchmark result.
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }

    /// Print a human-readable summary.
    pub fn print_summary(&self) {
        eprintln!();
        eprintln!("=== Benchmark Results ===");
        eprintln!("Model:       {}", self.model);
        if self.deterministic {
            eprintln!(
                "Mode:        deterministic (samples={}, cooldown={}ms)",
                self.samples, self.cooldown_ms
            );
        }
        eprintln!(
            "Prefill:     {} tokens → median {:.1} tok/s (mean {:.1})",
            self.prompt_tokens, self.prefill_tok_per_sec_median, self.prefill_tok_per_sec,
        );
        if self.qwen35_shared_timeline_slots > 1 {
            eprintln!(
                "Qwen35Fanout:{:>4} shared-timeline slots",
                self.qwen35_shared_timeline_slots
            );
            if let Some(source_slot) = self.qwen35_shared_timeline_source_slot {
                eprintln!("Qwen35Src:   slot {source_slot}");
            }
            eprintln!(
                "PrefillEff:  {} slot-tok → median {:.1} tok/s (mean {:.1})",
                self.effective_prefill_tokens_resolved(),
                self.effective_prefill_tok_per_sec_median_resolved(),
                self.effective_prefill_tok_per_sec_resolved(),
            );
        }
        eprintln!(
            "Decode:      {} tokens → median {:.1} tok/s (mean {:.1})",
            self.decode_tokens, self.decode_tok_per_sec_median, self.decode_tok_per_sec,
        );
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
        if self.decode_intent == "latency" {
            eprintln!(
                "Latency:     P50 {:.2}ms, P95 {:.2}ms, P99 {:.2}ms",
                self.p50_latency.as_secs_f64() * 1000.0,
                self.p95_latency.as_secs_f64() * 1000.0,
                self.p99_latency.as_secs_f64() * 1000.0,
            );
        } else {
            eprintln!("Latency:     n/a (throughput mode)");
        }
        let kv_dtype = kv_dtype_label_from_plan_summary(&self.decode_plan)
            .or_else(|| kv_dtype_label_from_plan_summary(&self.prefill_plan))
            .unwrap_or(if self.kv_f16 { "f16" } else { "f32" });
        eprintln!("KV dtype:    {kv_dtype}");
        eprintln!(
            "GPU Sync:    prefill cmd_buf {:.1}, barriers {:.1} | decode cmd_buf {:.1}, barriers {:.1}",
            self.prefill_command_buffers,
            self.prefill_buffer_barriers,
            self.decode_command_buffers,
            self.decode_buffer_barriers,
        );
        if self.decode_command_buffers_per_tok > 0.0 || self.decode_buffer_barriers_per_tok > 0.0 {
            eprintln!(
                "DecodeShape: CB avg/max {:.2}/{:.0} | barriers avg/max {:.2}/{:.0}",
                self.decode_command_buffers_per_tok,
                self.decode_command_buffers_per_tok_max,
                self.decode_buffer_barriers_per_tok,
                self.decode_buffer_barriers_per_tok_max,
            );
        }
    }
}

impl SpecBenchResult {
    /// Serialize to JSON.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Parse a JSON-encoded speculative benchmark result.
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }

    /// Print a human-readable summary.
    pub fn print_summary(&self) {
        eprintln!();
        eprintln!("=== Speculative Benchmark Results ===");
        eprintln!("Target:      {}", self.model);
        eprintln!("Draft:       {}", self.draft_model);
        eprintln!("K:           {}", self.speculative_k);
        if self.deterministic {
            eprintln!(
                "Mode:        deterministic (samples={}, cooldown={}ms)",
                self.samples, self.cooldown_ms
            );
        }
        eprintln!(
            "Prefill:     {} tokens → median {:.1} tok/s (mean {:.1})",
            self.prompt_tokens, self.prefill_tok_per_sec_median, self.prefill_tok_per_sec,
        );
        eprintln!(
            "Decode:      {} tokens → median {:.1} tok/s (mean {:.1})",
            self.decode_tokens, self.decode_tok_per_sec_median, self.decode_tok_per_sec,
        );
        eprintln!("PrefillPlan: {}", self.prefill_plan);
        eprintln!("VerifyMode:  {}", self.target_verify_mode);
        if let Some(mode) = &self.q5k_prefill_mode {
            eprintln!("Q5KPrefill:  {mode}");
        }
        let kv_dtype = kv_dtype_label_from_plan_summary(&self.prefill_plan)
            .unwrap_or(if self.kv_f16 { "f16" } else { "f32" });
        eprintln!("KV dtype:    {kv_dtype}");
        eprintln!(
            "Accepted:    {:.2} draft tokens/step",
            self.avg_accepted_per_step
        );
        eprintln!(
            "Draft:       {:.2} ms/step, {:.2} ms/drafted token",
            self.draft_ms_per_step, self.draft_ms_per_drafted_token,
        );
        eprintln!(
            "Verify:      {:.2} ms/step, {:.2} ms/position",
            self.verify_ms_per_step, self.verify_ms_per_position,
        );
        eprintln!(
            "VerifySub:   prep {:.2} ms | forward {:.2} ms | cleanup {:.2} ms",
            self.verify_prepare_ms_per_step,
            self.verify_forward_ms_per_step,
            self.verify_cleanup_ms_per_step,
        );
        eprintln!("Accept:      {:.2} ms/step", self.accept_ms_per_step);
        if let Some(note) = &self.support_note {
            eprintln!("Support:     {note}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use ax_engine_core::tokenizer::vocab::TokenType;
    use ax_engine_core::tokenizer::{Tokenizer, Vocab};

    fn make_test_tokenizer_with_eot() -> Tokenizer {
        let vocab = Vocab {
            tokens: vec![
                "<unk>".into(),
                "<bos>".into(),
                "<eos>".into(),
                "<eot>".into(),
            ],
            scores: vec![0.0; 4],
            types: vec![
                TokenType::Unknown,
                TokenType::Control,
                TokenType::Control,
                TokenType::Control,
            ],
            token_to_id: HashMap::from([
                ("<unk>".into(), 0),
                ("<bos>".into(), 1),
                ("<eos>".into(), 2),
                ("<eot>".into(), 3),
            ]),
            merge_ranks: None,
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            add_bos: false,
            add_eos: false,
            add_space_prefix: false,
            model_type: "gpt2".into(),
            eot_id: Some(3),
        };
        Tokenizer::from_vocab(vocab)
    }

    #[test]
    fn test_bench_config_defaults() {
        let c = BenchConfig::default();
        assert_eq!(c.prompt_tokens, 512);
        assert_eq!(c.decode_tokens, 128);
        assert_eq!(c.warmup_iters, 2);
        assert_eq!(c.measure_iters, 5);
        assert!(!c.deterministic);
        assert_eq!(c.samples, 1);
        assert_eq!(c.cooldown_ms, 0);
        assert_eq!(c.qwen35_shared_timeline_slots, 1);
        assert_eq!(c.qwen35_shared_timeline_source_slot, None);
    }

    #[test]
    fn test_bench_sampling_config_bans_stop_tokens() {
        let tokenizer = make_test_tokenizer_with_eot();
        let config = SamplingConfig::default();
        let bench = bench_sampling_config(&config, &tokenizer);
        assert!(bench.banned_token_ids.contains(&2));
        assert!(bench.banned_token_ids.contains(&3));
    }

    #[test]
    fn test_spec_bench_config_defaults() {
        let c = SpecBenchConfig::default();
        assert_eq!(c.prompt_tokens, 512);
        assert_eq!(c.decode_tokens, 128);
        assert_eq!(c.warmup_iters, 2);
        assert_eq!(c.measure_iters, 5);
        assert!(!c.deterministic);
        assert_eq!(c.samples, 1);
        assert_eq!(c.cooldown_ms, 0);
        assert_eq!(c.speculative_k, 4);
    }

    #[test]
    fn test_spec_bench_result_summary_formatting() {
        let result = SpecBenchResult {
            model: "target.gguf".into(),
            draft_model: "draft.gguf".into(),
            prompt_tokens: 512,
            decode_tokens: 64,
            speculative_k: 4,
            prefill_tok_per_sec: 900.0,
            prefill_tok_per_sec_median: 900.0,
            decode_tok_per_sec: 55.0,
            decode_tok_per_sec_median: 55.0,
            prefill_plan: "mode=gpu_batch q5k_prefill=small_n".into(),
            target_verify_mode: "qwen35_branch".into(),
            q5k_prefill_mode: Some("small_n".into()),
            avg_accepted_per_step: 1.5,
            draft_ms_per_step: 2.0,
            verify_ms_per_step: 3.0,
            verify_prepare_ms_per_step: 0.5,
            verify_forward_ms_per_step: 2.0,
            verify_cleanup_ms_per_step: 0.5,
            accept_ms_per_step: 0.2,
            verify_ms_per_position: 1.0,
            draft_ms_per_drafted_token: 0.5,
            support_note: Some(
                ax_engine_core::gguf::mmap::support_note_for_q5k_layer_presence(true)
                    .unwrap()
                    .to_string(),
            ),
            kv_f16: true,
            deterministic: false,
            samples: 1,
            cooldown_ms: 0,
        };

        result.print_summary();
    }

    #[test]
    fn test_bench_result_to_json() {
        let result = BenchResult {
            model: "test.gguf".into(),
            prompt_tokens: 16,
            effective_prefill_tokens: 16,
            decode_tokens: 32,
            prefill_tok_per_sec: 100.0,
            prefill_tok_per_sec_median: 99.0,
            effective_prefill_tok_per_sec: 100.0,
            effective_prefill_tok_per_sec_median: 99.0,
            decode_tok_per_sec: 35.0,
            decode_tok_per_sec_median: 34.5,
            p50_latency: Duration::from_millis(10),
            p95_latency: Duration::from_millis(12),
            p99_latency: Duration::from_millis(15),
            prefill_command_buffers: 1.0,
            prefill_buffer_barriers: 10.0,
            decode_command_buffers: 32.0,
            decode_buffer_barriers: 0.0,
            decode_command_buffers_per_tok: 1.0,
            decode_buffer_barriers_per_tok: 0.0,
            decode_command_buffers_per_tok_max: 1.0,
            decode_buffer_barriers_per_tok_max: 0.0,
            decode_intent: "throughput".into(),
            decode_mode: "pipelined".into(),
            prefill_plan: "mode=gpu_batch q5k_prefill=small_n".into(),
            prefill_mode: "gpu_batch".into(),
            prefill_route_family: "dense_gpu_batch".into(),
            prefill_route_detail: "generic_gpu_batch".into(),
            prefill_attention_route: None,
            prefill_qkv_plan: None,
            prefill_split_rope_append: None,
            q5k_prefill_mode: Some("small_n".into()),
            decode_plan: "sync=pipelined scratch=gpu_shared".into(),
            support_note: Some(
                ax_engine_core::gguf::mmap::support_note_for_q5k_layer_presence(true)
                    .unwrap()
                    .to_string(),
            ),
            decode_fallback_reason: None,
            qwen35_shared_timeline_slots: 1,
            qwen35_shared_timeline_source_slot: None,
            kv_f16: true,
            deterministic: false,
            samples: 1,
            cooldown_ms: 0,
        };
        let json = result.to_json().unwrap();
        assert!(json.contains("\"q5k_prefill_mode\": \"small_n\""));
        assert!(json.contains("\"prefill_plan\": \"mode=gpu_batch q5k_prefill=small_n\""));
        assert!(json.contains("\"prefill_mode\": \"gpu_batch\""));
        assert!(json.contains("\"prefill_route_family\": \"dense_gpu_batch\""));
    }

    #[test]
    fn test_bench_result_from_json_defaults_missing_optional_fields() {
        let json = r#"{
          "model": "test.gguf",
          "prompt_tokens": 16,
          "decode_tokens": 32,
          "prefill_tok_per_sec": 100.0,
          "prefill_tok_per_sec_median": 99.0,
          "decode_tok_per_sec": 35.0,
          "decode_tok_per_sec_median": 34.5,
          "p50_latency": { "secs": 0, "nanos": 0 },
          "p95_latency": { "secs": 0, "nanos": 0 },
          "p99_latency": { "secs": 0, "nanos": 0 },
          "prefill_command_buffers": 1.0,
          "prefill_buffer_barriers": 10.0,
          "decode_command_buffers": 32.0,
          "decode_buffer_barriers": 0.0,
          "decode_intent": "throughput",
          "kv_f16": true,
          "deterministic": false
        }"#;
        let result = BenchResult::from_json(json).unwrap();
        assert_eq!(result.model, "test.gguf");
        assert_eq!(result.q5k_prefill_mode, None);
        assert_eq!(result.qwen35_shared_timeline_slots, 1);
        assert_eq!(result.qwen35_shared_timeline_source_slot, None);
        assert_eq!(result.effective_prefill_tokens, 0);
        assert_eq!(result.effective_prefill_tok_per_sec, 0.0);
        assert_eq!(result.effective_prefill_tok_per_sec_median, 0.0);
        assert!(result.prefill_plan.is_empty());
        assert!(result.prefill_mode.is_empty());
        assert!(result.prefill_route_family.is_empty());
        assert!(result.prefill_route_detail.is_empty());
        assert_eq!(result.prefill_attention_route, None);
        assert_eq!(result.prefill_qkv_plan, None);
        assert_eq!(result.prefill_split_rope_append, None);
        assert!(result.decode_plan.is_empty());
        assert_eq!(result.samples, 0);
        assert_eq!(result.cooldown_ms, 0);
    }

    #[test]
    fn test_spec_bench_result_to_json() {
        let result = SpecBenchResult {
            model: "target.gguf".into(),
            draft_model: "draft.gguf".into(),
            prompt_tokens: 512,
            decode_tokens: 64,
            speculative_k: 4,
            prefill_tok_per_sec: 900.0,
            prefill_tok_per_sec_median: 900.0,
            decode_tok_per_sec: 55.0,
            decode_tok_per_sec_median: 55.0,
            prefill_plan: "mode=gpu_batch q5k_prefill=small_n".into(),
            target_verify_mode: "qwen35_branch".into(),
            q5k_prefill_mode: Some("small_n".into()),
            avg_accepted_per_step: 1.5,
            draft_ms_per_step: 2.0,
            verify_ms_per_step: 3.0,
            verify_prepare_ms_per_step: 0.5,
            verify_forward_ms_per_step: 2.0,
            verify_cleanup_ms_per_step: 0.5,
            accept_ms_per_step: 0.2,
            verify_ms_per_position: 1.0,
            draft_ms_per_drafted_token: 0.5,
            support_note: Some(
                ax_engine_core::gguf::mmap::support_note_for_q5k_layer_presence(true)
                    .unwrap()
                    .to_string(),
            ),
            kv_f16: true,
            deterministic: false,
            samples: 1,
            cooldown_ms: 0,
        };
        let json = result.to_json().unwrap();
        assert!(json.contains("\"q5k_prefill_mode\": \"small_n\""));
        assert!(json.contains("\"prefill_plan\": \"mode=gpu_batch q5k_prefill=small_n\""));
    }

    #[test]
    fn test_spec_bench_result_from_json_defaults_missing_optional_fields() {
        let json = r#"{
          "model": "target.gguf",
          "draft_model": "draft.gguf",
          "prompt_tokens": 512,
          "decode_tokens": 64,
          "speculative_k": 4,
          "prefill_tok_per_sec": 900.0,
          "prefill_tok_per_sec_median": 900.0,
          "decode_tok_per_sec": 55.0,
          "decode_tok_per_sec_median": 55.0,
          "avg_accepted_per_step": 1.5,
          "draft_ms_per_step": 2.0,
          "verify_ms_per_step": 3.0,
          "accept_ms_per_step": 0.2,
          "verify_ms_per_position": 1.0,
          "draft_ms_per_drafted_token": 0.5,
          "kv_f16": true,
          "deterministic": false
        }"#;
        let result = SpecBenchResult::from_json(json).unwrap();
        assert_eq!(result.model, "target.gguf");
        assert_eq!(result.q5k_prefill_mode, None);
        assert!(result.target_verify_mode.is_empty());
        assert!(result.prefill_plan.is_empty());
        assert_eq!(result.samples, 0);
        assert_eq!(result.cooldown_ms, 0);
    }
}
