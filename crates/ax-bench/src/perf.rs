//! Performance benchmark runner.
//!
//! Measures prefill tok/s, decode tok/s, and latency percentiles
//! for a given model and configuration.

use std::path::Path;
use std::thread;
use std::time::Duration;

use ax_core::gguf::MappedModel;
use ax_core::metrics::LatencyHistogram;
use ax_core::metrics::counters::OpTimer;
use ax_core::model::{
    DecodeControl, DecodeIntent, DecodeRunConfig, DecodeSelection, LlamaModel, ModelConfig,
    WeightStore, run_decode,
};
use ax_core::sampling::{Sampler, SamplingConfig};
use ax_core::speculative::SpeculativeDecoder;
use ax_core::tokenizer::Tokenizer;

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
#[derive(Debug)]
pub struct BenchResult {
    /// Model name/path.
    pub model: String,
    /// Prompt token count.
    pub prompt_tokens: usize,
    /// Decode token count.
    pub decode_tokens: usize,
    /// Average prefill throughput (tok/s).
    pub prefill_tok_per_sec: f64,
    /// Median prefill throughput (tok/s).
    pub prefill_tok_per_sec_median: f64,
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
    /// Benchmark intent (`throughput` or `latency`).
    pub decode_intent: String,
    /// Selected decode mode for measured runs.
    pub decode_mode: String,
    /// Fallback reason if a faster mode was requested but not used.
    pub decode_fallback_reason: Option<String>,
    /// Whether f16 KV cache was active.
    pub kv_f16: bool,
    /// True when deterministic mode was enabled.
    pub deterministic: bool,
    /// Number of repeated samples used.
    pub samples: usize,
    /// Cooldown between measured iterations (ms).
    pub cooldown_ms: u64,
}

/// Results of a speculative-decoding benchmark.
#[derive(Debug)]
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
    /// Average accepted draft tokens per speculative step.
    pub avg_accepted_per_step: f64,
    /// Average draft time per speculative step.
    pub draft_ms_per_step: f64,
    /// Average verification time per speculative step.
    pub verify_ms_per_step: f64,
    /// Average acceptance bookkeeping time per speculative step.
    pub accept_ms_per_step: f64,
    /// Average verification time per verified position.
    pub verify_ms_per_position: f64,
    /// Average draft time per drafted token.
    pub draft_ms_per_drafted_token: f64,
    /// Whether f16 KV cache was active.
    pub kv_f16: bool,
    /// True when deterministic mode was enabled.
    pub deterministic: bool,
    /// Number of repeated samples used.
    pub samples: usize,
    /// Cooldown between measured iterations (ms).
    pub cooldown_ms: u64,
}

/// Run a performance benchmark.
pub fn run_benchmark(config: &BenchConfig) -> anyhow::Result<BenchResult> {
    run_benchmark_with_backend(
        config,
        ax_core::backend::create_backend(ax_core::backend::BackendConfig::default())?,
    )
}

/// Run a speculative-decoding benchmark.
pub fn run_speculative_benchmark(config: &SpecBenchConfig) -> anyhow::Result<SpecBenchResult> {
    run_speculative_benchmark_with_backend(
        config,
        ax_core::backend::create_backend(ax_core::backend::BackendConfig::default())?,
    )
}

pub fn run_benchmark_with_backend(
    config: &BenchConfig,
    backend: Box<dyn ax_core::backend::Backend>,
) -> anyhow::Result<BenchResult> {
    let mapped = MappedModel::open(Path::new(&config.model_path))?;
    crate::init_kernel_profile(&config.model_path, &mapped);
    let model_config = ModelConfig::from_gguf(&mapped.header)?;
    let tokenizer = Tokenizer::from_gguf(&mapped.header)?;
    let model = LlamaModel::with_backend(model_config.clone(), backend);
    let weights = WeightStore::new(&mapped);

    let vocab_size = model_config.vocab_size as usize;
    let sampling = SamplingConfig {
        temperature: 0.0, // greedy for determinism
        ..Default::default()
    };

    let kv_f16 =
        ax_core::backend::metal::metal_f16_kv_cache_enabled(model_config.context_length as usize);

    eprintln!(
        "Benchmarking: {} layers, {:.0}MB, KV dtype: {}",
        model_config.n_layers,
        mapped.total_tensor_bytes() as f64 / 1024.0 / 1024.0,
        if kv_f16 { "f16" } else { "f32" },
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
    let mut decode_selection: Option<DecodeSelection> = None;

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
                iter_selection,
            ) = run_single_bench(
                &model,
                &weights,
                &tokenizer,
                &prompt_tokens,
                vocab_size,
                config.decode_tokens,
                &sampling,
                config.intent,
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

    let avg_u64 = |vals: &[u64]| -> f64 {
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().copied().sum::<u64>() as f64 / vals.len() as f64
        }
    };

    Ok(BenchResult {
        model: config.model_path.clone(),
        prompt_tokens: config.prompt_tokens,
        decode_tokens: config.decode_tokens,
        prefill_tok_per_sec,
        prefill_tok_per_sec_median,
        decode_tok_per_sec,
        decode_tok_per_sec_median,
        p50_latency: latency.p50().unwrap_or(Duration::ZERO),
        p95_latency: latency.p95().unwrap_or(Duration::ZERO),
        p99_latency: latency.p99().unwrap_or(Duration::ZERO),
        prefill_command_buffers: avg_u64(&prefill_cmd_buf_counts),
        prefill_buffer_barriers: avg_u64(&prefill_barrier_counts),
        decode_command_buffers: avg_u64(&decode_cmd_buf_counts),
        decode_buffer_barriers: avg_u64(&decode_barrier_counts),
        decode_intent: config.intent.to_string(),
        decode_mode: decode_selection
            .as_ref()
            .map(|s| s.mode.to_string())
            .unwrap_or_else(|| "sequential".to_string()),
        decode_fallback_reason: decode_selection.and_then(|s| s.fallback_reason),
        kv_f16,
        deterministic: config.deterministic,
        samples: sample_count,
        cooldown_ms: config.cooldown_ms,
    })
}

pub fn run_speculative_benchmark_with_backend(
    config: &SpecBenchConfig,
    backend: Box<dyn ax_core::backend::Backend>,
) -> anyhow::Result<SpecBenchResult> {
    let mapped = MappedModel::open(Path::new(&config.model_path))?;
    crate::init_kernel_profile(&config.model_path, &mapped);
    let model_config = ModelConfig::from_gguf(&mapped.header)?;
    let tokenizer = Tokenizer::from_gguf(&mapped.header)?;
    let model = LlamaModel::with_backend(model_config.clone(), backend);
    let weights = WeightStore::new(&mapped);

    let sampling = SamplingConfig {
        temperature: 0.0,
        ..Default::default()
    };
    let kv_f16 =
        ax_core::backend::metal::metal_f16_kv_cache_enabled(model_config.context_length as usize);

    eprintln!(
        "Spec benchmark: target={} draft={} k={} layers={} {:.0}MB KV dtype={}",
        config.model_path,
        config.draft_model_path,
        config.speculative_k,
        model_config.n_layers,
        mapped.total_tensor_bytes() as f64 / 1024.0 / 1024.0,
        if kv_f16 { "f16" } else { "f32" },
    );

    let prompt_tokens = build_fixed_prompt(&tokenizer, config.prompt_tokens);

    for _ in 0..config.warmup_iters {
        run_single_spec_bench(
            &model,
            &weights,
            &tokenizer,
            &prompt_tokens,
            config.decode_tokens,
            &sampling,
            &config.draft_model_path,
            config.speculative_k,
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
                &config.draft_model_path,
                config.speculative_k,
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
        avg_accepted_per_step: avg_metric(&accepted_per_step_values),
        draft_ms_per_step: avg_metric(&draft_ms_per_step_values),
        verify_ms_per_step: avg_metric(&verify_ms_per_step_values),
        accept_ms_per_step: avg_metric(&accept_ms_per_step_values),
        verify_ms_per_position: avg_metric(&verify_ms_per_position_values),
        draft_ms_per_drafted_token: avg_metric(&draft_ms_per_drafted_token_values),
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
///  decode_command_buffers, decode_buffer_barriers, decode_selection).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn run_single_bench(
    model: &LlamaModel,
    weights: &WeightStore,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    vocab_size: usize,
    decode_count: usize,
    sampling_config: &SamplingConfig,
    intent: DecodeIntent,
) -> anyhow::Result<(
    Duration,
    Duration,
    usize,
    Vec<Duration>,
    u64,
    u64,
    u64,
    u64,
    DecodeSelection,
)> {
    let mut kv = model.create_model_kv();
    let mut logits = vec![0.0f32; vocab_size];
    let mut sampler = Sampler::new(sampling_config.clone());

    // Prefill
    ax_core::backend::metal::reset_metal_perf_counters();
    let prefill_timer = OpTimer::start();
    logits.fill(0.0);
    model.forward_batch(prompt_tokens, &mut kv, weights, &mut logits)?;
    let prefill_dur = prefill_timer.elapsed();
    let prefill_counters = ax_core::backend::metal::read_metal_perf_counters();

    // Decode
    ax_core::backend::metal::reset_metal_perf_counters();
    let mut history = prompt_tokens.to_vec();
    let first_token = sampler.sample(&mut logits, &history);
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
        },
        |_tok, _info| Ok(DecodeControl::Continue),
    )?;
    let decode_dur = decode.decode_duration;
    let decode_counters = ax_core::backend::metal::read_metal_perf_counters();

    Ok((
        prefill_dur,
        decode_dur,
        decode.generated_tokens as usize,
        decode.latencies,
        prefill_counters.command_buffers,
        prefill_counters.buffer_barriers,
        decode_counters.command_buffers,
        decode_counters.buffer_barriers,
        decode.selection,
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

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_single_spec_bench(
    model: &LlamaModel,
    weights: &WeightStore,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    decode_count: usize,
    sampling_config: &SamplingConfig,
    draft_model_path: &str,
    speculative_k: usize,
) -> anyhow::Result<(Duration, Duration, usize, f64, f64, f64, f64, f64, f64)> {
    let vocab_size = model.config.vocab_size as usize;
    let mut kv = model.create_model_kv();
    let mut logits = vec![0.0f32; vocab_size];
    let mut sampler = Sampler::new(sampling_config.clone());
    let mut spec = SpeculativeDecoder::load(draft_model_path, speculative_k)?;

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
    let mut total_accept_duration = Duration::ZERO;
    let mut total_verified_positions = 0usize;
    let mut total_drafted_tokens = 0usize;

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
            let tok = sampler.sample(&mut logits, &[]);
            ax_core::speculative::SpecStep {
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
            generated_tokens = generated_tokens.saturating_add(1);
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
        eprintln!(
            "Decode:      {} tokens → median {:.1} tok/s (mean {:.1})",
            self.decode_tokens, self.decode_tok_per_sec_median, self.decode_tok_per_sec,
        );
        eprintln!("Intent:      {}", self.decode_intent);
        eprintln!("Mode:        {}", self.decode_mode);
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
        eprintln!("KV dtype:    {}", if self.kv_f16 { "f16" } else { "f32" });
        eprintln!(
            "GPU Sync:    prefill cmd_buf {:.1}, barriers {:.1} | decode cmd_buf {:.1}, barriers {:.1}",
            self.prefill_command_buffers,
            self.prefill_buffer_barriers,
            self.decode_command_buffers,
            self.decode_buffer_barriers,
        );
    }
}

impl SpecBenchResult {
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
        eprintln!("KV dtype:    {}", if self.kv_f16 { "f16" } else { "f32" });
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
        eprintln!("Accept:      {:.2} ms/step", self.accept_ms_per_step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
