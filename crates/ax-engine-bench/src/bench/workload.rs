//! Workload-oriented benchmark runner for completion and infill request paths.

use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, ensure};
use ax_engine_sdk::{
    FinishReason, GenerationOptions, LoadOptions, Model, PromptCacheStats, Session, SessionOptions,
    SessionSnapshot,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadKind {
    Completion,
    Infill,
}

impl WorkloadKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completion => "completion",
            Self::Infill => "infill",
        }
    }
}

#[derive(Debug, Clone)]
pub enum WorkloadInput {
    Completion { prompt: String },
    Infill { prefix: String, suffix: String },
}

impl WorkloadInput {
    fn resolve(&self, model: &Model) -> anyhow::Result<ResolvedWorkloadInput> {
        match self {
            Self::Completion { prompt } => {
                let tokens = model.tokenize(prompt, true);
                let prompt_tokens = tokens.len();
                ensure!(!tokens.is_empty(), "workload prompt produced no tokens");
                Ok(ResolvedWorkloadInput {
                    kind: WorkloadKind::Completion,
                    prompt_tokens: tokens,
                    summary: InputSummary {
                        prompt_tokens,
                        prompt_bytes: prompt.len(),
                        prefix_bytes: None,
                        suffix_bytes: None,
                    },
                })
            }
            Self::Infill { prefix, suffix } => {
                ensure!(
                    model.supports_infill(),
                    "loaded model does not support infill"
                );
                ensure!(
                    !(prefix.is_empty() && suffix.is_empty()),
                    "infill requires a non-empty prefix or suffix"
                );
                let rendered = model
                    .render_infill_prompt(prefix, suffix)
                    .context("failed to render infill prompt")?;
                let tokens = model.tokenize_with_options(&rendered, true, true);
                let prompt_tokens = tokens.len();
                ensure!(!tokens.is_empty(), "workload prompt produced no tokens");
                Ok(ResolvedWorkloadInput {
                    kind: WorkloadKind::Infill,
                    prompt_tokens: tokens,
                    summary: InputSummary {
                        prompt_tokens,
                        prompt_bytes: rendered.len(),
                        prefix_bytes: Some(prefix.len()),
                        suffix_bytes: Some(suffix.len()),
                    },
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkloadBenchConfig {
    pub model_path: String,
    pub label: Option<String>,
    pub workload: WorkloadKind,
    pub measured: WorkloadInput,
    pub prime: Option<WorkloadInput>,
    pub max_tokens: usize,
    pub warmup_iters: usize,
    pub measure_iters: usize,
    pub deterministic: bool,
    pub samples: usize,
    pub cooldown_ms: u64,
    pub context_length: Option<u32>,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadBenchResult {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub workload: String,
    pub primed: bool,
    pub prompt_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prime_prompt_tokens: Option<usize>,
    pub max_tokens: usize,
    pub request_ms_mean: f64,
    pub request_ms_median: f64,
    pub first_token_ms_mean: f64,
    pub first_token_ms_median: f64,
    pub output_tok_per_sec_mean: f64,
    pub output_tok_per_sec_median: f64,
    pub generated_tokens_mean: f64,
    pub generated_tokens_median: f64,
    pub cached_tokens_mean: f64,
    pub cached_tokens_median: f64,
    pub prompt_tokens_evaluated_mean: f64,
    pub prompt_tokens_evaluated_median: f64,
    pub cache_hit_ratio_mean: f64,
    pub cache_hit_ratio_median: f64,
    pub prompt_tokens_mean: f64,
    pub prompt_tokens_median: f64,
    pub finish_reason: String,
    pub deterministic: bool,
    pub samples: usize,
    pub measure_iters: usize,
    pub cooldown_ms: u64,
    pub input: InputSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prime_input: Option<InputSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub support_note: Option<String>,
    pub sample_request_ms: Vec<f64>,
    pub sample_first_token_ms: Vec<f64>,
    pub sample_output_tok_per_sec: Vec<f64>,
    pub sample_generated_tokens: Vec<f64>,
    pub sample_cached_tokens: Vec<f64>,
    pub sample_prompt_tokens_evaluated: Vec<f64>,
    pub sample_cache_hit_ratio: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSummary {
    pub prompt_tokens: usize,
    pub prompt_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix_bytes: Option<usize>,
}

#[derive(Debug, Clone)]
struct ResolvedWorkloadInput {
    kind: WorkloadKind,
    prompt_tokens: Vec<u32>,
    summary: InputSummary,
}

#[derive(Debug, Clone, Copy)]
struct IterationMetrics {
    request_ms: f64,
    first_token_ms: f64,
    output_tok_per_sec: f64,
    generated_tokens: usize,
    cache_stats: PromptCacheStats,
    finish_reason: FinishReason,
}

impl WorkloadBenchResult {
    pub fn print_summary(&self) {
        eprintln!(
            "Workload:    {}{}",
            self.workload,
            if self.primed { " (primed)" } else { "" }
        );
        if let Some(label) = self.label.as_deref() {
            eprintln!("Label:       {label}");
        }
        eprintln!("Model:       {}", self.model);
        eprintln!("Prompt:      {} tok", self.prompt_tokens,);
        if let Some(prime_prompt_tokens) = self.prime_prompt_tokens {
            eprintln!("Prime:       {} tok", prime_prompt_tokens);
        }
        eprintln!("Max tokens:  {}", self.max_tokens);
        eprintln!(
            "Request:     median {:.2} ms | mean {:.2} ms",
            self.request_ms_median, self.request_ms_mean
        );
        eprintln!(
            "First token: median {:.2} ms | mean {:.2} ms",
            self.first_token_ms_median, self.first_token_ms_mean
        );
        eprintln!(
            "Output:      median {:.2} tok/s | mean {:.2} tok/s",
            self.output_tok_per_sec_median, self.output_tok_per_sec_mean
        );
        eprintln!(
            "Cache:       median {:.1}% hit | {:.1} cached | {:.1} eval",
            self.cache_hit_ratio_median * 100.0,
            self.cached_tokens_median,
            self.prompt_tokens_evaluated_median,
        );
        if let Some(note) = self.support_note.as_deref() {
            eprintln!("Support:     {note}");
        }
    }

    pub fn to_json(&self) -> anyhow::Result<String> {
        serde_json::to_string_pretty(self).context("failed to serialize workload result")
    }
}

pub fn run_workload_benchmark(config: &WorkloadBenchConfig) -> anyhow::Result<WorkloadBenchResult> {
    ensure!(
        config.max_tokens > 0,
        "max_tokens must be greater than zero"
    );
    ensure!(
        config.measure_iters > 0,
        "measure_iters must be greater than zero"
    );

    let model = Model::load(
        &config.model_path,
        LoadOptions {
            backend: Default::default(),
            context_length: config.context_length,
        },
    )?;
    let measured = config.measured.resolve(&model)?;
    ensure!(
        measured.kind == config.workload,
        "measured workload does not match the requested workload mode"
    );
    let prime = config
        .prime
        .as_ref()
        .map(|input| input.resolve(&model))
        .transpose()?;
    if let Some(prime) = prime.as_ref() {
        ensure!(
            prime.kind == config.workload,
            "prime workload does not match the requested workload mode"
        );
    }

    for _ in 0..config.warmup_iters {
        let session = model
            .session(SessionOptions::default())
            .context("failed to create workload warmup session")?;
        let prime_snapshot = prepare_prime_snapshot(&session, prime.as_ref())?;
        if let Some(snapshot) = prime_snapshot.as_ref() {
            session
                .restore_snapshot(snapshot)
                .context("failed to restore warmup snapshot")?;
        } else {
            session.reset().context("failed to reset warmup session")?;
        }
        run_iteration(
            &session,
            &measured.prompt_tokens,
            config.max_tokens,
            config.seed,
        )?;
    }

    let sample_count = if config.deterministic {
        config.samples.max(1)
    } else {
        1
    };
    let cooldown = Duration::from_millis(config.cooldown_ms);
    let mut sample_request_ms = Vec::with_capacity(sample_count);
    let mut sample_first_token_ms = Vec::with_capacity(sample_count);
    let mut sample_output_tok_per_sec = Vec::with_capacity(sample_count);
    let mut sample_generated_tokens = Vec::with_capacity(sample_count);
    let mut sample_cached_tokens = Vec::with_capacity(sample_count);
    let mut sample_prompt_tokens = Vec::with_capacity(sample_count);
    let mut sample_prompt_tokens_evaluated = Vec::with_capacity(sample_count);
    let mut sample_cache_hit_ratio = Vec::with_capacity(sample_count);
    let mut finish_reason = FinishReason::Stop;

    for sample_idx in 0..sample_count {
        let session = model.session(SessionOptions::default()).with_context(|| {
            format!(
                "failed to create workload session for sample {}",
                sample_idx + 1
            )
        })?;
        let prime_snapshot = prepare_prime_snapshot(&session, prime.as_ref())?;

        let mut iter_request_ms = Vec::with_capacity(config.measure_iters);
        let mut iter_first_token_ms = Vec::with_capacity(config.measure_iters);
        let mut iter_output_tok_per_sec = Vec::with_capacity(config.measure_iters);
        let mut iter_generated_tokens = Vec::with_capacity(config.measure_iters);
        let mut iter_cached_tokens = Vec::with_capacity(config.measure_iters);
        let mut iter_prompt_tokens = Vec::with_capacity(config.measure_iters);
        let mut iter_prompt_tokens_evaluated = Vec::with_capacity(config.measure_iters);
        let mut iter_cache_hit_ratio = Vec::with_capacity(config.measure_iters);

        for iter_idx in 0..config.measure_iters {
            if let Some(snapshot) = prime_snapshot.as_ref() {
                session
                    .restore_snapshot(snapshot)
                    .context("failed to restore workload prime snapshot")?;
            } else {
                session
                    .reset()
                    .context("failed to reset workload session")?;
            }

            let metrics = run_iteration(
                &session,
                &measured.prompt_tokens,
                config.max_tokens,
                config.seed,
            )?;
            finish_reason = metrics.finish_reason;

            iter_request_ms.push(metrics.request_ms);
            iter_first_token_ms.push(metrics.first_token_ms);
            iter_output_tok_per_sec.push(metrics.output_tok_per_sec);
            iter_generated_tokens.push(metrics.generated_tokens as f64);
            iter_cached_tokens.push(metrics.cache_stats.cached_tokens as f64);
            iter_prompt_tokens.push(metrics.cache_stats.prompt_tokens as f64);
            iter_prompt_tokens_evaluated.push(metrics.cache_stats.prompt_tokens_evaluated as f64);
            iter_cache_hit_ratio.push(cache_hit_ratio(metrics.cache_stats));

            if config.cooldown_ms > 0 && iter_idx + 1 < config.measure_iters {
                thread::sleep(cooldown);
            }
        }

        sample_request_ms.push(mean(&iter_request_ms));
        sample_first_token_ms.push(mean(&iter_first_token_ms));
        sample_output_tok_per_sec.push(mean(&iter_output_tok_per_sec));
        sample_generated_tokens.push(mean(&iter_generated_tokens));
        sample_cached_tokens.push(mean(&iter_cached_tokens));
        sample_prompt_tokens.push(mean(&iter_prompt_tokens));
        sample_prompt_tokens_evaluated.push(mean(&iter_prompt_tokens_evaluated));
        sample_cache_hit_ratio.push(mean(&iter_cache_hit_ratio));

        if config.deterministic && config.cooldown_ms > 0 && sample_idx + 1 < sample_count {
            thread::sleep(cooldown);
        }
    }

    let mut request_ms_for_median = sample_request_ms.clone();
    let mut first_token_ms_for_median = sample_first_token_ms.clone();
    let mut output_tok_per_sec_for_median = sample_output_tok_per_sec.clone();
    let mut generated_tokens_for_median = sample_generated_tokens.clone();
    let mut cached_tokens_for_median = sample_cached_tokens.clone();
    let mut prompt_tokens_for_median = sample_prompt_tokens.clone();
    let mut prompt_tokens_evaluated_for_median = sample_prompt_tokens_evaluated.clone();
    let mut cache_hit_ratio_for_median = sample_cache_hit_ratio.clone();

    Ok(WorkloadBenchResult {
        model: config.model_path.clone(),
        label: config.label.clone(),
        workload: config.workload.as_str().to_string(),
        primed: prime.is_some(),
        prompt_tokens: measured.summary.prompt_tokens,
        prime_prompt_tokens: prime.as_ref().map(|input| input.summary.prompt_tokens),
        max_tokens: config.max_tokens,
        request_ms_mean: mean(&sample_request_ms),
        request_ms_median: median(&mut request_ms_for_median),
        first_token_ms_mean: mean(&sample_first_token_ms),
        first_token_ms_median: median(&mut first_token_ms_for_median),
        output_tok_per_sec_mean: mean(&sample_output_tok_per_sec),
        output_tok_per_sec_median: median(&mut output_tok_per_sec_for_median),
        generated_tokens_mean: mean(&sample_generated_tokens),
        generated_tokens_median: median(&mut generated_tokens_for_median),
        cached_tokens_mean: mean(&sample_cached_tokens),
        cached_tokens_median: median(&mut cached_tokens_for_median),
        prompt_tokens_evaluated_mean: mean(&sample_prompt_tokens_evaluated),
        prompt_tokens_evaluated_median: median(&mut prompt_tokens_evaluated_for_median),
        cache_hit_ratio_mean: mean(&sample_cache_hit_ratio),
        cache_hit_ratio_median: median(&mut cache_hit_ratio_for_median),
        prompt_tokens_mean: mean(&sample_prompt_tokens),
        prompt_tokens_median: median(&mut prompt_tokens_for_median),
        finish_reason: finish_reason_name(finish_reason).to_string(),
        deterministic: config.deterministic,
        samples: sample_count,
        measure_iters: config.measure_iters,
        cooldown_ms: config.cooldown_ms,
        input: measured.summary.clone(),
        prime_input: prime.as_ref().map(|input| input.summary.clone()),
        support_note: model.support_note().map(str::to_owned),
        sample_request_ms,
        sample_first_token_ms,
        sample_output_tok_per_sec,
        sample_generated_tokens,
        sample_cached_tokens,
        sample_prompt_tokens_evaluated,
        sample_cache_hit_ratio,
    })
}

fn prepare_prime_snapshot(
    session: &Session,
    prime: Option<&ResolvedWorkloadInput>,
) -> anyhow::Result<Option<SessionSnapshot>> {
    let Some(prime) = prime else {
        return Ok(None);
    };
    session
        .load_prompt_tokens(&prime.prompt_tokens)
        .context("failed to load primed workload prompt")?;
    let snapshot = session
        .snapshot()
        .context("failed to capture primed workload snapshot")?;
    Ok(Some(snapshot))
}

fn run_iteration(
    session: &Session,
    prompt_tokens: &[u32],
    max_tokens: usize,
    seed: u64,
) -> anyhow::Result<IterationMetrics> {
    let options = GenerationOptions::default()
        .max_tokens(max_tokens)
        .temperature(0.0)
        .top_k(1)
        .top_p(1.0)
        .seed(seed);

    let started = Instant::now();
    let (mut stream, cache_stats) = session
        .stream_with_prefix_reuse(prompt_tokens, options)
        .context("failed to start workload stream")?;
    let mut first_token_ms = None;

    while let Some(chunk) = stream
        .next_chunk()
        .context("failed to decode workload chunk")?
    {
        if first_token_ms.is_none() && !chunk.is_empty() {
            first_token_ms = Some(started.elapsed().as_secs_f64() * 1_000.0);
        }
    }

    let output = stream
        .output()
        .cloned()
        .context("workload stream ended without output")?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let generated_tokens = output.usage.completion_tokens;
    let output_tok_per_sec = if elapsed_ms > 0.0 {
        generated_tokens as f64 / (elapsed_ms / 1_000.0)
    } else {
        0.0
    };

    Ok(IterationMetrics {
        request_ms: elapsed_ms,
        first_token_ms: first_token_ms.unwrap_or(elapsed_ms),
        output_tok_per_sec,
        generated_tokens,
        cache_stats,
        finish_reason: output.finish_reason,
    })
}

fn cache_hit_ratio(stats: PromptCacheStats) -> f64 {
    if stats.prompt_tokens == 0 {
        0.0
    } else {
        stats.cached_tokens as f64 / stats.prompt_tokens as f64
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn finish_reason_name(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit_ratio_uses_prompt_token_count() {
        assert!(
            (cache_hit_ratio(PromptCacheStats {
                cached_tokens: 6,
                prompt_tokens: 10,
                prompt_tokens_evaluated: 4,
            }) - 0.6)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn test_median_even_length() {
        let mut values = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&mut values), 2.5);
    }
}
