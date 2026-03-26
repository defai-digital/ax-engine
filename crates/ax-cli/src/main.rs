// AX Engine CLI — llama.cpp-compatible interface
//
// Binary name: ax-llama
// Can be symlinked as `llama` for drop-in use.

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use clap::Parser;
use serde_json::json;

use ax_core::chat::render_user_prompt;
use ax_core::gguf::MappedModel;
use ax_core::memory::MemoryBudget;
use ax_core::metrics::counters::OpTimer;
use ax_core::metrics::{InferenceMetrics, LatencyHistogram, current_rss_bytes};
use ax_core::model::{
    DecodeControl, DecodeIntent, DecodeRunConfig, LlamaModel, ModelConfig, WeightStore, run_decode,
};
use ax_core::sampling::{LogitBias, SampledTokenInfo, Sampler, SamplingConfig};
use ax_core::speculative::{SpecStep, SpeculativeDecoder};
use ax_core::tokenizer::Tokenizer;

mod args;
mod interactive;
mod stream;

use stream::{StreamAction, StreamPrinter};

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on"))
}

fn main() -> anyhow::Result<()> {
    // Check for known unsupported llama.cpp flags before clap parsing
    args::check_unsupported_flags();

    let args = args::CliArgs::parse();

    validate_mode_combinations(&args)?;

    if args.speculative_draft.is_some() && !args.experimental {
        anyhow::bail!(
            "--speculative-draft is experimental. Re-run with --experimental to enable it."
        );
    }

    // Initialize tracing
    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("ax_core=debug,ax_cli=debug")
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter("ax_core=warn,ax_cli=info")
            .init();
    }

    tracing::info!("AX Engine v{}", env!("CARGO_PKG_VERSION"));

    // Bootstrap Metal before any scheduler/model-loading side effects so
    // device availability is probed in a clean process state.
    let backend =
        ax_core::backend::create_backend(ax_core::backend::resolve_backend_config_from_env())?;
    ax_core::scheduler::init_global_threadpool();

    if args.interactive {
        interactive::run(&args, backend)?;
    } else {
        run_single(&args, backend)?;
    }

    Ok(())
}

fn validate_mode_combinations(args: &args::CliArgs) -> anyhow::Result<()> {
    if !args.interactive {
        return Ok(());
    }

    if args.prompt.is_some() {
        anyhow::bail!("--prompt is not used in --interactive mode");
    }
    if args.reuse_bench || args.reuse_bench_json {
        anyhow::bail!(
            "--reuse-bench and --reuse-bench-json are not supported in --interactive mode"
        );
    }
    if args.speculative_draft.is_some() {
        anyhow::bail!("--speculative-draft is not supported in --interactive mode");
    }

    Ok(())
}

/// Load a GGUF model and return all components needed for inference.
pub(crate) fn load_model(
    model_path: &str,
    ctx_size: u32,
    verbose: bool,
    backend: Box<dyn ax_core::backend::Backend>,
) -> anyhow::Result<(MappedModel, ModelConfig, Tokenizer, LlamaModel)> {
    let rss_before = current_rss_bytes();
    let timer = OpTimer::start();
    let mapped = MappedModel::open(Path::new(model_path))?;
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
    backend.configure_for_model(&profile_model_name, &profile_quant, profile_architecture)?;

    let mut config = ModelConfig::from_gguf(&mapped.header)?;
    // Apply user-specified context size override
    if ctx_size > 0 && ctx_size != config.context_length {
        tracing::info!(
            "Overriding context length: {} -> {}",
            config.context_length,
            ctx_size
        );
        config.context_length = ctx_size;
    }
    let tokenizer = Tokenizer::from_gguf(&mapped.header)?;
    let model = LlamaModel::with_backend(config.clone(), backend);
    let kv_plan = model.kv_plan();
    let kv_memory = kv_plan.memory_estimate();
    let kv_capacity = kv_plan.capacity_policy();
    let model_bytes_u64 = mapped.total_tensor_bytes();
    MemoryBudget::check_combined(model_bytes_u64, kv_memory.initial_bytes as u64)?;
    if let Ok(max_summary) = MemoryBudget::summary(model_bytes_u64, kv_memory.max_bytes as u64)
        && max_summary.required_bytes > max_summary.allowed_bytes
    {
        eprintln!(
            "Warning: model + max planned KV footprint ({:.1}GB) exceeds budget ({:.1}GB). Initial allocation still fits, but long-context growth may pressure memory.",
            max_summary.required_bytes as f64 / 1e9,
            max_summary.allowed_bytes as f64 / 1e9,
        );
    }

    let load_time = timer.elapsed();
    let rss_after = current_rss_bytes();
    let model_mb = mapped.total_tensor_bytes() as f64 / 1024.0 / 1024.0;

    eprintln!(
        "Model loaded: {} layers, {} vocab, {:.0}MB, {:.2}s",
        config.n_layers,
        config.vocab_size,
        model_mb,
        load_time.as_secs_f64(),
    );
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
    if let Some(note) = mapped.support_note() {
        eprintln!("Support: {note}");
    }

    // Log architecture features
    let head_dim_source = if config.head_dim != config.embedding_dim / config.n_heads {
        "explicit"
    } else {
        "derived"
    };
    eprintln!(
        "Architecture: {} | QKV bias: {} | Head dim: {} ({}) | Sliding window: {} | Gate: {:?} | Tie embeddings: {}",
        model.arch_name(),
        if config.has_qkv_bias { "yes" } else { "no" },
        config.head_dim,
        head_dim_source,
        config
            .sliding_window_size
            .map_or("none".to_string(), |s| format!("{s}")),
        config.gate_activation,
        if config.tie_word_embeddings {
            "yes"
        } else {
            "no"
        },
    );

    if verbose {
        eprintln!(
            "RSS: {:.1}MB → {:.1}MB (load Δ{:.1}MB)",
            rss_before as f64 / 1024.0 / 1024.0,
            rss_after as f64 / 1024.0 / 1024.0,
            (rss_after.saturating_sub(rss_before)) as f64 / 1024.0 / 1024.0,
        );
    }

    Ok((mapped, config, tokenizer, model))
}

/// Build a SamplingConfig from CLI args.
fn sampling_config(args: &args::CliArgs) -> SamplingConfig {
    SamplingConfig {
        logit_bias: args
            .logit_bias
            .iter()
            .map(|bias| LogitBias {
                token: bias.token,
                bias: bias.bias,
            })
            .collect(),
        allowed_token_ids: args.allow_token_id.clone(),
        banned_token_ids: args.ban_token_id.clone(),
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        min_keep: args.min_keep.max(1),
        repeat_penalty: args.repeat_penalty,
        frequency_penalty: args.frequency_penalty,
        presence_penalty: args.presence_penalty,
        repeat_last_n: args.repeat_last_n,
        seed: if args.seed < 0 {
            u64::MAX
        } else {
            args.seed as u64
        },
    }
}

fn validate_sampling_token_ids(args: &args::CliArgs, vocab_size: usize) -> anyhow::Result<()> {
    for &token in &args.allow_token_id {
        if token as usize >= vocab_size {
            anyhow::bail!("--allow-token-id {token} is outside vocab range 0..{vocab_size}");
        }
    }

    for &token in &args.ban_token_id {
        if token as usize >= vocab_size {
            anyhow::bail!("--ban-token-id {token} is outside vocab range 0..{vocab_size}");
        }
    }

    for bias in &args.logit_bias {
        if bias.token as usize >= vocab_size {
            anyhow::bail!(
                "--logit-bias {}={} is outside vocab range 0..{vocab_size}",
                bias.token,
                bias.bias
            );
        }
    }

    if !args.allow_token_id.is_empty() {
        let allowed_after_ban = args
            .allow_token_id
            .iter()
            .filter(|token| !args.ban_token_id.contains(token))
            .count();
        if allowed_after_ban == 0 {
            anyhow::bail!("--allow-token-id and --ban-token-id leave no valid tokens to sample");
        }
    } else {
        let banned_unique = args
            .ban_token_id
            .iter()
            .copied()
            .filter(|&token| (token as usize) < vocab_size)
            .collect::<HashSet<_>>()
            .len();
        if banned_unique >= vocab_size {
            anyhow::bail!("--ban-token-id excludes the entire vocabulary");
        }
    }

    Ok(())
}

pub(crate) fn print_top_logprobs(
    tokenizer: &Tokenizer,
    infos: &[SampledTokenInfo],
) -> anyhow::Result<()> {
    let out = infos
        .iter()
        .map(|info| {
            json!({
                "token": info.token,
                "text": tokenizer.render_token(info.token),
                "logprob": info.logprob,
                "top_logprobs": info.top_logprobs.iter().map(|candidate| {
                    json!({
                        "token": candidate.token,
                        "text": tokenizer.render_token(candidate.token),
                        "logprob": candidate.logprob,
                    })
                }).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    eprintln!("--- Top Logprobs ---");
    eprintln!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

/// Run single-prompt generation.
fn run_single(
    args: &args::CliArgs,
    backend: Box<dyn ax_core::backend::Backend>,
) -> anyhow::Result<()> {
    let prompt = args.prompt.as_deref().unwrap_or("");
    if prompt.is_empty() {
        anyhow::bail!("No prompt provided. Use -p/--prompt or --interactive mode.");
    }

    let (mapped, _config, tokenizer, model) =
        load_model(&args.model, args.ctx_size, args.verbose, backend)?;
    validate_sampling_token_ids(args, model.config.vocab_size as usize)?;
    let weights = WeightStore::new(&mapped);
    let mut kv = model.create_model_kv_for_weights(&weights);
    let mut sampler = Sampler::new(sampling_config(args));
    let mut metrics = InferenceMetrics::new();
    let mut latency = LatencyHistogram::new();

    // Snapshot RSS after model load
    metrics.update_peak_rss();

    // Apply chat template if requested
    let effective_prompt = if args.chat {
        let arch = model.arch_name();
        let wrapped = render_user_prompt(prompt, arch);
        if args.verbose {
            eprintln!("Chat template applied ({arch}): {:?}", wrapped);
        }
        wrapped
    } else {
        prompt.to_string()
    };

    // Tokenize prompt
    let prompt_tokens = tokenizer.encode(&effective_prompt, true);
    let n_prompt = prompt_tokens.len();

    if args.verbose {
        eprintln!("Prompt tokens: {n_prompt}");
        eprintln!("Token IDs: {:?}", prompt_tokens);
    }

    // Determine max tokens to generate
    let max_tokens = if args.n_predict < 0 {
        (model.config.context_length as usize).saturating_sub(n_prompt)
    } else {
        args.n_predict as usize
    };

    if max_tokens == 0 {
        eprintln!(
            "Warning: prompt ({n_prompt} tokens) fills or exceeds context window ({} tokens). No room to generate.",
            model.config.context_length
        );
        return Ok(());
    }

    if args.reuse_bench || args.reuse_bench_json {
        return run_reuse_bench(&model, &weights, &prompt_tokens, args.reuse_bench_json);
    }

    // --- Speculative decoding path (if --speculative-draft provided) ---
    if let Some(ref draft_path) = args.speculative_draft {
        if args.top_logprobs > 0 {
            anyhow::bail!("--top-logprobs is not supported with speculative decoding");
        }
        if !args.stop.is_empty() || !args.stop_token_id.is_empty() {
            anyhow::bail!("--stop and --stop-token-id are not supported with speculative decoding");
        }
        if !args.allow_token_id.is_empty() {
            anyhow::bail!("--allow-token-id is not supported with speculative decoding");
        }
        if !args.ban_token_id.is_empty() {
            anyhow::bail!("--ban-token-id is not supported with speculative decoding");
        }
        if !args.logit_bias.is_empty() {
            anyhow::bail!("--logit-bias is not supported with speculative decoding");
        }
        return run_speculative(
            args,
            draft_path,
            &model,
            &weights,
            &tokenizer,
            &prompt_tokens,
        );
    }

    // --- Prefill: process prompt tokens, keep last logits for sampling ---
    let prefill_plan = model.prefill_plan_summary(&weights, &kv, prompt_tokens.len())?;
    let prefill_timer = OpTimer::start();
    let mut logits = vec![0.0f32; model.config.vocab_size as usize];
    model.forward_batch(&prompt_tokens, &mut kv, &weights, &mut logits)?;
    metrics.prefill_tokens = n_prompt as u64;
    metrics.prefill_duration = prefill_timer.elapsed();

    // Print prompt (in chat mode, show user's original prompt instead of template markup)
    if args.chat {
        print!("{prompt}\n\n");
    } else {
        let prompt_text = tokenizer.decode(&prompt_tokens);
        print!("{prompt_text}");
    }
    std::io::stdout().flush()?;

    // --- Decode: generate tokens ---
    // Pre-allocate token history for repetition penalty (avoids O(n²) reallocation)
    let mut all_tokens: Vec<u32> = Vec::with_capacity(n_prompt + max_tokens);
    all_tokens.extend_from_slice(&prompt_tokens);

    let debug_logits = env_flag_enabled("AX_DEBUG_LOGITS");
    let vocab_size = model.config.vocab_size as usize;

    // Debug helper: print top-5 logits and their token IDs
    let dump_top_logits = |logits: &[f32], step: &str, vocab_size: usize, tokenizer: &Tokenizer| {
        if !debug_logits {
            return;
        }
        let mut indexed: Vec<(usize, f32)> = logits[..vocab_size]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let max_v = indexed[0].1;
        let min_v = indexed.last().map(|x| x.1).unwrap_or(0.0);
        let has_nan = logits[..vocab_size].iter().any(|v| v.is_nan());
        let has_inf = logits[..vocab_size].iter().any(|v| v.is_infinite());
        eprint!("[{step}] max={max_v:.2} min={min_v:.2} nan={has_nan} inf={has_inf} top5: ");
        for (id, val) in indexed.iter().take(5) {
            let tok = tokenizer
                .render_token(*id as u32)
                .unwrap_or_else(|| format!("<{id}>"));
            let tok_display = tok.replace('\n', "\\n");
            eprint!("{}({val:.2}) ", tok_display);
        }
        eprintln!();
    };

    dump_top_logits(&logits, "prefill", vocab_size, &tokenizer);

    // Sample first token from last prefill logits
    let first_decode_info = if args.top_logprobs > 0 {
        Some(sampler.sample_with_logprobs(&mut logits, &all_tokens, args.top_logprobs))
    } else {
        None
    };
    let first_decode_token = first_decode_info
        .as_ref()
        .map(|info| info.token)
        .unwrap_or_else(|| sampler.sample(&mut logits, &all_tokens));
    let mut sampled_infos = Vec::new();

    let mut decode_selection = None;
    let mut decode_plan = None;
    let stdout = std::io::stdout();
    let mut stream =
        StreamPrinter::with_stops(stdout.lock(), args.stop.clone(), args.stop_token_id.clone());
    // Debug path stays local because it dumps per-step logits.
    if debug_logits {
        let decode_timer = OpTimer::start();
        let mut next_token = first_decode_token;
        let mut next_token_info = first_decode_info;
        let mut position = n_prompt;
        let mut n_generated = 0u64;

        for _ in 0..max_tokens {
            // Check for EOS
            if tokenizer.is_eos(next_token) {
                break;
            }

            // Print token text
            if matches!(
                stream.push_token(&tokenizer, next_token)?,
                StreamAction::Stop
            ) {
                break;
            }
            if let Some(info) = next_token_info.as_ref() {
                sampled_infos.push(info.clone());
            }

            all_tokens.push(next_token);
            n_generated += 1;

            // Forward pass for next token (timed for latency histogram)
            let tok_timer = OpTimer::start();
            logits.fill(0.0);
            model.forward_single(next_token, position, &mut kv, &weights, &mut logits)?;
            latency.record(tok_timer.elapsed());
            position += 1;

            dump_top_logits(
                &logits,
                &format!("dec{n_generated}"),
                vocab_size,
                &tokenizer,
            );

            // Sample next token
            if args.top_logprobs > 0 {
                let info =
                    sampler.sample_with_logprobs(&mut logits, &all_tokens, args.top_logprobs);
                next_token = info.token;
                next_token_info = Some(info);
            } else {
                next_token = sampler.sample(&mut logits, &all_tokens);
                next_token_info = None;
            }
        }
        metrics.decode_tokens = n_generated;
        metrics.decode_duration = decode_timer.elapsed();
    } else {
        let outcome = run_decode(
            &model,
            &weights,
            &tokenizer,
            &mut kv,
            &mut sampler,
            &mut all_tokens,
            first_decode_token,
            first_decode_info,
            n_prompt,
            max_tokens,
            DecodeRunConfig {
                intent: DecodeIntent::Throughput,
                allow_pipelined: true,
                top_logprobs: args.top_logprobs,
            },
            |tok, info| {
                let action = stream
                    .push_token(&tokenizer, tok)
                    .map_err(anyhow::Error::from)?;
                if let Some(info) = info
                    && action == StreamAction::Continue
                {
                    sampled_infos.push(info.clone());
                }
                Ok(match action {
                    StreamAction::Continue => DecodeControl::Continue,
                    StreamAction::Stop => DecodeControl::Stop,
                })
            },
        )?;
        for sample in outcome.latencies {
            latency.record(sample);
        }
        metrics.decode_tokens = outcome.generated_tokens;
        metrics.decode_duration = outcome.decode_duration;
        decode_plan = Some(outcome.plan_summary.clone());
        decode_selection = Some(outcome.selection);
    }
    stream.flush()?;
    metrics.update_peak_rss();

    println!(); // final newline

    if args.top_logprobs > 0 {
        print_top_logprobs(&tokenizer, &sampled_infos)?;
    }

    // Print metrics
    if args.verbose {
        eprintln!();
        eprintln!("--- Metrics ---");
        eprintln!(
            "Prefill: {} tokens, {:.2}s ({:.1} tok/s)",
            metrics.prefill_tokens,
            metrics.prefill_duration.as_secs_f64(),
            metrics.prefill_tok_per_sec(),
        );
        eprintln!("PrefillPlan: {prefill_plan}");
        eprintln!(
            "Decode:  {} tokens, {:.2}s ({:.1} tok/s)",
            metrics.decode_tokens,
            metrics.decode_duration.as_secs_f64(),
            metrics.decode_tok_per_sec(),
        );
        if let Some(selection) = &decode_selection {
            eprintln!("Mode:    {} ({})", selection.mode, selection.intent);
            if let Some(plan) = &decode_plan {
                eprintln!("Plan:    {plan}");
            }
            if let Some(reason) = &selection.fallback_reason {
                eprintln!("Fallback:{reason}");
            }
        } else if debug_logits {
            eprintln!("Mode:    sequential (debug)");
        }
        if let (Some(p50), Some(p95), Some(p99)) = (latency.p50(), latency.p95(), latency.p99()) {
            eprintln!(
                "Latency: P50 {:.2}ms, P95 {:.2}ms, P99 {:.2}ms",
                p50.as_secs_f64() * 1000.0,
                p95.as_secs_f64() * 1000.0,
                p99.as_secs_f64() * 1000.0,
            );
        }
        eprintln!(
            "Memory:  peak RSS {:.1}MB",
            metrics.peak_rss_bytes as f64 / 1024.0 / 1024.0,
        );
    }

    Ok(())
}

/// Run generation using speculative decoding (CPU draft + GPU target verification).
///
/// Uses `SpeculativeDecoder` with a small draft model to propose K tokens per step.
/// The target model verifies all K+1 tokens (Leviathan et al. protocol).
#[allow(clippy::too_many_arguments)]
fn run_speculative(
    args: &args::CliArgs,
    draft_path: &str,
    model: &LlamaModel,
    weights: &WeightStore,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
) -> anyhow::Result<()> {
    let prompt = args.prompt.as_deref().unwrap_or("");
    let n_prompt = prompt_tokens.len();
    let max_tokens = if args.n_predict < 0 {
        (model.config.context_length as usize).saturating_sub(n_prompt)
    } else {
        args.n_predict as usize
    };

    eprintln!(
        "Speculative decoding: draft={draft_path}, k={}, max_tokens={max_tokens}",
        args.speculative_k
    );
    eprintln!(
        "Warning: speculative decoding is experimental and excluded from performance claims. Verification uses batched all-logits when the active architecture/backend supports it, with serial fallback retained for correctness."
    );

    let mut spec = SpeculativeDecoder::load(draft_path, args.speculative_k)?;
    let mut kv = model.create_model_kv_for_weights(weights);
    let mut sampler = Sampler::new(sampling_config(args));
    let mut metrics = InferenceMetrics::new();

    // Print prompt
    if args.chat {
        print!("{prompt}\n\n");
    } else {
        let prompt_text = tokenizer.decode(prompt_tokens);
        print!("{prompt_text}");
    }
    std::io::stdout().flush()?;

    // Prefill
    let prefill_timer = OpTimer::start();
    let mut logits = vec![0.0f32; model.config.vocab_size as usize];
    model.forward_batch(prompt_tokens, &mut kv, weights, &mut logits)?;
    metrics.prefill_tokens = n_prompt as u64;
    metrics.prefill_duration = prefill_timer.elapsed();

    // Sample first decode token from prefill logits
    let mut kv_history: Vec<u32> = prompt_tokens.to_vec();
    let mut last_token = sampler.sample(&mut logits, prompt_tokens);
    let mut position = n_prompt;
    let mut n_generated = 0u64;
    let mut total_accepted = 0u64;
    let mut total_steps = 0u64;
    let mut total_draft_duration = std::time::Duration::ZERO;
    let mut total_verify_duration = std::time::Duration::ZERO;
    let mut total_accept_duration = std::time::Duration::ZERO;
    let mut total_verified_positions = 0u64;
    let mut total_drafted_tokens = 0u64;

    let decode_timer = OpTimer::start();
    let stdout = std::io::stdout();
    let mut stream = StreamPrinter::new(stdout.lock());

    loop {
        if n_generated >= max_tokens as u64 {
            break;
        }
        if tokenizer.is_eos(last_token) {
            break;
        }

        // Print the last accepted token.
        let _ = stream.push_token(tokenizer, last_token)?;
        n_generated += 1;
        if n_generated >= max_tokens as u64 {
            break;
        }

        let remaining = max_tokens as u64 - n_generated;
        let step_k = spec.k().min(remaining as usize).max(1);

        // Run the speculative (or partial-fallback) step.
        // `position` must equal `target_kv.seq_len()` here — generate_step
        // feeds `last_token` as verify_tokens[0] at this position.
        let step = if step_k < spec.k() {
            // For the last partial step, just do one regular forward_single
            logits.fill(0.0);
            model.forward_single(last_token, position, &mut kv, weights, &mut logits)?;
            let tok = sampler.sample(&mut logits, &kv_history);
            SpecStep {
                tokens: vec![tok],
                n_accepted: 0,
                metrics: Default::default(),
            }
        } else {
            spec.generate_step(
                &kv_history,
                last_token,
                position,
                model,
                &mut kv,
                weights,
                &mut sampler,
            )?
        };

        total_accepted += step.n_accepted as u64;
        total_steps += 1;
        total_draft_duration += step.metrics.draft_duration;
        total_verify_duration += step.metrics.verify_duration;
        total_accept_duration += step.metrics.accept_duration;
        total_verified_positions += step.metrics.verified_positions as u64;
        total_drafted_tokens += step.metrics.drafted_tokens as u64;

        // last_token was fed into target KV by the step; record it in history.
        kv_history.push(last_token);

        // Emit all tokens from this step (except last which seeds next step)
        let n_emitted = step.tokens.len().saturating_sub(1);
        let mut saw_stop_token = false;
        for &tok in &step.tokens[..n_emitted] {
            if tokenizer.is_eos(tok) {
                saw_stop_token = true;
                break;
            }
            let _ = stream.push_token(tokenizer, tok)?;
            kv_history.push(tok);
            n_generated += 1;
            if n_generated >= max_tokens as u64 {
                break;
            }
        }

        // Last token seeds the next step
        if let Some(&tok) = step.tokens.last() {
            last_token = tok;
        } else {
            break;
        }

        if saw_stop_token || n_generated >= max_tokens as u64 {
            break;
        }

        // position tracks target_kv.seq_len(): last_token (1) + accepted tokens.
        position += step.tokens.len();
    }

    metrics.decode_tokens = n_generated;
    metrics.decode_duration = decode_timer.elapsed();

    stream.flush()?;
    println!();

    if args.verbose {
        eprintln!();
        eprintln!("--- Speculative Decoding Metrics ---");
        eprintln!(
            "Prefill: {} tokens, {:.2}s ({:.1} tok/s)",
            metrics.prefill_tokens,
            metrics.prefill_duration.as_secs_f64(),
            metrics.prefill_tok_per_sec(),
        );
        eprintln!(
            "Decode:  {} tokens, {:.2}s ({:.1} tok/s)",
            metrics.decode_tokens,
            metrics.decode_duration.as_secs_f64(),
            metrics.decode_tok_per_sec(),
        );
        if total_steps > 0 {
            eprintln!(
                "Speculative: {total_steps} steps, avg {:.2} accepted/step (k={})",
                total_accepted as f64 / total_steps as f64,
                spec.k()
            );
            eprintln!(
                "Draft:       {:.2} ms/step, {:.2} ms/drafted token",
                total_draft_duration.as_secs_f64() * 1000.0 / total_steps as f64,
                if total_drafted_tokens > 0 {
                    total_draft_duration.as_secs_f64() * 1000.0 / total_drafted_tokens as f64
                } else {
                    0.0
                }
            );
            eprintln!(
                "Verify:      {:.2} ms/step, {:.2} ms/position",
                total_verify_duration.as_secs_f64() * 1000.0 / total_steps as f64,
                if total_verified_positions > 0 {
                    total_verify_duration.as_secs_f64() * 1000.0 / total_verified_positions as f64
                } else {
                    0.0
                }
            );
            eprintln!(
                "Accept:      {:.2} ms/step",
                total_accept_duration.as_secs_f64() * 1000.0 / total_steps as f64
            );
        }
    }

    Ok(())
}

/// Run a two-pass prefill repeatability benchmark.
///
/// v2.0: Paged KV prefix cache is deferred to v2.1. Both passes run full prefill
/// from scratch. This measures prefill repeatability and warm-cache GPU performance.
fn run_reuse_bench(
    model: &LlamaModel,
    weights: &WeightStore,
    prompt_tokens: &[u32],
    json_output: bool,
) -> anyhow::Result<()> {
    if prompt_tokens.is_empty() {
        anyhow::bail!("prompt must tokenize to at least one token");
    }

    let vocab_size = model.config.vocab_size as usize;
    let mut run_results: Vec<(f64, f64)> = Vec::new(); // (elapsed_secs, tok_per_sec)

    for _ in 0..2u64 {
        let mut kv = model.create_model_kv_for_weights(weights);
        let mut logits = vec![0.0f32; vocab_size];

        let timer = OpTimer::start();
        model.forward_batch(prompt_tokens, &mut kv, weights, &mut logits)?;
        let elapsed = timer.elapsed().as_secs_f64();

        let tok_s = if elapsed > 0.0 {
            prompt_tokens.len() as f64 / elapsed
        } else {
            0.0
        };
        run_results.push((elapsed, tok_s));
    }

    let (r1_t, r1_s) = run_results[0];
    let (r2_t, r2_s) = run_results[1];
    let speedup = if r1_t > 0.0 { r1_t / r2_t } else { 0.0 };

    if json_output {
        let out = json!({
            "prompt_tokens": prompt_tokens.len(),
            "note": "prefix_cache deferred to v2.1 — both runs are full prefill",
            "run1": {
                "computed_prefill_tokens": prompt_tokens.len(),
                "prefill_seconds": r1_t,
                "prefill_tok_per_sec": r1_s,
            },
            "run2": {
                "computed_prefill_tokens": prompt_tokens.len(),
                "prefill_seconds": r2_t,
                "prefill_tok_per_sec": r2_s,
            },
            "prefill_speedup_run2_over_run1": speedup,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        eprintln!("--- Reuse Bench (2-pass prefill, prefix cache deferred to v2.1) ---");
        eprintln!(
            "Run 1: tokens={} time={:.3}s ({:.1} tok/s)",
            prompt_tokens.len(),
            r1_t,
            r1_s,
        );
        eprintln!(
            "Run 2: tokens={} time={:.3}s ({:.1} tok/s)",
            prompt_tokens.len(),
            r2_t,
            r2_s,
        );
        eprintln!("Prefill speedup (run2/run1): {:.2}x", speedup);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    struct EnvVarRestore {
        key: String,
        previous: Option<std::ffi::OsString>,
    }

    impl Drop for EnvVarRestore {
        fn drop(&mut self) {
            match &self.previous {
                Some(prev) => unsafe {
                    std::env::set_var(&self.key, prev);
                },
                None => unsafe {
                    std::env::remove_var(&self.key);
                },
            }
        }
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("test env lock")
    }

    fn with_env_var<T>(key: &str, value: &str, f: impl FnOnce() -> T) -> T {
        let _guard = env_lock();
        let _restore = EnvVarRestore {
            key: key.to_string(),
            previous: std::env::var_os(key),
        };
        unsafe { std::env::set_var(key, value) };
        f()
    }

    fn make_args() -> args::CliArgs {
        args::CliArgs {
            model: "model.gguf".to_string(),
            prompt: Some("hello".to_string()),
            n_predict: 16,
            ctx_size: 4096,
            threads: 0,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            min_keep: 1,
            top_logprobs: 0,
            stop: Vec::new(),
            stop_token_id: Vec::new(),
            logit_bias: Vec::new(),
            allow_token_id: Vec::new(),
            ban_token_id: Vec::new(),
            seed: -1,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            interactive: false,
            verbose: false,
            experimental: false,
            chat: false,
            reuse_bench: false,
            reuse_bench_json: false,
            speculative_draft: None,
            speculative_k: 4,
        }
    }

    #[test]
    fn test_validate_sampling_token_ids_rejects_full_ban_without_allowlist() {
        let mut args = make_args();
        args.ban_token_id = vec![0, 1, 2];

        let err = validate_sampling_token_ids(&args, 3).unwrap_err();
        assert!(
            err.to_string()
                .contains("--ban-token-id excludes the entire vocabulary")
        );
    }

    #[test]
    fn test_validate_sampling_token_ids_rejects_allowlist_fully_banned() {
        let mut args = make_args();
        args.allow_token_id = vec![1, 2];
        args.ban_token_id = vec![1, 2];

        let err = validate_sampling_token_ids(&args, 8).unwrap_err();
        assert!(
            err.to_string()
                .contains("--allow-token-id and --ban-token-id leave no valid tokens to sample")
        );
    }

    #[test]
    fn test_validate_sampling_token_ids_allows_partial_overlap() {
        let mut args = make_args();
        args.allow_token_id = vec![1, 2];
        args.ban_token_id = vec![2];

        validate_sampling_token_ids(&args, 8).unwrap();
    }

    #[test]
    fn test_validate_mode_combinations_rejects_prompt_in_interactive_mode() {
        let mut args = make_args();
        args.interactive = true;

        let err = validate_mode_combinations(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("--prompt is not used in --interactive mode")
        );
    }

    #[test]
    fn test_validate_mode_combinations_rejects_reuse_bench_in_interactive_mode() {
        let mut args = make_args();
        args.interactive = true;
        args.prompt = None;
        args.reuse_bench = true;

        let err = validate_mode_combinations(&args).unwrap_err();
        assert!(err.to_string().contains("--reuse-bench"));
    }

    #[test]
    fn test_validate_mode_combinations_rejects_speculative_in_interactive_mode() {
        let mut args = make_args();
        args.interactive = true;
        args.prompt = None;
        args.speculative_draft = Some("draft.gguf".to_string());

        let err = validate_mode_combinations(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("--speculative-draft is not supported in --interactive mode")
        );
    }

    #[test]
    fn test_support_note_for_quant_marks_conservative_q5k_prefill() {
        assert!(
            ax_core::gguf::mmap::support_note_for_q5k_layer_presence(true)
                .unwrap()
                .contains("conservative GPU prefill route")
        );
        assert_eq!(
            ax_core::gguf::mmap::support_note_for_q5k_layer_presence(false),
            None
        );
    }

    #[test]
    fn test_env_flag_enabled_supports_truthy_values_and_rejects_invalid() {
        let key = "AX_DEBUG_LOGITS";
        with_env_var(key, "1", || {
            assert!(env_flag_enabled(key));
        });
        with_env_var(key, "true", || {
            assert!(env_flag_enabled(key));
        });
        with_env_var(key, "on", || {
            assert!(env_flag_enabled(key));
        });
        with_env_var(key, "0", || {
            assert!(!env_flag_enabled(key));
        });
        with_env_var(key, "bad", || {
            assert!(!env_flag_enabled(key));
        });
    }
}
