use ax_engine_core::chat::render_user_turn;
use ax_engine_core::metrics::counters::OpTimer;
use ax_engine_core::metrics::{LatencyHistogram, current_rss_bytes};
use ax_engine_core::model::{
    DecodeControl, DecodeIntent, DecodeRunConfig, WeightStore, run_decode,
};
use ax_engine_core::sampling::{LogitBias, Sampler, SamplingConfig};
use ax_engine_sdk::{
    LoadOptions as SdkLoadOptions, Model as SdkModel, SessionOptions as SdkSessionOptions,
};
use std::io::{BufRead, Write};

use crate::args::CliArgs;
use crate::stream::{StreamAction, StreamPrinter};

fn resolved_generation_budget(n_predict: i32, remaining_decode_capacity: usize) -> usize {
    if n_predict < 0 {
        remaining_decode_capacity
    } else {
        (n_predict as usize).min(remaining_decode_capacity)
    }
}

/// Run interactive multi-turn REPL mode.
pub fn run(
    args: &CliArgs,
    backend: Box<dyn ax_engine_core::backend::Backend>,
) -> anyhow::Result<()> {
    let (mapped, config, tokenizer, model) =
        crate::load_model(&args.model, args.ctx_size, args.verbose, backend)?;
    crate::validate_sampling_token_ids(args, config.vocab_size as usize)?;
    let weights = WeightStore::new(&mapped);

    let sampling_config = SamplingConfig {
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
    };

    eprintln!("Interactive mode. Type your prompt and press Enter. Ctrl-D to quit.");
    eprintln!();

    let context_length = config.context_length as usize;

    let stdin = std::io::stdin();
    let mut all_tokens: Vec<u32> = Vec::new();
    let mut position = 0usize;
    let mut kv = model.create_model_kv_for_weights(&weights);
    let mut sampler = Sampler::new(sampling_config);
    let mut logits = vec![0.0f32; config.vocab_size as usize];
    let mut latency = LatencyHistogram::new();
    let stdout = std::io::stdout();
    let mut stream =
        StreamPrinter::with_stops(stdout.lock(), args.stop.clone(), args.stop_token_id.clone());

    loop {
        // Prompt
        eprint!("> ");
        std::io::stderr().flush()?;

        let mut line = String::new();
        let bytes_read = stdin.lock().read_line(&mut line)?;
        if bytes_read == 0 {
            // EOF (Ctrl-D)
            eprintln!();
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Handle special commands
        if line == "/reset" || line == "/clear" {
            kv.clear();
            all_tokens.clear();
            position = 0;
            latency.clear();
            eprintln!("[Context cleared]");
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }

        // Apply chat template if requested, then tokenize
        let add_bos = all_tokens.is_empty();
        let input_text = if args.chat {
            let arch = model.arch_name();
            render_user_turn(line, arch, add_bos)
        } else {
            line.to_string()
        };
        let input_tokens = tokenizer.encode(&input_text, add_bos);

        let remaining_context = context_length.saturating_sub(position);

        // Check context length before prefill
        if input_tokens.len() > remaining_context {
            eprintln!(
                "[Context full: position={}, input={} tokens, limit={}. Use /reset to start over.]",
                position,
                input_tokens.len(),
                context_length
            );
            continue;
        }

        // Prefill user tokens
        let prefill_plan = model.prefill_plan_summary(&weights, &kv, input_tokens.len())?;
        let prefill_timer = OpTimer::start();
        model.forward_batch(&input_tokens, &mut kv, &weights, &mut logits)?;
        all_tokens.extend_from_slice(&input_tokens);
        position += input_tokens.len();
        let prefill_time = prefill_timer.elapsed();
        let remaining_decode_capacity = context_length.saturating_sub(position);
        if remaining_decode_capacity == 0 {
            println!();
            eprintln!(
                "[Context full: no room to decode. position={}, limit={}. Use /reset to start over.]",
                position, context_length
            );
            continue;
        }

        // Generate response (limit to remaining context capacity)
        let max_tokens = resolved_generation_budget(args.n_predict, remaining_decode_capacity);
        if max_tokens == 0 {
            continue;
        }
        let first_token_info = if args.top_logprobs > 0 {
            Some(sampler.sample_with_logprobs(&mut logits, &all_tokens, args.top_logprobs))
        } else {
            None
        };
        let next_token = first_token_info
            .as_ref()
            .map(|info| info.token)
            .unwrap_or_else(|| sampler.sample(&mut logits, &all_tokens));
        let mut sampled_infos = Vec::new();
        let outcome = run_decode(
            &model,
            &weights,
            &tokenizer,
            &mut kv,
            &mut sampler,
            &mut all_tokens,
            next_token,
            first_token_info,
            position,
            max_tokens,
            DecodeRunConfig {
                intent: DecodeIntent::Throughput,
                allow_pipelined: true,
                top_logprobs: args.top_logprobs,
                collect_metal_perf: crate::env_flag_enabled("AX_METAL_DECODE_PERF_GATES"),
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
        stream.flush()?;
        let gen_count = outcome.generated_tokens as u32;
        for sample in outcome.latencies {
            latency.record(sample);
        }
        let decode_time = outcome.decode_duration;
        position += gen_count as usize;

        println!(); // newline after generation

        if args.top_logprobs > 0 {
            crate::print_top_logprobs(&tokenizer, &sampled_infos)?;
        }

        if args.verbose {
            let tok_per_sec = if decode_time.as_secs_f64() > 0.0 {
                gen_count as f64 / decode_time.as_secs_f64()
            } else {
                0.0
            };
            eprint!(
                "[prefill: {} tok, {:.2}s | decode: {} tok, {:.2}s ({:.1} tok/s)",
                input_tokens.len(),
                prefill_time.as_secs_f64(),
                gen_count,
                decode_time.as_secs_f64(),
                tok_per_sec,
            );
            eprint!(" | prefill {prefill_plan}");
            if let Some(route) = crate::prefill_route_summary(&prefill_plan) {
                eprint!(" | route {route}");
            }
            eprint!(
                " | mode {} ({})",
                outcome.selection.mode, outcome.selection.intent
            );
            eprint!(" | plan {}", outcome.plan_summary);
            if let Some(perf) = outcome.metal_perf {
                eprint!(
                    " | metal cb {:.2}/{} barrier {:.2}/{}",
                    perf.avg_command_buffers_per_token,
                    perf.max_command_buffers_per_token,
                    perf.avg_buffer_barriers_per_token,
                    perf.max_buffer_barriers_per_token,
                );
            }
            if let (Some(p50), Some(p95)) = (latency.p50(), latency.p95()) {
                eprint!(
                    " | P50 {:.1}ms P95 {:.1}ms",
                    p50.as_secs_f64() * 1000.0,
                    p95.as_secs_f64() * 1000.0,
                );
            }
            eprintln!(
                " | RSS {:.1}MB]",
                current_rss_bytes() as f64 / 1024.0 / 1024.0,
            );
            if let Some(reason) = &outcome.selection.fallback_reason {
                eprintln!("[decode fallback: {reason}]");
            }
        }
    }

    Ok(())
}

pub fn run_routed(args: &CliArgs) -> anyhow::Result<()> {
    crate::validate_routed_cli_args(args)?;

    let model = SdkModel::load(
        &args.model,
        SdkLoadOptions {
            context_length: (args.ctx_size > 0).then_some(args.ctx_size),
            ..SdkLoadOptions::default()
        },
    )?;
    let session = model.session(SdkSessionOptions {
        context_length: (args.ctx_size > 0).then_some(args.ctx_size as usize),
        seed: crate::seed_or_none(args.seed),
    })?;

    eprintln!("Interactive mode. Type your prompt and press Enter. Ctrl-D to quit.");
    eprintln!();

    let user_max_tokens = if args.n_predict < 0 {
        session.context_length()
    } else {
        args.n_predict as usize
    };

    let stdin = std::io::stdin();
    let mut latency = LatencyHistogram::new();
    let mut total_turns = 0usize;

    loop {
        eprint!("> ");
        std::io::stderr().flush()?;

        let mut line = String::new();
        let bytes_read = stdin.lock().read_line(&mut line)?;
        if bytes_read == 0 {
            eprintln!();
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line == "/reset" || line == "/clear" {
            session.reset()?;
            latency.clear();
            total_turns = 0;
            eprintln!("[Context cleared]");
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }

        let add_bos = session.position()? == 0;
        let input_text = if args.chat {
            render_user_turn(line, model.architecture(), add_bos)
        } else {
            line.to_string()
        };
        let input_tokens = model.tokenize(&input_text, add_bos);
        let position = session.position()?;
        let remaining_context = session.context_length().saturating_sub(position);

        if input_tokens.len() > remaining_context {
            eprintln!(
                "[Context full: position={}, input={} tokens, limit={}. Use /reset to start over.]",
                position,
                input_tokens.len(),
                session.context_length()
            );
            continue;
        }

        let decode_limit = user_max_tokens.min(
            session
                .context_length()
                .saturating_sub(position + input_tokens.len()),
        );
        if decode_limit == 0 {
            eprintln!(
                "[Context full: no room to decode. position={}, limit={}. Use /reset to start over.]",
                position + input_tokens.len(),
                session.context_length()
            );
            continue;
        }

        let timer = OpTimer::start();
        let mut stream = session.stream(
            &input_text,
            crate::sdk_generation_options(args, decode_limit),
        )?;
        while let Some(chunk) = stream.next_chunk()? {
            print!("{chunk}");
            std::io::stdout().flush()?;
        }
        let output = stream.into_output()?;
        let elapsed = timer.elapsed();
        latency.record(elapsed);
        total_turns += 1;

        println!();

        if args.verbose {
            let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                output.usage.completion_tokens as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            eprint!(
                "[backend llama.cpp | prompt {} tok | decode {} tok | {:.2}s ({:.1} tok/s)",
                output.usage.prompt_tokens,
                output.usage.completion_tokens,
                elapsed.as_secs_f64(),
                tok_per_sec,
            );
            if let (Some(p50), Some(p95)) = (latency.p50(), latency.p95()) {
                eprint!(
                    " | P50 {:.1}ms P95 {:.1}ms",
                    p50.as_secs_f64() * 1000.0,
                    p95.as_secs_f64() * 1000.0,
                );
            }
            eprintln!(
                " | turns {} | RSS {:.1}MB]",
                total_turns,
                current_rss_bytes() as f64 / 1024.0 / 1024.0,
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn can_fit_input(position: usize, input_tokens: usize, context_length: usize) -> bool {
        input_tokens <= context_length.saturating_sub(position)
    }

    #[test]
    fn test_context_check_allows_exact_fit() {
        assert!(can_fit_input(4, 6, 10));
        assert!(!can_fit_input(4, 7, 10));
        assert!(can_fit_input(10, 0, 10));
        assert!(!can_fit_input(10, 1, 10));
    }

    #[test]
    fn test_resolved_generation_budget_clamps_positive_request() {
        assert_eq!(resolved_generation_budget(32, 8), 8);
    }

    #[test]
    fn test_resolved_generation_budget_allows_zero_or_unlimited() {
        assert_eq!(resolved_generation_budget(0, 8), 0);
        assert_eq!(resolved_generation_budget(-1, 8), 8);
    }
}
