//! Compare a two-token Qwen linear-attention verify graph with two singleton
//! production-decode graphs on the same real checkpoint and cache state.
//!
//! Usage:
//!   linear_mtp_state_oracle_probe <model_dir> <comma-separated-prompt-tokens>

use std::env;
use std::path::Path;
use std::process::ExitCode;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill_with_final_hidden},
    kv_cache::MlxKVCache,
    model::{
        ModelConfig, embed_tokens, forward_all_positions, forward_argmax, layer_forward,
        layer_forward_last_only,
    },
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::ModelWeights,
    weights::load_weights,
};
use mlx_sys::{MlxArray, MlxDtype, argmax, astype, eval, multiply, slice};

fn parse_tokens(raw: &str) -> Result<Vec<u32>, String> {
    let tokens = raw
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u32>()
                .map_err(|error| format!("invalid token {part:?}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.is_empty() {
        return Err("prompt token list must not be empty".to_string());
    }
    Ok(tokens)
}

fn materialized_argmax(logits: &MlxArray, cache: &MlxKVCache) -> u32 {
    let token = argmax(logits, None);
    let cache_refs = cache.collect_eval_refs();
    let mut refs = Vec::with_capacity(1 + cache_refs.len());
    refs.push(&token);
    refs.extend(cache_refs);
    eval(&refs);
    token.data_u32().first().copied().unwrap_or(0)
}

fn max_abs_diff(left: &MlxArray, right: &MlxArray) -> f32 {
    let left = astype(left, MlxDtype::Float32, None);
    let right = astype(right, MlxDtype::Float32, None);
    eval(&[&left, &right]);
    left.data_f32()
        .iter()
        .zip(right.data_f32())
        .fold(0.0_f32, |maximum, (left, right)| {
            maximum.max((left - right).abs())
        })
}

fn model_embeddings(cfg: &ModelConfig, weights: &ModelWeights, tokens: &[u32]) -> MlxArray {
    let hidden = astype(
        &embed_tokens(tokens, &weights.token_embedding, cfg.hidden_size),
        MlxDtype::Bfloat16,
        None,
    );
    if let Some(scale) = cfg.hidden_states_scale {
        multiply(
            &hidden,
            &mlx_sys::ops::cached_scalar(scale, hidden.dtype()),
            None,
        )
    } else {
        hidden
    }
}

fn print_first_layer_hidden_diff(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    base_cache: &MlxKVCache,
    token_offset: usize,
    primary: u32,
    draft: u32,
    skip_ffn: bool,
) {
    let mut singleton_cache = base_cache.clone();
    let singleton_first = if skip_ffn {
        layer_forward_last_only(
            cfg,
            &weights.layers[0],
            &model_embeddings(cfg, weights, &[primary]),
            &mut singleton_cache,
            0,
            token_offset,
            None,
            None,
            true,
        )
    } else {
        layer_forward(
            cfg,
            &weights.layers[0],
            &model_embeddings(cfg, weights, &[primary]),
            &mut singleton_cache,
            0,
            token_offset,
            None,
            None,
        )
    };
    singleton_cache.advance(1);
    let singleton_second = if skip_ffn {
        layer_forward_last_only(
            cfg,
            &weights.layers[0],
            &model_embeddings(cfg, weights, &[draft]),
            &mut singleton_cache,
            0,
            token_offset + 1,
            None,
            None,
            true,
        )
    } else {
        layer_forward(
            cfg,
            &weights.layers[0],
            &model_embeddings(cfg, weights, &[draft]),
            &mut singleton_cache,
            0,
            token_offset + 1,
            None,
            None,
        )
    };
    singleton_cache.advance(1);

    let mut batched_cache = base_cache.clone();
    let batched = if skip_ffn {
        layer_forward_last_only(
            cfg,
            &weights.layers[0],
            &model_embeddings(cfg, weights, &[primary, draft]),
            &mut batched_cache,
            0,
            token_offset,
            None,
            None,
            true,
        )
    } else {
        layer_forward(
            cfg,
            &weights.layers[0],
            &model_embeddings(cfg, weights, &[primary, draft]),
            &mut batched_cache,
            0,
            token_offset,
            None,
            None,
        )
    };
    batched_cache.advance(2);
    let hidden_size = cfg.hidden_size as i32;
    let batched_first = slice(&batched, &[0, 0, 0], &[1, 1, hidden_size], &[1, 1, 1], None);
    let batched_second = slice(&batched, &[0, 1, 0], &[1, 2, hidden_size], &[1, 1, 1], None);
    eval(&[
        &singleton_first,
        &singleton_second,
        &batched_first,
        &batched_second,
    ]);
    println!(
        "layer=0 stage={} hidden_first_max_abs={:.9e} hidden_second_max_abs={:.9e}",
        if skip_ffn { "attention" } else { "full" },
        max_abs_diff(&singleton_first, &batched_first),
        max_abs_diff(&singleton_second, &batched_second),
    );
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let model_dir = args.next().ok_or_else(|| {
        "usage: linear_mtp_state_oracle_probe <model_dir> <prompt-tokens>".to_string()
    })?;
    let prompt = parse_tokens(
        &args
            .next()
            .ok_or_else(|| "missing comma-separated prompt tokens".to_string())?,
    )?;
    if let Some(unexpected) = args.next() {
        return Err(format!("unexpected argument: {unexpected}"));
    }

    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load model artifacts: {error}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights =
        load_weights(&artifacts).map_err(|error| format!("failed to load weights: {error}"))?;
    if cfg.linear_attention.is_none() {
        return Err("probe requires a linear-attention checkpoint".to_string());
    }

    let mut base_cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let (primary, _) = chunked_prefill_with_final_hidden(
        &cfg,
        &weights,
        &prompt,
        &mut base_cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), &prompt),
        &mut rng,
    );

    let token_offset = base_cache.seq_len();
    let mut first_cache = base_cache.clone();
    let first_logits = forward_argmax(&cfg, &weights, &[primary], &mut first_cache, token_offset);
    first_cache.advance(1);
    let draft = materialized_argmax(&first_logits, &first_cache);
    print_first_layer_hidden_diff(
        &cfg,
        &weights,
        &base_cache,
        token_offset,
        primary,
        draft,
        true,
    );
    print_first_layer_hidden_diff(
        &cfg,
        &weights,
        &base_cache,
        token_offset,
        primary,
        draft,
        false,
    );

    let mut singleton_cache = base_cache.clone();
    let singleton_first = forward_argmax(
        &cfg,
        &weights,
        &[primary],
        &mut singleton_cache,
        token_offset,
    );
    singleton_cache.advance(1);
    let singleton_first_token = materialized_argmax(&singleton_first, &singleton_cache);
    let singleton_second = forward_argmax(
        &cfg,
        &weights,
        &[draft],
        &mut singleton_cache,
        token_offset + 1,
    );
    singleton_cache.advance(1);
    let singleton_second_token = materialized_argmax(&singleton_second, &singleton_cache);

    let mut batched_cache = base_cache;
    batched_cache.begin_linear_prefix_capture(1);
    let batched_logits = forward_all_positions(
        &cfg,
        &weights,
        &[primary, draft],
        &mut batched_cache,
        token_offset,
    );
    batched_cache.advance(2);
    let batched_tokens = argmax(&batched_logits, None);
    let batched_refs = batched_cache.collect_eval_refs();
    let mut refs = Vec::with_capacity(1 + batched_refs.len());
    refs.push(&batched_tokens);
    refs.extend(batched_refs);
    eval(&refs);
    let batched_tokens = batched_tokens.data_u32();

    println!(
        "primary={primary} draft={draft} singleton=[{singleton_first_token},{singleton_second_token}] batched={batched_tokens:?}"
    );
    for layer in 0..cfg.layer_count {
        let (singleton_conv, singleton_recurrent) = singleton_cache.linear_state(layer);
        let (batched_conv, batched_recurrent) = batched_cache.linear_state(layer);
        let conv_diff = match (singleton_conv, batched_conv) {
            (Some(singleton), Some(batched)) => max_abs_diff(singleton, batched),
            (None, None) => 0.0,
            _ => f32::INFINITY,
        };
        let recurrent_diff = match (singleton_recurrent, batched_recurrent) {
            (Some(singleton), Some(batched)) => max_abs_diff(singleton, batched),
            (None, None) => 0.0,
            _ => f32::INFINITY,
        };
        if conv_diff != 0.0 || recurrent_diff != 0.0 {
            println!(
                "layer={layer} conv_max_abs={conv_diff:.9e} recurrent_max_abs={recurrent_diff:.9e}"
            );
        }
    }
    if !batched_cache.restore_linear_prefix_checkpoint() {
        return Err("batched verifier did not produce a complete prefix checkpoint".to_string());
    }
    for layer in 0..cfg.layer_count {
        let (singleton_conv, singleton_recurrent) = first_cache.linear_state(layer);
        let (checkpoint_conv, checkpoint_recurrent) = batched_cache.linear_state(layer);
        let conv_diff = match (singleton_conv, checkpoint_conv) {
            (Some(singleton), Some(checkpoint)) => max_abs_diff(singleton, checkpoint),
            (None, None) => 0.0,
            _ => f32::INFINITY,
        };
        let recurrent_diff = match (singleton_recurrent, checkpoint_recurrent) {
            (Some(singleton), Some(checkpoint)) => max_abs_diff(singleton, checkpoint),
            (None, None) => 0.0,
            _ => f32::INFINITY,
        };
        if conv_diff != 0.0 || recurrent_diff != 0.0 {
            println!(
                "checkpoint_layer={layer} conv_max_abs={conv_diff:.9e} \
                 recurrent_max_abs={recurrent_diff:.9e}"
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
