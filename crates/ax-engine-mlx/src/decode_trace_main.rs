//! Decode a fixed number of tokens through the direct-pipeline path and
//! print the per-step host-wall breakdown surfaced by the new
//! `forward_layer_loop_wall_us` / `forward_head_wall_us` counters.
//!
//! Usage: cargo run --release --bin decode-trace -- <model_dir> [decode_steps]
//!
//! No mlx_lm comparison, no SSE/server stack — exercises
//! `advance_direct_pipeline_with_timings` in a tight
//! loop so the host-wall split is visible without a full bench sweep.

use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    diagnostics::{take_linear_attention_profile_snapshot, take_moe_router_fused_snapshot},
    generate::{
        DEFAULT_PREFILL_CHUNK, advance_direct_pipeline_with_timings, chunked_prefill,
        start_direct_pipeline,
    },
    kv_cache::MlxKVCache,
    model::ModelConfig,
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::load_weights,
};
use mlx_sys::clear_cache;

/// Busy-spin for `dur_us` microseconds of host wall time. Diagnostic-only:
/// emulates extra host graph-build cost per decode step so the overlap
/// between host build and in-flight GPU work can be measured empirically —
/// if the pipeline overlaps, total wall stays flat until the injected spin
/// exceeds the GPU-side headroom (`AX_DECODE_TRACE_HOST_SPIN_US`).
fn spin_host_us(dur_us: u64) {
    if dur_us == 0 {
        return;
    }
    let start = Instant::now();
    while (start.elapsed().as_micros() as u64) < dur_us {
        std::hint::spin_loop();
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let model_dir = args
        .next()
        .ok_or_else(|| "usage: decode-trace <model_dir> [steps]".to_string())?;
    let decode_steps = args
        .next()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("steps must be a positive integer, got {value:?}"))
        })
        .transpose()?
        .unwrap_or(64);
    if decode_steps == 0 {
        return Err("steps must be greater than zero".to_string());
    }
    if let Some(unexpected) = args.next() {
        return Err(format!("unexpected argument: {unexpected}"));
    }
    let warmup_steps: usize = 8;
    let host_spin_us: u64 = env::var("AX_DECODE_TRACE_HOST_SPIN_US")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // Sleep variant of the injection: `thread::sleep` draws no package power,
    // so it separates true host/GPU serialization (total grows 1:1 with the
    // injected delay) from busy-spin power coupling (GPU clocks drop under a
    // spinning P-core, inflating the GPU bucket beyond the injected delay).
    let host_sleep_us: u64 = env::var("AX_DECODE_TRACE_HOST_SLEEP_US")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    println!("loading {model_dir}");
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load model artifacts: {error}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights =
        load_weights(&artifacts).map_err(|error| format!("failed to load weights: {error}"))?;

    // Short, deterministic random-style prompt to mimic readme bench shape.
    let prompt: Vec<u32> = (1..=128u32).collect();
    let mut rng = Xorshift64::new(0);
    let mut cache = MlxKVCache::new(cfg.layer_count);

    let prefill_start = Instant::now();
    let bootstrap_tok = chunked_prefill(
        &cfg,
        &weights,
        &prompt,
        &mut cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), &prompt),
        &mut rng,
    );
    let prefill_us = prefill_start.elapsed().as_micros();
    println!("prefill {} tokens → {} µs", prompt.len(), prefill_us);

    // FNV-1a over the full decoded token stream (bootstrap + warmup + timed).
    // Deterministic for greedy decode, so two runs of the same checkpoint must
    // print the same hash — the cross-process parity check for fastpath A/Bs.
    let mut token_hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut token_count: u64 = 0;
    let mut hash_token = |tok: u32| {
        token_hash ^= u64::from(tok);
        token_hash = token_hash.wrapping_mul(0x0000_0100_0000_01b3);
        token_count += 1;
    };
    hash_token(bootstrap_tok);

    let mut pending = start_direct_pipeline(&cfg, &weights, bootstrap_tok, &mut cache);

    // Warmup: drain JIT / kernel-cache cost so the reported numbers reflect
    // steady-state direct-pipeline cost.
    for _ in 0..warmup_steps {
        let advanced = advance_direct_pipeline_with_timings(&cfg, &weights, &pending, &mut cache);
        hash_token(advanced.token);
        pending = advanced.next_pending;
    }

    let mut sums = [0u64; 8];
    let mut tok = 0u32;
    let mut linear_layer_ops_sum = 0u64;
    let mut linear_layer_count_sum = 0u64;
    let mut full_layer_ops_sum = 0u64;
    let mut full_layer_count_sum = 0u64;
    let bench_start = Instant::now();
    for _ in 0..decode_steps {
        spin_host_us(host_spin_us);
        if host_sleep_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(host_sleep_us));
        }
        let advanced = advance_direct_pipeline_with_timings(&cfg, &weights, &pending, &mut cache);
        linear_layer_ops_sum =
            linear_layer_ops_sum.saturating_add(advanced.timings.linear_attention_layer_ops);
        linear_layer_count_sum = linear_layer_count_sum
            .saturating_add(advanced.timings.linear_attention_layer_count as u64);
        full_layer_ops_sum =
            full_layer_ops_sum.saturating_add(advanced.timings.full_attention_layer_ops);
        full_layer_count_sum =
            full_layer_count_sum.saturating_add(advanced.timings.full_attention_layer_count as u64);
        tok = advanced.token;
        hash_token(tok);
        pending = advanced.next_pending;
        sums[0] = sums[0].saturating_add(advanced.timings.forward_wall_us as u64);
        sums[1] = sums[1].saturating_add(advanced.timings.forward_layer_loop_wall_us as u64);
        sums[2] = sums[2].saturating_add(advanced.timings.forward_head_wall_us as u64);
        sums[3] = sums[3].saturating_add(advanced.timings.argmax_wall_us as u64);
        sums[4] = sums[4].saturating_add(advanced.timings.async_eval_wall_us as u64);
        sums[5] = sums[5].saturating_add(advanced.timings.pending_eval_wall_us as u64);
        sums[6] = sums[6].saturating_add(advanced.timings.pending_read_wall_us as u64);
        sums[7] = sums[7].saturating_add(advanced.timings.next_complete_wall_us as u64);
    }
    let total_us = bench_start.elapsed().as_micros();
    clear_cache();
    let _ = tok;

    let avg = |s: u64| s as f64 / decode_steps as f64;
    let forward_avg = avg(sums[0]);
    let layer_loop_avg = avg(sums[1]);
    let head_avg = avg(sums[2]);
    let residual_avg = forward_avg - layer_loop_avg - head_avg;

    println!();
    if host_spin_us > 0 {
        println!("host spin injected: {host_spin_us} µs/step (AX_DECODE_TRACE_HOST_SPIN_US)");
    }
    if host_sleep_us > 0 {
        println!("host sleep injected: {host_sleep_us} µs/step (AX_DECODE_TRACE_HOST_SLEEP_US)");
    }
    println!("steady-state ({decode_steps} steps after {warmup_steps} warmup):");
    println!(
        "  total wall              {:>8.1} µs/tok",
        total_us as f64 / decode_steps as f64
    );
    println!("  forward                 {forward_avg:>8.1} µs/tok");
    println!(
        "    layer_loop (64×)      {layer_loop_avg:>8.1} µs/tok  ({:>5.1}% of forward, ~{:.1} µs/layer)",
        layer_loop_avg / forward_avg * 100.0,
        layer_loop_avg / cfg.layer_count as f64
    );
    println!(
        "    head (norm+lm_head)   {head_avg:>8.1} µs/tok  ({:>5.1}% of forward)",
        head_avg / forward_avg * 100.0
    );
    println!(
        "    residual (embed+...)  {residual_avg:>8.1} µs/tok  ({:>5.1}% of forward)",
        residual_avg / forward_avg * 100.0
    );
    println!("  argmax                  {:>8.1} µs/tok", avg(sums[3]));
    println!("  async_eval (submit)     {:>8.1} µs/tok", avg(sums[4]));
    println!("  pending eval (gpu wait) {:>8.1} µs/tok", avg(sums[5]));
    println!("  pending read            {:>8.1} µs/tok", avg(sums[6]));
    if sums[7] > 0 {
        println!("  next-complete barrier   {:>8.1} µs/tok", avg(sums[7]));
    }
    println!();
    println!("per-layer ops/tok (FFI dispatch count from mlx_sys::op_count):");
    if linear_layer_count_sum > 0 {
        let avg = linear_layer_ops_sum as f64 / linear_layer_count_sum as f64;
        let layers_per_tok = linear_layer_count_sum as f64 / decode_steps as f64;
        println!(
            "  linear-attention layers: {avg:>5.1} ops/layer  ({layers_per_tok:>4.1} layers/tok, {:>6.1} ops/tok)",
            avg * layers_per_tok
        );
    }
    if full_layer_count_sum > 0 {
        let avg = full_layer_ops_sum as f64 / full_layer_count_sum as f64;
        let layers_per_tok = full_layer_count_sum as f64 / decode_steps as f64;
        println!(
            "  full-attention layers:   {avg:>5.1} ops/layer  ({layers_per_tok:>4.1} layers/tok, {:>6.1} ops/tok)",
            avg * layers_per_tok
        );
    }
    let linear_profile = take_linear_attention_profile_snapshot();
    println!();
    println!("linear-attention direct C++ counters:");
    println!(
        "  inputs:     attempts={} hits={} fallbacks={} profile_blocked={}",
        linear_profile.direct_cpp_inputs_attempts,
        linear_profile.direct_cpp_inputs_hits,
        linear_profile.direct_cpp_inputs_fallbacks,
        linear_profile.direct_cpp_inputs_profile_blocked
    );
    println!(
        "  post-input: attempts={} hits={} fallbacks={} profile_blocked={}",
        linear_profile.direct_cpp_post_input_attempts,
        linear_profile.direct_cpp_post_input_hits,
        linear_profile.direct_cpp_post_input_fallbacks,
        linear_profile.direct_cpp_post_input_profile_blocked
    );
    let router_fused = take_moe_router_fused_snapshot();
    println!();
    println!(
        "moe router fused: attempts={} hits={} fallbacks={}",
        router_fused.attempts, router_fused.hits, router_fused.fallbacks
    );
    println!("token stream: n={token_count} fnv1a64={token_hash:016x}");
    println!();
    println!(
        "decode tok/s = {:.2}",
        1_000_000.0 * decode_steps as f64 / total_us as f64
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}
