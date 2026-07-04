//! Batched decode — milestone 2a static-harness probe.
//!
//! Proves, end to end on a real dense model, the two things that gate the whole
//! batched-decode feature before any runner surgery:
//!
//! 1. **Token-exact.** Decoding B equal-length prompts all together through the
//!    batched path (`model::decode_batched_forward` + `argmax_batched`) yields,
//!    for every request, the identical greedy token stream as decoding each one
//!    independently through the single-sequence path (`model::forward` + MLX
//!    argmax). This is the load-bearing correctness oracle.
//! 2. **Amortized.** The batched arm's aggregate throughput (B tokens per step)
//!    vs B sequential single-stream decodes, same-session interleaved, should
//!    approach the Phase 0 prediction (~1.9× @B2, ~2.4× @B4).
//!
//! 2a scope: full-attention **dense** families (Qwen3/Llama/Mistral), equal-
//! length prompts (uniform RoPE offset). Ragged positions, MoE/MLA/linear,
//! sliding window, and the runner wiring are 2b.
//!
//! Usage:
//!   cargo run --release --bin batched_decode_2a_probe -- <dense_model_dir>
//! Env: AX_BATCH (default 4), AX_PROMPT_LEN (default 32), AX_GEN (default 32),
//!      AX_SEED (prompt base seed, default 0).

use std::env;
use std::path::Path;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    batched_kv_cache::BatchedKvCache,
    batched_sampling::argmax_batched,
    generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill_with_final_hidden},
    kv_cache::MlxKVCache,
    model::{ModelConfig, decode_batched_forward, forward},
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::{ModelWeights, load_weights},
};
use mlx_sys::{argmax, clear_cache, eval};

/// B distinct but equal-length prompts (equal length ⇒ uniform decode position,
/// the 2a constraint). Distinct content ⇒ distinct streams ⇒ the oracle checks
/// real per-row independence, not a trivial all-rows-identical case.
fn build_prompts(batch: usize, len: usize, vocab: usize) -> Vec<Vec<u32>> {
    let base: usize = env::var("AX_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (0..batch)
        .map(|r| {
            (0..len)
                .map(|i| (((base + r) * 17 + i * 5 + 3) % (vocab - 1)) as u32 + 1)
                .collect()
        })
        .collect()
}

fn prefill(cfg: &ModelConfig, w: &ModelWeights, prompt: &[u32]) -> (MlxKVCache, u32) {
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let (tok0, _hidden) = chunked_prefill_with_final_hidden(
        cfg,
        w,
        prompt,
        &mut cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), prompt),
        &mut rng,
    );
    (cache, tok0)
}

/// Single-sequence greedy decode of `gen_len` tokens after a prefilled cache —
/// the reference arm, mirroring production greedy (`forward` + MLX argmax).
fn single_decode(
    cfg: &ModelConfig,
    w: &ModelWeights,
    cache: &mut MlxKVCache,
    tok0: u32,
    gen_len: usize,
) -> Vec<u32> {
    let mut stream = vec![tok0];
    let mut tok = tok0;
    for _ in 0..gen_len {
        let offset = cache.seq_len;
        let logits = forward(cfg, w, &[tok], cache, offset);
        let idx = argmax(&logits, None);
        eval(&[&idx]);
        cache.seq_len += 1;
        tok = idx.data_u32()[0];
        stream.push(tok);
    }
    stream
}

/// Seed a fresh `BatchedKvCache` from B independent prefills; returns the cache
/// and the per-row first token.
fn seed_batched(
    cfg: &ModelConfig,
    w: &ModelWeights,
    prompts: &[Vec<u32>],
    layers: usize,
) -> (BatchedKvCache, Vec<u32>) {
    let mut bcache = BatchedKvCache::new(layers, prompts.len());
    let mut cur = Vec::with_capacity(prompts.len());
    for (r, prompt) in prompts.iter().enumerate() {
        let (cache, tok0) = prefill(cfg, w, prompt);
        for layer in 0..layers {
            let (k, v) = cache.peek_layer_kv(layer).expect("prefilled layer KV");
            bcache.seed_row_layer(layer, r, &k, &v);
        }
        cur.push(tok0);
    }
    (bcache, cur)
}

fn main() {
    let model_dir = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: batched_decode_2a_probe <dense_model_dir>");
        std::process::exit(2);
    });
    let batch: usize = env::var("AX_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let prompt_len: usize = env::var("AX_PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let gen_len: usize = env::var("AX_GEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("load artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).expect("load weights");
    let layers = cfg.layer_count;
    let prompts = build_prompts(batch, prompt_len, cfg.vocab_size);

    println!("# batched decode 2a probe");
    println!(
        "model_family {}  layers {}  batch {batch}  prompt_len {prompt_len}  gen_len {gen_len}",
        cfg.model_family, layers
    );

    // ── Correctness: reference (per-request) vs batched, token-exact ──
    let mut ref_streams: Vec<Vec<u32>> = Vec::with_capacity(batch);
    for prompt in &prompts {
        let (mut cache, tok0) = prefill(&cfg, &weights, prompt);
        ref_streams.push(single_decode(&cfg, &weights, &mut cache, tok0, gen_len));
    }
    clear_cache();

    let (mut bcache, mut cur) = seed_batched(&cfg, &weights, &prompts, layers);
    let mut batched_streams: Vec<Vec<u32>> = cur.iter().map(|&t| vec![t]).collect();
    for _ in 0..gen_len {
        let offset = bcache.row_len(0); // uniform across rows (equal length)
        let logits = decode_batched_forward(&cfg, &weights, &cur, &mut bcache, offset);
        let toks = argmax_batched(&logits);
        for r in 0..batch {
            batched_streams[r].push(toks[r]);
            cur[r] = toks[r];
        }
    }
    clear_cache();

    let mut mismatches = 0usize;
    for r in 0..batch {
        if batched_streams[r] != ref_streams[r] {
            mismatches += 1;
            let first = (0..batched_streams[r].len())
                .find(|&i| batched_streams[r].get(i) != ref_streams[r].get(i));
            eprintln!(
                "  row {r} MISMATCH at pos {first:?}: batched {:?} ref {:?}",
                batched_streams[r], ref_streams[r]
            );
        }
    }
    if mismatches == 0 {
        println!("TOKEN-EXACT: PASS ({batch}/{batch} rows identical to single-sequence decode)");
    } else {
        println!("TOKEN-EXACT: FAIL ({mismatches}/{batch} rows differ)");
        std::process::exit(1);
    }

    // ── Perf: aggregate tok/s, batched vs B× single-stream (interleaved) ──
    let time_single = || {
        let mut caches: Vec<(MlxKVCache, u32)> =
            prompts.iter().map(|p| prefill(&cfg, &weights, p)).collect();
        clear_cache();
        let t0 = Instant::now();
        for (cache, tok0) in caches.iter_mut() {
            let _ = single_decode(&cfg, &weights, cache, *tok0, gen_len);
        }
        t0.elapsed().as_secs_f64()
    };
    let time_batched = || {
        let (mut bc, mut cur) = seed_batched(&cfg, &weights, &prompts, layers);
        clear_cache();
        let t0 = Instant::now();
        for _ in 0..gen_len {
            let offset = bc.row_len(0);
            let logits = decode_batched_forward(&cfg, &weights, &cur, &mut bc, offset);
            cur.copy_from_slice(&argmax_batched(&logits));
        }
        t0.elapsed().as_secs_f64()
    };

    // 1 warmup + 3 measured (median), interleaved to share thermal conditions.
    let _ = time_single();
    let _ = time_batched();
    let mut single = Vec::new();
    let mut batched = Vec::new();
    for _ in 0..3 {
        single.push(time_single());
        batched.push(time_batched());
    }
    single.sort_by(|a, b| a.partial_cmp(b).unwrap());
    batched.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (single_s, batched_s) = (single[1], batched[1]);
    let toks = (batch * gen_len) as f64;
    println!();
    println!("{:<24} {:>10} {:>14}", "arm", "wall_s", "agg tok/s");
    println!(
        "{:<24} {:>10.3} {:>14.1}",
        "single (B sequential)",
        single_s,
        toks / single_s
    );
    println!(
        "{:<24} {:>10.3} {:>14.1}",
        "batched (B together)",
        batched_s,
        toks / batched_s
    );
    println!();
    println!(
        "aggregate speedup B={batch}: {:.2}x  (single_wall / batched_wall)",
        single_s / batched_s
    );
    println!("# Phase 0 kernel-ceiling prediction: ~1.9x @B2, ~2.4x @B4. End-to-end includes");
    println!(
        "# per-row KV reads, seed copy (excluded from the timed loop), per-row RoPE, and argmax."
    );
}
