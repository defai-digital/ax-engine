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
    batched_decode_session::BatchedDecodeSession,
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
    let row_stride: usize = env::var("AX_ROW_STRIDE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(17);
    let token_stride: usize = env::var("AX_TOKEN_STRIDE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let prompt_bias: usize = env::var("AX_PROMPT_BIAS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    // AX_RAGGED: give each row a different prompt length (row r → len - 3*r,
    // floored at len/2) so the cohort decodes at ragged sequence positions.
    let ragged = env::var("AX_RAGGED").is_ok();
    (0..batch)
        .map(|r| {
            let row_len = if env::var("AX_RAGGED_ASCENDING").is_ok() {
                len.saturating_sub(5 * (batch - 1 - r)).max(1)
            } else if ragged {
                (len.saturating_sub(3 * r)).max(len / 2).max(1)
            } else {
                len
            };
            (0..row_len)
                .map(|i| {
                    (((base + r) * row_stride + i * token_stride + prompt_bias) % (vocab - 1))
                        as u32
                        + 1
                })
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
        let offset = cache.seq_len();
        let logits = forward(cfg, w, &[tok], cache, offset);
        let idx = argmax(&logits, None);
        eval(&[&idx]);
        cache.advance(1);
        tok = idx.data_u32()[0];
        stream.push(tok);
    }
    stream
}

/// Batched streams via a [`BatchedDecodeSession`] — the production path, which
/// seeds and advances per-row linear-attention recurrent state for hybrid
/// (gated-delta) models. Each request is admitted from its own prefill and the
/// full cohort steps `gen_len` times. Row `i`'s stream starts with its prefill
/// token, exactly like [`single_decode`].
fn batched_streams_via_session(
    cfg: &ModelConfig,
    w: &ModelWeights,
    prompts: &[Vec<u32>],
    max_batch: usize,
    gen_len: usize,
) -> Vec<Vec<u32>> {
    let prefills: Vec<(MlxKVCache, u32)> = prompts.iter().map(|p| prefill(cfg, w, p)).collect();
    let mut session = BatchedDecodeSession::new(cfg.layer_count, max_batch.max(prompts.len()));
    let mut streams: Vec<Vec<u32>> = Vec::with_capacity(prompts.len());
    for (i, (cache, tok0)) in prefills.iter().enumerate() {
        session.add(i as u64, cache, *tok0);
        streams.push(vec![*tok0]);
    }
    for _ in 0..gen_len {
        for (id, tok) in session.step(cfg, w) {
            streams[id as usize].push(tok);
        }
    }
    clear_cache();
    streams
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

/// Continuous-batching oracle: requests of ragged length join and leave a
/// `BatchedDecodeSession` at different times; each request's produced stream
/// must equal a standalone single-sequence decode of the same length. Proves
/// join/leave/slot-swap don't perturb any request's output.
fn continuous_oracle(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    layers: usize,
    max_batch: usize,
    gen_len: usize,
) -> bool {
    let n_requests = max_batch + 2;
    let vocab = cfg.vocab_size;
    let prompts: Vec<Vec<u32>> = (0..n_requests)
        .map(|r| {
            let len = 16 + (r % 3) * 5; // ragged lengths
            (0..len)
                .map(|i| ((r * 29 + i * 7 + 5) % (vocab - 1)) as u32 + 1)
                .collect()
        })
        .collect();

    // Reference: each request decoded alone for `gen_len` steps.
    let refs: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            let (mut c, t0) = prefill(cfg, weights, p);
            single_decode(cfg, weights, &mut c, t0, gen_len)
        })
        .collect();
    clear_cache();

    // Prefill all up front; keep the caches alive for seeding.
    let prefills: Vec<(MlxKVCache, u32)> =
        prompts.iter().map(|p| prefill(cfg, weights, p)).collect();

    let mut session = BatchedDecodeSession::new(layers, max_batch);
    let mut produced: Vec<Vec<u32>> = vec![Vec::new(); n_requests];
    let mut steps = vec![0usize; n_requests];
    let mut next_pending = 0usize;

    loop {
        // Admit pending requests into any free slots.
        while session.len() < max_batch && next_pending < n_requests {
            let (cache, tok0) = &prefills[next_pending];
            session.add(next_pending as u64, cache, *tok0);
            produced[next_pending].push(*tok0);
            next_pending += 1;
        }
        if session.is_empty() {
            break;
        }
        let outs = session.step(cfg, weights);
        let mut finished = Vec::new();
        for (id, tok) in outs {
            let r = id as usize;
            produced[r].push(tok);
            steps[r] += 1;
            if steps[r] >= gen_len {
                finished.push(id);
            }
        }
        for id in finished {
            session.remove(id);
        }
    }
    clear_cache();

    let mut ok = true;
    for r in 0..n_requests {
        if produced[r] != refs[r] {
            ok = false;
            let first = (0..produced[r].len().max(refs[r].len()))
                .find(|&i| produced[r].get(i) != refs[r].get(i));
            eprintln!(
                "  CONTINUOUS req {r} MISMATCH at {first:?}: produced {:?} ref {:?}",
                produced[r], refs[r]
            );
        }
    }
    ok
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
    // Hybrid = has gated-delta linear-attention layers (Qwen3-Next). Those carry
    // per-row recurrent state that only the `BatchedDecodeSession` seeds/advances,
    // so hybrid models certify + benchmark through the session, not the raw
    // `decode_batched_forward(None)` manual path (which handles full-attention KV).
    let hybrid = weights.layers.iter().any(|w| w.linear_attn.is_some());
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

    let batched_streams: Vec<Vec<u32>> = if hybrid {
        batched_streams_via_session(&cfg, &weights, &prompts, batch, gen_len)
    } else {
        let (mut bcache, mut cur) = seed_batched(&cfg, &weights, &prompts, layers);
        let mut streams: Vec<Vec<u32>> = cur.iter().map(|&t| vec![t]).collect();
        for _ in 0..gen_len {
            let logits = decode_batched_forward(&cfg, &weights, &cur, &mut bcache, None);
            let toks = argmax_batched(&logits);
            for r in 0..batch {
                streams[r].push(toks[r]);
                cur[r] = toks[r];
            }
        }
        clear_cache();
        streams
    };

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

    // Continuous batching: dynamic join/leave over a BatchedDecodeSession.
    if env::var("AX_CONTINUOUS").is_ok() {
        let ok = continuous_oracle(&cfg, &weights, layers, batch, gen_len);
        if ok {
            println!(
                "CONTINUOUS: PASS ({} reqs join/leave a max-{batch} session, all token-exact)",
                batch + 2
            );
        } else {
            println!("CONTINUOUS: FAIL");
            std::process::exit(1);
        }
    }

    // ── Perf: aggregate tok/s, batched vs B× single-stream (interleaved) ──
    // Hybrid models benchmark through the session (per-row recurrent state), which
    // the batched-decode-ceiling-probe already drives; the manual arm below is the
    // full-attention path, so skip it for hybrids.
    if hybrid {
        println!();
        println!(
            "# hybrid model: run `batched-decode-ceiling-probe {model_dir}` for aggregate tok/s"
        );
        return;
    }
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
            let logits = decode_batched_forward(&cfg, &weights, &cur, &mut bc, None);
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
