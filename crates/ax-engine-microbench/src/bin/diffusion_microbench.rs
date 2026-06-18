//! DiffusionGemma denoise-step hotspot microbench.
//!
//! Profiles operations that run on every denoiser step at realistic
//! DiffusionGemma / Gemma4 dimensions.
//!
//! Hotspots:
//!   1. **Entropy-bound sampling** — sort canvas positions by entropy,
//!      accept confident positions, update uncertain ones via argmax.
//!   2. **Bidirectional mask construction** — build the boolean attention
//!      mask for canvas self-attention + cross-attention to cached prompt.
//!   3. **Self-conditioning embedding (CPU)** — probability-weighted average
//!      of the full token embedding table on CPU (the legacy approach).
//!   4. **Self-conditioning embedding (GPU)** — same operation via MLX
//!      matmul on GPU (the optimized approach).
//!
//! Run:
//!   cargo run -p ax-engine-microbench --release --bin diffusion-microbench

use std::time::{Duration, Instant};

use mlx_sys::{MlxArray, MlxDtype, astype, eval, matmul};

// ── Realistic DiffusionGemma / Gemma4 dimensions ──────────────────────
const CANVAS_SIZE: usize = 256;
const VOCAB_SIZE: usize = 262_144; // Gemma4 SentencePiece
const HIDDEN_SIZE: usize = 3584; // Gemma4 26B hidden dim
const SLIDING_WINDOW: usize = 1024; // Gemma4 SWA half-window
const ENTROPY_BOUND: f32 = 0.1;

// Benchmark iteration counts (2 warmup + N measure, report median).
const WARMUP: usize = 2;
const MEASURE: usize = 5;

fn main() {
    println!("DiffusionGemma Denoise-Step Microbench");
    println!("  canvas_size  = {CANVAS_SIZE}");
    println!("  vocab_size   = {VOCAB_SIZE}");
    println!("  hidden_size  = {HIDDEN_SIZE}");
    println!("  sliding_window = {SLIDING_WINDOW}");
    println!("  entropy_bound  = {ENTROPY_BOUND}");
    println!();

    bench_entropy_sampling();
    bench_bidirectional_mask();
    bench_self_conditioning_embed();
}

// ── 1. Entropy-bound sampling ─────────────────────────────────────────

fn bench_entropy_sampling() {
    // Synthetic per-position entropy: mix of low (confident) and high
    // (uncertain) values to exercise the sort + greedy accept path.
    let entropy: Vec<f32> = (0..CANVAS_SIZE)
        .map(|i| {
            if i % 4 == 0 {
                0.001 + (i as f32 * 0.0001) // low entropy (confident)
            } else {
                2.0 + (i as f32 * 0.01) // high entropy (uncertain)
            }
        })
        .collect();

    let run = || {
        let mut position_order: Vec<usize> = (0..CANVAS_SIZE).collect();
        position_order.sort_by(|&a, &b| entropy[a].total_cmp(&entropy[b]));

        let mut accept_mask = vec![false; CANVAS_SIZE];
        let mut cumulative_entropy = 0.0_f32;
        for &pos in &position_order {
            if cumulative_entropy + entropy[pos] > ENTROPY_BOUND {
                break;
            }
            accept_mask[pos] = true;
            cumulative_entropy += entropy[pos];
        }
        if !accept_mask.iter().any(|&v| v) {
            accept_mask[position_order[0]] = true;
        }

        // Accepted positions keep current tokens, rejected positions
        // adopt the model's argmax prediction (preserves progress).
        let mut tokens = vec![0_u32; CANVAS_SIZE];
        for (pos, &accepted) in accept_mask.iter().enumerate() {
            if !accepted {
                // In production: new_tokens[pos] = argmax_data[pos]
                tokens[pos] = pos as u32; // simulate argmax
            }
        }

        accept_mask.iter().filter(|&&v| v).count()
    };

    let (warmup_result, median, all) = timed_bench(run);
    println!("[1] Entropy-bound sampling ({CANVAS_SIZE} positions)");
    println!("    accepted positions (warmup): {warmup_result}");
    print_timing(median, &all);
}

// ── 2. Bidirectional mask construction ────────────────────────────────

fn bench_bidirectional_mask() {
    // Simulate a growing cached prompt (1K, 4K, 16K tokens).
    let cached_seqs = [1024, 4096, 16384];

    for &cached_seq in &cached_seqs {
        let run = || {
            let total_keys = cached_seq + CANVAS_SIZE;
            let mut mask = vec![0_u8; CANVAS_SIZE * total_keys];
            for qi in 0..CANVAS_SIZE {
                for ki in 0..cached_seq {
                    mask[qi * total_keys + ki] = 1;
                }
                for ki in 0..CANVAS_SIZE {
                    if qi.abs_diff(ki) < SLIDING_WINDOW {
                        mask[qi * total_keys + cached_seq + ki] = 1;
                    }
                }
            }
            let ones: usize = mask.iter().map(|&b| b as usize).sum();
            (ones, mask.len())
        };

        let (warmup_result, median, all) = timed_bench(run);
        println!(
            "[2] Bidirectional mask ({CANVAS_SIZE} x {total_keys}, cached_seq={cached_seq})",
            total_keys = cached_seq + CANVAS_SIZE,
        );
        println!(
            "    mask ones/total (warmup): {}/{}",
            warmup_result.0, warmup_result.1
        );
        print_timing(median, &all);
    }
}

// ── 3. Self-conditioning embedding ────────────────────────────────────

fn bench_self_conditioning_embed() {
    // Test with two sparsity levels:
    //   - Sparse (top-10): simulates aggressive pruning / low temperature
    //   - Dense (top-1000): more realistic softmax output at moderate temperature
    let configs = [
        (10_usize, "sparse (top-10)"),
        (1000_usize, "dense (top-1000)"),
    ];

    // Synthetic embedding table: [vocab_size, hidden_size] as flat f32.
    println!("[3] Self-conditioning embedding ({CANVAS_SIZE} x {VOCAB_SIZE} x {HIDDEN_SIZE})");
    println!(
        "    NOTE: allocating {mem_gb:.2} GB embed table...",
        mem_gb = (VOCAB_SIZE * HIDDEN_SIZE * 4) as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let embed_table = build_embed_table(VOCAB_SIZE, HIDDEN_SIZE);

    for (top_k, label) in configs {
        let prob = build_sparse_prob(CANVAS_SIZE, VOCAB_SIZE, top_k);

        let run = || {
            let mut weighted = vec![0.0_f32; CANVAS_SIZE * HIDDEN_SIZE];
            for pos in 0..CANVAS_SIZE {
                let p_offset = pos * VOCAB_SIZE;
                let w_offset = pos * HIDDEN_SIZE;
                for v in 0..VOCAB_SIZE {
                    let p = prob[p_offset + v];
                    if p == 0.0 {
                        continue;
                    }
                    let e_offset = v * HIDDEN_SIZE;
                    for h in 0..HIDDEN_SIZE {
                        weighted[w_offset + h] += p * embed_table[e_offset + h];
                    }
                }
            }
            let checksum: f32 = weighted.iter().sum();
            (checksum, weighted.len())
        };

        let (warmup_result, median, all) = timed_bench(run);
        println!(
            "    [{label}] weighted sum checksum: {:.6}",
            warmup_result.0
        );
        print_timing(median, &all);
    }

    // Also benchmark the fully dense case (all vocab entries non-zero).
    // This is the worst-case for high-temperature softmax.
    let prob_dense = build_dense_prob(CANVAS_SIZE, VOCAB_SIZE);
    let run_dense = || {
        let mut weighted = vec![0.0_f32; CANVAS_SIZE * HIDDEN_SIZE];
        for pos in 0..CANVAS_SIZE {
            let p_offset = pos * VOCAB_SIZE;
            let w_offset = pos * HIDDEN_SIZE;
            for v in 0..VOCAB_SIZE {
                let p = prob_dense[p_offset + v];
                let e_offset = v * HIDDEN_SIZE;
                for h in 0..HIDDEN_SIZE {
                    weighted[w_offset + h] += p * embed_table[e_offset + h];
                }
            }
        }
        let checksum: f32 = weighted.iter().sum();
        (checksum, weighted.len())
    };

    println!("    [fully dense (all {VOCAB_SIZE})] estimating (1 run)...");
    let start = Instant::now();
    let (checksum, _len) = run_dense();
    let elapsed = start.elapsed();
    println!("    checksum: {:.6}", checksum);
    println!(
        "    single run: {}ms (estimated per-denoise-step cost)",
        elapsed.as_millis()
    );
    println!();

    // GPU matmul benchmark (the optimized approach).
    bench_gpu_matmul_self_cond();
}

fn bench_gpu_matmul_self_cond() {
    println!("[4] Self-conditioning via GPU matmul ({CANVAS_SIZE} x {VOCAB_SIZE} x {HIDDEN_SIZE})");

    // Build prob array on GPU: [1, canvas_size, vocab_size] (sparse top-1000).
    let prob_data = build_sparse_prob(CANVAS_SIZE, VOCAB_SIZE, 1000);
    let prob = MlxArray::from_raw_data(
        prob_data.as_ptr() as *const u8,
        prob_data.len() * std::mem::size_of::<f32>(),
        &[1, CANVAS_SIZE as i32, VOCAB_SIZE as i32],
        MlxDtype::Float32,
    );
    let prob = astype(&prob, MlxDtype::Bfloat16, None);

    // Build embed table on GPU: [vocab_size, hidden_size].
    let embed_data = build_embed_table(VOCAB_SIZE, HIDDEN_SIZE);
    let embed = MlxArray::from_raw_data(
        embed_data.as_ptr() as *const u8,
        embed_data.len() * std::mem::size_of::<f32>(),
        &[VOCAB_SIZE as i32, HIDDEN_SIZE as i32],
        MlxDtype::Float32,
    );
    let embed = astype(&embed, MlxDtype::Bfloat16, None);

    // Warmup: materialize both arrays.
    eval(&[&prob, &embed]);
    let warmup = matmul(&prob, &embed, None);
    eval(&[&warmup]);

    // Measure GPU matmul: prob [1, 256, 262144] x embed [262144, 3584]
    // = [1, 256, 3584].
    let run = || {
        let result = matmul(&prob, &embed, None);
        eval(&[&result]);
        result.shape()[2] as usize // return hidden_size as checksum
    };

    let (warmup_result, median, all) = timed_bench(run);
    println!("    result hidden dim (warmup): {warmup_result}");
    print_timing(median, &all);
}

// ── Helpers ───────────────────────────────────────────────────────────

fn build_sparse_prob(canvas_size: usize, vocab_size: usize, top_k: usize) -> Vec<f32> {
    // Each row has `top_k` non-zero entries that sum to 1.0.
    let mut prob = vec![0.0_f32; canvas_size * vocab_size];
    for pos in 0..canvas_size {
        let base = (pos * 37) % vocab_size;
        let weight = 1.0 / top_k as f32;
        for k in 0..top_k {
            let idx = (base + k * 1024) % vocab_size;
            prob[pos * vocab_size + idx] = weight;
        }
    }
    prob
}

fn build_dense_prob(canvas_size: usize, vocab_size: usize) -> Vec<f32> {
    // Uniform distribution: every entry = 1/vocab_size. Worst case for the
    // skip-zero optimization.
    let uniform = 1.0 / vocab_size as f32;
    vec![uniform; canvas_size * vocab_size]
}

fn build_embed_table(vocab_size: usize, hidden_size: usize) -> Vec<f32> {
    // Deterministic pseudo-random table: small values in [-0.02, 0.02].
    let mut table = Vec::with_capacity(vocab_size * hidden_size);
    let mut state: u64 = 0xCAFE_BABE;
    for _ in 0..vocab_size * hidden_size {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let f = ((state >> 32) as f32 / u32::MAX as f32) * 0.04 - 0.02;
        table.push(f);
    }
    table
}

fn timed_bench<F, R>(f: F) -> (R, Duration, Vec<Duration>)
where
    F: Fn() -> R,
{
    // Warmup.
    let mut warmup_result = None;
    for i in 0..WARMUP {
        let r = f();
        if i == WARMUP - 1 {
            warmup_result = Some(r);
        }
    }

    // Measure.
    let mut durations = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let start = Instant::now();
        let _ = f();
        durations.push(start.elapsed());
    }
    durations.sort();
    let median = durations[MEASURE / 2];
    (warmup_result.unwrap(), median, durations)
}

fn print_timing(median: Duration, all: &[Duration]) {
    let min = all.iter().min().unwrap();
    let max = all.iter().max().unwrap();
    println!(
        "    min={min_us:>8}us  median={med_us:>8}us  max={max_us:>8}us  ({MEASURE} runs)",
        min_us = min.as_micros(),
        med_us = median.as_micros(),
        max_us = max.as_micros(),
    );
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_always_accepts_at_least_one() {
        let entropy: Vec<f32> = vec![100.0; CANVAS_SIZE]; // all very high
        let mut position_order: Vec<usize> = (0..CANVAS_SIZE).collect();
        position_order.sort_by(|&a, &b| entropy[a].total_cmp(&entropy[b]));
        let mut accept_mask = vec![false; CANVAS_SIZE];
        let mut cumulative_entropy = 0.0_f32;
        for &pos in &position_order {
            if cumulative_entropy + entropy[pos] > ENTROPY_BOUND {
                break;
            }
            accept_mask[pos] = true;
            cumulative_entropy += entropy[pos];
        }
        if !accept_mask.iter().any(|&v| v) {
            accept_mask[position_order[0]] = true;
        }
        assert!(accept_mask.iter().filter(|&&v| v).count() >= 1);
    }

    #[test]
    fn mask_prompt_prefix_always_attended() {
        let cached_seq = 512;
        let total_keys = cached_seq + CANVAS_SIZE;
        let mut mask = vec![0_u8; CANVAS_SIZE * total_keys];
        for qi in 0..CANVAS_SIZE {
            for ki in 0..cached_seq {
                mask[qi * total_keys + ki] = 1;
            }
        }
        // Every query row must have all cached_seq prefix columns set.
        for qi in 0..CANVAS_SIZE {
            let row = &mask[qi * total_keys..qi * total_keys + cached_seq];
            assert!(row.iter().all(|&b| b == 1));
        }
    }
}
