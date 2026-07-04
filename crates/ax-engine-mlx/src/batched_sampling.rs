//! Batched token sampling — Phase 1 of batched MLX decode.
//!
//! After a batched decode forward, the LM head yields logits `[B, vocab]` (or
//! `[B, 1, vocab]`) for the B concurrent requests in the step. This module turns
//! those into B token ids, matching what B independent single-sequence decodes
//! would produce — the last isolated piece before the runner wires batched
//! decode together (Phase 2).
//!
//! Sampling is ~0.01% of decode wall time (Phase 0 profile), so this does not
//! need a fused kernel; it needs to be **token-exact** with the single-sequence
//! paths. There are two, mirroring the two the single decode path uses:
//!
//! - [`argmax_batched`] — greedy. Uses the SAME last-axis MLX `argmax` kernel as
//!   the single greedy decode path (`generate.rs` `argmax(&logits)`), so row `r`
//!   is identical to a batch=1 greedy decode of row `r`'s logits, tie-breaking
//!   included. It reads back only B indices, not the `B × vocab` logits — the
//!   right path for the common all-greedy batch.
//! - [`sample_batched_host`] — general per-row sampling with per-row params, RNG,
//!   and repetition context. Reads `[B, vocab]` back to host and applies the
//!   single-row [`sample_categorical`] to each row, so row `r` is identical to a
//!   batch=1 sample of row `r`.
//!
//! Both are exposed so the runner can route all-greedy batches through the
//! cheap GPU argmax and everything else through the host sampler. Not yet wired
//! into the runner (Phase 2).

use mlx_sys::{MlxArray, MlxDtype, argmax, astype, eval, reshape};

use crate::sampling::{MlxSamplingParams, Xorshift64, sample_categorical};

/// `(batch, vocab)` for a logits tensor that is `[B, vocab]` or `[B, 1, vocab]`.
fn batched_logits_dims(logits: &MlxArray) -> (i32, i32) {
    let shape = logits.shape();
    match shape.as_slice() {
        [b, vocab] => (*b, *vocab),
        [b, 1, vocab] => (*b, *vocab),
        other => panic!("batched logits must be [B, vocab] or [B, 1, vocab], got {other:?}"),
    }
}

/// Greedy batched decode: per-row argmax over `[B, vocab]` (or `[B, 1, vocab]`)
/// logits → B token ids.
///
/// Casts to f32 then applies the last-axis MLX `argmax` — byte-for-byte the same
/// reduction the single-sequence greedy path runs on `[vocab]`, so each row's
/// token (tie-breaks included) matches a batch=1 greedy decode of that row.
/// Only the B result indices are read back, not the full logits.
pub fn argmax_batched(logits: &MlxArray) -> Vec<u32> {
    let (batch, vocab) = batched_logits_dims(logits);
    // Match the single path, which casts logits to f32 before argmax so tie /
    // rounding behavior is identical.
    let logits_f32 = astype(logits, MlxDtype::Float32, None);
    let logits_bv = reshape(&logits_f32, &[batch, vocab], None);
    let idx = argmax(&logits_bv, None); // last-axis → [B]
    eval(&[&idx]);
    idx.data_u32().to_vec()
}

/// General batched sampling with per-row parameters, RNG, and repetition
/// context. Reads `[B, vocab]` logits back to host and applies the single-row
/// [`sample_categorical`] to each row, so row `r`'s token is identical to a
/// batch=1 sample of row `r` with the same `params[r]`, `recent[r]`, and RNG.
///
/// `params`, `recent`, and `rngs` are indexed by row and must each have length
/// `B`. RNG state advances per row exactly as the single path would.
///
/// # Panics
/// If `params`, `recent`, or `rngs` length differs from the batch size.
pub fn sample_batched_host(
    logits: &MlxArray,
    params: &[MlxSamplingParams],
    recent: &[&[u32]],
    rngs: &mut [Xorshift64],
) -> Vec<u32> {
    let (batch, vocab) = batched_logits_dims(logits);
    let batch = batch as usize;
    let vocab = vocab as usize;
    assert_eq!(params.len(), batch, "params must have one entry per row");
    assert_eq!(recent.len(), batch, "recent must have one entry per row");
    assert_eq!(rngs.len(), batch, "rngs must have one entry per row");

    // Single host readback of the full logits (as f32, contiguous), then per-row
    // slicing — the same [vocab] slice each single-sequence decode would sample.
    let logits_f32 = astype(logits, MlxDtype::Float32, None);
    let logits_bv = reshape(&logits_f32, &[batch as i32, vocab as i32], None);
    eval(&[&logits_bv]);
    let flat = logits_bv.data_f32();

    (0..batch)
        .map(|row| {
            let row_logits = &flat[row * vocab..(row + 1) * vocab];
            sample_categorical(row_logits, params[row], recent[row], &mut rngs[row])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_arr(data: &[f32], shape: &[i32]) -> MlxArray {
        assert_eq!(data.len(), shape.iter().product::<i32>() as usize);
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    /// The oracle: `argmax_batched([B, vocab])` row `r` equals the single-row
    /// MLX `argmax` of row `r`'s logits. Distinct per-row maxima (no ties) so the
    /// expected index is unambiguous.
    #[test]
    fn argmax_batched_matches_per_row_single_argmax() {
        let vocab = 6i32;
        // Row r's max sits at column (2*r + 1) mod vocab, value 9.0.
        let rows = [1usize, 3usize, 5usize, 0usize];
        let mut data = Vec::new();
        for &peak in &rows {
            let mut row = vec![0.0f32; vocab as usize];
            for (j, v) in row.iter_mut().enumerate() {
                *v = (j as f32) * 0.1;
            }
            row[peak] = 9.0;
            data.extend_from_slice(&row);
        }
        let logits = f32_arr(&data, &[rows.len() as i32, vocab]);
        let got = argmax_batched(&logits);
        assert_eq!(got, rows.iter().map(|&r| r as u32).collect::<Vec<_>>());

        // Cross-check each row against a single-row MLX argmax (same kernel).
        for (r, &peak) in rows.iter().enumerate() {
            let row = &data[r * vocab as usize..(r + 1) * vocab as usize];
            let single = f32_arr(row, &[vocab]);
            let idx = argmax(&single, None);
            eval(&[&idx]);
            assert_eq!(idx.data_u32()[0], peak as u32);
            assert_eq!(got[r], peak as u32);
        }
    }

    /// `[B, 1, vocab]` (the raw LM-head decode shape) is accepted and reduced
    /// over the vocab axis like `[B, vocab]`.
    #[test]
    fn argmax_batched_accepts_b1v_shape() {
        let vocab = 4i32;
        let data = vec![
            0.0, 1.0, 0.5, 0.2, // row 0 → 1
            3.0, 0.1, 0.2, 0.3, // row 1 → 0
        ];
        let logits = f32_arr(&data, &[2, 1, vocab]);
        assert_eq!(argmax_batched(&logits), vec![1, 0]);
    }

    /// The oracle: `sample_batched_host` row `r` equals the single-row
    /// `sample_categorical` of row `r`'s logits with the same params, recent
    /// tokens, and RNG — covering a greedy row, a repetition-penalized row, and a
    /// temperature-sampled row whose RNG must advance independently.
    #[test]
    fn sample_batched_host_matches_per_row_sample() {
        let vocab = 5usize;
        let data = vec![
            0.1, 5.0, 1.0, 2.0, 0.5, // row 0
            4.0, 4.5, 0.2, 0.3, 0.1, // row 1
            1.0, 1.1, 0.9, 3.0, 0.4, // row 2
        ];
        let logits = f32_arr(&data, &[3, vocab as i32]);

        let params = [
            MlxSamplingParams::greedy(),
            MlxSamplingParams::greedy().with_repetition_penalty(2.0, None),
            MlxSamplingParams::new(0.8, 1.0, 0), // temperature-sampled
        ];
        let recent: [&[u32]; 3] = [&[], &[1], &[]];
        let seeds = [7u64, 11u64, 23u64];

        let mut rngs: Vec<Xorshift64> = seeds.iter().map(|&s| Xorshift64::new(s)).collect();
        let got = sample_batched_host(&logits, &params, &recent, &mut rngs);

        for row in 0..3 {
            let row_logits = &data[row * vocab..(row + 1) * vocab];
            let mut ref_rng = Xorshift64::new(seeds[row]);
            let want = sample_categorical(row_logits, params[row], recent[row], &mut ref_rng);
            assert_eq!(got[row], want, "row {row} sample differs from single path");
            // RNG state must have advanced exactly as the single path's did.
            assert_eq!(rngs[row].0, ref_rng.0, "row {row} RNG advanced differently");
        }
    }

    /// A greedy tie row resolves through `sample_batched_host` exactly as the
    /// single-row `sample_categorical` does (host argmax, last-max on ties) —
    /// which can differ from the GPU `argmax_batched` path, so the two entry
    /// points stay faithful to their respective single-sequence counterparts.
    #[test]
    fn sample_batched_host_greedy_tie_matches_single_path() {
        let vocab = 4usize;
        let data = vec![0.1, 0.1, 0.1, 0.1]; // all-tie row
        let logits = f32_arr(&data, &[1, vocab as i32]);
        let params = [MlxSamplingParams::greedy()];
        let recent: [&[u32]; 1] = [&[]];
        let mut rngs = vec![Xorshift64::new(1)];
        let got = sample_batched_host(&logits, &params, &recent, &mut rngs);

        let mut ref_rng = Xorshift64::new(1);
        let want = sample_categorical(&data, params[0], &[], &mut ref_rng);
        assert_eq!(got[0], want);
    }
}
