use mlx_sys::{
    MlxArray, MlxDtype, add, arange, eval, expand_dims, greater_equal, less, logical_and, reshape,
};

/// Build the boolean causal mask used by mlx-lm's `create_causal_mask`.
///
/// Shape is `[seq_len, offset + seq_len]`. `window_size` applies the same
/// sliding-window rule as mlx-lm: `linds >= rinds && linds < rinds + window`.
pub fn create_causal_mask(seq_len: usize, offset: usize, window_size: Option<usize>) -> MlxArray {
    let key_len = offset + seq_len;
    let rinds = arange(0.0, key_len as f64, 1.0, MlxDtype::Int32, None);
    let linds = if offset == 0 {
        rinds.clone()
    } else {
        arange(offset as f64, key_len as f64, 1.0, MlxDtype::Int32, None)
    };
    let linds = expand_dims(&linds, 1, None);
    let rinds = expand_dims(&rinds, 0, None);

    let causal = greater_equal(&linds, &rinds, None);
    if let Some(window_size) = window_size {
        let window = scalar_i32(window_size as i32);
        let upper = add(&rinds, &window, None);
        let sliding = less(&linds, &upper, None);
        logical_and(&causal, &sliding, None)
    } else {
        causal
    }
}

/// Boolean per-row validity mask for **batched decode** attention (Phase 1 of
/// batched MLX decode).
///
/// In a batched decode step the queries are `[B, n_heads, 1, head_dim]` and the
/// batched KV view (from [`crate::batched_kv_cache::BatchedKvCache::layer_view`])
/// is `[B, n_kv_heads, key_len, head_dim]`, where row `r` holds only
/// `valid_lengths[r]` real key positions — the remaining `key_len -
/// valid_lengths[r]` are ragged padding whose contents must not affect the
/// result. This returns a boolean mask `[B, 1, 1, key_len]` for MLX fast SDPA
/// ([`mlx_sys::ScaledDotProductAttentionMask::Array`]) whose entry
/// `[r, 0, 0, j]` is `true` (attend) iff `j < valid_lengths[r]`, broadcasting
/// over heads and the single query position. `true` = attend matches
/// [`create_causal_mask`]'s convention.
///
/// A single decode query needs no causal triangle — it is the newest position
/// and legitimately attends every real key — so per-row padding exclusion is the
/// only masking required. `valid_lengths[r]` is therefore the row's full KV
/// length including the just-appended current token.
///
/// The mask depends only on `valid_lengths` and `arange(key_len)`, not on the
/// model graph, so it is materialized here (one small `eval`, built once per
/// decode step and reused across all layers) and owns its data on return —
/// `from_raw_data` borrows `valid_lengths`, so it must be consumed before this
/// returns.
///
/// # Panics
/// If `valid_lengths` is empty, `key_len == 0`, or any length is 0 or exceeds
/// `key_len`. A zero-length row would make SDPA softmax over an all-masked row
/// (NaN), so it is rejected rather than silently produced.
// Not yet called from the decode path: it is wired into the batched runner in
// Phase 2. Exercised now only by the token-exact SDPA oracle in tests.
#[allow(dead_code)]
pub fn batched_decode_validity_mask(valid_lengths: &[usize], key_len: usize) -> MlxArray {
    assert!(
        !valid_lengths.is_empty(),
        "batched decode mask requires at least one row"
    );
    assert!(key_len > 0, "batched decode mask requires key_len > 0");
    let batch = valid_lengths.len() as i32;
    let lens_i32: Vec<i32> = valid_lengths
        .iter()
        .map(|&len| {
            assert!(
                len > 0 && len <= key_len,
                "row valid length {len} out of range 1..={key_len}"
            );
            len as i32
        })
        .collect();

    // positions `[1, 1, 1, key_len]` (broadcasts over batch/heads/query).
    let positions = arange(0.0, key_len as f64, 1.0, MlxDtype::Int32, None);
    let positions = reshape(&positions, &[1, 1, 1, key_len as i32], None);
    // lengths `[B, 1, 1, 1]` (broadcasts over the key axis).
    let lengths = MlxArray::from_raw_data(
        lens_i32.as_ptr().cast(),
        std::mem::size_of_val(lens_i32.as_slice()),
        &[batch, 1, 1, 1],
        MlxDtype::Int32,
    );
    let mask = less(&positions, &lengths, None);
    // Materialize while `lens_i32` is still alive so the mask owns its data and
    // does not dangle once the borrowed `valid_lengths` slice is gone.
    eval(&[&mask]);
    mask
}

/// Boolean SDPA validity mask for a **bounded-rollback rotating ring**
/// ([`crate::kv_cache::SlidingRingLayout`]).
///
/// The ring stores token `t` at slot `t % capacity` with
/// `capacity = window + slack`, so the newest resident token in slot `s`
/// after this forward appends `seq_len` tokens (`write_end = write_start +
/// seq_len`) is `t(s) = (write_end - 1) - ((write_end - 1 - s) mod capacity)`.
/// Query `i` (absolute position `p = write_start + i`) may attend slot `s`
/// iff that token is inside its sliding window and not in its causal future:
///
/// `t(s) >= 0 && t(s) > p - window && t(s) <= p`
///
/// The `t(s) >= 0` term is load-bearing: right after ring conversion the
/// `slack` newest slots may be zero-filled and never written, and a zero key
/// still contributes `exp(0)` softmax mass if left unmasked. Slots holding
/// expired tokens (older than any query's window) and slots holding tokens
/// rolled back by `trim_to` (their `t(s)` maps below `write_end - capacity`
/// under the post-trim `write_end`) fail the window term automatically.
///
/// Shape `[seq_len, capacity]`, `true` = attend — the same convention as
/// [`create_causal_mask`], broadcasting over batch and heads in SDPA. Built
/// on the host (`seq_len <= slack + 1`, so at most a few KB) and evaluated
/// before return so the array owns its data.
pub fn create_ring_sliding_mask(
    seq_len: usize,
    window: usize,
    capacity: usize,
    write_start: usize,
) -> MlxArray {
    assert!(seq_len > 0, "ring mask requires at least one query");
    assert!(
        capacity >= window && window > 0,
        "ring capacity {capacity} must contain the sliding window {window}"
    );
    let write_end = (write_start + seq_len) as i64;
    let capacity_i = capacity as i64;
    let mut data = vec![0u8; seq_len * capacity];
    for i in 0..seq_len {
        let p = (write_start + i) as i64;
        let row = &mut data[i * capacity..(i + 1) * capacity];
        for (s, keep) in row.iter_mut().enumerate() {
            let newest = write_end - 1;
            let t = newest - (newest - s as i64).rem_euclid(capacity_i);
            *keep = (t >= 0 && t > p - window as i64 && t <= p) as u8;
        }
    }
    let mask = MlxArray::from_raw_data(
        data.as_ptr(),
        data.len(),
        &[seq_len as i32, capacity as i32],
        MlxDtype::Bool,
    );
    // Materialize while `data` is alive so the mask owns its bytes.
    eval(&[&mask]);
    mask
}

pub(crate) fn scalar_i32(value: i32) -> MlxArray {
    MlxArray::from_raw_data(
        &value as *const i32 as *const u8,
        std::mem::size_of::<i32>(),
        &[1_i32],
        MlxDtype::Int32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::eval;

    fn mask_data(mask: &MlxArray) -> Vec<u8> {
        eval(&[mask]);
        let len = mask.nbytes();
        let ptr = mask.data_raw();
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    }

    #[test]
    fn causal_mask_matches_mlx_lm_lower_triangle() {
        let mask = create_causal_mask(4, 0, None);

        assert_eq!(mask.shape(), vec![4, 4]);
        assert_eq!(
            mask_data(&mask),
            vec![1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]
        );
    }

    #[test]
    fn causal_mask_applies_sliding_window_rule() {
        let mask = create_causal_mask(4, 0, Some(2));

        assert_eq!(mask.shape(), vec![4, 4]);
        assert_eq!(
            mask_data(&mask),
            vec![1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
        );
    }

    #[test]
    fn causal_mask_uses_cache_offset_for_key_length() {
        let mask = create_causal_mask(2, 3, None);

        assert_eq!(mask.shape(), vec![2, 5]);
        assert_eq!(mask_data(&mask), vec![1, 1, 1, 1, 0, 1, 1, 1, 1, 1]);
    }

    // ── Ring sliding mask ──

    /// Reference semantics: query at absolute position `p` attends exactly
    /// the tokens in `(p - window, p]` that exist (`>= 0`) and are still
    /// resident in the ring under end `write_start + seq`.
    fn ring_mask_reference(
        seq: usize,
        window: usize,
        capacity: usize,
        write_start: usize,
    ) -> Vec<u8> {
        let write_end = write_start + seq;
        let mut want = vec![0u8; seq * capacity];
        for i in 0..seq {
            let p = write_start + i;
            for t in p.saturating_sub(window - 1)..=p {
                // Token t is resident iff no newer token < write_end shares
                // its slot, i.e. t + capacity >= write_end.
                if t + capacity >= write_end {
                    want[i * capacity + t % capacity] = 1;
                }
            }
        }
        want
    }

    #[test]
    fn ring_mask_matches_ordered_sliding_semantics() {
        // (window, slack, write_start, seq): steady-state single-token,
        // steady-state verify, crossing forwards, wrap-around ends, and a
        // post-rollback re-forward (write_start rewound below a prior end).
        let cases = [
            (8usize, 4usize, 20usize, 1usize),
            (8, 4, 20, 4),
            (8, 4, 6, 3),    // crossing: write_start < window < write_end
            (8, 4, 11, 1),   // write_end == capacity boundary
            (8, 4, 12, 3),   // wraps past capacity
            (8, 4, 23, 2),   // arbitrary wrap phase
            (4, 3, 3, 2),    // small ring, crossing
            (16, 8, 100, 7), // gemma-shaped slack with max verify width
        ];
        for (window, slack, write_start, seq) in cases {
            let capacity = window + slack;
            let mask = create_ring_sliding_mask(seq, window, capacity, write_start);
            assert_eq!(mask.shape(), vec![seq as i32, capacity as i32]);
            assert_eq!(
                mask_data(&mask),
                ring_mask_reference(seq, window, capacity, write_start),
                "window={window} slack={slack} write_start={write_start} seq={seq}"
            );
        }
    }

    #[test]
    fn ring_mask_excludes_never_written_slots_at_crossing() {
        // First ring forward: window 8, slack 4, prompt of 8 tokens, one new
        // token at position 8. Slots for tokens 1..=8 are live; the three
        // remaining slots (which would decode to t < 0) must be masked even
        // though they sit "within" a naive window span — they hold zeros the
        // conversion never wrote, and an unmasked zero key still receives
        // exp(0) softmax mass.
        let mask = create_ring_sliding_mask(1, 8, 12, 8);
        let data = mask_data(&mask);
        let live: usize = data.iter().map(|&b| b as usize).sum();
        assert_eq!(live, 8, "exactly the window's tokens are attendable");
        for t in 1..=8usize {
            assert_eq!(data[t % 12], 1, "token {t} must be attendable");
        }
    }

    #[test]
    fn ring_mask_full_pure_ring_is_all_true() {
        // Pure ring (capacity == window) deep in decode: every slot holds a
        // live window token, so the mask is all-true — the geometry that
        // justifies pure rings running mask-free.
        let mask = create_ring_sliding_mask(1, 8, 8, 100);
        assert_eq!(mask_data(&mask), vec![1u8; 8]);
    }

    // ── Batched decode validity mask ──

    fn arr4(data: &[f32], b: i32, h: i32, t: i32, d: i32) -> MlxArray {
        assert_eq!(data.len(), (b * h * t * d) as usize);
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            &[b, h, t, d],
            MlxDtype::Float32,
        )
    }

    #[test]
    fn batched_decode_mask_has_expected_shape_and_bits() {
        // 2 rows, key_len 4, valid lengths 2 and 4.
        let mask = batched_decode_validity_mask(&[2, 4], 4);
        assert_eq!(mask.shape(), vec![2, 1, 1, 4]);
        // true(1) where j < valid_len[r]: row0 = 1,1,0,0 ; row1 = 1,1,1,1.
        assert_eq!(mask_data(&mask), vec![1, 1, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn batched_decode_mask_rejects_zero_length_row() {
        // An all-masked row would make SDPA softmax over nothing (NaN).
        let _ = batched_decode_validity_mask(&[0, 3], 3);
    }

    /// The oracle: masked batched SDPA over a ragged KV view produces, for each
    /// row, the same output as an unmasked SDPA over just that row's real keys —
    /// even though the padding key slots hold different garbage. This is the
    /// property that lets batched decode replace per-request decode.
    #[test]
    fn batched_decode_mask_isolates_per_row_attention() {
        use mlx_sys::{
            ScaledDotProductAttentionMask, scaled_dot_product_attention_with_mask, slice,
        };

        let (b, h, d, key_len) = (2usize, 1usize, 4usize, 3usize);
        let valid = [2usize, 3usize];
        let scale = 1.0f32 / (d as f32).sqrt();

        // Distinct values per (row, kind, token, dim); row 0's key/value slot at
        // position 2 (>= valid_len 0) is deliberate garbage that must be ignored.
        let q_data: Vec<f32> = (0..(b * h * d)).map(|i| (i as f32) * 0.03 - 0.2).collect();
        let mut k_data = Vec::new();
        let mut v_data = Vec::new();
        for row in 0..b {
            for _hd in 0..h {
                for t in 0..key_len {
                    for dim in 0..d {
                        // Garbage in row 0's padding slot (t == 2) is large/distinct.
                        let pad = row == 0 && t >= valid[0];
                        let base = (row * 100 + t * 10 + dim) as f32 * 0.01;
                        k_data.push(if pad { 999.0 + base } else { base });
                        v_data.push(if pad { -999.0 - base } else { base + 0.5 });
                    }
                }
            }
        }

        let q = arr4(&q_data, b as i32, h as i32, 1, d as i32);
        let k = arr4(&k_data, b as i32, h as i32, key_len as i32, d as i32);
        let v = arr4(&v_data, b as i32, h as i32, key_len as i32, d as i32);

        let mask = batched_decode_validity_mask(&valid, key_len);
        let masked = scaled_dot_product_attention_with_mask(
            &q,
            &k,
            &v,
            scale,
            ScaledDotProductAttentionMask::Array(&mask),
            None,
        );
        eval(&[&masked]);
        let masked_out = masked.data_f32().to_vec();

        // Reference: each row attends only its first `valid[r]` keys, unmasked.
        for row in 0..b {
            let vl = valid[row] as i32;
            let r = row as i32;
            let ones = [1, 1, 1, 1];
            let qr = slice(
                &q,
                &[r, 0, 0, 0],
                &[r + 1, h as i32, 1, d as i32],
                &ones,
                None,
            );
            let kr = slice(
                &k,
                &[r, 0, 0, 0],
                &[r + 1, h as i32, vl, d as i32],
                &ones,
                None,
            );
            let vr = slice(
                &v,
                &[r, 0, 0, 0],
                &[r + 1, h as i32, vl, d as i32],
                &ones,
                None,
            );
            let reference = scaled_dot_product_attention_with_mask(
                &qr,
                &kr,
                &vr,
                scale,
                ScaledDotProductAttentionMask::None,
                None,
            );
            eval(&[&reference]);
            let ref_out = reference.data_f32();
            for dim in 0..d {
                let got = masked_out[row * d + dim];
                let want = ref_out[dim];
                assert!(
                    (got - want).abs() < 1e-5,
                    "row {row} dim {dim}: masked {got} vs reference {want}"
                );
            }
        }
    }
}
