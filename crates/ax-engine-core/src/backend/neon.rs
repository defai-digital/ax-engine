//! Fused dequant+matvec kernels using ARM NEON intrinsics.
//!
//! These kernels avoid the intermediate f32 buffer by dequantizing quantized
//! weights on-the-fly during the dot product. Each row's quantized blocks
//! are decoded directly into NEON registers and accumulated with the input
//! vector, so the dequantized values never touch memory.
//!
//! Supported formats: Q4_K, Q5_K, Q8_0, Q6_K.
//! Unsupported formats fall back to the default dequantize-then-matmul path.

use rayon::prelude::*;

/// Minimum row count to use parallel dispatch. Below this threshold,
/// sequential iteration avoids rayon scheduling overhead. All production
/// matmuls (smallest is K projection at 1024 rows) exceed this.
const PARALLEL_ROW_THRESHOLD: usize = 128;

// --- Q4_K constants ---
const Q4_K_BLOCK_SIZE: usize = 256;
const Q4_K_BYTES_PER_BLOCK: usize = 144;

// --- Q5_K constants ---
const Q5_K_BLOCK_SIZE: usize = 256;
const Q5_K_BYTES_PER_BLOCK: usize = 176;

// --- Q8_0 constants ---
const Q8_0_BLOCK_SIZE: usize = 32;
const Q8_0_BYTES_PER_BLOCK: usize = 34;

// --- Q6_K constants ---
const Q6_K_BLOCK_SIZE: usize = 256;
const Q6_K_BYTES_PER_BLOCK: usize = 210;

// ---------------------------------------------------------------------------
// Q4_K fused matvec
// ---------------------------------------------------------------------------

/// Fused dequant+matvec for Q4_K: c[i] = dot(dequant(A_row[i]), b) for each row.
pub fn fused_matvec_q4_k(a_quant: &[u8], b: &[f32], c: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(
        k % Q4_K_BLOCK_SIZE,
        0,
        "k must be a multiple of Q4_K block size"
    );
    let blocks_per_row = k / Q4_K_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_K_BYTES_PER_BLOCK;
    let expected_bytes = m * row_bytes;
    assert!(
        a_quant.len() >= expected_bytes,
        "Q4_K tensor data too small: have {} bytes, need {} (m={m}, k={k})",
        a_quant.len(),
        expected_bytes,
    );
    assert!(
        b.len() >= k,
        "Q4_K input vector too small: have {}, need {k}",
        b.len(),
    );

    if m >= PARALLEL_ROW_THRESHOLD {
        c[..m].par_iter_mut().enumerate().for_each(|(row, c_val)| {
            let row_data = &a_quant[row * row_bytes..(row + 1) * row_bytes];
            *c_val = fused_dot_q4_k(row_data, b, blocks_per_row);
        });
    } else {
        for row in 0..m {
            let row_data = &a_quant[row * row_bytes..(row + 1) * row_bytes];
            c[row] = fused_dot_q4_k(row_data, b, blocks_per_row);
        }
    }
}

/// Compute dot product of one Q4_K-quantized row with an f32 vector.
#[cfg(target_arch = "aarch64")]
fn fused_dot_q4_k(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    unsafe { fused_dot_q4_k_neon(quant_row, b, n_blocks) }
}

#[cfg(not(target_arch = "aarch64"))]
fn fused_dot_q4_k(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    fused_dot_q4_k_scalar(quant_row, b, n_blocks)
}

/// NEON-optimized Q4_K fused dot product.
///
/// For each Q4_K super-block (256 values, 8 sub-blocks of 32):
///   - Decomposes: value = d * sc * nibble - dmin * m
///   - For dot product, factorizes per sub-block:
///     sub_dot = d * sc * sum(q*b) - dmin * m * sum(b)
///   - Accumulates nibble*b and sum(b) in NEON, applies scales at sub-block end.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn fused_dot_q4_k_neon(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    use std::arch::aarch64::*;

    unsafe {
        let mut total = 0.0f32;

        for blk in 0..n_blocks {
            let block = &quant_row[blk * Q4_K_BYTES_PER_BLOCK..][..Q4_K_BYTES_PER_BLOCK];
            let b_base = blk * Q4_K_BLOCK_SIZE;

            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales = &block[4..16];
            let qs = &block[16..144];

            let mut q_idx = 0;
            let mut out_idx = 0;
            let mut is = 0;

            // 4 pairs of sub-blocks
            for _ in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);

                // Sub-block 1 (low nibbles): accumulate nibble*b and sum(b)
                let mut qb_acc1 = vdupq_n_f32(0.0);
                let mut b_sum1 = vdupq_n_f32(0.0);

                // Sub-block 2 (high nibbles)
                let mut qb_acc2 = vdupq_n_f32(0.0);
                let mut b_sum2 = vdupq_n_f32(0.0);

                let qs_base = qs.as_ptr().add(q_idx);
                let b_ptr1 = b.as_ptr().add(b_base + out_idx);
                let b_ptr2 = b.as_ptr().add(b_base + out_idx + 32);

                // Process 32 bytes -> 32 low + 32 high nibble values
                // 8 groups of 4
                for g in 0..8 {
                    // Load 4 quant bytes and unpack nibbles to i32
                    let byte0 = *qs_base.add(g * 4);
                    let byte1 = *qs_base.add(g * 4 + 1);
                    let byte2 = *qs_base.add(g * 4 + 2);
                    let byte3 = *qs_base.add(g * 4 + 3);

                    // Low nibbles -> sub-block 1
                    let lo_arr: [i32; 4] = [
                        (byte0 & 0x0F) as i32,
                        (byte1 & 0x0F) as i32,
                        (byte2 & 0x0F) as i32,
                        (byte3 & 0x0F) as i32,
                    ];
                    let flo = vcvtq_f32_s32(vld1q_s32(lo_arr.as_ptr()));
                    let bv1 = vld1q_f32(b_ptr1.add(g * 4));
                    qb_acc1 = vfmaq_f32(qb_acc1, flo, bv1);
                    b_sum1 = vaddq_f32(b_sum1, bv1);

                    // High nibbles -> sub-block 2
                    let hi_arr: [i32; 4] = [
                        (byte0 >> 4) as i32,
                        (byte1 >> 4) as i32,
                        (byte2 >> 4) as i32,
                        (byte3 >> 4) as i32,
                    ];
                    let fhi = vcvtq_f32_s32(vld1q_s32(hi_arr.as_ptr()));
                    let bv2 = vld1q_f32(b_ptr2.add(g * 4));
                    qb_acc2 = vfmaq_f32(qb_acc2, fhi, bv2);
                    b_sum2 = vaddq_f32(b_sum2, bv2);
                }

                // Combine: dot = d * sc * sum(q*b) - dmin * m * sum(b)
                let qb1 = vaddvq_f32(qb_acc1);
                let bs1 = vaddvq_f32(b_sum1);
                total += d * sc1 as f32 * qb1 - dmin * m1 as f32 * bs1;

                let qb2 = vaddvq_f32(qb_acc2);
                let bs2 = vaddvq_f32(b_sum2);
                total += d * sc2 as f32 * qb2 - dmin * m2 as f32 * bs2;

                out_idx += 64;
                q_idx += 32;
                is += 2;
            }
        }

        total
    }
}

/// Scalar fallback for Q4_K fused dot product.
#[allow(dead_code)]
fn fused_dot_q4_k_scalar(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    let mut total = 0.0f32;

    for blk in 0..n_blocks {
        let block = &quant_row[blk * Q4_K_BYTES_PER_BLOCK..][..Q4_K_BYTES_PER_BLOCK];
        let b_base = blk * Q4_K_BLOCK_SIZE;

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut q_idx = 0;
        let mut out_idx = 0;
        let mut is = 0;

        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);

            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;

            for l in 0..32 {
                let byte = qs[q_idx + l];
                let v1 = d1 * (byte & 0xF) as f32 - m1f;
                let v2 = d2 * (byte >> 4) as f32 - m2f;
                total += v1 * b[b_base + out_idx + l];
                total += v2 * b[b_base + out_idx + l + 32];
            }

            out_idx += 64;
            q_idx += 32;
            is += 2;
        }
    }

    total
}

// ---------------------------------------------------------------------------
// Q5_K fused matvec
// ---------------------------------------------------------------------------

/// Fused dequant+matvec for Q5_K: c[i] = dot(dequant(A_row[i]), b) for each row.
pub fn fused_matvec_q5_k(a_quant: &[u8], b: &[f32], c: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(
        k % Q5_K_BLOCK_SIZE,
        0,
        "k must be a multiple of Q5_K block size"
    );
    let blocks_per_row = k / Q5_K_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q5_K_BYTES_PER_BLOCK;
    let expected_bytes = m * row_bytes;
    assert!(
        a_quant.len() >= expected_bytes,
        "Q5_K tensor data too small: have {} bytes, need {} (m={m}, k={k})",
        a_quant.len(),
        expected_bytes,
    );
    assert!(
        b.len() >= k,
        "Q5_K input vector too small: have {}, need {k}",
        b.len(),
    );

    if m >= PARALLEL_ROW_THRESHOLD {
        c[..m].par_iter_mut().enumerate().for_each(|(row, c_val)| {
            let row_data = &a_quant[row * row_bytes..(row + 1) * row_bytes];
            *c_val = fused_dot_q5_k(row_data, b, blocks_per_row);
        });
    } else {
        for row in 0..m {
            let row_data = &a_quant[row * row_bytes..(row + 1) * row_bytes];
            c[row] = fused_dot_q5_k(row_data, b, blocks_per_row);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn fused_dot_q5_k(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    unsafe { fused_dot_q5_k_neon(quant_row, b, n_blocks) }
}

#[cfg(not(target_arch = "aarch64"))]
fn fused_dot_q5_k(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    fused_dot_q5_k_scalar(quant_row, b, n_blocks)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn fused_dot_q5_k_neon(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    use std::arch::aarch64::*;

    unsafe {
        let mut total = 0.0f32;

        for blk in 0..n_blocks {
            let block = &quant_row[blk * Q5_K_BYTES_PER_BLOCK..][..Q5_K_BYTES_PER_BLOCK];
            let b_base = blk * Q5_K_BLOCK_SIZE;

            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales = &block[4..16];
            let qh = &block[16..48];
            let qs = &block[48..176];

            for subblock in 0..8 {
                let (sc, m) = get_scale_min_k4(subblock, scales);
                let d_scaled = d * sc as f32;
                let d_min_scaled = dmin * m as f32;
                let qs_group = subblock / 2;
                let high_nibble = (subblock & 1) == 1;

                let mut qb_acc = vdupq_n_f32(0.0);
                let mut b_sum = vdupq_n_f32(0.0);
                let b_ptr = b.as_ptr().add(b_base + subblock * 32);

                for g in 0..8 {
                    let mut q_arr = [0i32; 4];
                    for (i, q_val) in q_arr.iter_mut().enumerate() {
                        let idx = g * 4 + i;
                        let ql_byte = qs[qs_group * 32 + idx];
                        let ql = if high_nibble {
                            (ql_byte >> 4) & 0x0F
                        } else {
                            ql_byte & 0x0F
                        };
                        let qh_bit = (qh[idx] >> subblock) & 0x01;
                        *q_val = (ql | (qh_bit << 4)) as i32;
                    }

                    let qv = vcvtq_f32_s32(vld1q_s32(q_arr.as_ptr()));
                    let bv = vld1q_f32(b_ptr.add(g * 4));
                    qb_acc = vfmaq_f32(qb_acc, qv, bv);
                    b_sum = vaddq_f32(b_sum, bv);
                }

                let qb = vaddvq_f32(qb_acc);
                let bs = vaddvq_f32(b_sum);
                total += d_scaled * qb - d_min_scaled * bs;
            }
        }

        total
    }
}

#[allow(dead_code)]
fn fused_dot_q5_k_scalar(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    let mut total = 0.0f32;

    for blk in 0..n_blocks {
        let block = &quant_row[blk * Q5_K_BYTES_PER_BLOCK..][..Q5_K_BYTES_PER_BLOCK];
        let b_base = blk * Q5_K_BLOCK_SIZE;

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        for subblock in 0..8 {
            let (sc, m) = get_scale_min_k4(subblock, scales);
            let d_scaled = d * sc as f32;
            let d_min_scaled = dmin * m as f32;
            let qs_group = subblock / 2;
            let high_nibble = (subblock & 1) == 1;
            let mut qb = 0.0f32;
            let mut bs = 0.0f32;

            for i in 0..32 {
                let ql_byte = qs[qs_group * 32 + i];
                let ql = if high_nibble {
                    (ql_byte >> 4) & 0x0F
                } else {
                    ql_byte & 0x0F
                };
                let qh_bit = (qh[i] >> subblock) & 0x01;
                let q = (ql | (qh_bit << 4)) as f32;
                let bv = b[b_base + subblock * 32 + i];
                qb += q * bv;
                bs += bv;
            }

            total += d_scaled * qb - d_min_scaled * bs;
        }
    }

    total
}

// ---------------------------------------------------------------------------
// Q8_0 fused matvec
// ---------------------------------------------------------------------------

/// Fused dequant+matvec for Q8_0: c[i] = dot(dequant(A_row[i]), b) for each row.
///
/// Q8_0 block layout (34 bytes → 32 values):
///   - 2 bytes: f16 scale (d)
///   - 32 bytes: 32 × i8 quantized values
///
/// Dequantize: value[i] = d * qs[i]
pub fn fused_matvec_q8_0(a_quant: &[u8], b: &[f32], c: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(
        k % Q8_0_BLOCK_SIZE,
        0,
        "k must be a multiple of Q8_0 block size"
    );
    let blocks_per_row = k / Q8_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q8_0_BYTES_PER_BLOCK;
    let expected_bytes = m * row_bytes;
    assert!(
        a_quant.len() >= expected_bytes,
        "Q8_0 tensor data too small: have {} bytes, need {} (m={m}, k={k})",
        a_quant.len(),
        expected_bytes,
    );
    assert!(
        b.len() >= k,
        "Q8_0 input vector too small: have {}, need {k}",
        b.len(),
    );

    if m >= PARALLEL_ROW_THRESHOLD {
        c[..m].par_iter_mut().enumerate().for_each(|(row, c_val)| {
            let row_data = &a_quant[row * row_bytes..(row + 1) * row_bytes];
            *c_val = fused_dot_q8_0(row_data, b, blocks_per_row);
        });
    } else {
        for row in 0..m {
            let row_data = &a_quant[row * row_bytes..(row + 1) * row_bytes];
            c[row] = fused_dot_q8_0(row_data, b, blocks_per_row);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn fused_dot_q8_0(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    unsafe { fused_dot_q8_0_neon(quant_row, b, n_blocks) }
}

#[cfg(not(target_arch = "aarch64"))]
fn fused_dot_q8_0(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    fused_dot_q8_0_scalar(quant_row, b, n_blocks)
}

/// NEON-optimized Q8_0 fused dot product.
///
/// For each Q8_0 block (32 values in 34 bytes):
///   1. Load f16 scale → f32
///   2. Load 32 i8 values in two vld1q_s8 loads
///   3. Widen: s8 → s16 → s32 → f32 using vmovl chain
///   4. FMA with b slice into block accumulator (8 groups of 4)
///   5. Multiply block accumulator by scale (deferred) → add to global acc
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn fused_dot_q8_0_neon(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    use std::arch::aarch64::*;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);

        let mut q_ptr = quant_row.as_ptr();
        let mut b_ptr = b.as_ptr();

        for _ in 0..n_blocks {
            // Load f16 scale → f32
            let d_bits = (q_ptr as *const u16).read_unaligned();
            let d = half::f16::from_bits(d_bits).to_f32();
            q_ptr = q_ptr.add(2);

            // Load 32 i8 values (two 16-element loads)
            let qs0 = vld1q_s8(q_ptr as *const i8);
            let qs1 = vld1q_s8((q_ptr as *const i8).add(16));
            q_ptr = q_ptr.add(32);

            let mut blk_acc = vdupq_n_f32(0.0);

            // Expand qs0: s8[0..16] → 4 × f32x4
            let s00 = vmovl_s8(vget_low_s8(qs0)); // s8[0..8] → s16[0..8]
            let s01 = vmovl_s8(vget_high_s8(qs0)); // s8[8..16] → s16[8..16]

            let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s00)));
            blk_acc = vfmaq_f32(blk_acc, f0, vld1q_f32(b_ptr));
            let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s00)));
            blk_acc = vfmaq_f32(blk_acc, f1, vld1q_f32(b_ptr.add(4)));
            let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s01)));
            blk_acc = vfmaq_f32(blk_acc, f2, vld1q_f32(b_ptr.add(8)));
            let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s01)));
            blk_acc = vfmaq_f32(blk_acc, f3, vld1q_f32(b_ptr.add(12)));

            // Expand qs1: s8[16..32] → 4 × f32x4
            let s10 = vmovl_s8(vget_low_s8(qs1));
            let s11 = vmovl_s8(vget_high_s8(qs1));

            let f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s10)));
            blk_acc = vfmaq_f32(blk_acc, f4, vld1q_f32(b_ptr.add(16)));
            let f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s10)));
            blk_acc = vfmaq_f32(blk_acc, f5, vld1q_f32(b_ptr.add(20)));
            let f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s11)));
            blk_acc = vfmaq_f32(blk_acc, f6, vld1q_f32(b_ptr.add(24)));
            let f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s11)));
            blk_acc = vfmaq_f32(blk_acc, f7, vld1q_f32(b_ptr.add(28)));

            // Deferred scale multiply
            acc = vfmaq_n_f32(acc, blk_acc, d);

            b_ptr = b_ptr.add(32);
        }

        vaddvq_f32(acc)
    }
}

#[allow(dead_code)]
fn fused_dot_q8_0_scalar(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    let mut sum = 0.0f32;
    for blk in 0..n_blocks {
        let block = &quant_row[blk * Q8_0_BYTES_PER_BLOCK..][..Q8_0_BYTES_PER_BLOCK];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..34];
        let b_off = blk * Q8_0_BLOCK_SIZE;
        let blk_sum: f32 = qs
            .iter()
            .zip(b[b_off..].iter())
            .map(|(&q, &bv)| (q as i8) as f32 * bv)
            .sum();
        sum += d * blk_sum;
    }
    sum
}

// ---------------------------------------------------------------------------
// Q6_K fused matvec
// ---------------------------------------------------------------------------

/// Fused dequant+matvec for Q6_K: c[i] = dot(dequant(A_row[i]), b) for each row.
///
/// Q6_K super-block layout (210 bytes → 256 values):
///   - 128 bytes: `ql[128]` — lower 4 bits of each 6-bit quant
///   - 64 bytes:  `qh[64]`  — upper 2 bits of each 6-bit quant
///   - 16 bytes:  `scales[16]` — per-sub-block i8 scales
///   - 2 bytes:   `d` — f16 super-block scale
///
/// Each 6-bit quant: q = (ql_nibble | (qh_bits << 4)) - 32
/// Dequantized value: d * scale * q
pub fn fused_matvec_q6_k(a_quant: &[u8], b: &[f32], c: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(
        k % Q6_K_BLOCK_SIZE,
        0,
        "k must be a multiple of Q6_K block size"
    );
    let blocks_per_row = k / Q6_K_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q6_K_BYTES_PER_BLOCK;
    let expected_bytes = m * row_bytes;
    assert!(
        a_quant.len() >= expected_bytes,
        "Q6_K tensor data too small: have {} bytes, need {} (m={m}, k={k})",
        a_quant.len(),
        expected_bytes,
    );
    assert!(
        b.len() >= k,
        "Q6_K input vector too small: have {}, need {k}",
        b.len(),
    );

    if m >= PARALLEL_ROW_THRESHOLD {
        c[..m].par_iter_mut().enumerate().for_each(|(row, c_val)| {
            let row_data = &a_quant[row * row_bytes..(row + 1) * row_bytes];
            *c_val = fused_dot_q6_k(row_data, b, blocks_per_row);
        });
    } else {
        for row in 0..m {
            let row_data = &a_quant[row * row_bytes..(row + 1) * row_bytes];
            c[row] = fused_dot_q6_k(row_data, b, blocks_per_row);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn fused_dot_q6_k(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    unsafe { fused_dot_q6_k_neon(quant_row, b, n_blocks) }
}

#[cfg(not(target_arch = "aarch64"))]
fn fused_dot_q6_k(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    fused_dot_q6_k_scalar(quant_row, b, n_blocks)
}

/// NEON-optimized Q6_K fused dot product.
///
/// Per super-block, processes two groups of 128 output values. Within each
/// group the inner loop (l=0..32) is unrolled 4× (l4 steps of 4). For each
/// 4-element step the 6-bit quant reconstruction is done in scalar (complex
/// bit-packing), then NEON FMA is used for the multiply-accumulate.
///
/// Scales split by `is = l / 16`: separate NEON accumulators for each
/// (is, quadrant) pair allow the scale multiplication to be deferred to the
/// end of each sub-group (8 multiplies per group instead of 32×4).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn fused_dot_q6_k_neon(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    use std::arch::aarch64::*;

    unsafe {
        let mut total = 0.0f32;

        for blk in 0..n_blocks {
            let block = &quant_row[blk * Q6_K_BYTES_PER_BLOCK..][..Q6_K_BYTES_PER_BLOCK];
            let b_base = blk * Q6_K_BLOCK_SIZE;

            let ql = &block[0..128];
            let qh = &block[128..192];
            let scales = &block[192..208];
            let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();

            let mut ql_idx = 0usize;
            let mut qh_idx = 0usize;
            let mut sc_idx = 0usize;
            let mut out_idx = 0usize;

            // Two groups of 128 values each
            for _group in 0..2 {
                // 8 NEON f32x4 accumulators: [is=0|is=1] × [quadrant 0..4]
                // is=0: l=0..15, is=1: l=16..31 — deferred scale application
                let mut acc00 = vdupq_n_f32(0.0);
                let mut acc01 = vdupq_n_f32(0.0);
                let mut acc02 = vdupq_n_f32(0.0);
                let mut acc03 = vdupq_n_f32(0.0);
                let mut acc10 = vdupq_n_f32(0.0);
                let mut acc11 = vdupq_n_f32(0.0);
                let mut acc12 = vdupq_n_f32(0.0);
                let mut acc13 = vdupq_n_f32(0.0);

                // Process 32 inner iterations in steps of 4
                for l4 in 0..8usize {
                    let l = l4 * 4;
                    let is = l / 16; // 0 for l<16, 1 for l>=16

                    // Scalar 6-bit reconstruction for 4 consecutive l values.
                    // Each qh byte covers 4 quant values via 2-bit fields.
                    let mut q1_arr = [0i32; 4];
                    let mut q2_arr = [0i32; 4];
                    let mut q3_arr = [0i32; 4];
                    let mut q4_arr = [0i32; 4];
                    for i in 0..4 {
                        let ql_a = ql[ql_idx + l + i];
                        let ql_b = ql[ql_idx + l + i + 32];
                        let qh_v = qh[qh_idx + l + i];
                        q1_arr[i] = ((ql_a & 0xF) | ((qh_v & 3) << 4)) as i32 - 32;
                        q2_arr[i] = ((ql_b & 0xF) | (((qh_v >> 2) & 3) << 4)) as i32 - 32;
                        q3_arr[i] = ((ql_a >> 4) | (((qh_v >> 4) & 3) << 4)) as i32 - 32;
                        q4_arr[i] = ((ql_b >> 4) | (((qh_v >> 6) & 3) << 4)) as i32 - 32;
                    }

                    // Convert to f32x4 via s32
                    let f1 = vcvtq_f32_s32(vld1q_s32(q1_arr.as_ptr()));
                    let f2 = vcvtq_f32_s32(vld1q_s32(q2_arr.as_ptr()));
                    let f3 = vcvtq_f32_s32(vld1q_s32(q3_arr.as_ptr()));
                    let f4 = vcvtq_f32_s32(vld1q_s32(q4_arr.as_ptr()));

                    // Load 4 b values per quadrant (strided by 32 in output)
                    let b_ptr = b.as_ptr().add(b_base + out_idx);
                    let bv1 = vld1q_f32(b_ptr.add(l));
                    let bv2 = vld1q_f32(b_ptr.add(32 + l));
                    let bv3 = vld1q_f32(b_ptr.add(64 + l));
                    let bv4 = vld1q_f32(b_ptr.add(96 + l));

                    // Accumulate into the right is-bucket (branch-free due to compile-time loop)
                    if is == 0 {
                        acc00 = vfmaq_f32(acc00, f1, bv1);
                        acc01 = vfmaq_f32(acc01, f2, bv2);
                        acc02 = vfmaq_f32(acc02, f3, bv3);
                        acc03 = vfmaq_f32(acc03, f4, bv4);
                    } else {
                        acc10 = vfmaq_f32(acc10, f1, bv1);
                        acc11 = vfmaq_f32(acc11, f2, bv2);
                        acc12 = vfmaq_f32(acc12, f3, bv3);
                        acc13 = vfmaq_f32(acc13, f4, bv4);
                    }
                }

                // Apply sub-block scales (deferred — 8 multiplies per group)
                let sc00 = scales[sc_idx] as i8 as f32;
                let sc01 = scales[sc_idx + 2] as i8 as f32;
                let sc02 = scales[sc_idx + 4] as i8 as f32;
                let sc03 = scales[sc_idx + 6] as i8 as f32;
                let sc10 = scales[sc_idx + 1] as i8 as f32;
                let sc11 = scales[sc_idx + 3] as i8 as f32;
                let sc12 = scales[sc_idx + 5] as i8 as f32;
                let sc13 = scales[sc_idx + 7] as i8 as f32;

                total += d * sc00 * vaddvq_f32(acc00);
                total += d * sc01 * vaddvq_f32(acc01);
                total += d * sc02 * vaddvq_f32(acc02);
                total += d * sc03 * vaddvq_f32(acc03);
                total += d * sc10 * vaddvq_f32(acc10);
                total += d * sc11 * vaddvq_f32(acc11);
                total += d * sc12 * vaddvq_f32(acc12);
                total += d * sc13 * vaddvq_f32(acc13);

                out_idx += 128;
                ql_idx += 64;
                qh_idx += 32;
                sc_idx += 8;
            }
        }

        total
    }
}

#[allow(dead_code)]
fn fused_dot_q6_k_scalar(quant_row: &[u8], b: &[f32], n_blocks: usize) -> f32 {
    let mut total = 0.0f32;

    for blk in 0..n_blocks {
        let block = &quant_row[blk * Q6_K_BYTES_PER_BLOCK..][..Q6_K_BYTES_PER_BLOCK];
        let b_base = blk * Q6_K_BLOCK_SIZE;

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();

        let mut ql_idx = 0usize;
        let mut qh_idx = 0usize;
        let mut sc_idx = 0usize;
        let mut out_idx = 0usize;

        for _group in 0..2 {
            for l in 0..32usize {
                let is = l / 16;
                let sc1 = scales[sc_idx + is] as i8 as f32;
                let sc2 = scales[sc_idx + is + 2] as i8 as f32;
                let sc3 = scales[sc_idx + is + 4] as i8 as f32;
                let sc4 = scales[sc_idx + is + 6] as i8 as f32;

                let ql_a = ql[ql_idx + l];
                let ql_b = ql[ql_idx + l + 32];
                let qh_v = qh[qh_idx + l];

                let q1 = ((ql_a & 0xF) | ((qh_v & 3) << 4)) as i32 - 32;
                let q2 = ((ql_b & 0xF) | (((qh_v >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql_a >> 4) | (((qh_v >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql_b >> 4) | (((qh_v >> 6) & 3) << 4)) as i32 - 32;

                total += d * sc1 * q1 as f32 * b[b_base + out_idx + l];
                total += d * sc2 * q2 as f32 * b[b_base + out_idx + l + 32];
                total += d * sc3 * q3 as f32 * b[b_base + out_idx + l + 64];
                total += d * sc4 * q4 as f32 * b[b_base + out_idx + l + 96];
            }

            out_idx += 128;
            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
        }
    }

    total
}

// ---------------------------------------------------------------------------
// Q4_K scale/min extraction (inlined from quant::q4_k to keep this
// module self-contained).
// ---------------------------------------------------------------------------

/// Extract 6-bit scale and min values from the packed Q4_K scales array.
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    debug_assert!(j < 8, "Q4_K scale index out of range: {j}");
    if j < 4 {
        let sc = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (sc, m)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Q4_K tests ---

    fn make_q4_k_block_simple(d: f32, dmin: f32, sc: u8, m: u8, nibble: u8) -> Vec<u8> {
        let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];
        // Scales for sub-blocks 0-3
        block[4] = sc;
        block[5] = sc;
        block[6] = sc;
        block[7] = sc;
        // Mins for sub-blocks 0-3
        block[8] = m;
        block[9] = m;
        block[10] = m;
        block[11] = m;
        // Pack nibble into both lo and hi
        block[16..144].fill((nibble << 4) | nibble);
        block
    }

    #[test]
    fn test_fused_dot_q4_k_zeros() {
        let block = make_q4_k_block_simple(0.0, 0.0, 1, 0, 5);
        let b = [1.0f32; 256];
        let result = fused_dot_q4_k(&block, &b, 1);
        assert!(result.abs() < 1e-4, "expected 0, got {result}");
    }

    #[test]
    fn test_fused_dot_q4_k_matches_dequant() {
        let block = make_q4_k_block_simple(1.0, 0.5, 2, 1, 7);
        let mut b = [0.0f32; 256];
        for (i, val) in b.iter_mut().enumerate() {
            *val = ((i % 17) as f32 - 8.0) * 0.1;
        }

        let mut dequanted = [0.0f32; 256];
        crate::quant::q4_k::dequantize(&block, &mut dequanted);
        let ref_dot: f32 = dequanted.iter().zip(b.iter()).map(|(a, b)| a * b).sum();

        let fused_result = fused_dot_q4_k(&block, &b, 1);
        assert!(
            (fused_result - ref_dot).abs() < 1.0,
            "fused={fused_result}, ref={ref_dot}"
        );
    }

    #[test]
    fn test_fused_matvec_q4_k_single_row() {
        let block = make_q4_k_block_simple(1.0, 0.0, 1, 0, 5);
        let b = [1.0f32; 256];
        let mut c = [0.0f32; 1];
        fused_matvec_q4_k(&block, &b, &mut c, 1, 256);

        // Sub-blocks 0-3: 128 values * 5.0 = 640.0
        // Sub-blocks 4-7: scale=0, so 0.0
        assert!((c[0] - 640.0).abs() < 1.0, "expected 640, got {}", c[0]);
    }

    #[test]
    fn test_fused_dot_q4_k_scalar_matches_neon() {
        let block = make_q4_k_block_simple(2.0, 0.25, 3, 2, 9);
        let mut b = [0.0f32; 256];
        for (i, val) in b.iter_mut().enumerate() {
            *val = (i as f32 + 1.0) * 0.01;
        }

        let scalar = fused_dot_q4_k_scalar(&block, &b, 1);
        let neon = fused_dot_q4_k(&block, &b, 1);

        assert!((scalar - neon).abs() < 1.0, "scalar={scalar}, neon={neon}");
    }

    #[test]
    #[should_panic(expected = "Q4_K tensor data too small")]
    fn test_fused_matvec_q4_k_validates_data_size() {
        let too_small = [0u8; 10]; // way too small for any valid Q4_K data
        let b = [1.0f32; 256];
        let mut c = [0.0f32; 2];
        fused_matvec_q4_k(&too_small, &b, &mut c, 2, 256);
    }

    #[test]
    #[should_panic(expected = "Q4_K input vector too small")]
    fn test_fused_matvec_q4_k_validates_input_size() {
        let block = make_q4_k_block_simple(1.0, 0.0, 1, 0, 5);
        let b_too_small = [1.0f32; 128]; // need 256
        let mut c = [0.0f32; 1];
        fused_matvec_q4_k(&block, &b_too_small, &mut c, 1, 256);
    }

    // --- Q8_0 tests ---

    fn make_q8_0_block(d: f32, qs: i8) -> [u8; Q8_0_BYTES_PER_BLOCK] {
        let mut block = [0u8; Q8_0_BYTES_PER_BLOCK];
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2..34].fill(qs as u8);
        block
    }

    #[test]
    fn test_fused_dot_q8_0_zeros() {
        let block = make_q8_0_block(1.0, 0);
        let b = [1.0f32; 32];
        let result = fused_dot_q8_0(&block, &b, 1);
        assert!(result.abs() < 1e-6, "expected 0, got {result}");
    }

    #[test]
    fn test_fused_dot_q8_0_unit() {
        // d=1.0, qs=2, b=1.0 → dot = 32 * 1.0 * 2 = 64.0
        let block = make_q8_0_block(1.0, 2);
        let b = [1.0f32; 32];
        let result = fused_dot_q8_0(&block, &b, 1);
        assert!((result - 64.0).abs() < 0.1, "expected 64.0, got {result}");
    }

    #[test]
    fn test_fused_dot_q8_0_negative_qs() {
        // d=1.0, qs=-3, b=1.0 → dot = 32 * (-3) = -96.0
        let block = make_q8_0_block(1.0, -3);
        let b = [1.0f32; 32];
        let result = fused_dot_q8_0(&block, &b, 1);
        assert!(
            (result - (-96.0)).abs() < 0.1,
            "expected -96.0, got {result}"
        );
    }

    #[test]
    fn test_fused_dot_q8_0_matches_dequant() {
        let mut block = make_q8_0_block(0.5, 0);
        // Set varied qs values
        for i in 0..32 {
            block[2 + i] = ((i as i32 - 16) as i8) as u8;
        }
        let mut b = [0.0f32; 32];
        for (i, v) in b.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.1;
        }

        let mut dequanted = [0.0f32; 32];
        crate::quant::q8_0::dequantize(&block, &mut dequanted);
        let ref_dot: f32 = dequanted.iter().zip(b.iter()).map(|(a, b)| a * b).sum();

        let fused = fused_dot_q8_0(&block, &b, 1);
        assert!(
            (fused - ref_dot).abs() < 0.01,
            "fused={fused}, ref={ref_dot}"
        );
    }

    #[test]
    fn test_fused_dot_q8_0_scalar_matches_neon() {
        let mut block = make_q8_0_block(1.5, 0);
        for i in 0..32 {
            block[2 + i] = ((i as i32 % 7 - 3) as i8) as u8;
        }
        let mut b = [0.0f32; 32];
        for (i, v) in b.iter_mut().enumerate() {
            *v = (i as f32 - 16.0) * 0.05;
        }
        let scalar = fused_dot_q8_0_scalar(&block, &b, 1);
        let neon = fused_dot_q8_0(&block, &b, 1);
        assert!((scalar - neon).abs() < 0.01, "scalar={scalar}, neon={neon}");
    }

    #[test]
    fn test_fused_matvec_q8_0_single_row() {
        let block = make_q8_0_block(1.0, 1);
        let b = [1.0f32; 32];
        let mut c = [0.0f32; 1];
        fused_matvec_q8_0(&block, &b, &mut c, 1, 32);
        assert!((c[0] - 32.0).abs() < 0.1, "expected 32.0, got {}", c[0]);
    }

    #[test]
    fn test_fused_matvec_q8_0_multi_row() {
        let block1 = make_q8_0_block(1.0, 1); // dot = 32*1 = 32
        let block2 = make_q8_0_block(2.0, 1); // dot = 32*2 = 64
        let mut quant = Vec::new();
        quant.extend_from_slice(&block1);
        quant.extend_from_slice(&block2);

        let b = [1.0f32; 32];
        let mut c = [0.0f32; 2];
        fused_matvec_q8_0(&quant, &b, &mut c, 2, 32);

        assert!((c[0] - 32.0).abs() < 0.1, "row0: got {}", c[0]);
        assert!((c[1] - 64.0).abs() < 0.1, "row1: got {}", c[1]);
    }

    #[test]
    #[should_panic(expected = "Q8_0 tensor data too small")]
    fn test_fused_matvec_q8_0_validates_data_size() {
        let too_small = [0u8; 10];
        let b = [1.0f32; 32];
        let mut c = [0.0f32; 1];
        fused_matvec_q8_0(&too_small, &b, &mut c, 1, 32);
    }

    #[test]
    #[should_panic(expected = "Q8_0 input vector too small")]
    fn test_fused_matvec_q8_0_validates_input_size() {
        let block = make_q8_0_block(1.0, 1);
        let b_too_small = [1.0f32; 16]; // need 32
        let mut c = [0.0f32; 1];
        fused_matvec_q8_0(&block, &b_too_small, &mut c, 1, 32);
    }

    // --- Q6_K tests ---

    fn make_q6_k_block_zeros_d(d: f32) -> Vec<u8> {
        let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];
        block
    }

    #[test]
    fn test_fused_dot_q6_k_zero_d() {
        let block = make_q6_k_block_zeros_d(0.0);
        let b = [1.0f32; 256];
        let result = fused_dot_q6_k(&block, &b, 1);
        assert!(result.abs() < 1e-4, "expected 0, got {result}");
    }

    #[test]
    fn test_fused_dot_q6_k_matches_dequant() {
        let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];

        // d = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];

        // All scales = 1
        block[192..208].fill(1);

        // Set ql to varied values and qh to give q=5-32=-27 for first value
        for (i, b) in block[..128].iter_mut().enumerate() {
            *b = (i % 16) as u8; // varied lower nibbles
        }
        block[128..192].fill(0xAA); // upper 2 bits = 2 for all quants

        let mut b = [0.0f32; 256];
        for (i, v) in b.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 0.1;
        }

        let mut dequanted = [0.0f32; 256];
        crate::quant::q6_k::dequantize(&block, &mut dequanted);
        let ref_dot: f32 = dequanted.iter().zip(b.iter()).map(|(a, b)| a * b).sum();

        let fused = fused_dot_q6_k(&block, &b, 1);
        assert!(
            (fused - ref_dot).abs() < 1.0,
            "fused={fused}, ref={ref_dot}"
        );
    }

    #[test]
    fn test_fused_dot_q6_k_scalar_matches_neon() {
        let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];
        let d_bytes = half::f16::from_f32(2.0).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];
        block[192..208].fill(3);
        for (i, b) in block[..128].iter_mut().enumerate() {
            *b = ((i * 7 + 3) % 16) as u8;
        }
        for (i, b) in block[128..192].iter_mut().enumerate() {
            *b = ((i * 5 + 1) % 4) as u8;
        }

        let mut b = [0.0f32; 256];
        for (i, v) in b.iter_mut().enumerate() {
            *v = (i as f32 - 128.0) * 0.01;
        }

        let scalar = fused_dot_q6_k_scalar(&block, &b, 1);
        let neon = fused_dot_q6_k(&block, &b, 1);
        assert!((scalar - neon).abs() < 1.0, "scalar={scalar}, neon={neon}");
    }

    #[test]
    fn test_fused_matvec_q6_k_single_row() {
        let block = make_q6_k_block_zeros_d(0.0);
        let b = [1.0f32; 256];
        let mut c = [0.0f32; 1];
        fused_matvec_q6_k(&block, &b, &mut c, 1, 256);
        assert!(c[0].abs() < 1e-4, "expected 0, got {}", c[0]);
    }

    #[test]
    #[should_panic(expected = "Q6_K tensor data too small")]
    fn test_fused_matvec_q6_k_validates_data_size() {
        let too_small = [0u8; 10];
        let b = [1.0f32; 256];
        let mut c = [0.0f32; 1];
        fused_matvec_q6_k(&too_small, &b, &mut c, 1, 256);
    }

    #[test]
    #[should_panic(expected = "Q6_K input vector too small")]
    fn test_fused_matvec_q6_k_validates_input_size() {
        let block = make_q6_k_block_zeros_d(1.0);
        let b_too_small = [1.0f32; 128]; // need 256
        let mut c = [0.0f32; 1];
        fused_matvec_q6_k(&block, &b_too_small, &mut c, 1, 256);
    }
}
