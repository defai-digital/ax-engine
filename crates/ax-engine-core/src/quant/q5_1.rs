//! Dequantize Q5_1 blocks to f32.
//!
//! Q5_1 block layout (24 bytes -> 32 values):
//!   - 2 bytes: f16 scale (d)
//!   - 2 bytes: f16 minimum (m)
//!   - 4 bytes: high bits (qh) — 1 bit per value, packed as u32
//!   - 16 bytes: 32 × 4-bit nibbles packed into 16 bytes (qs)
//!
//! Dequantize: output[j] = d * (qs_nibble | qh_bit << 4) + m

const Q5_1_BLOCK_SIZE: usize = 32;
const Q5_1_BYTES_PER_BLOCK: usize = 24; // 2 (d) + 2 (m) + 4 (qh) + 16 (qs)

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(Q5_1_BYTES_PER_BLOCK),
        "Q5_1 src length {} is not a multiple of block size {}",
        src.len(),
        Q5_1_BYTES_PER_BLOCK,
    );

    let n_blocks = src.len() / Q5_1_BYTES_PER_BLOCK;
    assert!(
        dst.len() >= n_blocks * Q5_1_BLOCK_SIZE,
        "Q5_1 dst too small: need {}, have {}",
        n_blocks * Q5_1_BLOCK_SIZE,
        dst.len(),
    );

    for block_idx in 0..n_blocks {
        let block = &src[block_idx * Q5_1_BYTES_PER_BLOCK..][..Q5_1_BYTES_PER_BLOCK];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();

        // High bits: 4 bytes → u32, one bit per value
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24]; // 16 bytes of packed 4-bit nibbles

        let out = &mut dst[block_idx * Q5_1_BLOCK_SIZE..][..Q5_1_BLOCK_SIZE];
        for j in 0..Q5_1_BLOCK_SIZE / 2 {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;

            let x0 = (qs[j] as u32 & 0x0F) | xh_0;
            let x1 = (qs[j] as u32 >> 4) | xh_1;

            out[j] = x0 as f32 * d + m;
            out[j + Q5_1_BLOCK_SIZE / 2] = x1 as f32 * d + m;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(d: f32, m: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q5_1_BYTES_PER_BLOCK);
        block.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        block.extend_from_slice(&half::f16::from_f32(m).to_le_bytes());
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    #[test]
    fn test_q5_1_zeros() {
        // d=1.0, m=0.0, qh=0, all qs=0 → all values = 1.0 * 0 + 0.0 = 0.0
        let block = make_block(1.0, 0.0, 0, &[0; 16]);
        let mut dst = vec![999.0f32; Q5_1_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert!((v - 0.0).abs() < 1e-3, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_q5_1_with_min() {
        // d=1.0, m=-5.0, qh=0, all qs=0 → all values = 1.0 * 0 + (-5.0) = -5.0
        let block = make_block(1.0, -5.0, 0, &[0; 16]);
        let mut dst = vec![999.0f32; Q5_1_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert!((v - (-5.0)).abs() < 1e-2, "expected -5.0, got {v}");
        }
    }

    #[test]
    fn test_q5_1_high_bits() {
        // d=1.0, m=0.0, qh=0xFFFFFFFF (all high bits set), all qs=0
        // x0 = (0 | 0x10) = 16 → 16.0 + 0.0 = 16.0
        let block = make_block(1.0, 0.0, 0xFFFFFFFF, &[0; 16]);
        let mut dst = vec![999.0f32; Q5_1_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert!((v - 16.0).abs() < 1e-3, "expected 16.0, got {v}");
        }
    }

    #[test]
    fn test_q5_1_max_value() {
        // d=1.0, m=0.0, qh=0xFFFFFFFF, all qs=0xFF (both nibbles = 15)
        // x0 = (15 | 0x10) = 31 → 31.0
        let block = make_block(1.0, 0.0, 0xFFFFFFFF, &[0xFF; 16]);
        let mut dst = vec![999.0f32; Q5_1_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert!((v - 31.0).abs() < 1e-3, "expected 31.0, got {v}");
        }
    }
}
