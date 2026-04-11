//! Dequantize Q5_0 blocks to f32.
//!
//! Q5_0 block layout (22 bytes -> 32 values):
//!   - 2 bytes: f16 scale (d)
//!   - 4 bytes: high bits (qh) — 1 bit per value, packed as u32
//!   - 16 bytes: 32 × 4-bit nibbles packed into 16 bytes (qs)
//!
//! Dequantize: output[j] = d * ((qs_nibble | qh_bit << 4) - 16)

const Q5_0_BLOCK_SIZE: usize = 32;
const Q5_0_BYTES_PER_BLOCK: usize = 22; // 2 (d) + 4 (qh) + 16 (qs)

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(Q5_0_BYTES_PER_BLOCK),
        "Q5_0 src length {} is not a multiple of block size {}",
        src.len(),
        Q5_0_BYTES_PER_BLOCK,
    );

    let n_blocks = src.len() / Q5_0_BYTES_PER_BLOCK;
    assert!(
        dst.len() >= n_blocks * Q5_0_BLOCK_SIZE,
        "Q5_0 dst too small: need {}, have {}",
        n_blocks * Q5_0_BLOCK_SIZE,
        dst.len(),
    );

    for block_idx in 0..n_blocks {
        let block = &src[block_idx * Q5_0_BYTES_PER_BLOCK..][..Q5_0_BYTES_PER_BLOCK];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();

        // High bits: 4 bytes → u32, one bit per value
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22]; // 16 bytes of packed 4-bit nibbles

        let out = &mut dst[block_idx * Q5_0_BLOCK_SIZE..][..Q5_0_BLOCK_SIZE];
        for j in 0..Q5_0_BLOCK_SIZE / 2 {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;

            let x0 = ((qs[j] as u32 & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((qs[j] as u32 >> 4) | xh_1) as i32 - 16;

            out[j] = x0 as f32 * d;
            out[j + Q5_0_BLOCK_SIZE / 2] = x1 as f32 * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(d: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q5_0_BYTES_PER_BLOCK);
        block.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(qs);
        block
    }

    #[test]
    fn test_q5_0_zeros() {
        // d=1.0, qh=0, all qs nibbles=0 → all values = (0 - 16) * 1.0 = -16.0
        let block = make_block(1.0, 0, &[0; 16]);
        let mut dst = vec![0.0f32; Q5_0_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert!((v - (-16.0)).abs() < 1e-3, "expected -16.0, got {v}");
        }
    }

    #[test]
    fn test_q5_0_midpoint() {
        // d=1.0, qh=0xFFFFFFFF (all high bits set), all qs=0
        // x0 = (0 | 0x10) - 16 = 0 → 0.0
        // x1 = (0 | 0x10) - 16 = 0 → 0.0
        let block = make_block(1.0, 0xFFFFFFFF, &[0; 16]);
        let mut dst = vec![999.0f32; Q5_0_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert!(v.abs() < 1e-3, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_q5_0_max_value() {
        // d=1.0, qh=0xFFFFFFFF, all qs=0xFF (nibble 0xF for both halves)
        // x0 = (0xF | 0x10) - 16 = 31 - 16 = 15
        // x1 = (0xF | 0x10) - 16 = 15
        let block = make_block(1.0, 0xFFFFFFFF, &[0xFF; 16]);
        let mut dst = vec![0.0f32; Q5_0_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert!((v - 15.0).abs() < 1e-3, "expected 15.0, got {v}");
        }
    }

    #[test]
    fn test_q5_0_scale() {
        // d=0.5, qh=0, all qs=0 → all values = (0 - 16) * 0.5 = -8.0
        let block = make_block(0.5, 0, &[0; 16]);
        let mut dst = vec![0.0f32; Q5_0_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert!((v - (-8.0)).abs() < 1e-3, "expected -8.0, got {v}");
        }
    }

    #[test]
    fn test_q5_0_multiple_blocks() {
        let b1 = make_block(1.0, 0xFFFFFFFF, &[0; 16]);
        let b2 = make_block(2.0, 0xFFFFFFFF, &[0; 16]);
        let mut src = Vec::new();
        src.extend_from_slice(&b1);
        src.extend_from_slice(&b2);
        let mut dst = vec![0.0f32; Q5_0_BLOCK_SIZE * 2];
        dequantize(&src, &mut dst);
        // Block 0: (0|0x10)-16 = 0, * 1.0 = 0
        for &v in &dst[..32] {
            assert!(v.abs() < 1e-3, "block 0: expected 0.0, got {v}");
        }
        // Block 1: (0|0x10)-16 = 0, * 2.0 = 0
        for &v in &dst[32..64] {
            assert!(v.abs() < 0.1, "block 1: expected 0.0, got {v}");
        }
    }
}
