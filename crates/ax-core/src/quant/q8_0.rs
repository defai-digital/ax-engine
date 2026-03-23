//! Dequantize Q8_0 blocks to f32.
//!
//! Q8_0 block layout (34 bytes -> 32 values):
//!   - 2 bytes: f16 scale (d)
//!   - 32 bytes: 32 × i8 quantized values
//!
//! Dequantize: output[i] = d * qs[i]

const Q8_0_BLOCK_SIZE: usize = 32;
const Q8_0_BYTES_PER_BLOCK: usize = 34;

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(Q8_0_BYTES_PER_BLOCK),
        "Q8_0 src length {} is not a multiple of block size {}",
        src.len(),
        Q8_0_BYTES_PER_BLOCK,
    );

    let n_blocks = src.len() / Q8_0_BYTES_PER_BLOCK;
    assert!(
        dst.len() >= n_blocks * Q8_0_BLOCK_SIZE,
        "Q8_0 dst too small: need {}, have {}",
        n_blocks * Q8_0_BLOCK_SIZE,
        dst.len(),
    );

    for block_idx in 0..n_blocks {
        let block = &src[block_idx * Q8_0_BYTES_PER_BLOCK..][..Q8_0_BYTES_PER_BLOCK];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..34]; // 32 signed i8 values

        let out = &mut dst[block_idx * Q8_0_BLOCK_SIZE..][..Q8_0_BLOCK_SIZE];
        for i in 0..Q8_0_BLOCK_SIZE {
            out[i] = d * (qs[i] as i8) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_0_zeros() {
        // d=1.0, all qs=0 → all zeros
        let mut block = [0u8; Q8_0_BYTES_PER_BLOCK];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        // qs already all 0

        let mut dst = vec![999.0f32; Q8_0_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_q8_0_positive_scale() {
        // d=0.5, all qs=2 → all values = 0.5 * 2 = 1.0
        let mut block = [0u8; Q8_0_BYTES_PER_BLOCK];
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2..34].fill(2); // i8 value 2

        let mut dst = vec![0.0f32; Q8_0_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert!((v - 1.0).abs() < 1e-3, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_q8_0_negative_values() {
        // d=1.0, qs=-1 (0xFF as i8) → all values = -1.0
        let mut block = [0u8; Q8_0_BYTES_PER_BLOCK];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2..34].fill(0xFF); // i8 -1

        let mut dst = vec![0.0f32; Q8_0_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert!((v - (-1.0)).abs() < 1e-3, "expected -1.0, got {v}");
        }
    }

    #[test]
    fn test_q8_0_known_values() {
        // d=2.0, first qs=10, second qs=-5
        let mut block = [0u8; Q8_0_BYTES_PER_BLOCK];
        let d_bytes = half::f16::from_f32(2.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2] = 10u8; // i8 10
        block[3] = (-5i8) as u8; // i8 -5

        let mut dst = vec![0.0f32; Q8_0_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        assert!((dst[0] - 20.0).abs() < 0.1, "expected 20.0, got {}", dst[0]);
        assert!(
            (dst[1] - (-10.0)).abs() < 0.1,
            "expected -10.0, got {}",
            dst[1]
        );
    }

    #[test]
    fn test_q8_0_multiple_blocks() {
        // 2 blocks
        let mut src = vec![0u8; Q8_0_BYTES_PER_BLOCK * 2];

        // Block 0: d=1.0, all qs=1 → all 1.0
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        src[0] = d_bytes[0];
        src[1] = d_bytes[1];
        src[2..34].fill(1);

        // Block 1: d=3.0, all qs=2 → all 6.0
        let d_bytes = half::f16::from_f32(3.0).to_le_bytes();
        src[34] = d_bytes[0];
        src[35] = d_bytes[1];
        src[36..68].fill(2);

        let mut dst = vec![0.0f32; Q8_0_BLOCK_SIZE * 2];
        dequantize(&src, &mut dst);

        for &v in &dst[..32] {
            assert!((v - 1.0).abs() < 1e-3, "block 0: expected 1.0, got {v}");
        }
        for &v in &dst[32..64] {
            assert!((v - 6.0).abs() < 0.1, "block 1: expected 6.0, got {v}");
        }
    }
}
