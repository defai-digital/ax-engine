//! Dequantize Q4_0 blocks to f32.
//!
//! Q4_0 block layout (18 bytes -> 32 values):
//!   - 2 bytes: f16 scale (d)
//!   - 16 bytes: 32 x 4-bit quantized values (packed, 2 per byte)
//!
//! Each 4-bit value is unsigned [0, 15]. To dequantize:
//!   output = (q - 8) * d
//!
//! Byte packing: for block of 32 values in 16 bytes,
//!   byte[i] stores: low nibble = element[i], high nibble = element[i+16]
//!   where i in 0..16

const Q4_0_BLOCK_SIZE: usize = 32;
const Q4_0_BYTES_PER_BLOCK: usize = 18;

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(Q4_0_BYTES_PER_BLOCK),
        "Q4_0 src length {} is not a multiple of block size {}",
        src.len(),
        Q4_0_BYTES_PER_BLOCK
    );

    let n_blocks = src.len() / Q4_0_BYTES_PER_BLOCK;
    let n_values = n_blocks * Q4_0_BLOCK_SIZE;
    assert!(
        dst.len() >= n_values,
        "Q4_0 dst length {} too small for {} values",
        dst.len(),
        n_values
    );

    for block_idx in 0..n_blocks {
        let block = &src[block_idx * Q4_0_BYTES_PER_BLOCK..][..Q4_0_BYTES_PER_BLOCK];
        let out = &mut dst[block_idx * Q4_0_BLOCK_SIZE..][..Q4_0_BLOCK_SIZE];

        // First 2 bytes: f16 scale
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();

        // Next 16 bytes: 32 × 4-bit quants
        let qs = &block[2..18];

        for i in 0..16 {
            let byte = qs[i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            out[i] = d * lo as f32;
            out[i + 16] = d * hi as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_zeros() {
        // Scale = 0 → all outputs should be 0
        let mut block = [0u8; Q4_0_BYTES_PER_BLOCK];
        // d = 0.0 as f16
        let d_bytes = half::f16::from_f32(0.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        // quants: all 0x88 → nibbles are 8, so (8 - 8) * 0 = 0
        block[2..18].fill(0x88);

        let mut dst = [f32::NAN; 32];
        dequantize(&block, &mut dst);

        for &v in &dst {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q4_0_known_values() {
        let mut block = [0u8; Q4_0_BYTES_PER_BLOCK];

        // d = 1.0 as f16
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];

        // All quant nibbles = 8 → (8-8)*1.0 = 0.0
        block[2..18].fill(0x88);

        let mut dst = [f32::NAN; 32];
        dequantize(&block, &mut dst);

        for &v in &dst {
            assert!((v - 0.0).abs() < 1e-6, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_q4_0_scale_and_offset() {
        let mut block = [0u8; Q4_0_BYTES_PER_BLOCK];

        // d = 2.0
        let d_bytes = half::f16::from_f32(2.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];

        // First quant byte: low nibble = 0xF (15), high nibble = 0x0 (0)
        // → lo = (15 - 8) * 2.0 = 14.0
        // → hi = (0 - 8) * 2.0 = -16.0
        block[2] = 0x0F;
        // Rest: 0x88 → (8-8)*2 = 0
        block[3..18].fill(0x88);

        let mut dst = [0.0f32; 32];
        dequantize(&block, &mut dst);

        assert!((dst[0] - 14.0).abs() < 0.01, "dst[0] = {}", dst[0]);
        assert!((dst[16] - (-16.0)).abs() < 0.01, "dst[16] = {}", dst[16]);
        // Other values should be 0
        assert!((dst[1] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_q4_0_multiple_blocks() {
        // 2 blocks = 36 bytes → 64 values
        let mut src = vec![0u8; Q4_0_BYTES_PER_BLOCK * 2];

        // Block 0: d=1.0, all quants = 0x99 → lo=(9-8)*1=1, hi=(9-8)*1=1
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        src[0] = d_bytes[0];
        src[1] = d_bytes[1];
        src[2..18].fill(0x99);

        // Block 1: d=0.5, all quants = 0xAA → lo=(10-8)*0.5=1, hi=(10-8)*0.5=1
        let d2_bytes = half::f16::from_f32(0.5).to_le_bytes();
        src[18] = d2_bytes[0];
        src[19] = d2_bytes[1];
        src[20..36].fill(0xAA);

        let mut dst = vec![0.0f32; 64];
        dequantize(&src, &mut dst);

        // Block 0: all values should be ~1.0
        for &v in &dst[0..32] {
            assert!((v - 1.0).abs() < 0.01, "block 0: expected 1.0, got {v}");
        }
        // Block 1: all values should be ~1.0
        for &v in &dst[32..64] {
            assert!((v - 1.0).abs() < 0.01, "block 1: expected 1.0, got {v}");
        }
    }
}
