//! Dequantize Q5_0 blocks to f32.
//!
//! Q5_0 block layout (22 bytes -> 32 values):
//!   - 2 bytes: f16 scale (d)
//!   - 4 bytes: packed high bits for 32 values (little-endian u32)
//!   - 16 bytes: packed low 4 bits (2 values per byte)
//!
//! Each 5-bit quant is reconstructed as:
//!   q = low_nibble | (high_bit << 4)
//! Then centered and scaled:
//!   output = (q - 16) * d

const Q5_0_BLOCK_SIZE: usize = 32;
const Q5_0_BYTES_PER_BLOCK: usize = 22;

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(Q5_0_BYTES_PER_BLOCK),
        "Q5_0 src length {} is not a multiple of block size {}",
        src.len(),
        Q5_0_BYTES_PER_BLOCK
    );

    let n_blocks = src.len() / Q5_0_BYTES_PER_BLOCK;
    let n_values = n_blocks * Q5_0_BLOCK_SIZE;
    assert!(
        dst.len() >= n_values,
        "Q5_0 dst length {} too small for {} values",
        dst.len(),
        n_values
    );

    for block_idx in 0..n_blocks {
        let block = &src[block_idx * Q5_0_BYTES_PER_BLOCK..][..Q5_0_BYTES_PER_BLOCK];
        let out = &mut dst[block_idx * Q5_0_BLOCK_SIZE..][..Q5_0_BLOCK_SIZE];

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];

        for i in 0..16 {
            let byte = qs[i];
            let lo = byte & 0x0F;
            let hi = byte >> 4;

            let q0 = lo | (((qh >> i) as u8) & 0x01) << 4;
            let q1 = hi | (((qh >> (i + 16)) as u8) & 0x01) << 4;

            out[i] = d * (q0 as i32 - 16) as f32;
            out[i + 16] = d * (q1 as i32 - 16) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q5_0_zero_scale() {
        let block = [0u8; Q5_0_BYTES_PER_BLOCK];
        let mut dst = [f32::NAN; Q5_0_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q5_0_centered_zero() {
        let mut block = [0u8; Q5_0_BYTES_PER_BLOCK];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2..6].fill(0xFF);
        block[6..22].fill(0x00);

        let mut dst = [f32::NAN; Q5_0_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            assert!(v.abs() < 1e-6, "dst[{i}] = {v}, expected 0.0");
        }
    }

    #[test]
    fn test_q5_0_known_values() {
        let mut block = [0u8; Q5_0_BYTES_PER_BLOCK];
        let d_bytes = half::f16::from_f32(2.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];

        // Element 0: low nibble 15, high bit 0 -> q=15 -> (15-16)*2 = -2
        // Element 16: high nibble 0, high bit 1 -> q=16 -> 0
        block[2] = 0x00;
        block[3] = 0x00;
        block[4] = 0x01;
        block[5] = 0x00;
        block[6] = 0x0F;

        let mut dst = [0.0f32; Q5_0_BLOCK_SIZE];
        dequantize(&block, &mut dst);

        assert!((dst[0] - (-2.0)).abs() < 0.01, "dst[0] = {}", dst[0]);
        assert!(dst[16].abs() < 0.01, "dst[16] = {}", dst[16]);
    }

    #[test]
    fn test_q5_0_multiple_blocks() {
        let mut src = vec![0u8; Q5_0_BYTES_PER_BLOCK * 2];

        let one = half::f16::from_f32(1.0).to_le_bytes();
        src[0] = one[0];
        src[1] = one[1];
        src[2..6].fill(0x00);
        src[6..22].fill(0xFF);

        src[22] = one[0];
        src[23] = one[1];
        src[24..28].fill(0xFF);
        src[28..44].fill(0x00);

        let mut dst = vec![0.0f32; Q5_0_BLOCK_SIZE * 2];
        dequantize(&src, &mut dst);

        for &v in &dst[..32] {
            assert!((v - (-1.0)).abs() < 0.01, "expected -1.0, got {v}");
        }
        for &v in &dst[32..64] {
            assert!(v.abs() < 0.01, "expected 0.0, got {v}");
        }
    }
}
