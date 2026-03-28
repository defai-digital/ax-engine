//! Dequantize Q6_K blocks to f32.
//!
//! Q6_K super-block layout (210 bytes -> 256 values):
//!   - 128 bytes: `ql[128]` -- lower 4 bits of each 6-bit quant
//!   - 64 bytes:  `qh[64]`  -- upper 2 bits of each 6-bit quant
//!   - 16 bytes:  `scales[16]` -- per-sub-block 8-bit signed scales
//!   - 2 bytes:   `d` -- f16 super-block scale
//!
//! 16 sub-blocks of 16 values each. Processed in groups of 128
//! (two groups for the full 256 values).
//!
//! Each 6-bit quant is reconstructed from ql and qh, then centered:
//!   q = (ql_nibble | (qh_bits << 4)) - 32
//!   value = d * scale * q

const Q6_K_BLOCK_SIZE: usize = 256;
const Q6_K_BYTES_PER_BLOCK: usize = 210;

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(Q6_K_BYTES_PER_BLOCK),
        "Q6_K src length {} is not a multiple of block size {}",
        src.len(),
        Q6_K_BYTES_PER_BLOCK
    );

    let n_blocks = src.len() / Q6_K_BYTES_PER_BLOCK;
    let n_values = n_blocks * Q6_K_BLOCK_SIZE;
    assert!(
        dst.len() >= n_values,
        "Q6_K dst length {} too small for {} values",
        dst.len(),
        n_values
    );

    for block_idx in 0..n_blocks {
        let block = &src[block_idx * Q6_K_BYTES_PER_BLOCK..][..Q6_K_BYTES_PER_BLOCK];
        let out = &mut dst[block_idx * Q6_K_BLOCK_SIZE..][..Q6_K_BLOCK_SIZE];

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();

        let mut ql_idx = 0;
        let mut qh_idx = 0;
        let mut sc_idx = 0;
        let mut out_idx = 0;

        // Two groups of 128 values
        for _group in 0..2 {
            // Each group: 32 iterations, producing 4 values each = 128 values
            for l in 0..32 {
                let is = l / 16; // which pair of sub-block scales to use

                // Reconstruct 6-bit quants from lower 4 bits (ql) and upper 2 bits (qh)
                let q1 = ((ql[ql_idx + l] & 0xF) | ((qh[qh_idx + l] & 3) << 4)) as i32 - 32;
                let q2 =
                    ((ql[ql_idx + l + 32] & 0xF) | (((qh[qh_idx + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_idx + l] >> 4) | (((qh[qh_idx + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 =
                    ((ql[ql_idx + l + 32] >> 4) | (((qh[qh_idx + l] >> 6) & 3) << 4)) as i32 - 32;

                let sc1 = scales[sc_idx + is] as i8 as f32;
                let sc2 = scales[sc_idx + is + 2] as i8 as f32;
                let sc3 = scales[sc_idx + is + 4] as i8 as f32;
                let sc4 = scales[sc_idx + is + 6] as i8 as f32;

                out[out_idx + l] = d * sc1 * q1 as f32;
                out[out_idx + l + 32] = d * sc2 * q2 as f32;
                out[out_idx + l + 64] = d * sc3 * q3 as f32;
                out[out_idx + l + 96] = d * sc4 * q4 as f32;
            }

            out_idx += 128;
            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q6_k_zeros() {
        // d=0 → all outputs 0
        let block = [0u8; Q6_K_BYTES_PER_BLOCK];
        let mut dst = [f32::NAN; Q6_K_BLOCK_SIZE];
        dequantize(&block, &mut dst);

        for (i, &v) in dst.iter().enumerate() {
            // When d=0, everything is 0 regardless of quant values
            assert!(v.abs() < 1e-6, "dst[{i}] = {v}, expected ~0.0");
        }
    }

    #[test]
    fn test_q6_k_centered_zero() {
        let mut block = [0u8; Q6_K_BYTES_PER_BLOCK];

        // d = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];

        // All scales = 1 (as signed i8)
        block[192..208].fill(1);

        // To get q=0 after centering (q - 32 = 0), we need raw q = 32 = 0b100000
        // ql nibble = 0 (lower 4 bits of 32), qh 2-bits = 2 (bits 4-5 of 32 = 10)
        // So ql bytes: 0x00 (both nibbles zero)
        block[0..128].fill(0x00);
        // qh bytes: need bits 0,1 = 2 (binary 10) for q1, bits 2,3 = 2 for q2, etc.
        // For all 4 quants from each qh byte: 0b10_10_10_10 = 0xAA
        block[128..192].fill(0xAA);

        let mut dst = [f32::NAN; Q6_K_BLOCK_SIZE];
        dequantize(&block, &mut dst);

        // All values should be d * scale * 0 = 0
        for (i, &v) in dst.iter().enumerate() {
            assert!(v.abs() < 0.01, "dst[{i}] = {v}, expected ~0.0");
        }
    }

    #[test]
    fn test_q6_k_known_value() {
        let mut block = [0u8; Q6_K_BYTES_PER_BLOCK];

        // d = 0.5
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];

        // All scales = 2
        block[192..208].fill(2);

        // For the first value (l=0, group=0):
        // q1 = (ql[0] & 0xF) | ((qh[0] & 3) << 4) - 32
        // Set ql[0] = 0x05 (low nibble = 5), qh[0] = 0x02 (bits 0-1 = 2)
        // q1 = 5 | (2 << 4) - 32 = 5 | 32 - 32 = 37 - 32 = 5
        // value = 0.5 * 2 * 5 = 5.0
        block[0] = 0x05;
        block[128] = 0x02;

        let mut dst = [0.0f32; Q6_K_BLOCK_SIZE];
        dequantize(&block, &mut dst);

        assert!(
            (dst[0] - 5.0).abs() < 0.01,
            "dst[0] = {}, expected 5.0",
            dst[0]
        );
    }

    #[test]
    fn test_q6_k_multiple_blocks() {
        let src = vec![0u8; Q6_K_BYTES_PER_BLOCK * 2];
        let mut dst = vec![0.0f32; Q6_K_BLOCK_SIZE * 2];
        // d=0 for both blocks → all zeros
        dequantize(&src, &mut dst);

        for (i, &v) in dst.iter().enumerate() {
            assert!(v.abs() < 1e-6, "dst[{i}] = {v}");
        }
    }
}
