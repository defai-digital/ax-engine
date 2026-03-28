//! Dequantize Q5_K blocks to f32.
//!
//! Q5_K uses the same packed scale/min representation as Q4_K, plus one extra
//! high-bit plane for the 5th quant bit.

const Q5_K_BLOCK_SIZE: usize = 256;
const Q5_K_BYTES_PER_BLOCK: usize = 176;

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(Q5_K_BYTES_PER_BLOCK),
        "Q5_K src length {} is not a multiple of block size {}",
        src.len(),
        Q5_K_BYTES_PER_BLOCK
    );

    let n_blocks = src.len() / Q5_K_BYTES_PER_BLOCK;
    let n_values = n_blocks * Q5_K_BLOCK_SIZE;
    assert!(
        dst.len() >= n_values,
        "Q5_K dst length {} too small for {} values",
        dst.len(),
        n_values
    );

    for block_idx in 0..n_blocks {
        let block = &src[block_idx * Q5_K_BYTES_PER_BLOCK..][..Q5_K_BYTES_PER_BLOCK];
        let out = &mut dst[block_idx * Q5_K_BLOCK_SIZE..][..Q5_K_BLOCK_SIZE];

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

            for i in 0..32 {
                let ql_byte = qs[qs_group * 32 + i];
                let ql = if high_nibble {
                    (ql_byte >> 4) & 0x0F
                } else {
                    ql_byte & 0x0F
                };
                let qh_bit = (qh[i] >> subblock) & 0x01;
                let q = (ql | (qh_bit << 4)) as f32;
                out[subblock * 32 + i] = d_scaled * q - d_min_scaled;
            }
        }
    }
}

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q5_k_zeros() {
        let block = [0u8; Q5_K_BYTES_PER_BLOCK];
        let mut dst = [f32::NAN; Q5_K_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        assert!(dst.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn test_q5_k_simple_value() {
        let mut block = [0u8; Q5_K_BYTES_PER_BLOCK];
        let one = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = one[0];
        block[1] = one[1];
        block[4] = 1;
        block[5] = 1;
        block[6] = 1;
        block[7] = 1;
        block[48..80].fill(0x55);
        let mut dst = [0.0f32; Q5_K_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        for &v in &dst[..64] {
            assert!((v - 5.0).abs() < 0.01, "expected 5.0, got {v}");
        }
    }

    #[test]
    fn test_q5_k_high_bit() {
        let mut block = [0u8; Q5_K_BYTES_PER_BLOCK];
        let one = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = one[0];
        block[1] = one[1];
        block[4] = 1;
        block[16..48].fill(0x01);
        let mut dst = [0.0f32; Q5_K_BLOCK_SIZE];
        dequantize(&block, &mut dst);
        assert!(
            (dst[0] - 16.0).abs() < 0.01,
            "expected 16.0, got {}",
            dst[0]
        );
    }
}
